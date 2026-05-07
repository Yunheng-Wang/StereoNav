"""WorldModel Server

NOTE: This module is an experimental attempt at automated goal-point prediction
for vision-and-language navigation. The WorldModel takes panoramic observations
and a language instruction, then predicts a target waypoint in the scene.
This line of work is considered promising future research and is not yet part
of the main StereoVLN evaluation pipeline.

Run:
    python scripts/server_world.py --model_checkpoint /path/to/checkpoint --port 5000 --gpu 0

Multi-GPU: launch one instance per GPU with different --gpu and --port:
    python scripts/server_world.py --model_checkpoint ... --port 5000 --gpu 0 &
    python scripts/server_world.py --model_checkpoint ... --port 5001 --gpu 1 &

Endpoints:
    GET  /health     - liveness check
    POST /inference  - predict goal point from panoramic input (torch-serialized)
"""
import os
import sys
import io
import json
import torch
import argparse
import numpy as np
from flask import Flask, request, jsonify, Response

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.WorldModel import WorldModel
from model.WorldModelConfig import WorldModelConfig

app = Flask(__name__)
model = None
device = None


def load_model(args):
    """Load WorldModel from checkpoint."""
    global device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"Loading WorldModel on {device}...")
    config = WorldModelConfig(
        vlm_checkpoint_path=os.path.join(PROJECT_ROOT, "checkpoints/Qwen3.5-4B")
    )
    model = WorldModel(config)

    # Load checkpoint
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        print(f"Loading checkpoint from {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint + "/pytorch_model_0.bin", map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[WARNING] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        if not missing_keys and not unexpected_keys:
            print("[INFO] All keys matched successfully!")
        del checkpoint, state_dict
    else:
        print(f"Warning: Checkpoint not found at {args.model_checkpoint}")

    model.requires_grad_(False)
    model.eval()
    model.to(device)
    torch.cuda.empty_cache()
    return model


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/inference', methods=['POST'])
def inference():
    try:
        print(f"[GPU {device}] Received inference request")
        # Decode binary payload from client
        data = torch.load(io.BytesIO(request.data), weights_only=False)

        instruction = data['instruction']
        panoramic = data['panoramic']

        print(f"Panoramic dtype before conversion: {panoramic.dtype}")
        print(f"Panoramic shape: {panoramic.shape}")
        print(f"Panoramic device: {panoramic.device}")
        print(f"Panoramic min/max: {panoramic.min()}/{panoramic.max()}")

        # Ensure proper dtype before moving to GPU
        if panoramic.dtype != torch.float32:
            panoramic = panoramic.float()

        panoramic = panoramic.to(device=device)

        print(f"Instruction: {instruction}")
        print(f"Panoramic shape after conversion: {panoramic.shape}")

        with torch.no_grad():
            outputs = model.inference(
                instruction=instruction,
                Panoramic=panoramic
            )

        # Pack response as binary
        response_data = {
            'responses': outputs['responses'],
        }
        buf = io.BytesIO()
        torch.save(response_data, buf)
        return Response(buf.getvalue(), mimetype='application/octet-stream')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    args = parser.parse_args()

    print(f"Loading model from {args.model_checkpoint} on GPU {args.gpu}...")
    model = load_model(args)
    print(f"Model loaded on {device}")

    print(f"Starting server on {args.host}:{args.port} (GPU {args.gpu})")
    app.run(host=args.host, port=args.port, threaded=False)
