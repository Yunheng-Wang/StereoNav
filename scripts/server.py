"""StereoVLN Model Server

Serves a StereoVLN model over HTTP for distributed evaluation.
Each instance handles inference requests from one evaluation worker.

Run:
    python scripts/server.py --checkpoint_path /path/to/checkpoint --port 5000 --gpu 0

Endpoints:
    GET  /health     - liveness check
    POST /inference  - run model inference (torch-serialized request/response)
"""
import os
import sys
import io
import torch
import argparse
import numpy as np
from flask import Flask, request, jsonify, Response
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.StereoVLNConfig import StereoVLNConfig
from model.StereoVLN import StereoVLN

app = Flask(__name__)
model = None
device = None


def load_model(args):
    """Load StereoVLN model from checkpoint onto the specified GPU."""
    global device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(os.path.join(args.checkpoint_path, "config.json"))
    img_size = config.main.image_size
    # Compute focal length from image size and fixed 79-degree HFOV
    fx = img_size / (2.0 * np.tan(np.deg2rad(79.0) / 2.0))
    camera_k = torch.tensor([[fx, 0, img_size/2.0], [0, fx, img_size/2.0], [0, 0, 1]], dtype=torch.float32)
    camera_baseline = args.camera_baseline

    # Checkpoint sub-paths follow the expected directory layout under PROJECT_ROOT
    model_config = StereoVLNConfig(
        image_size=(img_size, img_size),
        dtype=torch.bfloat16 if config.main.dtype == "bf16" else torch.float16,
        dim=config.model.dim,
        max_tokens=config.model.vlm.max_tokens,
        vlm_checkpoints_path=os.path.join(PROJECT_ROOT, "checkpoints/InternVL3_5-2B"),
        camera_k=camera_k,
        camera_baseline=camera_baseline,
        foundationstereo_checkpoints_path=os.path.join(PROJECT_ROOT, "checkpoints/foundationstereo/23-51-11"),
        foundationstereo_edgenext_path=os.path.join(PROJECT_ROOT, "checkpoints/edgenext_small/model.safetensors"),
        dino=os.path.join(PROJECT_ROOT, "checkpoints/dinov2_large/model.safetensors"),
        mlp_ratio=config.model.pointhead.mlp_ratio,
        dropout=config.model.pointhead.dropout,
        prediction_steps=config.main.prediction_steps,
        depth_token_weight=config.model.depth_token_weight,
        dino_token_weight=config.model.dino_token_weight
    )
    model = StereoVLN(model_config)

    checkpoint_dir = args.checkpoint_path
    bin_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.bin')]
    if bin_files:
        model_files = [f for f in bin_files if f.startswith('pytorch_model')]
        checkpoint_file = os.path.join(checkpoint_dir, model_files[0] if model_files else bin_files[0])
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[WARNING] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        del checkpoint, state_dict
    else:
        print(f"Warning: No .bin file found in {checkpoint_dir}")

    model.requires_grad_(False)
    model.eval()
    model.to(device)
    torch.cuda.empty_cache()
    return model


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/inference', methods=['POST'])
def inference():
    try:
        data = torch.load(io.BytesIO(request.data), weights_only=False)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with torch.no_grad():
            outputs = model.inference(
                instruction=data['instruction'],
                history_action=data['history_action'],
                left_current_frame=data['left_current_frame'].to(device=device, dtype=torch.bfloat16),
                right_current_frame=data['right_current_frame'].to(device=device, dtype=torch.bfloat16),
                left_history_video=data['left_history_video'].to(device=device, dtype=torch.bfloat16),
                right_history_video=data['right_history_video'].to(device=device, dtype=torch.bfloat16),
                max_new_tokens=data.get('max_new_tokens', 24),
                depth_iters=data.get('depth_iters', 16),
                temperature=data.get('temperature', 0.0),
                top_p=data.get('top_p', 1.0),
                output_point=data.get('output_point', False),
                output_depth=data.get('output_depth', False),
            )
        response_data = {
            'action': outputs['action'],
            'left_point': outputs['left_point'].cpu() if outputs.get('left_point') is not None else None,
            'right_point': outputs['right_point'].cpu() if outputs.get('right_point') is not None else None,
            'depth': outputs['depth'].cpu() if outputs.get('depth') is not None else None,
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
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--camera_baseline', type=float, default=0.1, help='stereo camera baseline distance in meters')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint_path} on GPU {args.gpu}...")
    model = load_model(args)
    print(f"Model loaded on {device}")
    print(f"Starting server on {args.host}:{args.port} (GPU {args.gpu})")
    app.run(host=args.host, port=args.port, threaded=False)
