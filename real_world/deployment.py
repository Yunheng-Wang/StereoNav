"""
deployment.py — Runs on the host machine, connecting the G1 robot and the remote inference server.

Flow:
  1. Fetch stereo frames from the ZED camera server.
  2. Build a history buffer of past frames.
  3. Send frames + instruction to the StereoNav inference server.
  4. Parse the returned action sequence and forward it to the robot action server.
  5. Repeat until a stop action is received or max_steps is reached.
"""

import sys
import os
import io
import json
import argparse
import requests
import torch
import numpy as np
import cv2
from datetime import datetime
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

transform = transforms.ToTensor()

ACTIONS2IDX = OrderedDict({
    'stop here': 0,
    'move forward': 1,
    'turn left': 2,
    'turn right': 3,
})

IDX2ACTION = {v: k for k, v in ACTIONS2IDX.items()}


def fetch_image(url, image_size) -> torch.Tensor:
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()

    arr = np.frombuffer(resp.content, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise RuntimeError(
            f"Failed to decode image from {url}. "
            f"Response size={len(resp.content)} bytes, "
            f"content starts with={resp.content[:80]!r}"
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize(image_size, Image.BILINEAR)

    return (transform(img) * 255.0).unsqueeze(0)


def build_history(frames, history_num, image_size):
    n = len(frames)

    if n == 0:
        return torch.zeros(history_num, 3, image_size[0], image_size[1])

    if n < history_num:
        pad = [torch.zeros(3, image_size[0], image_size[1])] * (history_num - n)
        frames = pad + frames

    elif n > history_num:
        # uniformly sample history frames, always including the last one
        indices = np.linspace(0, n - 1, history_num).astype(int)
        frames = [frames[i] for i in indices]

    return torch.stack(frames, dim=0)


def parse_actions(text):
    actions = []

    for part in text.strip().split(','):
        p = part.strip().lower().strip('.,;:!?')
        if p in ACTIONS2IDX:
            actions.append(ACTIONS2IDX[p])

    return actions


def set_zed_target(zed_base, target):
    resp = requests.post(
        f"{zed_base}/target",
        json={"x": target[0], "y": target[1]},
        timeout=5,
    )
    resp.raise_for_status()


def send_actions(action_server_url, actions):
    """Send a list of actions to the robot action server."""
    resp = requests.post(
        f"{action_server_url}/execute",
        json={"actions": actions},
        timeout=60,
    )
    resp.raise_for_status()


def run_inference(server_url, instruction, history_action,
                  left_cur, right_cur, left_hist, right_hist):
    buf = io.BytesIO()

    torch.save({
        'instruction': [instruction],
        'history_action': [history_action],
        'left_current_frame': left_cur.unsqueeze(0),
        'right_current_frame': right_cur.unsqueeze(0),
        'left_history_video': left_hist.unsqueeze(0),
        'right_history_video': right_hist.unsqueeze(0),
        'max_new_tokens': 24,
        'depth_iters': 8,
        'temperature': 0.0,
        'top_p': 1.0,
        'output_point': False,
        'output_depth': False,
    }, buf)

    resp = requests.post(
        f"{server_url}/inference",
        data=buf.getvalue(),
        headers={'Content-Type': 'application/octet-stream'},
        timeout=120,
    )
    resp.raise_for_status()

    return torch.load(io.BytesIO(resp.content), weights_only=False)


def tensor_to_uint8_image(tensor_img):
    """
    tensor_img:
        shape [1, 3, H, W] or [3, H, W]
        value range roughly 0~255

    return:
        RGB uint8 ndarray [H, W, 3]
    """
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)

    img = tensor_img.detach().cpu().numpy()   # [3, H, W]
    img = np.transpose(img, (1, 2, 0))        # [H, W, 3]
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def save_run_config(results_dir, args, timestamp):
    """Save the deployment configuration for this run."""
    config = {
        "run_timestamp": timestamp,
        "raw_instruction": args.raw_instruction,
        "instruction": args.instruction,
        "target": args.target,
        "zed_ip": args.zed_ip,
        "action_url": args.action_url,
        "server_url": args.server_url,
        "image_size": args.image_size,
        "history_num": args.history_num,
        "execute_steps": args.execute_steps,
        "max_steps": args.max_steps,
    }

    config_path = os.path.join(results_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Run config saved to: {config_path}")


def save_step_result(results_dir, step_id, left_cur, right_cur, generated, actions):
    """Save the input stereo frames, raw model output, and parsed actions for each step."""
    step_name = f"step_{step_id:04d}"

    left_rgb = tensor_to_uint8_image(left_cur)
    right_rgb = tensor_to_uint8_image(right_cur)

    left_path = os.path.join(results_dir, f"{step_name}_left.jpg")
    right_path = os.path.join(results_dir, f"{step_name}_right.jpg")
    action_path = os.path.join(results_dir, f"{step_name}_action.txt")

    Image.fromarray(left_rgb).save(left_path)
    Image.fromarray(right_rgb).save(right_path)

    action_names = [IDX2ACTION[a] for a in actions]

    with open(action_path, "w", encoding="utf-8") as f:
        f.write(f"step_id: {step_id}\n")
        f.write(f"model_output: {generated}\n")
        f.write(f"parsed_actions: {actions}\n")
        f.write(f"parsed_action_names: {action_names}\n")

    print(f"  Saved input images/action to: {results_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--raw_instruction", type=str, default="")
    parser.add_argument("--target", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--zed_ip", type=str, default="192.168.123.164")
    parser.add_argument("--action_url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:7200")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--history_num", type=int, default=8)
    parser.add_argument("--execute_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500)

    args = parser.parse_args()

    image_size = (args.image_size, args.image_size)

    zed_base = f"http://{args.zed_ip}:8000"
    action_base = args.action_url

    # results directory: saved alongside deployment.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(script_dir, "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    save_run_config(results_dir, args, timestamp)

    print(f"Instruction: {args.instruction}")
    print(f"Target: {args.target}")
    print(f"ZED server: {zed_base}")
    print(f"Action server: {action_base}")
    print(f"Inference server: {args.server_url}")
    print(f"Results dir: {results_dir}")

    print(f"Setting ZED target to {args.target}...")
    set_zed_target(zed_base, args.target)

    left_history, right_history = [], []
    executed_actions = []
    action_seq = []
    step_id = 0

    while step_id < args.max_steps:
        left_cur = fetch_image(f"{zed_base}/left.jpg", image_size)
        right_cur = fetch_image(f"{zed_base}/right.jpg", image_size)

        if len(action_seq) == 0:
            left_hist = build_history(left_history, args.history_num, image_size)
            right_hist = build_history(right_history, args.history_num, image_size)

            if step_id == 0:
                history_action = "This is the initial timestep, so no previous action sequence is available."
            else:
                history_action = ",".join(IDX2ACTION[a] for a in executed_actions)

            print(f"Step {step_id}: calling inference...")

            result = run_inference(
                args.server_url,
                args.instruction,
                history_action,
                left_cur,
                right_cur,
                left_hist,
                right_hist,
            )

            generated = result['action'][0]
            print(f"  Model output: {generated}")

            action_seq = parse_actions(generated)[:args.execute_steps]

            if not action_seq:
                action_seq = [0]

            print(f"  Actions: {[IDX2ACTION[a] for a in action_seq]}")

            # save the input stereo frames and model output for this inference step
            save_step_result(
                results_dir=results_dir,
                step_id=step_id,
                left_cur=left_cur,
                right_cur=right_cur,
                generated=generated,
                actions=action_seq.copy(),
            )

            # dispatch the full action sequence to the robot action server
            send_actions(action_base, action_seq.copy())

        action = action_seq.pop(0)
        executed_actions.append(action)

        left_history.append(left_cur.squeeze(0))
        right_history.append(right_cur.squeeze(0))

        step_id += 1

        if action == 0:
            print("Stop action received. Navigation complete.")
            break

    print(f"Done. Total steps: {step_id}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()