import sys
import os
import random
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

from omegaconf import OmegaConf
import habitat
from habitat import Env
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from data.utils import measures  # noqa: F401
from scripts.client import StereoVLNClient

_to_tensor = transforms.ToTensor()
IMAGE_SIZE = (448, 448)


def depth_to_colormap(depth_tensor):
    """depth_tensor: [H, W] float, any range -> BGR colormap (INFERNO)."""
    d = depth_tensor.float().cpu().numpy()
    mn, mx = d.min(), d.max()
    if mx - mn < 1e-6:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    else:
        d_norm = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)


def build_config(config_path, split, data_path=None, scenes_dir=None):
    config = get_habitat_config(config_path)
    with habitat.config.read_write(config):
        OmegaConf.set_struct(config, False)
        config.habitat.dataset.split = split
        if data_path:
            config.habitat.dataset.data_path = data_path
        if scenes_dir:
            config.habitat.dataset.scenes_dir = scenes_dir
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
            "type": "HabitatSimRGBSensor", "uuid": "rgb_left",
            "width": 448, "height": 448, "hfov": 79,
            "position": [-0.05, 1.25, 0.0], "orientation": [0.0, 0.0, 0.0]
        })
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
            "type": "HabitatSimRGBSensor", "uuid": "rgb_right",
            "width": 448, "height": 448, "hfov": 79,
            "position": [0.05, 1.25, 0.0], "orientation": [0.0, 0.0, 0.0]
        })
        config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3, map_resolution=1024,
                draw_source=True, draw_border=True,
                draw_shortest_path=True, draw_view_points=True,
                draw_goal_positions=True, draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        OmegaConf.set_struct(config, True)
    return config


def sample_episodes(env, num_samples):
    scene_ep = {}
    for ep in env.episodes:
        if hasattr(ep, 'instruction') and not getattr(ep.instruction, 'language', 'en').startswith('en'):
            continue
        if ep.scene_id not in scene_ep:
            scene_ep[ep.scene_id] = ep
    episodes = list(scene_ep.values())
    random.shuffle(episodes)
    return episodes[:num_samples]


def preprocess_rgb(rgb_np):
    """numpy HxWx3 uint8 -> [1, 3, 448, 448] float tensor 0~255"""
    from PIL import Image
    img = Image.fromarray(rgb_np).convert('RGB').resize(IMAGE_SIZE, Image.BILINEAR)
    return (_to_tensor(img) * 255.0).unsqueeze(0)


def run_episode(env, episode, model, max_steps, output_dir, tag, history_num, execute_steps):
    env.current_episode = episode
    obs = env.reset()

    instruction = episode.instruction.instruction_text if hasattr(episode, 'instruction') else ""

    stereo_frames = []
    depth_frames = []

    left_history, right_history = [], []
    executed_actions = []
    action_seq = []
    step = 0

    actions2idx = {'stop here': 0, 'move forward': 1, 'turn left': 2, 'turn right': 3}
    idx2action = {v: k for k, v in actions2idx.items()}

    while not env.episode_over and step < max_steps:
        left_t = preprocess_rgb(obs['rgb_left'][:, :, :3])   # [1,3,448,448]
        right_t = preprocess_rgb(obs['rgb_right'][:, :, :3])

        # Build history tensors
        def build_hist(frames):
            n = len(frames)
            if n < history_num:
                pad = [torch.zeros(3, *IMAGE_SIZE)] * (history_num - n)
                frames = pad + frames
            elif n > history_num:
                step_s = n / history_num
                frames = [frames[int(i * step_s)] for i in range(history_num)]
            return torch.stack(frames, dim=0)  # [history_num, 3, 448, 448]

        if len(action_seq) == 0:
            lh = build_hist(left_history).unsqueeze(0).to(torch.bfloat16)   # [1,H,3,448,448]
            rh = build_hist(right_history).unsqueeze(0).to(torch.bfloat16)
            lc = left_t.unsqueeze(0).to(torch.bfloat16)   # [1,1,3,448,448]
            rc = right_t.unsqueeze(0).to(torch.bfloat16)

            hist_str = "This is the initial timestep, so no previous action sequence is available." \
                if step == 0 else ",".join(idx2action[a] for a in executed_actions)

            outputs = model.inference(
                instruction=[instruction],
                history_action=[hist_str],
                left_current_frame=lc,
                right_current_frame=rc,
                left_history_video=lh,
                right_history_video=rh,
                depth_iters=8,
                max_new_tokens=24,
                temperature=0.0,
                top_p=1.0,
                output_point=False,
                output_depth=True,
            )

            # Parse actions
            parts = outputs['action'][0].strip().split(',')
            action_seq = []
            for p in parts:
                p = p.strip().lower().strip('.,;:!?')
                if p in actions2idx:
                    action_seq.append(actions2idx[p])
            if execute_steps > 0:
                action_seq = action_seq[:execute_steps]
            if not action_seq:
                action_seq = [0]

            # Depth from model: outputs['depth'] shape varies, normalize and colorize
            depth_out = outputs.get('depth', None)
            if depth_out is not None:
                if isinstance(depth_out, torch.Tensor):
                    d = depth_out.squeeze()  # [H, W] or [1, H, W]
                    if d.dim() == 3:
                        d = d[0]
                    depth_frame = depth_to_colormap(d)
                else:
                    depth_frame = np.zeros((448, 448, 3), dtype=np.uint8)
            else:
                depth_frame = np.zeros((448, 448, 3), dtype=np.uint8)

            # Stereo frame: left | right
            left_np = (left_t.squeeze(0).permute(1, 2, 0).numpy()).astype(np.uint8)
            right_np = (right_t.squeeze(0).permute(1, 2, 0).numpy()).astype(np.uint8)
            stereo_frame = np.concatenate([left_np, right_np], axis=1)

            stereo_frames.append(stereo_frame)
            depth_frames.append(depth_frame)

        action = action_seq.pop(0)
        executed_actions.append(action)
        left_history.append(left_t.squeeze(0))
        right_history.append(right_t.squeeze(0))

        obs = env.step(action)
        step += 1

    os.makedirs(output_dir, exist_ok=True)

    if stereo_frames:
        h, w2 = stereo_frames[0].shape[:2]
        stereo_path = os.path.join(output_dir, f"{tag}_stereo.mp4")
        vw = cv2.VideoWriter(stereo_path, cv2.VideoWriter_fourcc(*'mp4v'), 6, (w2, h))
        for f in stereo_frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"Saved: {stereo_path}")

    if depth_frames:
        dh, dw = depth_frames[0].shape[:2]
        depth_path = os.path.join(output_dir, f"{tag}_depth.mp4")
        vw2 = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'mp4v'), 6, (dw, dh))
        for f in depth_frames:
            vw2.write(f)
        vw2.release()
        print(f"Saved: {depth_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://localhost:7200")
    parser.add_argument("--r2r_config", type=str,
                        default="StereoVLN_InternVL_3_5_2_B/config/eval_r2r.yaml")
    parser.add_argument("--rxr_config", type=str,
                        default="StereoVLN_InternVL_3_5_2_B/config/eval_rxr.yaml")
    parser.add_argument("--split", type=str, default="val_unseen")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--execute_steps", type=int, default=4)
    parser.add_argument("--history_num", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./vis_output")
    parser.add_argument("--datasets", type=str, default="r2r,rxr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r2r_data_path", type=str, default=None)
    parser.add_argument("--rxr_data_path", type=str, default=None)
    parser.add_argument("--scenes_dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = StereoVLNClient(server_url=args.server_url)

    dataset_configs = {
        "r2r": (args.r2r_config, args.r2r_data_path),
        "rxr": (args.rxr_config, args.rxr_data_path),
    }

    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        if dataset not in dataset_configs:
            continue

        config_path, data_path = dataset_configs[dataset]
        print(f"\n=== Dataset: {dataset.upper()} ===")

        config = build_config(config_path, args.split, data_path, args.scenes_dir)
        env = Env(config=config)

        episodes = sample_episodes(env, args.num_samples)
        print(f"Sampled {len(episodes)} episodes from {len(set(ep.scene_id for ep in episodes))} scenes")

        for i, ep in enumerate(episodes):
            scene_id = ep.scene_id.split('/')[-2]
            ep_id = ep.episode_id
            tag = f"{dataset}_{scene_id}_{ep_id}"
            print(f"\n[{i+1}/{len(episodes)}] scene={scene_id} episode={ep_id}")
            run_episode(env, ep, model, args.max_steps, args.output_dir, tag,
                        args.history_num, args.execute_steps)

        env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()



# python vis.py \
#     --r2r_config /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/config/eval_r2r.yaml \
#     --rxr_config /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/config/eval_rxr.yaml \
#     --split val_unseen \
#     --num_samples 10 \
#     --max_steps 500 \
#     --output_dir /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/vis \
#     --datasets r2r,rxr \
#     --r2r_data_path /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/data/task/r2r/val_unseen/val_unseen.json.gz \
#     --rxr_data_path /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/data/task/rxr/val_unseen/val_unseen_guide.json.gz \
#     --scenes_dir /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/data/scene/



