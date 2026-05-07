import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import quaternion
import habitat
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
import torchvision.transforms as transforms

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
from utils.dist import get_rank, get_world_size, init_distributed_mode
from data.utils import rxr_dataset  # noqa: F401
from data.utils import measures     # noqa: F401


# ── constants (same as gen_trajectory.py) ──────────────────────────────────
IMG_W, IMG_H, HFOV = 448, 448, 79
CAM_LEFT  = np.array([-0.05, 1.25, 0.0])
CAM_RIGHT = np.array([ 0.05, 1.25, 0.0])
MIDGOAL_RADIUS = 0.5
GOAL_RADIUS    = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD         = 0.98
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.90
ACTIONS_MAP = {0: 'stop here', 1: 'move forward', 2: 'turn left', 3: 'turn right'}
ACTIONS_INV = {v: k for k, v in ACTIONS_MAP.items()}


def get_camera_intrinsics(w, h, hfov):
    fx = w / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))
    return np.array([[fx, 0, w/2.0], [0, fx, h/2.0], [0, 0, 1]])


K_GLOBAL = get_camera_intrinsics(IMG_W, IMG_H, HFOV)


def project_3d_to_2d(world_pt, agent_pos, agent_rot, cam_offset, K, w, h):
    R = quaternion.as_rotation_matrix(agent_rot)
    pt_cam = R.T @ (world_pt - (agent_pos + R @ cam_offset))
    pt_cam[2] = -pt_cam[2]
    cam_x = pt_cam[0]
    if pt_cam[2] <= 0:
        return None, None, cam_x
    p = K @ pt_cam
    return p[0]/p[2], h - p[1]/p[2], cam_x


def clip_to_image_bounds(u, v, w, h, cam_x):
    vc = h / 2.0
    if u is None or v is None:
        return (0.0, vc) if cam_x < 0 else (w - 1.0, vc)
    um, vm = w * 0.2, h * 0.2
    if u < -um or u > w - 1 + um or v < -vm or v > h - 1 + vm:
        return (0.0, vc) if cam_x < 0 else (w - 1.0, vc)
    return float(np.clip(u, 0, w - 1)), float(np.clip(v, 0, h - 1))


def project_goal(env, goal_pos):
    st = env.sim.get_agent_state()
    ap, ar = np.array(st.position), st.rotation
    gp = goal_pos + np.array([0.0, 1.25, 0.0])
    ul, vl, cxl = project_3d_to_2d(gp, ap, ar, CAM_LEFT,  K_GLOBAL, IMG_W, IMG_H)
    ur, vr, cxr = project_3d_to_2d(gp, ap, ar, CAM_RIGHT, K_GLOBAL, IMG_W, IMG_H)
    return (clip_to_image_bounds(ul, vl, IMG_W, IMG_H, cxl),
            clip_to_image_bounds(ur, vr, IMG_W, IMG_H, cxr))


def draw_point(rgb_np, point, radius=5):
    img = Image.fromarray(rgb_np).convert("RGB")
    u, v_tl = float(point[0]), IMG_H - float(point[1])
    draw = ImageDraw.Draw(img)
    draw.ellipse([(u-radius, v_tl-radius), (u+radius, v_tl+radius)], fill='red', outline='red')
    return np.array(img)


def preprocess(rgb_np):
    """numpy HxWx3 -> [3,H,W] float tensor 0~255"""
    t = transforms.ToTensor()(Image.fromarray(rgb_np).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR))
    return t * 255.0


def build_history_video(frames, history_num):
    n = len(frames)
    if n < history_num:
        frames = [torch.zeros(3, IMG_H, IMG_W)] * (history_num - n) + frames
    elif n > history_num:
        frames = [frames[int(i * n / history_num)] for i in range(history_num)]
    return torch.stack(frames, dim=0)  # [history_num, 3, H, W]


def parse_actions(text):
    text = text.lower()
    seq = [idx for name, idx in ACTIONS_INV.items() if name in text]
    return seq if seq else [0]


# ── DAggerCollector ─────────────────────────────────────────────────────────

class DAggerCollector:
    def __init__(self, args, rank, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.dataset = args.dagger_dataset.lower()
        self.output_path = args.dagger_output_path
        self.config = get_habitat_config(args.habitat_config_path)

        with read_write(self.config):
            OmegaConf.set_struct(self.config, False)
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
                "type": "HabitatSimRGBSensor", "uuid": "rgb_left",
                "width": IMG_W, "height": IMG_H, "hfov": HFOV,
                "position": [-0.05, 1.25, 0.0], "orientation": [0.0, 0.0, 0.0],
            })
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
                "type": "HabitatSimRGBSensor", "uuid": "rgb_right",
                "width": IMG_W, "height": IMG_H, "hfov": HFOV,
                "position": [0.05, 1.25, 0.0], "orientation": [0.0, 0.0, 0.0],
            })
            self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor_left = OmegaConf.create({
                "type": "HabitatSimDepthSensor", "uuid": "depth_left",
                "width": IMG_W, "height": IMG_H, "hfov": HFOV,
                "min_depth": 0.0, "max_depth": 10.0, "normalize_depth": True,
                "position": [-0.05, 1.25, 0.0], "orientation": [0.0, 0.0, 0.0],
            })
            self.config.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3, map_resolution=1024,
                    draw_source=True, draw_border=True,
                    draw_shortest_path=True, draw_view_points=True,
                    draw_goal_positions=True, draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            })
            OmegaConf.set_struct(self.config, True)

    def config_env(self):
        with read_write(self.config):
            self.config.habitat.dataset.data_path = self.args.dagger_data_path
        return habitat.Env(config=self.config)

    def generate(self, env, model, episode, force_expert, data_it):
        beta = 0.0 if self.args.dagger_p == 0 else self.args.dagger_p ** data_it

        scene_id   = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = episode.trajectory_id
        ref_path   = episode.reference_path
        goal_pos   = np.array(episode.goals[0].position)

        if self.dataset == 'rxr':
            instr_obj = episode.instruction
            if hasattr(instr_obj, 'language') and not instr_obj.language.startswith('en'):
                return None
            instructions = instr_obj.instruction_text
        else:
            instructions = episode.instruction.instruction_text

        env.current_episode = episode
        env.current_episode.goals[0].radius = MIDGOAL_RADIUS
        obs = env.reset()

        agent = ShortestPathFollower(sim=env.sim, goal_radius=MIDGOAL_RADIUS, return_one_hot=False)

        scene_dir     = os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}")
        rgb_left_dir  = os.path.join(scene_dir, "rgb_left")
        rgb_right_dir = os.path.join(scene_dir, "rgb_right")
        depth_dir     = os.path.join(scene_dir, "depth_left")

        rgb_left_buf, rgb_right_buf, depth_buf = [], [], []
        label_points_data = []
        actions = [-1]
        next_wp = 1
        step_id = 0
        action_seq = []
        left_hist, right_hist, executed_actions = [], [], []
        accumulated_error = 0
        model_success = True
        from_expert = force_expert
        left_expert_num = 0
        force_end = False
        ref_actions_len = max(1, len(ref_path) - 1)
        metrics = None

        while not env.episode_over:
            rgb_l  = obs["rgb_left"]
            rgb_r  = obs["rgb_right"]
            dep_l  = obs["depth_left"]

            lp, rp = project_goal(env, goal_pos)
            rgb_l_d = draw_point(rgb_l, lp)
            rgb_r_d = draw_point(rgb_r, rp)

            rgb_left_buf.append((rgb_l_d,  os.path.join(rgb_left_dir,  f"{step_id:03d}.jpg")))
            rgb_right_buf.append((rgb_r_d, os.path.join(rgb_right_dir, f"{step_id:03d}.jpg")))
            depth_buf.append((dep_l,       os.path.join(depth_dir,     f"{step_id:03d}.png")))
            label_points_data.append({
                "frame_id": step_id,
                "label_left_point":  [lp[0], lp[1]],
                "label_right_point": [rp[0], rp[1]],
            })

            # decide action source
            if len(action_seq) == 0 and left_expert_num == 0:
                from_expert = force_expert or (random.random() < beta)

            if len(action_seq) == 0:
                if left_expert_num > 0:
                    action_seq = [agent.get_next_action(ref_path[next_wp])]
                    left_expert_num -= 1
                elif from_expert:
                    action_seq = [agent.get_next_action(ref_path[next_wp])]
                    left_expert_num = self.args.num_future_steps - 1
                else:
                    lhv = build_history_video(left_hist,  self.args.history_num).unsqueeze(0).to(torch.bfloat16)
                    rhv = build_history_video(right_hist, self.args.history_num).unsqueeze(0).to(torch.bfloat16)
                    lfc = preprocess(rgb_l_d).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
                    rfc = preprocess(rgb_r_d).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
                    hist_str = ("This is the initial timestep, so no previous action sequence is available."
                                if step_id == 0 else ",".join(ACTIONS_MAP[a] for a in executed_actions))
                    with torch.no_grad():
                        out = model.inference(
                            instruction=[instructions],
                            history_action=[hist_str],
                            left_current_frame=lfc,
                            right_current_frame=rfc,
                            left_history_video=lhv,
                            right_history_video=rhv,
                            depth_iters=8, max_new_tokens=24,
                            temperature=0.0, top_p=1.0,
                            output_point=False, output_depth=False,
                        )
                    action_seq = parse_actions(out['action'][0])[:self.args.execute_steps]

            if not action_seq:
                action_seq = [0]

            action = action_seq.pop(0)
            if action != agent.get_next_action(ref_path[next_wp]):
                accumulated_error += 1

            # advance waypoint
            while agent.get_next_action(ref_path[next_wp]) == 0:
                next_wp += 1
                force_expert = False
                left_expert_num = 0
                if next_wp == len(ref_path) - 1:
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_wp >= len(ref_path):
                    force_end = True
                    action = 0
                    from_expert = True
                    break

            metrics = env.get_metrics()
            error_bad = (
                (not from_expert and action == 0 and metrics["distance_to_goal"] >= 3.0)
                or (accumulated_error / ref_actions_len > 0.8)
                or accumulated_error > 12
            )
            if next_wp < len(ref_path) and error_bad:
                model_success = False
                force_expert = True
                accumulated_error = 0
                action = agent.get_next_action(ref_path[next_wp])
                from_expert = True
                action_seq = []

            if action == 0 and not force_end:
                action = agent.get_next_action(ref_path[next_wp])

            left_hist.append(preprocess(rgb_l_d))
            right_hist.append(preprocess(rgb_r_d))
            executed_actions.append(action)
            actions.append(action)

            obs = env.step(action)
            metrics = env.get_metrics()
            step_id += 1

            if force_end:
                break

        assert len(rgb_left_buf) + 1 == len(actions), \
            f"Length mismatch: rgb={len(rgb_left_buf)}, actions={len(actions)}"

        episode_save = (
            metrics["distance_to_goal"] < MIDGOAL_RADIUS
            and (
                (not model_success and metrics["pl"] < RELATIVE_PATH_LENGTH_THRESHOLD)
                or metrics["pl"] < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD
            )
        )

        if episode_save:
            os.makedirs(rgb_left_dir,  exist_ok=True)
            os.makedirs(rgb_right_dir, exist_ok=True)
            os.makedirs(depth_dir,     exist_ok=True)
            for rgb_d, path in rgb_left_buf:
                Image.fromarray(rgb_d).convert("RGB").save(path)
            for rgb_d, path in rgb_right_buf:
                Image.fromarray(rgb_d).convert("RGB").save(path)
            for dep, path in depth_buf:
                depth_mm = (dep * 10.0 * 1000.0).astype(np.uint16)
                Image.fromarray(np.squeeze(depth_mm)).save(path)
            with open(os.path.join(scene_dir, "label_points.json"), 'w') as f:
                json.dump(label_points_data, f, indent=4)
            with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                f.write(json.dumps({
                    "id": episode_id,
                    "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                    "trajectory_id": trajectory_id,
                    "scene_id": scene_id,
                }) + "\n")

        metrics.update({
            "step_id": step_id,
            "ref_actions_len": ref_actions_len,
            "accumulated_error": accumulated_error,
            "save": int(episode_save),
            "model_success": model_success,
            "force_episode_end": force_end,
        })
        return {
            "anno": {
                "id": episode_id,
                "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                "instructions": instructions if isinstance(instructions, list) else [instructions],
                "actions": actions,
            },
            "metrics": metrics,
        }

    def _flush(self, annotations):
        path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
        existing = json.load(open(path)) if os.path.exists(path) else []
        existing.extend(annotations)
        seen, deduped = set(), []
        for item in existing:
            if item["video"] not in seen:
                seen.add(item["video"])
                deduped.append(item)
        with open(path, "w") as f:
            json.dump(deduped, f, indent=4)
        annotations.clear()

    def _merge_annotations(self):
        merged = []
        for fname in os.listdir(self.output_path):
            if fname.startswith("annotations_") and fname.endswith(".json"):
                merged.extend(json.load(open(os.path.join(self.output_path, fname))))
        merged = sorted(merged, key=lambda x: x["id"])
        seen, deduped = set(), []
        for item in merged:
            if item["video"] not in seen:
                seen.add(item["video"])
                deduped.append(item)
        with open(os.path.join(self.output_path, "annotations.json"), "w") as f:
            json.dump(deduped, f, indent=4)

    def update_dataset(self, model):
        random.seed(self.rank)
        np.random.seed(self.rank)
        os.makedirs(self.output_path, exist_ok=True)

        env = self.config_env()
        scene_episode_dict = {}
        for ep in env.episodes:
            scene_episode_dict.setdefault(ep.scene_id, []).append(ep)

        annotations = []
        num_collected = 0
        total = min(self.args.dagger_update_size,
                    sum(len(v) for v in scene_episode_dict.values())) // self.world_size

        with tqdm.tqdm(total=total, dynamic_ncols=True) as pbar, torch.no_grad():
            for scene_id in sorted(scene_episode_dict.keys()):
                for episode in scene_episode_dict[scene_id][self.rank::self.world_size]:
                    result = self.generate(
                        env=env, model=model, episode=episode,
                        force_expert=self.args.force_expert,
                        data_it=self.args.dagger_data_it,
                    )
                    if result is None:
                        continue

                    m = result["metrics"]
                    with open(os.path.join(self.output_path, "result.json"), "a") as f:
                        f.write(json.dumps({
                            "scene": scene_id.split('/')[-2],
                            "episode_id": episode.episode_id,
                            "save": m["save"],
                            "model_success": m["model_success"],
                            "success": m["success"],
                            "relative_pl": m["pl"],
                            "step_id": m["step_id"],
                            "accumulated_error": m["accumulated_error"],
                        }) + "\n")

                    pbar.update()
                    if not m["save"]:
                        continue

                    annotations.append(result["anno"])
                    num_collected += 1

                    if num_collected % self.args.dagger_commit_freq == 0:
                        self._flush(annotations)
                    if num_collected >= self.args.dagger_update_size:
                        break
                if num_collected >= self.args.dagger_update_size:
                    break

            self._flush(annotations)

        dist.barrier()
        if get_rank() == 0:
            self._merge_annotations()


# ── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--dagger_dataset", type=str, default="R2R")
    parser.add_argument("--dagger_data_path", type=str, default="task/r2r/train/train.json.gz")
    parser.add_argument("--dagger_output_path", type=str, default="data/dagger")
    parser.add_argument("--dagger_update_size", type=int, default=160000)
    parser.add_argument("--dagger_commit_freq", type=int, default=50)
    parser.add_argument("--dagger_p", type=float, default=0.0)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--num_future_steps", type=int, default=1)
    parser.add_argument("--execute_steps", type=int, default=4)
    parser.add_argument("--history_num", type=int, default=8)
    parser.add_argument("--server_urls", type=str, default="http://localhost:7200",
                        help="Comma-separated server URLs matching server.sh")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()

    from scripts.client import StereoVLNClient
    urls = [u.strip() for u in args.server_urls.split(",")]
    model = StereoVLNClient(server_url=urls[rank % len(urls)])

    collector = DAggerCollector(args=args, rank=rank, world_size=world_size)
    collector.update_dataset(model=model)