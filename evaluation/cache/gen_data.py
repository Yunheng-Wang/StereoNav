"""Ground-truth trajectory data generator for VLN evaluation.

This script generates navigation trajectory data from Habitat simulator by following
the shortest path to goal waypoints. For each episode, it:
  - Follows the reference path using ShortestPathFollower
  - Captures RGB observations at each step
  - Projects goal positions onto image frames (optional red dot rendering)
  - Saves trajectory metadata (instructions, actions, scene info) to summary.json

Supports incremental generation with filtering and resume capabilities.

Usage:
    python evaluation/cache/gen_data.py --dataset R2R --config_path config/vln_r2r.yaml \
        --output_path cache/R2R --data_path task/r2r/train/train.json.gz
"""
import habitat
import json
import numpy as np
import argparse
import os
import shutil
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write
import quaternion
from typing import Tuple


def get_camera_intrinsics(width: int, height: int, hfov: float) -> np.ndarray:
    """Compute camera intrinsic matrix from image dimensions and horizontal FOV."""
    hfov_rad = np.deg2rad(hfov)
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])


def project_3d_to_2d(
    world_point: np.ndarray,
    agent_position: np.ndarray,
    agent_rotation: np.quaternion,
    camera_offset: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[float, float, float]:
    """Project 3D world point to 2D image coordinates.

    Returns (u, v, cam_x) with bottom-left origin. u/v are None if point is behind camera.
    """
    R_agent = quaternion.as_rotation_matrix(agent_rotation)
    camera_position = agent_position + R_agent @ camera_offset
    point_rel = world_point - camera_position
    point_cam_opengl = R_agent.T @ point_rel
    point_cam = point_cam_opengl.copy()
    point_cam[2] = -point_cam_opengl[2]
    cam_x = point_cam[0]

    if point_cam[2] <= 0:
        return None, None, cam_x

    p = K @ point_cam
    u = p[0] / p[2]
    v = height - p[1] / p[2]
    return u, v, cam_x


def clip_to_image_bounds(u: float, v: float, width: int, height: int, cam_x: float) -> Tuple[float, float]:
    """Clip projected coordinates to image bounds. Points outside margin are clamped to edges."""
    v_center = height / 2.0
    if u is None or v is None:
        return (0.0, v_center) if cam_x < 0 else (width - 1.0, v_center)

    u_margin = width * 0.2
    v_margin = height * 0.2
    if u < -u_margin or u > width - 1 + u_margin or v < -v_margin or v > height - 1 + v_margin:
        return (0.0, v_center) if cam_x < 0 else (width - 1.0, v_center)

    return float(np.clip(u, 0, width - 1)), float(np.clip(v, 0, height - 1))


class StreamVLNHabitatRunner:
    """Generates ground-truth VLN trajectory data from Habitat simulator.

    Follows reference paths using ShortestPathFollower and captures RGB observations
    with optional goal-point rendering. Supports incremental generation with filtering
    and automatic cleanup of stale episodes.
    """
    def __init__(self, dataset: str, config_path: str, output_path: str, data_path: str = None, scenes_dir: str = None, filter_json: str = None):
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path
        self.scenes_dir = scenes_dir
        self.filter_json = filter_json
        self.config = get_habitat_config(self.config_path)
        with read_write(self.config):
            sensors = self.config.habitat.simulator.agents.main_agent.sim_sensors
            sensors.rgb_sensor.width = 640
            sensors.rgb_sensor.height = 480
            sensors.rgb_sensor.hfov = 79
            sensors.rgb_sensor.position = [0.0, 1.25, 0.0]
            sensors.depth_sensor.width = 640
            sensors.depth_sensor.height = 480
            sensors.depth_sensor.hfov = 79
            sensors.depth_sensor.position = [0.0, 1.25, 0.0]

    def config_env(self) -> habitat.Env:
        with read_write(self.config):
            if self.data_path is not None:
                self.config.habitat.dataset.update({"data_path": self.data_path})
            if self.scenes_dir is not None:
                self.config.habitat.dataset.update({"scenes_dir": self.scenes_dir})
            measurements = self.config.habitat.task.measurements
            for key in ["oracle_success", "oracle_navigation_error", "pl"]:
                if key in measurements:
                    del measurements[key]
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)

    def generate(self, rank: int = 0, world_size: int = 1, render_points: bool = False) -> None:
        filter_set = None
        if self.filter_json is not None:
            filter_set = set()
            with open(self.filter_json) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        filter_set.add((d['scene_id'], int(d['episode_id'])))

        os.makedirs(self.output_path, exist_ok=True)

        # 读取已有 summary.json 中的条目
        summary_path = os.path.join(self.output_path, "summary.json")
        existing_set = set()
        existing_lines = []
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        existing_set.add((d['scene_id'], int(d['id'])))
                        existing_lines.append((d['scene_id'], int(d['id']), line))

        # 情况2：output 中有但 filter_set 中没有 → 删除 summary 条目 + images 目录
        if filter_set is not None:
            stale = existing_set - filter_set
            if stale:
                kept_lines = [l for sc, eid, l in existing_lines if (sc, eid) not in stale]
                with open(summary_path, 'w') as f:
                    for l in kept_lines:
                        f.write(l + '\n')
                for sc, eid in stale:
                    img_dir = os.path.join(self.output_path, "images", f"{sc}_{self.dataset}_{eid:06d}")
                    if os.path.exists(img_dir):
                        shutil.rmtree(img_dir)
                existing_set -= stale

        img_width, img_height, hfov = 640, 480, 79
        K = get_camera_intrinsics(img_width, img_height, hfov)

        intrinsics_path = os.path.join(self.output_path, "intrinsics.txt")
        with open(intrinsics_path, 'w') as f:
            for row in K:
                f.write(' '.join([f"{val:.6f}" for val in row]) + '\n')

        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        annotations = []
        for scene_id in sorted(scene_episode_dict.keys()):
            scan = scene_id.split("/")[-2]
            episodes = scene_episode_dict[scene_id]
            print(f"scene_id: {scene_id}, scan: {scan}")
            for episode in episodes[rank::world_size]:
                ep_scan = episode.scene_id.split('/')[-2]
                ep_id = int(episode.episode_id)
                if filter_set is not None and (ep_scan, ep_id) not in filter_set:
                    continue
                # 情况1：filter_set 有且已存在 → 跳过
                if (ep_scan, ep_id) in existing_set:
                    continue
                env.current_episode = episode
                agent = ShortestPathFollower(sim=env.sim, goal_radius=0.5, return_one_hot=False)

                if self.dataset == 'rxr':
                    instruction_obj = episode.instruction
                    if hasattr(instruction_obj, 'language') and not instruction_obj.language.startswith('en'):
                        continue
                    instructions = instruction_obj.instruction_text
                else:
                    instructions = episode.instruction.instruction_text

                trajectory_id = episode.trajectory_id
                scene_id = episode.scene_id.split('/')[-2]
                episode_id = int(episode.episode_id)
                ref_path = episode.reference_path
                observation = env.reset()

                camera_offset = np.array([0.0, 1.25, 0.0])
                rgb_list = []
                actions = [-1]
                next_waypoint_id = 1
                scene_dir = os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}")
                rgb_dir = os.path.join(scene_dir, "rgb")
                os.makedirs(rgb_dir, exist_ok=True)

                while not env.episode_over:
                    rgb = observation["rgb"]
                    rgb_list.append(rgb)

                    next_action = agent.get_next_action(ref_path[next_waypoint_id])

                    agent_state = env.sim.get_agent_state()
                    agent_position = agent_state.position
                    agent_rotation = agent_state.rotation
                    goal_camera_center = np.array(ref_path[-1]) + np.array([0.0, 1.25, 0.0])

                    u, v, cam_x = project_3d_to_2d(
                        goal_camera_center, agent_position, agent_rotation,
                        camera_offset, K, img_width, img_height)
                    u, v = clip_to_image_bounds(u, v, img_width, img_height, cam_x)

                    frame_idx = len(rgb_list)
                    if render_points:
                        v_tl = img_height - v
                        img = Image.fromarray(rgb).convert("RGB")
                        draw = ImageDraw.Draw(img)
                        r = 5
                        draw.ellipse([(u-r, v_tl-r), (u+r, v_tl+r)], fill='red', outline='red')
                        img.save(os.path.join(rgb_dir, f"{frame_idx:03d}.jpg"))
                    else:
                        Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{frame_idx:03d}.jpg"))

                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(ref_path[next_waypoint_id])
                    if force_episode_over:
                        break
                    observation = env.step(next_action)
                    actions.append(next_action)

                if len(actions) > 498:
                    continue

                assert len(actions) == len(rgb_list)

                video = os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}")
                instr_list = instructions if isinstance(instructions, list) else [instructions]
                annotations.append({"id": episode_id, "video": video, "instructions": instr_list, "actions": actions})

                with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                    f.write(json.dumps({
                        "id": episode_id, "video": video, "instructions": instr_list,
                        "actions": actions, "trajectory_id": trajectory_id, "scene_id": scene_id,
                    }) + "\n")

            with open(os.path.join(self.output_path, f"annotations_{rank}.json"), "w") as f:
                json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="R2R")
    parser.add_argument("--config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--output_path", type=str, default="cache/R2R")
    parser.add_argument("--data_path", type=str, default="task/r2r/train/train.json.gz")
    parser.add_argument("--render_points", action="store_true")
    parser.add_argument("--scenes_dir", type=str, default=None)
    parser.add_argument("--filter_json", type=str, default=None)
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))

    runner = StreamVLNHabitatRunner(
        dataset=args.dataset,
        config_path=args.config_path,
        output_path=args.output_path,
        data_path=args.data_path,
        scenes_dir=args.scenes_dir,
        filter_json=args.filter_json,
    )
    runner.generate(rank, world_size, render_points=args.render_points)
