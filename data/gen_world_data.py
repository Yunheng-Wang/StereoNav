import os
import sys
import numpy as np
import argparse
from PIL import Image
import json
import quaternion
from omegaconf import OmegaConf

import habitat
from habitat import Env
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config

# 导入自定义测量器以注册到 Habitat registry
sys.path.insert(0, '/home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B')
from data.utils import measures  # noqa: F401
from data.utils import rxr_dataset  # noqa: F401


def save_initial_frames(config_path: str, split: str, output_dir: str, data_path: str = None, dataset: str = 'r2r'):
    """
    Load each episode, save 6 frames (every 60 degrees), instruction,
    and metadata (clockwise angle to goal, straight-line distance).
    """
    config = get_habitat_config(config_path)
    dataset = dataset.lower()

    # Configure stereo cameras
    with habitat.config.read_write(config):
        OmegaConf.set_struct(config, False)
        config.habitat.dataset.split = split
        if data_path:
            config.habitat.dataset.data_path = data_path

        # Left camera
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
            "type": "HabitatSimRGBSensor",
            "uuid": "rgb_left",
            "width": 448,
            "height": 448,
            "hfov": 79,
            "position": [-0.05, 1.25, 0.0],
            "orientation": [0.0, 0.0, 0.0]
        })
        # Right camera
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
            "type": "HabitatSimRGBSensor",
            "uuid": "rgb_right",
            "width": 448,
            "height": 448,
            "hfov": 79,
            "position": [0.05, 1.25, 0.0],
            "orientation": [0.0, 0.0, 0.0]
        })
        OmegaConf.set_struct(config, True)

    env = Env(config=config)
    os.makedirs(output_dir, exist_ok=True)

    # 按场景分组排序episodes，减少场景切换次数
    episodes_sorted = sorted(env.episodes, key=lambda ep: ep.scene_id)
    print(f"Total episodes: {len(episodes_sorted)}, sorted by scene_id")

    for episode in episodes_sorted:
        episode_id = episode.episode_id
        scene_id = episode.scene_id.split('/')[-2]

        # Filter RXR for English only BEFORE creating folder
        if dataset == 'rxr':
            instruction_obj = episode.instruction
            if hasattr(instruction_obj, 'language'):
                language = instruction_obj.language
                if not language.startswith('en'):
                    continue

        # Create output folder for this episode
        episode_dir = os.path.join(output_dir, f"{scene_id}_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)

        print(f"Processing episode {scene_id}_{episode_id}")

        # Reset to episode start
        env.current_episode = episode
        observations = env.reset()

        # Save instruction
        if dataset == 'rxr':
            instruction = instruction_obj.instruction_text
        else:
            instruction = episode.instruction.instruction_text if hasattr(episode, 'instruction') else ""

        # Save initial frame (0 degrees)
        Image.fromarray(observations['rgb_right'][:, :, :3]).save(
            os.path.join(episode_dir, "0.jpg")
        )

        # Get initial state
        initial_position = env.sim.get_agent_state().position
        initial_rotation = env.sim.get_agent_state().rotation
        print(f"Agent position Y (height): {initial_position[1]:.3f}m")

        # Save 5 more frames at 60° intervals (60°, 120°, 180°, 240°, 300°)
        for i in range(1, 6):
            angle_deg = i * 60
            angle_rad = np.deg2rad(angle_deg)

            # Create rotation quaternion around Y axis (clockwise)
            new_rotation = initial_rotation * np.quaternion(np.cos(-angle_rad/2), 0, np.sin(-angle_rad/2), 0)

            # Set agent state
            agent_state = env.sim.get_agent_state()
            agent_state.position = initial_position
            agent_state.rotation = new_rotation
            env.sim.set_agent_state(agent_state.position, agent_state.rotation)

            # Get new observations
            observations = env.sim.get_sensor_observations()

            # Save frame
            Image.fromarray(observations['rgb_right'][:, :, :3]).save(
                os.path.join(episode_dir, f"{i}.jpg")
            )

        # Calculate clockwise angle from agent's facing direction to goal
        goal_pos = np.array(episode.goals[0].position)
        agent_pos = np.array(initial_position)
        distance = float(np.linalg.norm(goal_pos - agent_pos))

        # Agent's forward direction (forward = -Z in Habitat)
        forward = quaternion.rotate_vectors(initial_rotation, np.array([0.0, 0.0, -1.0]))
        fx, fz = forward[0], forward[2]
        dx, dz = goal_pos[0] - agent_pos[0], goal_pos[2] - agent_pos[2]
        heading_fwd = np.arctan2(fx, -fz)
        heading_goal = np.arctan2(dx, -dz)
        angle_to_goal = float(np.rad2deg(heading_goal - heading_fwd) % 360)

        # Calculate height difference (goal - agent ground position)
        height_diff = float(goal_pos[1] - agent_pos[1])

        # Save all info to a single JSON
        metadata = {
            "instruction": instruction,
            "angle_to_goal_deg": round(angle_to_goal, 2),
            "distance_to_goal_m": round(distance, 2),
            "height_diff_m": round(height_diff, 2),
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Saved 6 frames + metadata for episode {scene_id}_{episode_id}")

    env.close()
    print(f"All episodes processed. Frames saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--habitat_config_path", type=str,
                        default='/home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/config/eval.yaml')
    parser.add_argument("--split", type=str, default='val_unseen')
    parser.add_argument("--output_dir", type=str,
                        default='/home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/target/init_frame_r2r')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='r2r', help='Dataset name: r2r or rxr')

    args = parser.parse_args()
    save_initial_frames(args.habitat_config_path, args.split, args.output_dir, args.data_path, args.dataset)

    # python data/world.py   --habitat_config_path /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/config/eval.yaml   --split val_unseen   --output_dir /home/CONNECT/yfang870/yunhengwang/StereoVLN_InternVL_3_5_2_B/target/init_frame_r2r

