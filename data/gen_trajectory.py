import habitat
import logging
import random
import json
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from utils import measures
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write
import quaternion
from typing import Tuple, List
# Import custom RxR dataset to register it with habitat
from utils import rxr_dataset  # noqa: F401


def get_camera_intrinsics(width: int, height: int, hfov: float) -> np.ndarray:
    """
    Calculate camera intrinsic matrix from image dimensions and horizontal FOV.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        hfov: Horizontal field of view in degrees

    Returns:
        3x3 camera intrinsic matrix
    """
    hfov_rad = np.deg2rad(hfov)
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx  # Assuming square pixels
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def quaternion_to_rotation_matrix(q: np.quaternion) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion

    Returns:
        3x3 rotation matrix
    """
    return quaternion.as_rotation_matrix(q)


def project_3d_to_2d(
    world_point: np.ndarray,
    agent_position: np.ndarray,
    agent_rotation: np.quaternion,
    camera_offset: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    debug: bool = False
) -> Tuple[float, float, float]:
    """
    Project a 3D world point to 2D pixel coordinates in camera view.
    Uses bottom-left as origin (0,0) as specified.

    Args:
        world_point: 3D point in world coordinates [x, y, z]
        agent_position: Agent position in world coordinates [x, y, z]
        agent_rotation: Agent rotation as quaternion
        camera_offset: Camera offset from agent center [x, y, z]
        K: Camera intrinsic matrix (3x3)
        width: Image width
        height: Image height
        debug: Print debug information

    Returns:
        Tuple of (u, v, cam_x) where:
        - u, v: pixel coordinates with origin at bottom-left (None if behind camera)
        - cam_x: X coordinate in camera frame (negative=left, positive=right)
    """
    # Get rotation matrix from agent's quaternion
    R_agent = quaternion_to_rotation_matrix(agent_rotation)

    # Calculate camera position in world coordinates
    # In Habitat, camera offset is in agent's local frame
    camera_position = agent_position + R_agent @ camera_offset

    # Transform world point to camera coordinates
    point_rel = world_point - camera_position

    # Rotate point into camera frame
    # IMPORTANT: Habitat uses OpenGL convention where camera looks toward -Z
    # We need to convert to standard CV convention (camera looks toward +Z)
    point_cam_opengl = R_agent.T @ point_rel

    # Convert from OpenGL (camera looks at -Z) to CV convention (camera looks at +Z)
    # by negating the Z axis
    point_cam = point_cam_opengl.copy()
    point_cam[2] = -point_cam_opengl[2]  # Negate Z to convert from OpenGL to CV

    # cam_x is used to determine if target is on left (negative) or right (positive)
    cam_x = point_cam[0]

    if debug:
        print(f"\n=== Debug Projection ===")
        print(f"World point: {world_point}")
        print(f"Agent position: {agent_position}")
        print(f"Camera position: {camera_position}")
        print(f"Point relative: {point_rel}")
        print(f"Point in camera (OpenGL): {point_cam_opengl}")
        print(f"Point in camera (CV): {point_cam}")
        print(f"Camera X (left-/right+): {cam_x:.2f}")
        print(f"Distance: {np.linalg.norm(point_rel):.2f}m")

    # Check if point is behind camera (in CV convention, Z should be positive)
    if point_cam[2] <= 0:
        if debug:
            print(f"Point is behind camera (Z={point_cam[2]:.2f})")
        return None, None, cam_x

    # Project to image plane
    point_2d_homogeneous = K @ point_cam
    u = point_2d_homogeneous[0] / point_2d_homogeneous[2]
    v = point_2d_homogeneous[1] / point_2d_homogeneous[2]

    # Convert from top-left origin to bottom-left origin
    v_bottom_left = height - v

    if debug:
        print(f"Projected u, v (top-left): ({u:.2f}, {v:.2f})")
        print(f"Projected u, v (bottom-left): ({u:.2f}, {v_bottom_left:.2f})")

    return u, v_bottom_left, cam_x


def clip_to_image_bounds(
    u: float,
    v: float,
    width: int,
    height: int,
    cam_x: float
) -> Tuple[float, float]:
    """
    Clip pixel coordinates to image bounds.
    When target is out of view, return left/right boundary based on target's relative position.

    Args:
        u: Horizontal pixel coordinate (0 = left edge)
        v: Vertical pixel coordinate (0 = bottom edge)
        width: Image width
        height: Image height
        cam_x: X coordinate in camera frame (negative=target on left, positive=target on right)

    Returns:
        Clipped (u, v) coordinates
    """
    v_center = height / 2.0

    # Case 1: Point is behind camera (projection failed, u or v is None)
    # Determine direction based on target's X coordinate in camera frame
    if u is None or v is None:
        if cam_x < 0:  # Target is on the left
            return 0.0, v_center
        else:  # Target is on the right
            return width - 1.0, v_center

    # Case 2: Point is in front of camera but far outside field of view
    # If projected coordinates exceed image bounds significantly, target is out of view
    out_of_view_threshold = 0.2
    u_margin = width * out_of_view_threshold
    v_margin = height * out_of_view_threshold

    # Check if significantly out of view
    is_far_left = u < -u_margin
    is_far_right = u > width - 1 + u_margin
    is_far_out_vertically = v < -v_margin or v > height - 1 + v_margin

    # If point is out of view, return corresponding boundary based on target's relative position
    if is_far_left or is_far_right or is_far_out_vertically:
        if cam_x < 0:  # Target is on the left
            return 0.0, v_center
        else:  # Target is on the right
            return width - 1.0, v_center

    # Case 3: Point is within or near field of view, clip to image boundaries normally
    u_clipped = np.clip(u, 0, width - 1)
    v_clipped = np.clip(v, 0, height - 1)

    return u_clipped, v_clipped


class StreamVLNHabitatRunner:
    def __init__(self, dataset: str, config_path: str, output_path: str, data_path: str = None):
        # 1. Basic configuration
        self.device = torch.device("cuda")
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path
        self.config = get_habitat_config(self.config_path)
        # 2. Stereo camera configuration
        with read_write(self.config):
            OmegaConf.set_struct(self.config, False)
            # 2.1 Left camera RGB configuration
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_left",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [-0.05, 1.25, 0.0],  # Left eye, offset 5cm to the left
                "orientation": [0.0, 0.0, 0.0]
            })
            # 2.2 Right camera RGB configuration
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_right",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [0.05, 1.25, 0.0],  # Right eye, offset 5cm to the right
                "orientation": [0.0, 0.0, 0.0]
            })
            # 2.3 Left camera depth configuration
            self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor_left = OmegaConf.create({
                "type": "HabitatSimDepthSensor",
                "uuid": "depth_left",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "min_depth": 0.0,
                "max_depth": 3.0,
                "normalize_depth": True,
                "position": [-0.05, 1.25, 0.0],  # Left eye, offset 5cm to the left
                "orientation": [0.0, 0.0, 0.0]
            })
            OmegaConf.set_struct(self.config, True)


    def config_env(self, scene: str = None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.update(
                    {
                        "data_path": self.data_path,
                    }
                )
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)


    def generate(self, rank: int = 0, world_size: int = 1, render_points: bool = False) -> None:
        # 1. Setup save paths
        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        # 1.1 Calculate and save camera intrinsics (shared by all cameras: left RGB, right RGB, left depth)
        img_width, img_height = 448, 448
        hfov = 79  # degrees
        K = get_camera_intrinsics(img_width, img_height, hfov)

        # Save intrinsic matrix to file (3x3 matrix, one row per line)
        intrinsics_path = os.path.join(self.output_path, "intrinsics.txt")
        with open(intrinsics_path, 'w') as f:
            for row in K:
                f.write(' '.join([f"{val:.6f}" for val in row]) + '\n')
        print(f"Camera intrinsics saved to {intrinsics_path}")

        # 1.2 Calculate and save stereo camera baseline distance
        # Left camera position: [-0.05, 1.25, 0.0], Right camera position: [0.05, 1.25, 0.0]
        baseline = 0.05 - (-0.05)  # Baseline distance = 0.1 meters
        baseline_path = os.path.join(self.output_path, "baseline.txt")
        with open(baseline_path, 'w') as f:
            f.write(f"{baseline:.6f}\n")
        print(f"Stereo baseline saved to {baseline_path}")

        # 2. Create environment & task (habitat-lab)
        env = self.config_env()
        # 3. Extract tasks for each environment
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        # Load completed episode_ids (for resuming from checkpoint)
        done_ids = set()
        summary_path = os.path.join(self.output_path, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done_ids.add(json.loads(line)["id"])
            print(f"Resuming: {len(done_ids)} episodes already done, skipping them.")

        # 4. Process episodes
        annotations = []
        for scene_id in sorted(scene_episode_dict.keys()):
            # 4.1 Extract current scene
            scan = scene_id.split("/")[-2]
            # 4.2 Extract tasks for current scene
            episodes = scene_episode_dict[scene_id]
            print(f"scene_id: {scene_id}, scan: {scan}")
            # 4.3 Core task collection & execution
            rank_episodes = episodes[rank::world_size]
            for episode in tqdm(rank_episodes, desc=f"rank{rank} {scene_id.split('/')[-2]}", dynamic_ncols=True):
                if int(episode.episode_id) in done_ids:
                    continue
                # 4.3.1 Prepare current task to execute
                env.current_episode = episode
                # 4.3.2 Initialize shortest path follower
                agent = ShortestPathFollower(
                    sim=env.sim, goal_radius=0.5, return_one_hot=False)
                # 4.3.3 Extract basic task information
                # Compatible with R2R and RXR data formats
                if self.dataset == 'rxr':
                    # RXR: instruction is an object with language attribute
                    instruction_obj = episode.instruction
                    # Check if has language attribute (RXR specific)
                    if hasattr(instruction_obj, 'language'):
                        language = instruction_obj.language
                        # Only process English tasks
                        if not language.startswith('en'):
                            continue
                    instructions = instruction_obj.instruction_text
                else:
                    # R2R: instruction is an object
                    instructions = episode.instruction.instruction_text

                trajectory_id = episode.trajectory_id
                scene_id = episode.scene_id.split('/')[-2]
                episode_id = int(episode.episode_id)
                ref_path = episode.reference_path
                # 4.3.4 Get initial observation
                observation = env.reset()
                # 4.3.5 Initialize camera offsets (reuse previously calculated intrinsics K)
                camera_offset_left = np.array([-0.05, 1.25, 0.0])
                camera_offset_right = np.array([0.05, 1.25, 0.0])
                camera_offset_center = np.array([0.0, 1.25, 0.0])  # Center position between two cameras
                # 4.3.6 Physical execution
                rgb_left_list = []
                rgb_right_list = []
                depth_left_list = []
                actions = [-1]
                next_waypoint_id = 1
                scene_dir = os.path.join(
                    self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}")
                rgb_left_dir = os.path.join(scene_dir, "rgb_left")
                rgb_right_dir = os.path.join(scene_dir, "rgb_right")
                depth_left_dir = os.path.join(scene_dir, "depth_left")
                label_points_file = os.path.join(scene_dir, "label_points.json")
                os.makedirs(rgb_left_dir, exist_ok=True)
                os.makedirs(rgb_right_dir, exist_ok=True)
                os.makedirs(depth_left_dir, exist_ok=True)
                # Initialize point label data structure
                label_points_data = []
                while not env.episode_over:
                    # 4.3.5.1 Get stereo RGB observations and left view depth observation
                    rgb_left = observation["rgb_left"]
                    rgb_right = observation["rgb_right"]
                    depth_left = observation["depth_left"]
                    rgb_left_list.append(rgb_left)
                    rgb_right_list.append(rgb_right)
                    depth_left_list.append(depth_left)

                    # Calculate next action first (used to decide label position when target is behind)
                    next_action = agent.get_next_action(
                        ref_path[next_waypoint_id])

                    # Calculate label_left_point and label_right_point
                    # Get current agent position and rotation
                    agent_state = env.sim.get_agent_state()
                    agent_position = agent_state.position  # [x, y, z]
                    agent_rotation = agent_state.rotation  # quaternion

                    # Get final goal position (last point in ref_path)
                    # Note: Points in ref_path are agent positions on ground, need to add camera height offset
                    goal_agent_position = np.array(ref_path[-1])
                    goal_camera_center = goal_agent_position + np.array([0.0, 1.25, 0.0])  # Camera center height

                    # Debug: Only print debug info on first frame
                    debug_mode = (len(rgb_left_list) == 1)
                    if debug_mode:
                        print(f"\n=== Frame {len(rgb_left_list)} Debug Info ===")
                        print(f"Current waypoint_id: {next_waypoint_id}")
                        print(f"Total waypoints: {len(ref_path)}")
                        print(f"Current agent position: {agent_position}")
                        print(f"Goal agent position: {goal_agent_position}")
                        print(f"Goal camera center: {goal_camera_center}")
                        print(f"Distance to goal: {np.linalg.norm(goal_agent_position - agent_position):.2f}m")
                        print(f"Next action: {next_action}")

                    # Project final goal camera center to current left view
                    u_left, v_left, cam_x_left = project_3d_to_2d(
                        world_point=goal_camera_center,
                        agent_position=agent_position,
                        agent_rotation=agent_rotation,
                        camera_offset=camera_offset_left,
                        K=K,
                        width=img_width,
                        height=img_height,
                        debug=debug_mode
                    )
                    # Clip to image boundaries (based on target's relative position)
                    u_left, v_left = clip_to_image_bounds(u_left, v_left, img_width, img_height, cam_x_left)

                    # Project final goal camera center to current right view
                    u_right, v_right, cam_x_right = project_3d_to_2d(
                        world_point=goal_camera_center,
                        agent_position=agent_position,
                        agent_rotation=agent_rotation,
                        camera_offset=camera_offset_right,
                        K=K,
                        width=img_width,
                        height=img_height,
                        debug=False
                    )
                    # Clip to image boundaries (based on target's relative position)
                    u_right, v_right = clip_to_image_bounds(u_right, v_right, img_width, img_height, cam_x_right)

                    # Save point label data for current frame
                    label_points_data.append({
                        "frame_id": len(rgb_left_list),
                        "label_left_point": [float(u_left), float(v_left)],
                        "label_right_point": [float(u_right), float(v_right)]
                    })

                    # Save RGB images (optionally render target points)
                    if render_points:
                        # Convert to PIL image and draw target points
                        # Note: v coordinate uses bottom-left origin, need to convert back to top-left for PIL drawing
                        v_left_topleft = img_height - v_left
                        v_right_topleft = img_height - v_right

                        # Left image
                        img_left = Image.fromarray(rgb_left).convert("RGB")
                        draw_left = ImageDraw.Draw(img_left)
                        radius = 5
                        draw_left.ellipse(
                            [(u_left - radius, v_left_topleft - radius),
                             (u_left + radius, v_left_topleft + radius)],
                            fill='red', outline='red'
                        )
                        img_left.save(os.path.join(rgb_left_dir, f"{len(rgb_left_list):03d}.jpg"))

                        # Right image
                        img_right = Image.fromarray(rgb_right).convert("RGB")
                        draw_right = ImageDraw.Draw(img_right)
                        draw_right.ellipse(
                            [(u_right - radius, v_right_topleft - radius),
                             (u_right + radius, v_right_topleft + radius)],
                            fill='red', outline='red'
                        )
                        img_right.save(os.path.join(rgb_right_dir, f"{len(rgb_right_list):03d}.jpg"))
                    else:
                        # Don't render target points, save original images directly
                        Image.fromarray(rgb_left).convert("RGB").save(
                            os.path.join(rgb_left_dir, f"{len(rgb_left_list):03d}.jpg"))
                        Image.fromarray(rgb_right).convert("RGB").save(
                            os.path.join(rgb_right_dir, f"{len(rgb_right_list):03d}.jpg"))
                    # Save depth image (unit: millimeters, uint16 format)
                    # Habitat depth is normalized (0-1), need to denormalize to real depth
                    depth_m = depth_left * 10.0  # Denormalize: depth range 0-10 meters
                    depth_mm = (depth_m * 1000.0).astype(np.uint16)  # Convert to millimeters
                    # Remove single dimension: from (H, W, 1) to (H, W)
                    depth_mm = np.squeeze(depth_mm)
                    Image.fromarray(depth_mm).save(
                        os.path.join(depth_left_dir, f"{len(depth_left_list):03d}.png"))

                    # After reaching first waypoint, switch to next waypoint
                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(
                                sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(
                            ref_path[next_waypoint_id])
                    # If all reference points are executed, break out
                    if force_episode_over:
                        break
                    # Execute action
                    observation = env.step(next_action)
                    actions.append(next_action)
                # 4.3.6 Skip if too many actions
                if len(actions) > 498:
                    continue
                # 4.3.7 Save point label data to JSON file
                with open(label_points_file, 'w') as f:
                    json.dump(label_points_data, f, indent=4)
                # 4.3.8 Collect & save content
                assert len(actions) == len(rgb_left_list) == len(rgb_right_list) == len(depth_left_list) == len(label_points_data), \
                    f"Data length mismatch - actions: {len(actions)}, rgb_left: {len(rgb_left_list)}, rgb_right: {len(rgb_right_list)}, depth_left: {len(depth_left_list)}, label_points: {len(label_points_data)}"
                annotations.append({
                    "id": episode_id,
                    "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                })

                with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                    result = {
                        "id": episode_id,
                        "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                        "instructions": instructions if isinstance(instructions, list) else [instructions],
                        "actions": actions,
                        "trajectory_id": trajectory_id,
                        "scene_id": scene_id,
                    }
                    f.write(json.dumps(result) + "\n")

            with open(os.path.join(self.output_path, f"annotations_{rank}.json"), "w") as f:
                json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="R2R")
    parser.add_argument("--config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--output_path", type=str, default="cache/R2R")
    parser.add_argument("--data_path", type=str, default="task/r2r/train/train.json.gz")
    parser.add_argument("--render_points", action="store_true",
                        help="If set, render target points on RGB images (default: False)")
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))

    runner = StreamVLNHabitatRunner(
        dataset=args.dataset,
        config_path=args.config_path,
        output_path=args.output_path,
        data_path=args.data_path
    )
    runner.generate(rank, world_size, render_points=args.render_points)

