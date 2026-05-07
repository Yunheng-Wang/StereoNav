import sys
import os
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile, ImageDraw
from collections import OrderedDict
import torchvision.transforms as transforms

import habitat
from habitat import logger, Env
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
from datetime import datetime
from scripts.client import StereoVLNClient

# Register custom measurements into the Habitat registry
from data.utils import measures  # noqa: F401


def get_camera_intrinsics(width: int, height: int, hfov: float) -> np.ndarray:
    """Calculate camera intrinsic matrix (same as gen_trajectory.py)."""
    hfov_rad = np.deg2rad(hfov)
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def project_3d_to_2d(world_point, agent_position, agent_rotation, camera_offset, K, width, height):
    """Project a 3D world point to 2D pixel coordinates (same as gen_trajectory.py).
    Returns (u, v, cam_x) with bottom-left origin. u/v=None if behind camera."""
    R_agent = quaternion.as_rotation_matrix(agent_rotation)
    camera_position = agent_position + R_agent @ camera_offset
    point_rel = world_point - camera_position
    point_cam_opengl = R_agent.T @ point_rel
    point_cam = point_cam_opengl.copy()
    point_cam[2] = -point_cam_opengl[2]  # OpenGL -> CV convention
    cam_x = point_cam[0]
    if point_cam[2] <= 0:
        return None, None, cam_x
    point_2d = K @ point_cam
    u = point_2d[0] / point_2d[2]
    v_bottom_left = height - (point_2d[1] / point_2d[2])
    return u, v_bottom_left, cam_x


def clip_to_image_bounds(u, v, width, height, cam_x):
    """Clip pixel coordinates to image bounds (same as gen_trajectory.py)."""
    v_center = height / 2.0
    if u is None or v is None:
        if cam_x < 0:
            return 0.0, v_center
        else:
            return width - 1.0, v_center
    out_of_view_threshold = 0.2
    u_margin = width * out_of_view_threshold
    v_margin = height * out_of_view_threshold
    if u < -u_margin or u > width - 1 + u_margin or v < -v_margin or v > height - 1 + v_margin:
        if cam_x < 0:
            return 0.0, v_center
        else:
            return width - 1.0, v_center
    return np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)

        # Configure stereo camera sensors for left and right eyes
        with habitat.config.read_write(self.config):
            OmegaConf.set_struct(self.config, False)
            self.config.habitat.dataset.split = self.split

            # Left RGB camera: offset 5cm to the left at eye height
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_left = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_left",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [-0.05, 1.25, 0.0],  # left eye, 5cm left of center
                "orientation": [0.0, 0.0, 0.0]
            })
            # Right RGB camera: offset 5cm to the right at eye height
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor_right = OmegaConf.create({
                "type": "HabitatSimRGBSensor",
                "uuid": "rgb_right",
                "width": 448,
                "height": 448,
                "hfov": 79,
                "position": [0.05, 1.25, 0.0],  # right eye, 5cm right of center
                "orientation": [0.0, 0.0, 0.0]
            })

            # Add task measurements for top-down map and collision tracking
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            OmegaConf.set_struct(self.config, True)

        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        self.model = model
        self.history_num = args.history_num
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.execute_steps = args.execute_steps
        self.max_steps = args.max_steps
        self.rotation_perturbation = args.rotation_perturbation
        self.perturbation_probability = args.perturbation_probability
        self.image_size = (448, 448)

        # Track perturbation direction: True for right (+), False for left (-)
        self.perturbation_direction = True

        # Set random seed for reproducibility
        seed = 42 + get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # StereoVLN action mapping
        self.actions2idx = OrderedDict({
            'stop here': 0,
            'move forward': 1,
            'turn left': 2,
            'turn right': 3,
        })
        self.actionsmapping = {
            0: 'stop here',
            1: 'move forward',
            2: 'turn left',
            3: 'turn right',
        }

        self.transform = transforms.ToTensor()

        # Stereo projection parameters (must match gen_trajectory.py)
        self.img_width = 448
        self.img_height = 448
        self.hfov = 79
        self.K = get_camera_intrinsics(self.img_width, self.img_height, self.hfov)
        self.camera_offset_left = np.array([-0.05, 1.25, 0.0])
        self.camera_offset_right = np.array([0.05, 1.25, 0.0])

    def draw_point_on_tensor(self, frame_tensor, point, radius=5):
        """Draw a red point on a [1, 3, H, W] tensor (values 0~255). Returns new tensor.
        point is (u, v) in bottom-left origin, converted to top-left for PIL drawing."""
        if point is None:
            return frame_tensor
        img = transforms.ToPILImage()(frame_tensor.squeeze(0) / 255.0)
        draw = ImageDraw.Draw(img)
        u, v_bottom_left = float(point[0]), float(point[1])
        # Convert bottom-left origin to top-left for PIL drawing
        v_top_left = self.img_height - v_bottom_left
        draw.ellipse(
            [(u - radius, v_top_left - radius), (u + radius, v_top_left + radius)],
            fill='red', outline='red'
        )
        tensor = self.transform(img) * 255.0
        return tensor.unsqueeze(0)

    def project_goal_to_stereo(self, env, goal_position):
        """Project goal position onto left and right camera views.
        Returns (left_point, right_point) each as (u, v) in bottom-left origin."""
        agent_state = env.sim.get_agent_state()
        agent_position = np.array(agent_state.position)
        agent_rotation = agent_state.rotation
        # Offset goal to camera height to match training data projection
        goal_camera_center = goal_position + np.array([0.0, 1.25, 0.0])

        u_left, v_left, cam_x_left = project_3d_to_2d(
            goal_camera_center, agent_position, agent_rotation,
            self.camera_offset_left, self.K, self.img_width, self.img_height
        )
        u_left, v_left = clip_to_image_bounds(u_left, v_left, self.img_width, self.img_height, cam_x_left)

        u_right, v_right, cam_x_right = project_3d_to_2d(
            goal_camera_center, agent_position, agent_rotation,
            self.camera_offset_right, self.K, self.img_width, self.img_height
        )
        u_right, v_right = clip_to_image_bounds(u_right, v_right, self.img_width, self.img_height, cam_x_right)

        return (u_left, v_left), (u_right, v_right)

    def preprocess_rgb(self, rgb_image):
        """Preprocess RGB image to [1, 3, 448, 448] tensor with values 0~255"""
        image = Image.fromarray(rgb_image).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)
        tensor = self.transform(image) * 255.0
        return tensor.unsqueeze(0)  # [1, 3, 448, 448]

    def apply_agent_rotation_perturbation(self, env):
        """Apply fixed camera roll perturbation to agent in simulator, alternating left and right."""
        if self.rotation_perturbation <= 0:
            return env.sim.get_observations_at(env.sim.get_agent_state())

        if random.random() > self.perturbation_probability:
            return env.sim.get_observations_at(env.sim.get_agent_state())

        agent_state = env.sim.get_agent_state()
        original_rotation = agent_state.rotation.copy()
        agent_position = agent_state.position.copy()

        # Use fixed perturbation value, alternating between right (+) and left (-)
        angle_degrees = self.rotation_perturbation if self.perturbation_direction else -self.rotation_perturbation
        angle_rad = np.deg2rad(angle_degrees)

        # Toggle direction for next call
        self.perturbation_direction = not self.perturbation_direction

        roll_rotation = quaternion.from_rotation_vector([0, 0, angle_rad])
        perturbed_rotation = original_rotation * roll_rotation

        env.sim.set_agent_state(agent_position, perturbed_rotation)
        perturbed_observations = env.sim.get_observations_at(env.sim.get_agent_state())
        env.sim.set_agent_state(agent_position, original_rotation)

        return perturbed_observations

    def get_stereo_images(self, env, observations):
        """
        Get stereo images from habitat observations.
        Uses dual RGB sensors configured for left and right eyes.
        If rotation_perturbation > 0, applies agent rotation perturbation in simulator.
        """
        if self.rotation_perturbation > 0:
            # Get observations with agent rotation perturbation
            observations = self.apply_agent_rotation_perturbation(env)

        rgb_left = observations["rgb_left"]
        rgb_right = observations["rgb_right"]

        left_frame = self.preprocess_rgb(rgb_left)
        right_frame = self.preprocess_rgb(rgb_right)
        return left_frame, right_frame

    def build_history_video(self, history_frames):
        """
        Build history video tensor from a list of RGB tensors.
        Returns [history_num, 3, 448, 448] tensor with zero padding if needed.
        """
        num_actual = len(history_frames)
        if num_actual < self.history_num:
            img_shape = (3, self.image_size[0], self.image_size[1])
            num_padding = self.history_num - num_actual
            padding_frames = [torch.zeros(img_shape) for _ in range(num_padding)]
            history_frames = padding_frames + history_frames
        elif num_actual > self.history_num:
            # Uniform sampling
            step = num_actual / self.history_num
            indices = [int(i * step) for i in range(self.history_num)]
            history_frames = [history_frames[i] for i in indices]

        return torch.stack(history_frames, dim=0)  # [history_num, 3, 448, 448]

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sucs, spls, oss, ones, tls = [], [], [], [], []
        done_res = []
        first_episode_saved = False  # Track if first episode video has been saved
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'), 'r') as f:
                for line in f.readlines():
                    try:
                        res = json.loads(line)
                        if "scene_id" in res:  # Skip summary lines
                            done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                            if get_rank() == 0:
                                sucs.append(res['success'])
                                spls.append(res['spl'])
                                oss.append(res['os'])
                                ones.append(res['ne'])
                                tls.append(res.get('tl', 0))
                    except:
                        continue

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")

            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start", episode_instruction)
                episode_id = episode.episode_id

                if [scene_id, episode_id, episode_instruction] in done_res:
                    process_bar.update(1)
                    continue

                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb_left']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))

                vis_frames = []
                step_id = 0
                is_first_episode = not first_episode_saved

                if self.save_video or is_first_episode:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)

                initial_height = env.sim.get_agent_state().position[1]

                # Trajectory length tracking
                prev_position = np.array(env.sim.get_agent_state().position)
                trajectory_length = 0.0

                # History tracking
                left_history_frames = []   # List of [3, 448, 448] tensors
                right_history_frames = []  # List of [3, 448, 448] tensors
                executed_actions = []  # List of action indices
                action_seq = []  # Pending actions to execute

                # Goal position from episode metadata (first goal waypoint)
                goal_position = np.array(episode.goals[0].position)

                while not env.episode_over:
                    if self.max_steps > 0 and step_id >= self.max_steps:
                        # Force stop when step limit is reached; counts as success if within 3m
                        observations = env.step(0)
                        break
                    self.model.eval()

                    # Get stereo images
                    left_frame, right_frame = self.get_stereo_images(env, observations)

                    # Project goal onto stereo views and draw as red dot (matches training data)
                    left_point, right_point = self.project_goal_to_stereo(env, goal_position)
                    left_frame = self.draw_point_on_tensor(left_frame, left_point)
                    right_frame = self.draw_point_on_tensor(right_frame, right_point)

                    # Save frames for video (convert tensor to numpy for visualization)
                    if self.save_video or is_first_episode:
                        # Combine left and right frames side by side
                        left_img = (left_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                        right_img = (right_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                        combined_frame = np.hstack([left_img, right_img])
                        vis_frames.append(combined_frame)

                    # Generate new actions if queue is empty
                    if len(action_seq) == 0:
                        # Build history video
                        left_history_video = self.build_history_video(left_history_frames)
                        right_history_video = self.build_history_video(right_history_frames)

                        # Build history action string
                        if step_id == 0:
                            history_action = "This is the initial timestep, so no previous action sequence is available."
                        else:
                            history_action_strs = [self.actionsmapping[a] for a in executed_actions]
                            history_action = ",".join(history_action_strs)

                        # Prepare inputs for model
                        left_current_frame = left_frame.unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # [1, 1, 3, 448, 448]
                        right_current_frame = right_frame.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                        left_history_video_tensor = left_history_video.unsqueeze(0).to(self.device, dtype=torch.bfloat16)  # [1, history_num, 3, 448, 448]
                        right_history_video_tensor = right_history_video.unsqueeze(0).to(self.device, dtype=torch.bfloat16)

                        # Run inference
                        outputs = self.model.inference(
                            instruction=[episode_instruction],
                            history_action=[history_action],
                            left_current_frame=left_current_frame,
                            right_current_frame=right_current_frame,
                            left_history_video=left_history_video_tensor,
                            right_history_video=right_history_video_tensor,
                            depth_iters = 8,
                            max_new_tokens=24,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            output_point = False,
                            output_depth = False,
                        )

                        generated_text = outputs['action'][0]
                        print(f"Step {step_id}, Generated: {generated_text}", flush=True)

                        # Parse actions from generated text
                        action_seq = self.parse_actions(generated_text)
                        if self.execute_steps > 0:
                            action_seq = action_seq[:self.execute_steps]
                        print(f"Parsed actions: {action_seq}", flush=True)

                        if len(action_seq) == 0:
                            # If no valid actions parsed, default to stop
                            action_seq = [0]

                    # Execute action
                    action = action_seq.pop(0)
                    executed_actions.append(action)

                    # Update history (add current left and right frames before action)
                    left_history_frames.append(left_frame.squeeze(0))   # [3, 448, 448]
                    right_history_frames.append(right_frame.squeeze(0))  # [3, 448, 448]

                    # Step environment
                    observations = env.step(action)
                    step_id += 1

                    # Update trajectory length
                    current_position = np.array(env.sim.get_agent_state().position)
                    trajectory_length += np.linalg.norm(current_position - prev_position)
                    prev_position = current_position

                process_bar.update(1)
                metrics = env.get_metrics()

                if self.save_video or is_first_episode:
                    try:
                        if len(vis_frames) > 0:
                            # Save frames as images first
                            frame_dir = os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}')
                            os.makedirs(frame_dir, exist_ok=True)
                            for frame_idx, frame in enumerate(vis_frames):
                                frame_path = os.path.join(frame_dir, f'frame_{frame_idx:06d}.jpg')
                                Image.fromarray(frame).save(frame_path)

                            # Try to convert to video
                            try:
                                images_to_video(
                                    vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                                )
                            except:
                                pass  # Video conversion failed, but frames are saved

                            if is_first_episode:
                                first_episode_saved = True
                                print(f"[Info] First episode saved: {scene_id}_{episode_id} with {len(vis_frames)} frames")
                        else:
                            print(f"[Warning] No frames for {scene_id}_{episode_id}")
                    except Exception as e:
                        print(f"[Error] Failed to save {scene_id}_{episode_id}: {e}")
                vis_frames.clear()

                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                tls.append(trajectory_length)

                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}, tl: {trajectory_length:.2f}")

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "tl": trajectory_length,
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }

                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(tls).to(self.device), torch.tensor(len(sucs)).to(self.device)

    def parse_actions(self, output):
        """
        Parse action sequence from model output.
        Expected format: "move forward,turn left,turn right,stop here"
        """
        parts = output.strip().split(',')
        actions = []
        for part in parts:
            p = part.strip().lower().strip('.,;:!?')
            if p in self.actions2idx:
                actions.append(self.actions2idx[p])
        return actions


def load_model(args):
    """Connect to remote StereoVLN server. Supports multi-server dispatch by rank."""
    server_urls = [u.strip() for u in args.server_urls.split(',')]
    rank = get_rank()
    url = server_urls[rank % len(server_urls)]
    print(f"[Rank {rank}] Connecting to StereoVLN server at {url}")
    return StereoVLNClient(server_url=url)


def evaluate(model, args):
    model.eval()

    world_size = get_world_size()
    args.output_path = os.path.join(args.output_path, datetime.now().strftime("%Y-%m-%d_%H"))
    os.makedirs(args.output_path, exist_ok=True)

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )

    sucs, spls, oss, ones, tls, ep_num = evaluator.eval_action(get_rank())

    # Gather results from all processes
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)

    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    tls_all = [torch.zeros(ep_num_all[i], dtype=tls.dtype).to(tls.device) for i in range(world_size)]

    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.all_gather(tls_all, tls)
    dist.barrier()

    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    tls_all = torch.cat(tls_all, dim=0)

    result_all = {
        "success_rate": (sum(sucs_all) / len(sucs_all)).item(),
        "spl": (sum(spls_all) / len(spls_all)).item(),
        "oracle_success": (sum(oss_all) / len(oss_all)).item(),
        "navigation_error": (sum(ones_all) / len(ones_all)).item(),
        "trajectory_length": (sum(tls_all) / len(tls_all)).item(),
        'num_episodes': len(sucs_all)
    }

    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))




def eval():
    global local_rank
    parser = argparse.ArgumentParser()

    parser.add_argument("--server_urls", type=str, default="http://localhost:5001",
                        help="Comma-separated server URLs, e.g. http://localhost:5000,http://localhost:5001")
    parser.add_argument("--habitat_config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default='./results')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling top_p")
    parser.add_argument("--execute_steps", type=int, default=4, help="max actions to execute per inference call (0=all)")
    parser.add_argument("--max_steps", type=int, default=500, help="max steps per episode (0=unlimited, use habitat default)")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--history_num", type=int, default=8)
    parser.add_argument("--rotation_perturbation", type=float, default=0, help="random rotation perturbation range in degrees (0=no perturbation)")
    parser.add_argument("--perturbation_probability", type=float, default=1.0, help="probability of applying perturbation (0-1)")

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    # Load model client
    model = load_model(args)

    # Run evaluation
    evaluate(model, args)


if __name__ == "__main__":
    eval()
