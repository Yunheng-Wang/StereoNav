import torch.utils.data as data
import torch
import os
import random
import json
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.utils.load import load_cur_rgb, load_history_rgb, load_depth_left, load_label_points, load_panoramic_images

logger = logging.getLogger(__name__)

class Dataset_Normal(data.Dataset):
    def __init__(self, config):
        self.valid_depth = config.main.valid_depth
        self.dataset_root = os.path.join(config.main.data_root, "train")
        self.predict_num = config.main.prediction_steps
        self.history_num = config.main.history_steps
        self.beta = config.main.beta
        self.window_size = config.main.window_size
        self.train_datasets = list(config.main.get("train_datasets", ["r2r", "rxr"]))
        self.image_size = (448, 448)
        self.actionsmapping = {
            '0': 'stop here',
            '1': 'move forward',
            '2': 'turn left',
            '3': 'turn right',
        }
        self.num_episodes = None
        self.all_episodes = self._load_episodes()


    def _load_episodes(self):
        episodes = []
        window_size = self.window_size
        n_samples = self.beta
        # 1. 收集所有含 summary.json 的叶子文件夹
        # dagger 下还有一层 r2r/rxr 子目录，其余数据集直接在 dataset_root/<name>/ 下
        allowed = set(d.lower() for d in self.train_datasets) if self.train_datasets else None
        leaf_folders = []
        for folder_name in os.listdir(self.dataset_root):
            if allowed and folder_name.lower() not in allowed:
                continue
            folder_path = os.path.join(self.dataset_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if os.path.exists(os.path.join(folder_path, "summary.json")):
                leaf_folders.append(folder_path)
            else:
                # 多一层子目录（如 dagger/r2r, dagger/rxr）
                for sub in os.listdir(folder_path):
                    sub_path = os.path.join(folder_path, sub)
                    if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "summary.json")):
                        leaf_folders.append(sub_path)

        for folder_path in leaf_folders:
            summary_path = os.path.join(folder_path, "summary.json")
            if not os.path.exists(summary_path):
                continue
            is_dagger = "dagger" in folder_path
            # 3. 读取每一行 trajectory
            with open(summary_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trajectory = json.loads(line)
                    # 4. 提取基本信息
                    video_path = os.path.join(folder_path, trajectory['video'])
                    instruction = trajectory['instructions'][0]  # 取第一条指令
                    actions = list(trajectory['actions'])  # 复制，避免污染原始数据
                    if not is_dagger:
                        actions.append(0)  # 添加停止动作
                    frame_offset = 0 if is_dagger else 1
                    # 5. 滑动窗口 + 随机采样
                    num_frames = len(actions)
                    max_start = num_frames - 2  # 合法起始帧索引上界
                    window_start = 0
                    while window_start <= max_start:
                        window_end = min(window_start + window_size - 1, max_start)
                        is_first = (window_start == 0)
                        is_last = (window_end >= max_start)
                        candidates = list(range(window_start, window_end + 1))
                        k = 3 * n_samples // 2 if (is_first or is_last) else n_samples
                        k = min(k, len(candidates))
                        sampled_starts = random.sample(candidates, k)
                        for start_idx in sampled_starts:
                            episode = {
                                "instruction": instruction,
                                "actions": actions,
                                "video_path": video_path,
                                "init_obers_idx": start_idx,
                                "init_action_idx": start_idx + 1,
                                "frame_offset": frame_offset,
                            }
                            episodes.append(episode)
                        window_start += window_size
        self.num_episodes = len(episodes)
        logger.info(f"Total amount of data: {len(episodes)}")
        return episodes
    

    def __len__(self):
        return len(self.all_episodes)


    def __getitem__(self, idx):
        # 1. 选择 episode
        episode = self.all_episodes[idx]
        init_frame_idx = episode['init_obers_idx']
        video_folder = episode['video_path']
        frame_offset = episode.get('frame_offset', 1)
        # 2. 获取指令
        instruction = episode['instruction']
        # 3. 获取左右视角的初始帧
        left_current_frame, right_current_frame = load_cur_rgb(init_frame_idx, video_folder, frame_offset)
        # 4. 获取左右视角的历史帧
        left_history_video = load_history_rgb(init_frame_idx, self.history_num, video_folder, side='left', frame_offset=frame_offset)
        right_history_video = load_history_rgb(init_frame_idx, self.history_num, video_folder, side='right', frame_offset=frame_offset)
        # 5. 获取深度图像 [1, 448, 448], 单位：毫米
        label_depth = load_depth_left(init_frame_idx, video_folder, self.valid_depth, frame_offset)
        # 6. 获取左右点标签
        label_left_point, label_right_point = load_label_points(init_frame_idx, video_folder, frame_offset)
        # 7. 获取历史动作
        init_action_idx = episode['init_action_idx']
        actions = episode['actions']
        if init_frame_idx == 0:
            # 没有历史动作
            history_action = "This is the initial timestep, so no previous action sequence is available."
        else:
            history_action_indices = actions[1:init_action_idx]
            history_action_strs = [self.actionsmapping[str(action)] for action in history_action_indices]
            history_action = ",".join(history_action_strs)
        # 8. 获取动作标签
        action_end_idx = min(init_action_idx + self.predict_num, len(actions))
        label_action_indices = actions[init_action_idx:action_end_idx]
        label_action_strs = [self.actionsmapping[str(action)] for action in label_action_indices]
        # 8.1 最后的数据块出现不足的情况 用 stop here 补齐
        while len(label_action_strs) < self.predict_num:
            label_action_strs.append('stop here')
        label_answer = ",".join(label_action_strs)
        # 9. 返回数据
        return {
            "instruction": instruction,
            "history_action": history_action,
            "left_current_frame": left_current_frame,
            "right_current_frame": right_current_frame,
            "left_history_video": left_history_video,
            "right_history_video": right_history_video,
            "label_left_point": label_left_point,
            "label_right_point": label_right_point,
            "label_depth": label_depth,
            "label_answer": label_answer,
        }


class Dataset_World(data.Dataset):
    def __init__(self, config):
        self.dataset_root = os.path.join(config.main.data_root, "train")
        self.num_episodes = None
        self.all_episodes = self._load_episodes()


    def _load_episodes(self):
        episodes = []

        # 1. 预先收集所有有效的文件夹路径
        all_folders = []
        for dataset_name in os.listdir(self.dataset_root):
            dataset_path = os.path.join(self.dataset_root, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            for folder_name in os.listdir(dataset_path):
                folder_path = os.path.join(dataset_path, folder_name)
                if os.path.isdir(folder_path):
                    all_folders.append(folder_path)

        # 2. 定义加载单个 episode 的函数
        def load_single_episode(folder_path):
            metadata_path = os.path.join(folder_path, "metadata.json")
            if not os.path.exists(metadata_path):
                return None
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    "folder_path": folder_path,
                    "instruction": metadata["instruction"],
                    "angle_to_goal_deg": metadata["angle_to_goal_deg"],
                    "distance_to_goal_m": metadata["distance_to_goal_m"],
                    "height_diff_m": metadata["height_diff_m"]
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {metadata_path}: {e}")
                return None

        # 3. 使用多线程并行加载（8 个 worker）
        logger.info(f"Loading {len(all_folders)} world model episodes with 8 workers...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(load_single_episode, folder): folder for folder in all_folders}
            for future in tqdm(as_completed(futures), total=len(all_folders), desc="Loading episodes", unit="folder"):
                result = future.result()
                if result:
                    episodes.append(result)

        logger.info(f"Total world model episodes: {len(episodes)}")
        self.num_episodes = len(episodes)
        return episodes


    def __len__(self):
        return len(self.all_episodes)


    def __getitem__(self, idx):
        episode = self.all_episodes[idx]
        folder_path = episode["folder_path"]
        panoramic_images = load_panoramic_images(folder_path, num_images=6)

        # 组织 label_answer，格式匹配 user_world 的期望
        label_answer = (
            f"Yaw: {round(episode['angle_to_goal_deg'])} deg; "
            f"Distance: {round(episode['distance_to_goal_m'])} m; "
            f"Height_Diff: {round(episode['height_diff_m'])} m"
        )

        return {
            "panoramic_images": panoramic_images,
            "instruction": episode["instruction"],
            "label_answer": label_answer
        }
    