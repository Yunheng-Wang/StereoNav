import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import json


def load_cur_rgb(idx, video_folder, frame_offset=1):
    frame_filename = f"{idx + frame_offset:03d}.jpg"
    left_image_path = os.path.join(video_folder, 'rgb_left', frame_filename)
    right_image_path = os.path.join(video_folder, 'rgb_right', frame_filename)
    left_image = Image.open(left_image_path).convert('RGB')
    right_image = Image.open(right_image_path).convert('RGB')
    transform = transforms.ToTensor()
    left_tensor = transform(left_image) * 255.0
    right_tensor = transform(right_image) * 255.0
    left_current_frame = left_tensor.unsqueeze(0)
    right_current_frame = right_tensor.unsqueeze(0)
    return left_current_frame, right_current_frame


def load_history_rgb(init_frame_idx, history_num, video_folder, side='right', frame_offset=1):
    available_indices = list(range(0, init_frame_idx)) if init_frame_idx > 0 else []
    if len(available_indices) == 0:
        sampled_indices = []
    elif len(available_indices) <= history_num:
        sampled_indices = available_indices
    else:
        step = len(available_indices) / history_num
        sampled_indices = [int(i * step) for i in range(history_num)]
    transform = transforms.ToTensor()
    history_frames = []
    for idx in sampled_indices:
        frame_filename = f"{idx + frame_offset:03d}.jpg"
        image_path = os.path.join(video_folder, f'rgb_{side}', frame_filename)
        image = Image.open(image_path).convert('RGB')
        history_frames.append(transform(image) * 255.0)
    num_actual_frames = len(history_frames)
    if num_actual_frames < history_num:
        img_shape = history_frames[0].shape if num_actual_frames > 0 else (3, 448, 448)
        num_padding = history_num - num_actual_frames
        history_frames = [torch.zeros(img_shape) for _ in range(num_padding)] + history_frames
    return torch.stack(history_frames, dim=0)


def load_depth_left(idx, video_folder, valid_depth, frame_offset=1):
    frame_filename = f"{idx + frame_offset:03d}.png"
    depth_image_path = os.path.join(video_folder, 'depth_left', frame_filename)
    # 2. Load depth image (uint16, unit: millimeters)
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image, dtype=np.float32)
    # 3. Convert unit: millimeters -> meters
    depth_array = depth_array / 1000.0
    # 4. Set pixels greater than valid_depth to inf (invalid value, will be filtered by valid mask)
    depth_array[depth_array > valid_depth] = np.inf
    # 5. Convert to tensor and add channel dimension [448, 448] -> [1, 448, 448]
    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
    return depth_tensor


def load_label_points(idx, video_folder, frame_offset=1):
    label_points_path = os.path.join(video_folder, 'label_points.json')
    with open(label_points_path, 'r') as f:
        label_points_data = json.load(f)
    target_frame_id = idx + frame_offset
    for frame_data in label_points_data:
        if frame_data['frame_id'] == target_frame_id:
            # 3. Extract point coordinates and convert to tensor
            label_left_point = torch.tensor(frame_data['label_left_point'], dtype=torch.float32)
            label_right_point = torch.tensor(frame_data['label_right_point'], dtype=torch.float32)
            return label_left_point, label_right_point


def load_panoramic_images(folder_path, num_images=6):
    """
    Load panoramic image sequence
    Args:
        folder_path: Folder path containing images
        num_images: Number of images to load (default 6)
    Returns:
        panoramic_video: [n, 3, H, W] tensor, value range [0, 255]
    """
    transform = transforms.ToTensor()
    panoramic_frames = []
    for i in range(num_images):
        image_path = os.path.join(folder_path, f"{i}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        panoramic_frames.append(transform(image) * 255.0)
    panoramic_video = torch.stack(panoramic_frames, dim=0)
    return panoramic_video