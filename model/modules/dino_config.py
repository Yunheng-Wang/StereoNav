import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass 
class DINOv2Config:
    image_size: Tuple[int, int] = (448, 448)
    dtype: torch.dtype = torch.bfloat16
    dino_path : str = "checkpoints/dinov2_large/model.safetensors"