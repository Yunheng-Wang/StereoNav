import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass 
class InternVLConfig:
    checkpoint_path: str ="/home/CONNECT/yfang870/yunhengwang/StereoVLN/model/base/InternVL3_5-2B"
    dtype: torch.dtype = torch.bfloat16
    image_size: tuple = (448, 448)
    max_tokens: int = 4096