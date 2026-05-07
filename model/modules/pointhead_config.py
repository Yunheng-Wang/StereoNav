import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass 
class PointHeadConfig:
    dtype: torch.dtype = torch.bfloat16
    dim: int = 2048
    img_size: Tuple = (448, 448)
    mlp_ratio: float = 0.5
    dropout: float = 0.1