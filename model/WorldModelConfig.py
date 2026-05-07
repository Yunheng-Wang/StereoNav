import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class WorldModelConfig:
    # VLM 配置
    vlm_checkpoint_path: str = "/home/CONNECT/yfang870/yunhengwang/StereoVLN/model/base/InternVL3_5-2B"
    vlm_dtype: str = "bfloat16"  # "bfloat16" 或 "float16"
    vlm_image_size: Tuple[int, int] = (448, 448)
    vlm_max_tokens: int = 4096