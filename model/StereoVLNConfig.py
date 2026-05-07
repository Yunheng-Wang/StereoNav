import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class StereoVLNConfig:
    image_size: Tuple = (448, 448)
    dtype: torch.dtype = torch.bfloat16
    dim: int = 2048

    # VLM Setting
    max_tokens: int = 4096
    vlm_checkpoints_path: str = "model/base/InternVL3_5-2B"

    # Depth Estimation Setting
    camera_k: torch.Tensor = field(default_factory=lambda: torch.tensor([[754.6681, 0.0, 489.3795], [0.0, 754.6681, 265.16162], [0.0, 0.0, 1.0]], dtype=torch.float32))
    camera_baseline: float = 0.1
    foundationstereo_checkpoints_path: str = "model/base/FoundationStereo/checkpoints/23-51-11"
    foundationstereo_edgenext_path : str = "checkpoints/edgenext_small/model.safetensors"

    # Point Head Setting
    mlp_ratio: float = 0.5
    dropout: float = 0.1

    # Dino Setting
    dino: str = "checkpoints/dinov2_large/model.safetensors"

    # Prediction Setting
    prediction_steps: int = 4

    # Weights
    depth_token_weight: float = 0.2
    dino_token_weight: float = 0.2


