import torch 
from torch import nn
from typing import List, Optional
from .pointhead_config import PointHeadConfig


class PointHead(nn.Module):
    def __init__(self, config: PointHeadConfig):
        super().__init__()
        self.config = config
        hidden_dim = max(64, int(self.config.dim * self.config.mlp_ratio))
        self.reduce = nn.Conv2d(self.config.dim, hidden_dim, kernel_size=1, bias=False).to(self.config.dtype)
        groups = 8
        if hidden_dim % groups != 0:
            groups = 4 if hidden_dim % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, hidden_dim).to(self.config.dtype)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.config.dtype)
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        ).to(self.config.dtype)
        self.heatmap = nn.Conv2d(hidden_dim, 1, kernel_size=1).to(self.config.dtype)
        self.offset = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        ).to(self.config.dtype)
        self.logit_scale = nn.Parameter(torch.tensor(0.0)).to(self.config.dtype)


    def _build_grid(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        return grid


    def forward(
        self, tokens: torch.Tensor, token_hw: Optional[tuple] = None
    ) -> torch.Tensor:
        tok_len = tokens.shape[1]
        w = h = int(tok_len**0.5)
        grid = self._build_grid(h, w, tokens.device, torch.float32)
        feat = tokens.reshape(tokens.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
        feat = self.reduce(feat)
        feat = self.norm(feat)
        pos = self.pos_mlp(grid.to(tokens.dtype)).reshape(1, h, w, -1).permute(0, 3, 1, 2)
        feat = feat + pos
        feat = feat + self.block(feat)
        logits = self.heatmap(feat).flatten(1)
        scale = self.logit_scale.exp().clamp(0.1, 10.0)
        weights = torch.softmax((logits * scale).float(), dim=1)
        coords_norm = weights @ grid
        offset = self.offset(feat).float()
        cell = torch.tensor([1.0 / w, 1.0 / h], device=coords_norm.device)
        coords_norm = (coords_norm + 0.5 * offset * cell).clamp(0.0, 1.0)
        img_h, img_w = self.config.img_size
        coords = coords_norm * torch.tensor(
            [img_w, img_h], device=coords_norm.device
        )
        return coords



