import torch 
from torch import nn
from transformers import AutoTokenizer, AutoModel
from .qwen_3_5_config import QwenConfig
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        # 1. 加载Qwen_3_5模型
        self.model = AutoModel.from_pretrained(config.checkpoint_path, dtype=config.dtype, attn_implementation="flash_attention_2", trust_remote_code=True)
        # 2. 冻结视觉编码器 & 激活其他层
        for param in self.model.visual.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if not name.startswith("visual"):
                param.requires_grad = True
        # 3. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path, trust_remote_code=True, use_fast=False, model_max_length = config.max_tokens, padding_side="right")
    

    @torch.no_grad()
    def Encoder_Vision(self, Video: torch.Tensor): # Video -> [B, n, 3, 480, 640]
        B, n, C, H, W = Video.shape
        visual = self.model.visual
        temporal_patch_size = visual.patch_embed.temporal_patch_size  # 2
        patch_size = visual.patch_embed.patch_size                    # 16
        spatial_merge_size = visual.spatial_merge_size                # 2

        # 1. 归一化: [0, 255] -> [0, 1] -> normalize(mean=0.5, std=0.5) -> [-1, 1]
        Video = Video.float() / 255.0
        Video = (Video - 0.5) / 0.5

        # 2. 帧数补齐为 temporal_patch_size 的倍数（不足则重复最后一帧）
        if n % temporal_patch_size != 0:
            pad_n = temporal_patch_size - (n % temporal_patch_size)
            pad_frames = Video[:, -1:].expand(B, pad_n, C, H, W)
            Video = torch.cat([Video, pad_frames], dim=1)
            n = Video.shape[1]

        # 3. 计算 grid_thw: 经过 patch 后的网格尺寸
        grid_t = n // temporal_patch_size
        grid_h = H // patch_size
        grid_w = W // patch_size
        # grid_thw: [B, 3]，每个视频一条
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * B,
            dtype=torch.long,
            device=Video.device
        )

        # 4. 将视频展平为 patch 序列，供 PatchEmbed 的 Conv3d 处理
        #    [B, n, C, H, W] -> [B, C, n, H, W]
        Video = Video.permute(0, 2, 1, 3, 4)
        #    reshape 为 patches: (B * grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size * patch_size)
        Video = Video.reshape(
            B,
            C, grid_t, temporal_patch_size,
            grid_h, patch_size,
            grid_w, patch_size
        )
        Video = Video.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B, grid_t, grid_h, grid_w, C, tp, p, p]
        hidden_states = Video.reshape(-1, C * temporal_patch_size * patch_size * patch_size)

        # 5. 送入视觉编码器
        outputs = visual(hidden_states=hidden_states, grid_thw=grid_thw)
        merged_tokens = outputs.pooler_output  # [total_tokens, out_hidden_size]

        # 6. 拆分 batch 并 reshape
        tokens_per_video = grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)
        visual_tokens = merged_tokens.reshape(B, tokens_per_video, -1)

        return visual_tokens

if __name__ == "__main__":
    pass
