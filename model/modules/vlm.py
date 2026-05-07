import torch 
from torch import nn
from transformers import AutoTokenizer, AutoModel
from .vlm_config import InternVLConfig
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


class InternVLModel(nn.Module):
    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config
        # 1. 加载InterVL模型
        self.model = AutoModel.from_pretrained(config.checkpoint_path, torch_dtype=config.dtype, use_flash_attn=True, attn_implementation="flash_attention_2", trust_remote_code=True)
        # 2. 冻结视觉编码器 & 激活其他层
        self.model.vision_model.eval()
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        self.model.language_model.train()
        for param in self.model.language_model.parameters():
            param.requires_grad = True
        # 3. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path, trust_remote_code=True, use_fast=False, model_max_length = config.max_tokens, padding_side="right")
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['<dep>', '</dep>']})
        # self.model.language_model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of = 64)
        # 4. 加载标准化层
        self.normalize = self._load_normalize()


    def _load_normalize(self):
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
        transform = T.Normalize(mean=MEAN, std=STD)
        return transform


    def Encoder_Vsion(self, Image: torch.Tensor): # Image -> [B, n, 3, 448, 448]
        # 0. 重塑形状
        B, N, C, H, W = Image.shape
        Image = Image.view(-1, C, H, W)
        # 1. 标准化处理
        Image = self.normalize(Image / 255.0)
        # 2. 编码图像
        vit_embeds = self.model.vision_model(pixel_values=Image, output_hidden_states=False, return_dict=True).last_hidden_state
        # 3. 剔除 CLS token
        vit_embeds = vit_embeds[:, 1:, :]
        # 4. 降采样
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.model.pixel_shuffle(vit_embeds, scale_factor=self.model.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.model.mlp1(vit_embeds)
        # 5. 恢复形状
        vit_embeds = vit_embeds.view(B, N, vit_embeds.shape[1], vit_embeds.shape[2])
        return vit_embeds[:, :-1, :, :], vit_embeds[:, -1:, :, :]


if __name__ == "__main__":
    pass
