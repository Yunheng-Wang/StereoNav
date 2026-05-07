import torch 
import timm
from torch import nn
import torchvision.transforms as tvtf
from .dino_config import DINOv2Config
from .utils.block import ResBlock

class DINOv2(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        # 1. 加载基本配置
        self.config = config
        # 2. 加载模型 & 冻结
        # self.model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size = self.config.image_size)
        self.model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=False, num_classes=0, checkpoint_path=self.config.dino_path)
        self.model.to(self.config.dtype)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # 3. 标准化层
        self.normalize = self._load_normalize()
        # 4. 获取模型维度
        self.dim = self.model.embed_dim
        # 5. 构建压缩映射层
        self.compressor = self._build_compressor()
        self.compressor.train()
        for param in self.compressor.parameters():
            param.requires_grad = True
        # 6. compressor 输出归一化
        self.token_norm = nn.LayerNorm(2048).to(self.config.dtype)



    def _build_compressor(self):
        return nn.Sequential(
            nn.Conv2d(self.dim, 1536, kernel_size=1),
            nn.GroupNorm(32, 1536),
            nn.SiLU(),

            # 这一步将 37x37 变成 19x19
            nn.Conv2d(1536, 2048, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 2048),
            nn.SiLU(),

            nn.AdaptiveMaxPool2d((16, 16)) 
        ).to(self.config.dtype)


    def _load_normalize(self):
        model_cfg = timm.data.resolve_model_data_config(self.model)
        model_cfg["input_size"] = (3, 518, 518)
        transform = timm.data.create_transform(**model_cfg, is_training=False)
        resize_transform = tvtf.Compose(
            [
                tvtf.Resize(
                    self.config.image_size, interpolation=transform.transforms[0].interpolation
                ),
                *transform.transforms[1:],
            ]
        )
        return resize_transform


    def DINOv2Encoder(self, left_current_frame):
        # 2. 标准化
        left_current_frame = left_current_frame / 255.0
        left_current_frame = self.normalize(left_current_frame) 
        # 3. 提取特征
        output = self.model.get_intermediate_layers(left_current_frame, n={len(self.model.blocks) - 2})[0] 
        output = output.reshape(output.shape[0], int(output.shape[1]**0.5), int(output.shape[1]**0.5), output.shape[2]).permute(0, 3, 1, 2).contiguous()
        # 3. 压缩映射
        output = self.compressor(output)
        output = output.permute(0, 2, 3, 1).reshape(output.shape[0], 256, 2048)
        output = self.token_norm(output)
        return output

        