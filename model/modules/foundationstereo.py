import torch 
import timm
import os,sys
from torch import nn
from omegaconf import OmegaConf
from ..base.FoundationStereo.core.foundation_stereo import FoundationStereo, normalize_image
from ..base.FoundationStereo.core.submodule import build_gwc_volume, build_concat_volume, disparity_regression
from ..base.FoundationStereo.core.geometry import Combined_Geo_Encoding_Volume
from .foundationstereo_config import FoundationStereoConfig
import torchvision.transforms as tvtf
from .utils.block import ResBlock
import torch.nn.functional as F


class FoundationStereoModel(nn.Module):
    def __init__(self, config: FoundationStereoConfig):
        super().__init__()
        # 1. 加载 FoundationStereo 模型
        self.config = config
        if torch.all(self.config.intrinsic == 0):
            raise ValueError("You must provide a valid intrinsic matrix.")
        self.model = self._load_model(self.config.edgenext_path)
        # 2. 冻结 FoundationStereo Encoder; 激活 FoundationStereo Decoder
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        decoder_modules = [self.model.classifier, self.model.cnet, self.model.cam,
                           self.model.sam, self.model.stem_2, self.model.update_block,
                           self.model.spx_2_gru, self.model.spx_gru]
        for module in decoder_modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
        if hasattr(self.model.cnet, 'gradient_checkpointing_enable'):
            self.model.cnet.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # # 2. 激活 FoundationStereo Encoder; 激活 FoundationStereo Decoder
        # decoder_modules = [self.model.classifier, self.model.cnet, self.model.cam,
        #                    self.model.sam, self.model.stem_2, self.model.update_block,
        #                    self.model.spx_2_gru, self.model.spx_gru]
        # for param in self.model.parameters():
        #     param.requires_grad = True
        # self.model.train()
        
        # 3. token 压缩卷积层
        self.compressor = self._build_compressor()
        self.compressor.train()
        for param in self.compressor.parameters():
            param.requires_grad = True
        # 4. token 还原卷积层
        self.decompressor = self._build_decompressor()
        self.decompressor.train()
        for param in self.decompressor.parameters():
            param.requires_grad = True
        # 5. compressor 输出归一化
        self.token_norm = nn.LayerNorm(self.config.dim).to(self.config.dtype)


    def _load_model(self, edgenext_path):
        cfg = OmegaConf.load(f'{self.config.checkpoint_path}/cfg.yaml')
        cfg['vit_size'] = 'vitl'
        args = OmegaConf.create(cfg)
        model = FoundationStereo(args, edgenext_path)
        ckpt = torch.load(os.path.join(self.config.checkpoint_path, "model_best_bp2.pth"), weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.to(self.config.dtype)
        return model
    

    def _build_compressor(self):
        return nn.Sequential(
            nn.ConstantPad2d(8, 0),  # 112 -> 128

            nn.Conv2d(2912, 2048, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.GroupNorm(32, 2048),
            nn.SiLU(),

            nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.GroupNorm(32, 2048),
            nn.SiLU(),

            nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.GroupNorm(32, 2048),
            nn.SiLU(),
        ).to(self.config.dtype)
    

    def _build_decompressor(self):
        return nn.Sequential(
            # 16 -> 32
            nn.Conv2d(2048, 1024 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(32, 1024), nn.SiLU(),

            # 32 -> 64
            nn.Conv2d(1024, 512 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(32, 512), nn.SiLU(),

            # 64 -> 128
            nn.Conv2d(512, 2912 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        ).to(self.config.dtype)

    def FoundationStereoEncoder(self, left_current_frame, right_current_frame): # left_current_frame/right_current_frame: [B, 3, 448, 448]
        # 1. 查看输入图片数量
        left_num = len(left_current_frame)
        # 2. 标准化图片
        left_current_frame = normalize_image(left_current_frame)
        right_current_frame = normalize_image(right_current_frame)
        # 3. 提取特征
        out, vit_feat = self.model.feature(torch.cat([left_current_frame, right_current_frame], dim=0))
        left_vit_feat = vit_feat[:left_num]
        features_left = [o[:left_num] for o in out]
        features_right = [o[left_num:] for o in out]
        # 4. 计算两图匹配度
        gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.model.args.max_disp//4, self.model.cv_group)  # Group-wise correlation volume (B, N_group, max_disp, H, W)
        left_tmp = self.model.proj_cmb(features_left[0])
        right_tmp = self.model.proj_cmb(features_right[0])
        concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.model.args.max_disp//4)
        del left_tmp, right_tmp
        # 5. 特征融合
        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        comb_volume = self.model.corr_stem(comb_volume)
        comb_volume = self.model.corr_feature_att(comb_volume, features_left[0])
        comb_volume = self.model.cost_agg(comb_volume, features_left)
        # 6. token 化 & token 压缩
        comb_volume_compressed = comb_volume.reshape(comb_volume.shape[0], -1, *comb_volume.shape[3:]).permute(0, 2, 3, 1)
        comb_volume_compressed = self.compressor(comb_volume_compressed.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        comb_volume_compressed = self.token_norm(comb_volume_compressed)
        return comb_volume, comb_volume_compressed, left_vit_feat, features_left, features_right
    

    def FoundationStereoDecoder(self, comb_volume, comb_volume_compressed, left_current_frame, left_vit_feat, features_left, features_right, iters = 10):
        B, n, d = comb_volume_compressed.shape
        # 1. 还原token (16x16 -> 128x128, 然后 crop 回 112x112)
        comb_volume_decompressed = self.decompressor(comb_volume_compressed.transpose(1, 2).reshape(B, d, int(n** 0.5), int(n** 0.5)))
        comb_volume_decompressed = comb_volume_decompressed[:, :, 8:120, 8:120]  # 128 -> 112 (去掉 pad)
        comb_volume_decompressed = comb_volume_decompressed.permute(0, 2, 3, 1)
        # 2. 残差计算
        comb_volume_decompressed = comb_volume_decompressed.permute(0, 3, 1, 2).reshape(comb_volume.shape[0], comb_volume.shape[1], comb_volume.shape[2], comb_volume.shape[3], comb_volume.shape[4])
        comb_volume = comb_volume + comb_volume_decompressed
        # 3. 初始化深度估计图
        prob = F.softmax(self.model.classifier(comb_volume).squeeze(1), dim=1) 
        init_disp = disparity_regression(prob, self.model.args.max_disp//4) 
        # 4. 多尺度特征提取
        cnet_list = self.model.cnet(left_current_frame, vit_feat=left_vit_feat, num_layers=self.model.args.n_gru_layers)   #(1/4, 1/8, 1/16)
        cnet_list = list(cnet_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.model.cam(x) * x for x in inp_list]
        att = [self.model.sam(x) for x in inp_list]
        stem_2x = self.model.stem_2(left_current_frame)
        # 5. GRU 迭代精修
        geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.model.args.corr_levels, dx=self.model.dx)
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = init_disp.float()
        for itr in range(iters):
            geo_feat = geo_fn(disp, coords, low_memory=False)
            with torch.cuda.amp.autocast(enabled=self.model.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.model.update_block(net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp.float()
            if itr < iters-1:
                continue
            disp_up = self.model.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
        # 6. 计算有效的视差 (left view)
        if disp_up.dim() == 4 and disp_up.size(1) == 1:
            disp_up = disp_up[:, 0]
        b, h, w = disp_up.shape
        x = torch.arange(w, device=disp_up.device, dtype=disp_up.dtype).view(1, 1, w).expand(b, h, w)
        invalid = (x - disp_up) < 0
        disp = disp_up.clamp_min(1e-6)
        # 7. 估计深度（左视角）
        fx = float(self.config.intrinsic[0, 0])
        baseline = float(self.config.camera_baseline)
        depth = (fx * baseline) / disp
        depth = depth.masked_fill(invalid, float("inf"))
        return depth