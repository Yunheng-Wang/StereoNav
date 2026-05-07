import torch
import timm
import logging
from torch import nn
import torch.nn.functional as F
from .modules.foundationstereo import FoundationStereoModel
from .modules.foundationstereo_config import FoundationStereoConfig
from .modules.vlm import InternVLModel
from .modules.vlm_config import InternVLConfig
from .modules.dino import DINOv2
from .modules.dino_config import DINOv2Config
from .modules.pointhead import PointHead
from .modules.pointhead_config import PointHeadConfig
from .utils.prompt import temple, system, user
from typing import List, Optional
from .StereoVLNConfig import StereoVLNConfig

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> float:
    """Count model parameters and return size in billions."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e9  # Convert to billions


class StereoVLN(nn.Module):
    def __init__(self, config: StereoVLNConfig):
        super().__init__()
        self.config = config
        # 1. Create VLM Backbone (with Gradient Checkpointing enabled)
        VLMConfig = InternVLConfig(
            checkpoint_path = config.vlm_checkpoints_path,
            image_size = config.image_size,
            dtype = config.dtype,
            max_tokens = config.max_tokens,
        )
        self.VLM = InternVLModel(VLMConfig)
        if hasattr(self.VLM.model, 'language_model') and hasattr(self.VLM.model.language_model, 'gradient_checkpointing_enable'):
            self.VLM.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info(f"VLM's LLM Backbone loaded: {count_parameters(self.VLM.model.language_model):.3f}B parameters")
        logger.info(f"VLM's Vision Encoder loaded: {count_parameters(self.VLM.model.vision_model):.3f}B parameters")
        # 2. Create FoundationStereo
        DepthConfig = FoundationStereoConfig(
            checkpoint_path = config.foundationstereo_checkpoints_path,
            edgenext_path = config.foundationstereo_edgenext_path,
            dtype = config.dtype,
            dim = config.dim,
            camera_baseline = config.camera_baseline,
            intrinsic = config.camera_k,
        )
        self.FoundationStereo = FoundationStereoModel(DepthConfig)
        logger.info(f"FoundationStereo loaded: {count_parameters(self.FoundationStereo):.3f}B parameters")
        # 3. Create DINOv2
        DINOConfig = DINOv2Config(
            image_size = config.image_size,
            dtype = config.dtype,
            dino_path = config.dino
        )
        self.DINO = DINOv2(DINOConfig)
        logger.info(f"DINOv2 loaded: {count_parameters(self.DINO):.3f}B parameters")
        # 4. Left Point Head
        LeftPointHeadConfig = PointHeadConfig(
            img_size = config.image_size,
            dtype = config.dtype,
            dim = config.dim,
            mlp_ratio = config.mlp_ratio,
            dropout = config.dropout,
        )
        self.LeftPointHead = PointHead(LeftPointHeadConfig)
        logger.info(f"LeftPointHead loaded: {count_parameters(self.LeftPointHead):.3f}B parameters")
        # 5. Right Point Head
        RightPointHeadConfig = PointHeadConfig(
            img_size = config.image_size,
            dtype = config.dtype,
            dim = config.dim,
            mlp_ratio = config.mlp_ratio,
            dropout = config.dropout,
        )
        self.RightPointHead = PointHead(RightPointHeadConfig)
        logger.info(f"RightPointHead loaded: {count_parameters(self.RightPointHead):.3f}B parameters")
        # 6. Print total parameters
        logger.info(f"StereoVLN Total: {count_parameters(self):.3f}B parameters")
        # 7. Save prediction_steps
        self.prediction_steps = config.prediction_steps


    def forward(
        self,
        instruction: List[str],
        history_action: List[str],
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448] - 0~255
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448] - 0~255
        left_history_video: torch.Tensor,   # [B, 8, 3, 448, 448] - 0~255
        right_history_video: torch.Tensor,  # [B, 8, 3, 448, 448] - 0~255
        label_left_point: torch.Tensor,     # [B, 2]
        label_right_point: torch.Tensor,    # [B, 2]
        label_depth: torch.Tensor,          # [B, 1, 448, 448]
        label_answer: List[str],             # [B]
        depth_iters: int = 3,
        enable_pointhead: bool = True,
        enable_depth: bool = True
    ) -> dict:
        B = right_history_video.shape[0]
        N = right_history_video.shape[1]  # Number of history frames
        # 0. Detect which history frames are all zeros in each sample
        zero_frame_mask = (right_history_video.view(B, N, -1).sum(dim=-1) == 0)
        # 1. FoundationStereo encoding: history frames use no_grad to save memory, current frame retains gradient
        with torch.no_grad():
            hist_left = left_history_video.view(B*N, 3, *left_history_video.shape[3:])
            hist_right = right_history_video.view(B*N, 3, *right_history_video.shape[3:])
            _, history_depth_token, _, _, _ = self.FoundationStereo.FoundationStereoEncoder(hist_left, hist_right)
            history_depth_token = history_depth_token.view(B, N, *history_depth_token.shape[1:])  # [B, N, 256, dim]
        depth_feature, depth_token, left_vit_feat, features_left, features_right = self.FoundationStereo.FoundationStereoEncoder(left_current_frame.squeeze(1), right_current_frame.squeeze(1))
        # 2. DINOv2 encoding: history frames use no_grad to save memory, current frame retains gradient
        with torch.no_grad():
            history_dino_token = self.DINO.DINOv2Encoder(hist_left)
            history_dino_token = history_dino_token.view(B, N, *history_dino_token.shape[1:])  # [B, N, 256, 2048]
        del hist_left, hist_right
        left_current_frame_token = self.DINO.DINOv2Encoder(left_current_frame.squeeze(1))  # [B, 256, 2048]
        # 3. Right current frame & history frame semantic encoding
        right_history_video_token, right_current_frame_token = self.VLM.Encoder_Vsion(torch.cat([right_history_video, right_current_frame], dim = 1))
        # 3.1 Fuse FoundationStereo and DINOv2 features into right eye tokens
        right_history_video_token = right_history_video_token + self.config.depth_token_weight * history_depth_token + self.config.dino_token_weight * history_dino_token
        del history_depth_token, history_dino_token
        right_current_frame_token = right_current_frame_token + self.config.depth_token_weight * depth_token.unsqueeze(1) + self.config.dino_token_weight * left_current_frame_token.unsqueeze(1)
        # 4. Organize prompt
        batch_prompts = []
        prompt_lengths = []  # Record each sample's prompt length (excluding answer)
        valid_history_counts = []  # Record number of valid (non-zero) history frames in each sample
        for b in range(B):
            # 4.1. History frames
            history_video_str = ""
            valid_frame_count = 0
            for i in range(N):
                # Check if all zeros (if all zeros, it's initial step, skip)
                if zero_frame_mask[b, i]:
                    continue
                valid_frame_count += 1
                t_count = right_history_video_token.shape[2]
                history_video_str += f"<img>" + "<IMG_CONTEXT>" * t_count + "</img>\n"
            if valid_frame_count == 0:
                history_video_str = "This is the initial timestep, so no historical observations is available.\n"
            valid_history_counts.append(valid_frame_count)
            # 4.2. Current frame
            cur_image = right_current_frame_token.shape[2]
            right_frame_str = "<img>" + "<IMG_CONTEXT>" * cur_image + "</img>"
            # 4.3 Organize prompt
            instruction_text = user.format(
                instruction = instruction[b],
            )
            prompt = temple.format(
                    system = system,
                    history_video = history_video_str,
                    current_image = right_frame_str,
                    user = instruction_text
                )
            # 4.4. Add label_answer
            prompt_with_answer = prompt + label_answer[b] + "<|im_end|>"
            batch_prompts.append(prompt_with_answer)
        # 5. Encode prompt
        # model_inputs = self.VLM.tokenizer(batch_prompts, return_tensors='pt', padding='max_length', max_length=self.VLM.tokenizer.model_max_length, truncation=True)
        model_inputs = self.VLM.tokenizer(batch_prompts, return_tensors='pt', padding='longest', max_length=self.VLM.tokenizer.model_max_length, truncation=True)
        input_ids = model_inputs['input_ids'].to(right_current_frame.device)
        attention_mask = model_inputs['attention_mask'].to(right_current_frame.device)
        # 5.1 Calculate prompt_lengths
        answer_start_marker = "<|im_start|>assistant\n"
        marker_tokens = self.VLM.tokenizer(answer_start_marker, add_special_tokens=False)['input_ids']
        marker_len = len(marker_tokens)
        valid_samples = torch.ones(B, dtype=torch.bool, device=left_current_frame.device)  # Mark which samples are valid
        for b in range(B):
            # Search for marker token sequence in input_ids (search backwards since it's at the end of prompt)
            found = False
            for pos in range(input_ids.shape[1] - marker_len, -1, -1):
                if input_ids[b, pos:pos+marker_len].tolist() == marker_tokens:
                    # Found marker, answer starts after marker
                    prompt_lengths.append(pos + marker_len)
                    found = True
                    break
            if not found:
                # Marker not found, prompt was truncated, mark as invalid but continue processing
                logger.warning(f"Sample {b}: Marker not found in tokenized sequence. Prompt was truncated. This sample will be excluded from loss calculation.")
                prompt_lengths.append(input_ids.shape[1])  # Set to sequence end to avoid calculation errors
                valid_samples[b] = False
        input_embeds = self.VLM.model.language_model.get_input_embeddings()(input_ids)
        # 6. Input visual tokens
        visual_features = []
        for b in range(B):
            # 6.1 History frame features
            for i in range(N):
                if not zero_frame_mask[b, i]:  # If not all-zero frame
                    # Add this history frame's tokens
                    visual_features.append(right_history_video_token[b, i])
            # 6.2 Current frame features
            visual_features.append(right_current_frame_token[b].view(-1, right_current_frame_token.shape[-1]))
        flatten_visual_feats = torch.cat(visual_features, dim=0)
        img_context_token_id = self.VLM.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        mask = (input_ids == img_context_token_id)
        input_embeds[mask] = flatten_visual_feats.to(input_embeds.dtype)
        # 7. Position encoding
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(input_embeds.device)
        # 8. Input to VLM Backbone (Flash Attention 2, only needs 2D padding mask)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.VLM.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        del input_embeds
        # 10. Organize output tokens
        output_tokens = output.hidden_states[-1]
        cur_token_len = right_current_frame_token.shape[2]
        current_output_tokens_list = []
        for b in range(B):
            visual_tokens = output_tokens[b][mask[b]]
            current_output_tokens_list.append(visual_tokens[-cur_token_len:])
        current_output_tokens = torch.stack(current_output_tokens_list, dim=0)  # [B, 256, hidden_dim]
        del output
        # 11. Language loss
        answer_hidden_list = []
        answer_label_list = []
        for b in range(B):
            if valid_samples[b]:
                answer_start = prompt_lengths[b]
                if answer_start < input_ids.shape[1]:
                    seq_end = int(attention_mask[b].sum())
                    answer_hidden_list.append(output_tokens[b, answer_start-1:seq_end-1])
                    answer_label_list.append(input_ids[b, answer_start:seq_end])
        del output_tokens
        if answer_hidden_list:
            answer_hidden = torch.cat(answer_hidden_list, dim=0)
            answer_labels = torch.cat(answer_label_list, dim=0)
            answer_logits = self.VLM.model.language_model.lm_head(answer_hidden)
            language_loss = F.cross_entropy(answer_logits, answer_labels)
            del answer_hidden, answer_logits
        else:
            language_loss = torch.tensor(0.0, device=left_current_frame.device, requires_grad=True)
        # 12. Point Loss
        if enable_pointhead:
            left_point = self.LeftPointHead(current_output_tokens + left_current_frame_token)
            right_point = self.RightPointHead(current_output_tokens + depth_token)
            if valid_samples.any():
                if torch.isnan(left_point[valid_samples]).any() or torch.isinf(left_point[valid_samples]).any():
                    logger.warning("NaN/Inf detected in left_point, cleaning values")
                    left_point_clean = torch.nan_to_num(left_point[valid_samples], nan=0.0, posinf=1e6, neginf=-1e6)
                    left_point_loss = F.smooth_l1_loss(left_point_clean, label_left_point[valid_samples], reduction="mean")
                else:
                    left_point_loss = F.smooth_l1_loss(left_point[valid_samples], label_left_point[valid_samples], reduction="mean")

                if torch.isnan(right_point[valid_samples]).any() or torch.isinf(right_point[valid_samples]).any():
                    logger.warning("NaN/Inf detected in right_point, cleaning values")
                    right_point_clean = torch.nan_to_num(right_point[valid_samples], nan=0.0, posinf=1e6, neginf=-1e6)
                    right_point_loss = F.smooth_l1_loss(right_point_clean, label_right_point[valid_samples], reduction="mean")
                else:
                    right_point_loss = F.smooth_l1_loss(right_point[valid_samples], label_right_point[valid_samples], reduction="mean")
            else:
                finite_left_point = torch.where(torch.isfinite(left_point), left_point, torch.zeros_like(left_point))
                finite_right_point = torch.where(torch.isfinite(right_point), right_point, torch.zeros_like(right_point))
                left_point_loss = (finite_left_point * 0.0).mean()
                right_point_loss = (finite_right_point * 0.0).mean()
            point_loss = (left_point_loss + right_point_loss) / 2.0     
        else:
            point_loss = torch.tensor(0.0, device=left_current_frame.device, requires_grad=True)
        # 13. Depth loss
        if enable_depth:
            depth = self.FoundationStereo.FoundationStereoDecoder(depth_feature, current_output_tokens, left_current_frame.squeeze(1), left_vit_feat, features_left, features_right, iters = depth_iters)
            label_depth_sq = label_depth.squeeze(1)
            valid = torch.isfinite(depth) & torch.isfinite(label_depth_sq) & (label_depth_sq > 0)
            for b in range(B):
                if not valid_samples[b]:
                    valid[b] = False
            if valid.any():
                eps = 1e-6
                sample_losses = []
                for b in range(B):
                    if valid[b].any():
                        sl = F.smooth_l1_loss(
                            torch.log(depth[b][valid[b]] + eps),
                            torch.log(label_depth_sq[b][valid[b]] + eps),
                            reduction="mean"
                        )
                        sample_losses.append(torch.clamp(sl, max=2.0))
                if sample_losses:
                    depth_loss = torch.stack(sample_losses).mean()
                else:
                    finite_depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
                    depth_loss = (finite_depth * 0.0).mean()
            else:
                finite_depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
                depth_loss = (finite_depth * 0.0).mean()
        else:
            depth_loss = torch.tensor(0.0, device=left_current_frame.device, requires_grad=True)
        # 14. Return all losses
        return {
            'point_loss': point_loss,
            'depth_loss': depth_loss,
            'language_loss': language_loss,
        }


    @torch.no_grad()
    def inference(
        self,
        instruction: List[str],
        history_action: List[str],
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448] - 0~255
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448] - 0~255
        left_history_video: torch.Tensor,   # [B, 8, 3, 448, 448] - 0~255
        right_history_video: torch.Tensor,  # [B, 8, 3, 448, 448] - 0~255
        max_new_tokens: int = 16,
        depth_iters: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        output_point: bool = False,
        output_depth: bool = False,
    ) -> dict:
        B = right_history_video.shape[0]
        N = right_history_video.shape[1]
        device = left_current_frame.device
        # 0. Detect which history frames are all zeros in each sample
        zero_frame_mask = (right_history_video.view(B, N, -1).sum(dim=-1) == 0)
        # 1. FoundationStereo encoding: history frames + current frame
        hist_left = left_history_video.view(B*N, 3, *left_history_video.shape[3:])
        hist_right = right_history_video.view(B*N, 3, *right_history_video.shape[3:])
        _, history_depth_token, _, _, _ = self.FoundationStereo.FoundationStereoEncoder(hist_left, hist_right)
        history_depth_token = history_depth_token.view(B, N, *history_depth_token.shape[1:])  # [B, N, 256, dim]
        depth_feature, depth_token, left_vit_feat, features_left, features_right = \
            self.FoundationStereo.FoundationStereoEncoder(left_current_frame.squeeze(1), right_current_frame.squeeze(1))
        # 2. DINOv2 encoding: history frames + current frame
        history_dino_token = self.DINO.DINOv2Encoder(hist_left)
        history_dino_token = history_dino_token.view(B, N, *history_dino_token.shape[1:])  # [B, N, 256, 2048]
        del hist_left, hist_right
        left_current_frame_token = self.DINO.DINOv2Encoder(left_current_frame.squeeze(1))  # [B, 256, 2048]
        # 3. Right current frame & history frame semantic encoding
        right_history_video_token, right_current_frame_token = \
            self.VLM.Encoder_Vsion(torch.cat([right_history_video, right_current_frame], dim=1))
        # 3.1 Fuse FoundationStereo and DINOv2 features into right eye tokens
        right_history_video_token = right_history_video_token + self.config.depth_token_weight * history_depth_token + self.config.dino_token_weight * history_dino_token
        del history_depth_token, history_dino_token
        right_current_frame_token = right_current_frame_token + self.config.depth_token_weight * depth_token.unsqueeze(1) + self.config.dino_token_weight * left_current_frame_token.unsqueeze(1)
        # 4. Organize prompt (without answer)
        batch_prompts = []
        valid_history_counts = []
        for b in range(B):
            # 4.1. History frames
            history_video_str = ""
            valid_frame_count = 0
            for i in range(N):
                if zero_frame_mask[b, i]:
                    continue
                valid_frame_count += 1
                t_count = right_history_video_token.shape[2]
                history_video_str += f"<img>" + "<IMG_CONTEXT>" * t_count + "</img>\n"
            if valid_frame_count == 0:
                history_video_str = "This is the initial timestep, so no historical observations is available.\n"
            valid_history_counts.append(valid_frame_count)
            # 4.2. Current frame
            cur_image = right_current_frame_token.shape[2]
            right_frame_str = "<img>" + "<IMG_CONTEXT>" * cur_image + "</img>"
            # 4.3 Organize prompt
            instruction_text = user.format(
                instruction=instruction[b],
            )
            prompt = temple.format(
                system=system,
                history_video=history_video_str,
                current_image=right_frame_str,
                user=instruction_text
            )
            batch_prompts.append(prompt)
        # 5. Tokenizer processing (inference doesn't need max_length padding, use longest)
        model_inputs = self.VLM.tokenizer(
            batch_prompts, return_tensors='pt', padding='longest', truncation=True,
            max_length=self.VLM.tokenizer.model_max_length
        )
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        # 6. Embed input tokens and inject visual features
        input_embeds = self.VLM.model.language_model.get_input_embeddings()(input_ids)
        visual_features = []
        for b in range(B):
            # 6.1 History frame features
            for i in range(N):
                if not zero_frame_mask[b, i]:
                    visual_features.append(right_history_video_token[b, i])
            # 6.2 Current frame features
            visual_features.append(right_current_frame_token[b].view(-1, right_current_frame_token.shape[-1]))
        flatten_visual_feats = torch.cat(visual_features, dim=0)
        img_context_token_id = self.VLM.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        mask = (input_ids == img_context_token_id)
        input_embeds[mask] = flatten_visual_feats.to(input_embeds.dtype)
        # 7. Position encoding
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(device)
        # 8. Prompt forward pass (Flash Attention 2, only needs 2D padding mask + KV Cache)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.VLM.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True,
            )
        del input_embeds
        output_tokens = output.hidden_states[-1]
        past_key_values = output.past_key_values
        del output
        # 9. Organize output tokens (same as forward: take last cur_token_len visual tokens)
        cur_token_len = right_current_frame_token.shape[2]
        current_output_tokens_list = []
        for b in range(B):
            visual_tokens = output_tokens[b][mask[b]]
            current_output_tokens_list.append(visual_tokens[-cur_token_len:])
        current_output_tokens = torch.stack(current_output_tokens_list, dim=0)  # [B, 256, hidden_dim]
        # 10. Autoregressive text generation (using KV Cache)
        prompt_lengths = attention_mask.sum(dim=1)  # [B]
        last_hidden_list = []
        for b in range(B):
            last_pos = int(prompt_lengths[b]) - 1
            last_hidden_list.append(output_tokens[b, last_pos])
        last_hidden = torch.stack(last_hidden_list, dim=0)  # [B, hidden_dim]
        first_logits = self.VLM.model.language_model.lm_head(last_hidden.to(torch.bfloat16))  # [B, vocab]
        del output_tokens
        # 11. Point Head prediction (same as forward: add residual connection)
        left_point, right_point = None, None
        if output_point:
            left_point = self.LeftPointHead(current_output_tokens + left_current_frame_token)
            right_point = self.RightPointHead(current_output_tokens + depth_token)
        # 12. Depth estimation (same as forward: use current_output_tokens)
        depth = None
        if output_depth:
            depth = self.FoundationStereo.FoundationStereoDecoder(
                depth_feature, current_output_tokens, left_current_frame.squeeze(1),
                left_vit_feat, features_left, features_right, iters=depth_iters
            )
        # 13.1 Get first generated token
        eos_token_id = self.VLM.tokenizer.convert_tokens_to_ids("<|im_end|>")
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated_ids = []

        def sample_from_logits(logits_2d):
            """Sample token from logits based on temperature and top_p"""
            if temperature <= 0 or top_p <= 0:
                return logits_2d.argmax(dim=-1)
            scaled = logits_2d / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above top_p (keep first one that exceeds)
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = -float('inf')
                # Restore to original order
                scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_token_ids = sample_from_logits(first_logits)
        for b in range(B):
            if next_token_ids[b].item() == eos_token_id:
                finished[b] = True
        generated_ids.append(next_token_ids.unsqueeze(1))
        # 13.2 Continue autoregressive generation (directly use complete past_key_values, no truncation)
        # Prompt position_ids are 0..prompt_lengths-1, first generated token should be at prompt_lengths
        current_positions = prompt_lengths.clone() - 1  # [B], initially points to last position of prompt
        for step in range(max_new_tokens - 1):
            if finished.all():
                break
            new_embeds = self.VLM.model.language_model.get_input_embeddings()(
                next_token_ids.unsqueeze(1)
            )
            current_positions += 1
            new_position_ids = current_positions.unsqueeze(1).to(device)
            # FA2: extend 2D attention mask, append a 1 for each generated token
            gen_attn = torch.cat([attention_mask, torch.ones((B, step + 1), device=device, dtype=attention_mask.dtype)], dim=1)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                gen_output = self.VLM.model.language_model(
                    inputs_embeds=new_embeds,
                    attention_mask=gen_attn,
                    position_ids=new_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = gen_output.past_key_values
            logits = gen_output.logits[:, -1, :]  # CausalLMOutputWithPast already includes lm_head output
            next_token_ids = sample_from_logits(logits)  # [B]
            # Force finished samples to fill eos
            next_token_ids[finished] = eos_token_id
            for b in range(B):
                if not finished[b] and next_token_ids[b].item() == eos_token_id:
                    finished[b] = True
            generated_ids.append(next_token_ids.unsqueeze(1))
        del past_key_values
        # 14. Decode generated text
        all_generated = torch.cat(generated_ids, dim=1)  # [B, gen_len]
        generated_texts = []
        for b in range(B):
            eos_positions = (all_generated[b] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = all_generated[b, :eos_positions[0]]
            else:
                gen_ids = all_generated[b]
            text = self.VLM.tokenizer.decode(gen_ids, skip_special_tokens=False)
            generated_texts.append(text)
        # 15. Return inference results
        return {
            'left_point': left_point,       # [B, 2]
            'right_point': right_point,     # [B, 2]
            'depth': depth,                 # [B, H, W]
            'action': generated_texts,      # List[str], len=B
        }
