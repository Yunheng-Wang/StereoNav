import torch
import logging
import torch.nn.functional as F
from torch import nn
from .modules.qwen_3_5 import QwenModel
from .modules.qwen_3_5_config import QwenConfig
from typing import List
from .WorldModelConfig import WorldModelConfig
from .utils.prompt import system_world, user_world, temple_world

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> float:
    """Count model parameters and return size in billions."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e9


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        # 0. 加载配置
        self.config = config
        # 1. 创建 VLM BackBone
        dtype = torch.bfloat16 if config.vlm_dtype == "bfloat16" else torch.float16
        qwen_config = QwenConfig(
            checkpoint_path=config.vlm_checkpoint_path,
            dtype=dtype,
            image_size=config.vlm_image_size,
            max_tokens=config.vlm_max_tokens
        )
        self.WM = QwenModel(qwen_config)
        if hasattr(self.WM.model, 'language_model') and hasattr(self.WM.model.language_model, 'gradient_checkpointing_enable'):
            self.WM.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info(f"WM's LLM Backbone loaded: {count_parameters(self.WM.model.language_model):.3f}B parameters")
        logger.info(f"WM's Vision Encoder loaded: {count_parameters(self.WM.model.visual):.3f}B parameters")
    

    def forward(
        self,
        instruction: List[str],
        Panoramic: torch.Tensor,                # [B, n, 3, 448, 448] - 0~255
        label_answer: List[str],                # [B]
    ) -> dict:
        B, n, C, H, W = Panoramic.shape
        device = Panoramic.device
        # 1. 编码全景图片
        panoramic_flat = Panoramic.reshape(B * n, C, H, W).unsqueeze(1)  # [B*n, 1, C, H, W]
        all_tokens = self.WM.Encoder_Vision(panoramic_flat)              # [B*n, tokens_per_frame, hidden]
        tokens_per_frame = all_tokens.shape[1]
        all_tokens = all_tokens.reshape(B, n, tokens_per_frame, -1)      # [B, n, tokens_per_frame, hidden]
        del panoramic_flat
        # 2. 构建 prompt
        batch_prompts = []
        for b in range(B):
            panoramic_str = ""
            for i in range(n):
                panoramic_str += (
                    "<|vision_start|>"
                    + "<|image_pad|>" * tokens_per_frame
                    + "<|vision_end|>"
                )
            user_text = user_world.format(instruction=instruction[b])
            prompt = temple_world.format(
                system_world=system_world,
                panoramic=panoramic_str,
                user_world=user_text,
            )
            prompt_with_answer = prompt + label_answer[b] + "<|im_end|>"
            batch_prompts.append(prompt_with_answer)
        # 3. Tokenize Prompt
        model_inputs = self.WM.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096,
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        # 4. 定位 answer 起始位置
        prompt_lengths = []
        answer_marker = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        marker_tokens = self.WM.tokenizer(answer_marker, add_special_tokens=False)["input_ids"]
        marker_len = len(marker_tokens)
        valid_samples = torch.ones(B, dtype=torch.bool, device=device)
        for b in range(B):
            found = False
            for pos in range(input_ids.shape[1] - marker_len, -1, -1):
                if input_ids[b, pos : pos + marker_len].tolist() == marker_tokens:
                    prompt_lengths.append(pos + marker_len)
                    found = True
                    break
            if not found:
                logger.warning(
                    f"Sample {b}: answer marker not found in tokenized sequence. "
                    f"Prompt was likely truncated. This sample will be excluded from loss."
                )
                prompt_lengths.append(input_ids.shape[1])
                valid_samples[b] = False
        # 5. 获取 input embeddings 并替换视觉 token
        input_embeds = self.WM.model.language_model.embed_tokens(input_ids)
        visual_features = []
        for b in range(B):
            for i in range(n):
                visual_features.append(all_tokens[b, i])

        if not visual_features:
            raise ValueError(f"No visual features extracted: B={B}, n={n}")

        flatten_visual = torch.cat(visual_features, dim=0)
        del all_tokens
        image_pad_id = self.WM.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mask = input_ids == image_pad_id
        assert mask.sum() == flatten_visual.shape[0], (
            f"视觉token数量不匹配: prompt中有 {mask.sum()} 个 <|image_pad|>, "
            f"但视觉编码产生了 {flatten_visual.shape[0]} 个token。"
            f"请检查 max_length 是否足够大，避免截断视觉token。"
        )
        input_embeds = input_embeds.clone()
        input_embeds[mask] = flatten_visual.to(input_embeds.dtype)
        del flatten_visual, mask
        # 6. 位置编码 & 前向传播
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        output = self.WM.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = output.last_hidden_state
        del input_embeds, output
        # 7. 计算 language loss
        lm_weight = self.WM.model.language_model.embed_tokens.weight
        answer_hidden_list = []
        answer_label_list = []
        for b in range(B):
            if valid_samples[b]:
                answer_start = prompt_lengths[b]
                # 确保 answer_start > 0 以避免负索引，且 answer_start < seq_len
                if 0 < answer_start < input_ids.shape[1]:
                    seq_end = int(attention_mask[b].sum())
                    answer_hidden_list.append(hidden_states[b, answer_start - 1 : seq_end - 1])
                    answer_label_list.append(input_ids[b, answer_start : seq_end])
        del hidden_states

        if answer_hidden_list:
            answer_hidden = torch.cat(answer_hidden_list, dim=0)
            answer_labels = torch.cat(answer_label_list, dim=0)
            # bfloat16 算 logits（省显存），loss 内部自动 upcast
            answer_logits = F.linear(answer_hidden, lm_weight)
            language_loss = F.cross_entropy(answer_logits.float(), answer_labels)
            del answer_hidden, answer_logits
        else:
            language_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {"language_loss": language_loss}


    def inference(
        self,
        instruction: List[str],
        Panoramic: torch.Tensor,                # [B, n, 3, 448, 448] - 0~255
    ) -> dict:
        B, n, C, H, W = Panoramic.shape
        device = Panoramic.device
        # 1. 编码全景图片
        panoramic_flat = Panoramic.reshape(B * n, C, H, W).unsqueeze(1)  # [B*n, 1, C, H, W]
        all_tokens = self.WM.Encoder_Vision(panoramic_flat)              # [B*n, tokens_per_frame, hidden]
        tokens_per_frame = all_tokens.shape[1]
        all_tokens = all_tokens.reshape(B, n, tokens_per_frame, -1)      # [B, n, tokens_per_frame, hidden]
        del panoramic_flat
        # 2. 构建推理提示
        batch_prompts = []
        for b in range(B):
            panoramic_str = ""
            for i in range(n):
                panoramic_str += (
                    "<|vision_start|>"
                    + "<|image_pad|>" * tokens_per_frame
                    + "<|vision_end|>"
                )
            user_text = user_world.format(instruction=instruction[b])
            prompt = temple_world.format(
                system_world=system_world,
                panoramic=panoramic_str,
                user_world=user_text,
            )
            prompt_for_generation = prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            batch_prompts.append(prompt_for_generation)
        # 3. Tokenize 提示
        model_inputs = self.WM.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096,
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        # 4. 获取 input embeddings 并替换视觉 token
        input_embeds = self.WM.model.language_model.embed_tokens(input_ids)
        visual_features = []
        for b in range(B):
            for i in range(n):
                visual_features.append(all_tokens[b, i])
        if not visual_features:
            raise ValueError(f"No visual features extracted: B={B}, n={n}")
        flatten_visual = torch.cat(visual_features, dim=0)
        del all_tokens
        image_pad_id = self.WM.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mask = input_ids == image_pad_id
        assert mask.sum() == flatten_visual.shape[0], (
            f"视觉token数量不匹配: prompt中有 {mask.sum()} 个 <|image_pad|>, "
            f"但视觉编码产生了 {flatten_visual.shape[0]} 个token。"
        )
        input_embeds = input_embeds.clone()
        input_embeds[mask] = flatten_visual.to(input_embeds.dtype)
        del flatten_visual, mask
        # 5. 位置编码（与 forward 保持一致）
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        # 6. 生成回答 - 自回归生成
        with torch.no_grad():
            self.WM.model.eval()  # Set model to eval mode

            # 前向传播获得输出
            output = self.WM.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

            # 获取最后一个 token 的 logits
            lm_weight = self.WM.model.language_model.embed_tokens.weight
            last_hidden = output.last_hidden_state[:, -1, :]  # [B, hidden_dim]
            first_logits = F.linear(last_hidden, lm_weight)  # [B, vocab]

            # 采样函数
            def sample_from_logits(logits_2d, temperature=0.7, top_p=0.9):
                if temperature <= 0 or top_p <= 0:
                    return logits_2d.argmax(dim=-1)
                scaled = logits_2d / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative probability above threshold
                    sorted_logits[cumulative_probs > top_p] = -float('inf')
                    # Scatter back to original order
                    scaled = torch.scatter(torch.full_like(scaled, -float('inf')), 1, sorted_indices, sorted_logits)
                probs = F.softmax(scaled, dim=-1)
                return torch.multinomial(probs, num_samples=1).squeeze(-1)

            eos_token_id = self.WM.tokenizer.convert_tokens_to_ids("<|im_end|>")
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            generated_ids = []

            # Accumulate embeddings for full sequence
            accumulated_embeds = input_embeds.clone()
            accumulated_attn = attention_mask.clone()
            seq_len = input_embeds.shape[1]

            # 采样第一个 token
            next_token_ids = sample_from_logits(first_logits, temperature=0.0, top_p=1.0)
            for b in range(B):
                if next_token_ids[b].item() == eos_token_id:
                    finished[b] = True
            generated_ids.append(next_token_ids.unsqueeze(1))

            # 继续自回归生成
            for step in range(512 - 1):
                if finished.all():
                    break

                # Add new token embeddings to accumulated sequence
                new_embeds = self.WM.model.language_model.embed_tokens(next_token_ids.unsqueeze(1))  # [B, 1, hidden]
                accumulated_embeds = torch.cat([accumulated_embeds, new_embeds], dim=1)  # [B, seq_len + step + 1, hidden]

                # Update attention mask
                new_attn = torch.ones((B, 1), device=device, dtype=attention_mask.dtype)
                accumulated_attn = torch.cat([accumulated_attn, new_attn], dim=1)  # [B, seq_len + step + 1]

                # Create position_ids for full sequence
                full_position_ids = torch.arange(accumulated_embeds.shape[1], device=device).unsqueeze(0).expand(B, -1)

                gen_output = self.WM.model.language_model(
                    inputs_embeds=accumulated_embeds,
                    attention_mask=accumulated_attn,
                    position_ids=full_position_ids,
                    use_cache=False,
                )

                last_hidden_gen = gen_output.last_hidden_state[:, -1, :]
                logits = F.linear(last_hidden_gen, lm_weight)
                next_token_ids = sample_from_logits(logits, temperature=0.0, top_p=1.0)

                next_token_ids[finished] = eos_token_id
                for b in range(B):
                    if not finished[b] and next_token_ids[b].item() == eos_token_id:
                        finished[b] = True
                generated_ids.append(next_token_ids.unsqueeze(1))

            generate_ids = torch.cat(generated_ids, dim=1)  # [B, gen_len]

        # 解码生成的文本
        responses = self.WM.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 提取 assistant 回答部分
        answers = []
        for response in responses:
            if "<|im_start|>assistant" in response:
                answer = response.split("<|im_start|>assistant")[-1]
                answer = answer.replace("<|im_end|>", "").strip()
            else:
                answer = response.strip()
            answers.append(answer)

        return {
            "responses": answers,
        }