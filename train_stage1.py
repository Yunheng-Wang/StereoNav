import torch
import random
import yaml
import logging
import os
import json
import time
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import wandb
import math
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from datetime import datetime
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from model.StereoVLNConfig import StereoVLNConfig
from model.StereoVLN import StereoVLN
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from data.datasetloader import Dataset_Normal
from utils.save import save_model_hook

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


logger = logging.getLogger(__name__)
logging.getLogger("accelerate").setLevel(logging.ERROR)


def setup_logging(rank, save_path):
    logging.basicConfig(level=logging.INFO, format=f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if rank == 0:
        log_file = os.path.join(save_path, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter) 
        logging.getLogger().addHandler(file_handler) 


def build_model_and_optimizer(config, num_all_episodes, world_size):
    # 1. Load camera parameters
    data_root = config.main.data_root
    train_dir = os.path.join(data_root, "train")
    subdir = os.listdir(train_dir)[0]
    param_dir = os.path.join(train_dir, subdir)
    intrinsics_path = os.path.join(param_dir, "intrinsics.txt")
    camera_k = torch.tensor(
        np.loadtxt(intrinsics_path), dtype=torch.float32
    )
    baseline_path = os.path.join(param_dir, "baseline.txt")
    camera_baseline = float(np.loadtxt(baseline_path))
    # 2. Create model
    model_config = StereoVLNConfig(
        image_size = (config.main.image_size, config.main.image_size),
        dtype = torch.bfloat16 if config.main.dtype == "bf16" else torch.float16,
        dim = config.model.dim,

        # VLM Setting
        max_tokens = config.model.vlm.max_tokens,
        vlm_checkpoints_path = config.model.vlm.vlm_checkpoints_path,

        # Depth Estimation Setting
        camera_k = camera_k,
        camera_baseline = camera_baseline,
        foundationstereo_checkpoints_path = config.model.foundationstereo.foundationstereo_checkpoints_path,
        foundationstereo_edgenext_path = config.model.foundationstereo.foundationstereo_edgenext_path,
        # Point Head Setting
        mlp_ratio = config.model.pointhead.mlp_ratio,
        dropout = config.model.pointhead.dropout,

        # dino setting 
        dino = config.model.dino.dino_path,

        # Prediction Setting
        prediction_steps = config.main.prediction_steps,

        # Weights
        depth_token_weight = config.model.depth_token_weight,
        dino_token_weight = config.model.dino_token_weight
    )
    model = StereoVLN(model_config)
    # 2. Create optimizer (FoundationStereo decoder layers use smaller learning rate)
    depth_decoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'FoundationStereo.model.' in name or 'FoundationStereo.compressor' in name or 'FoundationStereo.decompressor' in name:
            depth_decoder_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.training.optimizer.lr},
        {'params': depth_decoder_params, 'lr': config.training.optimizer.depth_lr},
    ],
        weight_decay=config.training.optimizer.weight_decay,
        betas=tuple(config.training.optimizer.betas),
        eps=config.training.optimizer.eps
    )
    # 3. Create scheduler
    # 3.1 Custom scheduler algorithm (with minimum value setting)
    sig_gpu_max_training_steps = math.ceil((config.main.training_epoch * num_all_episodes) / (config.main.batch_size * config.main.gradient.grad_accumulation_steps))
    min_lr_ratio = config.training.scheduler.get("min_lr_ratio", 0.0)
    num_cycles = config.training.scheduler.num_cycles
    warmup_steps = int(sig_gpu_max_training_steps * config.training.scheduler.warmup_ratio)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, sig_gpu_max_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    scheduler = LambdaLR(optimizer, lr_lambda)
    # 3.2 Official scheduler algorithm (without minimum value setting)
    # sig_gpu_max_training_steps = math.ceil((config.main.training_epoch * num_all_episodes) / (config.main.batch_size * config.main.gradient.grad_accumulation_steps))
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=int(sig_gpu_max_training_steps * config.training.scheduler.warmup_ratio),
    #     num_training_steps=sig_gpu_max_training_steps,
    #     num_cycles=config.training.scheduler.num_cycles
    # )
    return model, optimizer, scheduler, math.ceil(sig_gpu_max_training_steps / world_size)


def build_dataloader(config, world_size, rank):
    def seed_worker(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # 1. Load data
    train_dataset = Dataset_Normal(config)
    # 2. Distributed sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    else:
        train_sampler = None
    # 3. Build data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.main.batch_size,
        shuffle = True if train_sampler is None else False,
        sampler = train_sampler,
        num_workers = config.main.cpu_workers_num,
        pin_memory = True,
        drop_last = True,
        worker_init_fn = seed_worker,
    )

    return train_dataloader, train_dataset.num_episodes




def learning():
    # 1. Load configuration parameters & save root directory
    config = OmegaConf.load('train_stage1.yaml')
    os.makedirs(config.main.save_root, exist_ok=True)
    # 2. Configure distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps = config.main.gradient.grad_accumulation_steps,
        mixed_precision = config.main.dtype,
        project_dir = config.main.save_root,
        project_config = ProjectConfiguration(total_limit= 20),
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600)),
                        DistributedDataParallelKwargs(find_unused_parameters=True)])
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    save_path = os.path.join(config.main.save_root, datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(save_path, exist_ok=True)
    setup_logging(rank, save_path)
    # 3. Load data
    train_dataloader, num_all_episodes = build_dataloader(config, world_size, rank)
    # 4. Load model and optimizer
    model, optimizer, scheduler, max_training_steps = build_model_and_optimizer(config, num_all_episodes, world_size)
    # 5. Configure model save settings
    accelerator.register_save_state_pre_hook(lambda models, weights, output_dir: save_model_hook(models, weights, output_dir, accelerator))
    # 6. Distributed dispatch (including dataloader)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # 7. Initialize wandb (only on main process)
    if rank == 0:
        if config.main.wandb == "offline":
            wandb.init(
                project=config.main.get("wandb_project", "StereoVLN"),
                name=config.main.get("wandb_run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                config=OmegaConf.to_container(config, resolve=True),
                dir=save_path,
                mode='offline'
            )
        else: 
            wandb.init(
                project=config.main.get("wandb_project", "StereoVLN"),
                name=config.main.get("wandb_run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                config=OmegaConf.to_container(config, resolve=True),
                dir=save_path
            )
    torch.cuda.empty_cache()
    # 8. Training
    if rank == 0:
        print("Start Training ...")
    epoch = 0
    global_step = 0  # Records actual update steps (one update per global batch_size)
    start_time = time.time()  # Record training start time
    data_iter = iter(train_dataloader)
    epoch_completed = False
    accumulated_total_loss = 0.0
    accumulated_point_loss = 0.0
    accumulated_depth_loss = 0.0
    accumulated_language_loss = 0.0
    accumulation_count = 0
    while (global_step < max_training_steps):
        ## 8.1. Load one batch of data
        try:
            batch = next(data_iter)
            epoch_completed = False  # Reset flag
        except StopIteration:
            epoch += 1
            epoch_completed = True  # Mark epoch completed
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        ## 8.2 Preprocess batch
        device = accelerator.device
        dtype = torch.bfloat16 if config.main.dtype == "bf16" else torch.float16
        batch["left_current_frame"] = batch["left_current_frame"].to(device, dtype=dtype)
        batch["right_current_frame"] = batch["right_current_frame"].to(device, dtype=dtype)
        batch["left_history_video"] = batch["left_history_video"].to(device, dtype=dtype)
        batch["right_history_video"] = batch["right_history_video"].to(device, dtype=dtype)
        batch["label_left_point"] = batch["label_left_point"].to(device, dtype=dtype)
        batch["label_right_point"] = batch["label_right_point"].to(device, dtype=dtype)
        batch["label_depth"] = batch["label_depth"].to(device, dtype=dtype)
        ## 8.3 Forward pass
        with accelerator.accumulate(model):
            # Print model frozen status (only on first batch)
            if global_step == 0 and rank == 0:
                # Handle DistributedDataParallel wrapper
                actual_model = model.module if hasattr(model, 'module') else model
                print("\n" + "="*70)
                print("🔍 Model Frozen/Active Status:")
                print("="*70)

                def check_frozen(module):
                    """Check if module is frozen"""
                    params = list(module.parameters())
                    return len(params) > 0 and not any(p.requires_grad for p in params)

                def check_active(module):
                    """Check if module is active"""
                    params = list(module.parameters())
                    return len(params) > 0 and any(p.requires_grad for p in params)

                # FoundationStereo
                print("FoundationStereo:")
                # Check decoder modules frozen status
                decoder_modules = [actual_model.FoundationStereo.model.classifier,
                                  actual_model.FoundationStereo.model.cnet,
                                  actual_model.FoundationStereo.model.cam,
                                  actual_model.FoundationStereo.model.sam,
                                  actual_model.FoundationStereo.model.stem_2,
                                  actual_model.FoundationStereo.model.update_block,
                                  actual_model.FoundationStereo.model.spx_2_gru,
                                  actual_model.FoundationStereo.model.spx_gru]

                # Get all parameters in decoder modules
                decoder_params = set()
                for module in decoder_modules:
                    for param in module.parameters():
                        decoder_params.add(id(param))

                # Check encoder (parameters in model except decoder)
                encoder_frozen = True
                for param in actual_model.FoundationStereo.model.parameters():
                    if id(param) not in decoder_params and param.requires_grad:
                        encoder_frozen = False
                        break

                # Check if decoder is active
                decoder_active = any(check_active(m) for m in decoder_modules)

                print(f"  - encoder: {'❄️  FROZEN' if encoder_frozen else '🔥 ACTIVE'}")
                print(f"  - decoder: {'🔥 ACTIVE' if decoder_active else '❄️  FROZEN'}")
                print(f"  - compressor: {'❄️  FROZEN' if check_frozen(actual_model.FoundationStereo.compressor) else '🔥 ACTIVE'}")
                print(f"  - decompressor: {'❄️  FROZEN' if check_frozen(actual_model.FoundationStereo.decompressor) else '🔥 ACTIVE'}")

                # VLM
                print("VLM:")
                print(f"  - language_model: {'❄️  FROZEN' if check_frozen(actual_model.VLM.model.language_model) else '🔥 ACTIVE'}")
                print(f"  - vision_model: {'❄️  FROZEN' if check_frozen(actual_model.VLM.model.vision_model) else '🔥 ACTIVE'}")

                # DINO
                print("DINO:")
                print(f"  - model: {'❄️  FROZEN' if check_frozen(actual_model.DINO.model) else '🔥 ACTIVE'}")
                print(f"  - compressor: {'❄️  FROZEN' if check_frozen(actual_model.DINO.compressor) else '🔥 ACTIVE'}")

                # PointHeads
                print("PointHeads:")
                print(f"  - LeftPointHead: {'❄️  FROZEN' if check_frozen(actual_model.LeftPointHead) else '🔥 ACTIVE'}")
                print(f"  - RightPointHead: {'❄️  FROZEN' if check_frozen(actual_model.RightPointHead) else '🔥 ACTIVE'}")

                print("="*70 + "\n")

            outputs = model(
                instruction=batch["instruction"],
                history_action=batch["history_action"],
                left_current_frame=batch["left_current_frame"],
                right_current_frame=batch["right_current_frame"],
                left_history_video=batch["left_history_video"],
                right_history_video=batch["right_history_video"],
                label_left_point=batch["label_left_point"],
                label_right_point=batch["label_right_point"],
                label_depth=batch["label_depth"],
                label_answer=batch["label_answer"],
                depth_iters=config.main.depth_iters,
                enable_pointhead=config.model.enable_pointhead,
                enable_depth=config.model.enable_depth
            )
            total_loss = config.training.weight_pointsloss * outputs['point_loss'] + config.training.weight_depthloss * outputs['depth_loss'] + config.training.weight_languageloss * outputs['language_loss']
            accumulated_total_loss += total_loss.detach()
            accumulated_point_loss += outputs['point_loss'].detach()
            accumulated_depth_loss += outputs['depth_loss'].detach()
            accumulated_language_loss += outputs['language_loss'].detach()
            accumulation_count += 1
            ## 8.4 Backward pass
            accelerator.backward(total_loss)
            ## 8.5 Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.main.gradient.grad_clip_norm)
            ## 8.6 Update parameters
            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1
        ## 8.7 Log metrics
        if accelerator.sync_gradients:
            # Calculate average values
            avg_total_loss = accumulated_total_loss / accumulation_count
            avg_point_loss = accumulated_point_loss / accumulation_count
            avg_depth_loss = accumulated_depth_loss / accumulation_count
            avg_language_loss = accumulated_language_loss / accumulation_count

            if rank == 0:
                # Calculate estimated completion time
                elapsed_time = time.time() - start_time
                avg_time_per_step = elapsed_time / global_step
                remaining_steps = max_training_steps - global_step
                eta_seconds = avg_time_per_step * remaining_steps
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                logging.info(
                    f"Epoch: {epoch} | "
                    f"Step: {global_step}/{max_training_steps} | "
                    f"Total Loss: {avg_total_loss.item():.4f} | "
                    f"Point Loss: {avg_point_loss.item():.4f} | "
                    f"Depth Loss: {avg_depth_loss.item():.4f} | "
                    f"Language Loss: {avg_language_loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Depth LR: {scheduler.get_last_lr()[1]:.2e} | "
                    f"ETA: {eta_hours}h {eta_minutes}m"
                )
                wandb.log({
                    "train/total_loss": avg_total_loss.item(),
                    "train/point_loss": avg_point_loss.item(),
                    "train/depth_loss": avg_depth_loss.item(),
                    "train/language_loss": avg_language_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/depth_lr": scheduler.get_last_lr()[1],
                    "train/epoch": epoch,
                }, step=global_step)

            # Reset accumulated variables
            accumulated_total_loss = 0.0
            accumulated_point_loss = 0.0
            accumulated_depth_loss = 0.0
            accumulated_language_loss = 0.0
            accumulation_count = 0
        ## 8.8 Save model
        if epoch_completed:
            if rank == 0:
                logging.info(f"Saving model at epoch {epoch}, step {global_step}...")
            checkpoint_dir = os.path.join(save_path, "checkpoint_" + str(epoch))
            accelerator.save_state(str(checkpoint_dir))
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            # 2. Save configuration
            cfg = OmegaConf.to_container(config, resolve=True)
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
            dist.barrier()


    # 9. Close wandb
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    learning()

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision fp16 --main_process_port 29555 --dynamo_backend no train_stage1.py