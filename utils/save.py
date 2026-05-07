import os
import json
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)


def save_model_hook(models, weights, output_dir, accelerator):
    """Custom save hook to save model safely and avoid NCCL timeouts."""
    if accelerator.is_main_process:
        logger.info(f"Saving model to {output_dir}")
        for i, model_to_save in enumerate(models):
            unwrapped_model = accelerator.unwrap_model(model_to_save)
            model_save_path = os.path.join(output_dir, f"pytorch_model_{i}.bin")
            torch.save(unwrapped_model.state_dict(), model_save_path)
            logger.info(f"Model {i} saved to {model_save_path}")