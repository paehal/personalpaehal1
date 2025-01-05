from typing import Optional, Tuple, Any
import os
import logging
import traceback
import torch
import torch.nn as nn
import torch.optim
import librosa
import numpy as np
from torch import Tensor, tensor  # type: ignore

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skip_optimizer: bool = False,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], float, int]:
    """Load model checkpoint with error handling and logging.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the checkpoint into
        optimizer: Optional optimizer to load state
        skip_optimizer: Whether to skip loading optimizer state
        
    Returns:
        Tuple containing:
        - Loaded model
        - Optimizer with loaded state (if provided)
        - Learning rate from checkpoint
        - Iteration number from checkpoint
    """
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")  # type: ignore
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and not skip_optimizer and checkpoint_dict["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except:
            logger.error(f"Error loading key {k}: {traceback.format_exc()}")
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration

def load_wav_to_torch(full_path: str) -> Tuple[Tensor, int]:
    """Load audio file to torch tensor.
    
    Args:
        full_path: Path to the audio file
        
    Returns:
        Tuple containing:
        - Audio data as torch tensor
        - Sampling rate
    """
    data, sampling_rate = librosa.load(full_path, sr=22050)  # Set default sampling rate
    return torch.tensor(data, dtype=torch.float32), sampling_rate  # type: ignore
