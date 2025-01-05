from typing import Optional, Tuple
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor  # type: ignore
import librosa

# Add original GPT-SoVITS project path to sys.path for model loading
ORIGINAL_PROJECT_PATH = os.path.expanduser("~/repos/GPT-SoVITS")
if os.path.exists(ORIGINAL_PROJECT_PATH):
    sys.path.insert(0, ORIGINAL_PROJECT_PATH)

# Now we can import the original utils module if needed
try:
    import GPT_SoVITS.utils as gpt_sovits_utils

    HAS_ORIGINAL_UTILS = True
except ImportError:
    HAS_ORIGINAL_UTILS = False

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
    try:
        # Initialize default values
        iteration = 0
        learning_rate = 0.0
        saved_state_dict = None

        # Try loading with original utils if available
        if HAS_ORIGINAL_UTILS and hasattr(gpt_sovits_utils, "load_checkpoint"):
            try:
                logger.info("Using GPT-SoVITS utils for loading")
                model, optimizer, learning_rate, iteration = (
                    gpt_sovits_utils.load_checkpoint(
                        checkpoint_path, model, optimizer, skip_optimizer
                    )
                )
                return model, optimizer, learning_rate, iteration
            except Exception as e:
                logger.warning(f"Failed to load with original utils: {str(e)}")
                # Fall through to our implementation

        # Fallback to our implementation
        logger.info("Using custom checkpoint loading implementation")
        try:
            checkpoint_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )  # type: ignore
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint with weights_only=True: {str(e)}"
            )
            # Try without weights_only as fallback
            checkpoint_dict = torch.load(
                checkpoint_path, map_location="cpu"
            )  # type: ignore

        # Handle different checkpoint formats flexibly
        if isinstance(checkpoint_dict, dict):
            # Get optional metadata with defaults
            iteration = checkpoint_dict.get("iteration", 0)
            learning_rate = checkpoint_dict.get("learning_rate", 0.0)

            # Handle different model state dict locations
            if "model" in checkpoint_dict:
                saved_state_dict = checkpoint_dict["model"]
            elif "state_dict" in checkpoint_dict:
                saved_state_dict = checkpoint_dict["state_dict"]
            else:
                # Assume the dict itself is the state dict
                saved_state_dict = checkpoint_dict

            # Handle optimizer state if present
            if optimizer is not None and not skip_optimizer:
                optimizer_state = checkpoint_dict.get("optimizer")
                if optimizer_state is not None:
                    optimizer.load_state_dict(optimizer_state)
        else:
            # Handle case where checkpoint is just the model state dict
            saved_state_dict = checkpoint_dict
            iteration = 0
            learning_rate = 0.0
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        # Return model with default values instead of raising
        return model, optimizer, 0.0, 0
    # Ensure we have a valid state dict before proceeding
    if saved_state_dict is None:
        raise ValueError("No valid state dict found in checkpoint")

    try:
        # Get the current model state dict
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # Create new state dict with careful error handling
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                if k in saved_state_dict:
                    saved_v = saved_state_dict[k]
                    if saved_v.shape == v.shape:
                        new_state_dict[k] = saved_v
                    else:
                        logger.warning(f"Key {k}: shape mismatch")
                        logger.debug(f"Shapes: {v.shape} vs {saved_v.shape}")
                        new_state_dict[k] = v
                else:
                    logger.warning("Key %s not found, using default", k)
                    new_state_dict[k] = v
            except Exception as e:
                logger.error(f"Error loading key {k}: {str(e)}")
                new_state_dict[k] = v

        # Load the state dict
        try:
            if hasattr(model, "module"):
                model.module.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            logger.error(f"Error loading state dict: {str(e)}")
            raise

        logger.info(f"Loaded '{checkpoint_path}' at iter {iteration}")
        return model, optimizer, learning_rate, iteration

    except Exception as e:
        logger.error(f"Error during state dict processing: {str(e)}")
        raise


def load_wav_to_torch(full_path: str) -> Tuple[Tensor, int]:
    """Load audio file to torch tensor.

    Args:
        full_path: Path to the audio file

    Returns:
        Tuple containing:
        - Audio data as torch tensor
        - Sampling rate
    """
    # Load audio with default sampling rate
    data, sampling_rate = librosa.load(full_path, sr=22050)

    # Convert numpy array to torch tensor with float32 dtype
    tensor_data = torch.from_numpy(data.astype("float32"))
    return tensor_data, sampling_rate
