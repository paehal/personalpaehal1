import os
import torch
from pathlib import Path
from typing import Optional
from ..config import Settings

class GPTSoVITSModel:
    """Handler for GPT-SoVITS model operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = settings.MODEL_DIR / "gpt_sovits"
        self.model_path = self.model_dir / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        self.model: Optional[torch.nn.Module] = None
        
    def load_model(self) -> None:
        """Load the GPT-SoVITS model using torch.load()."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        try:
            # Load model using torch.load instead of torch.jit.load
            checkpoint = torch.load(self.model_path, map_location="cpu")
            self.model = checkpoint  # We'll refine this once we understand the model structure
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
