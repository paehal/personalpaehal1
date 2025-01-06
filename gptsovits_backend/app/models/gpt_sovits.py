from typing import Optional, List
import math
import torch
from torch import nn
from torch.nn import functional as F

from ..config import Settings
from .text_utils import JapaneseTextProcessor


class TextEncoder(nn.Module):
    """Text encoder for GPT-SoVITS."""
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        version: str = "v2"
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.version = version

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)


class Generator(nn.Module):
    """Generator for GPT-SoVITS."""
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)


class SynthesizerTrn(nn.Module):
    """Main GPT-SoVITS model architecture."""
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_sdp: bool = True,
        semantic_frame_rate: str = "25hz",
        version: str = "v2",
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.semantic_frame_rate = semantic_frame_rate
        self.version = version

        # Initialize components
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            version=version,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(768, 768, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(768, 768, 1, stride=1)


class GPTSoVITSModel:
    """Handler for GPT-SoVITS model operations."""

    def __init__(self, settings: Settings) -> None:
        """Initialize GPT-SoVITS model handler.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model_dir = settings.MODEL_DIR / "gpt_sovits"
        self.model_path = (
            self.model_dir / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        )
        self.model: Optional[SynthesizerTrn] = None
        self.text_processor = JapaneseTextProcessor()

    def load_model(self) -> None:
        """Load the GPT-SoVITS model using torch.load()."""
        if not self.model_path.exists():
            path_str = str(self.model_path)
            raise FileNotFoundError(f"Model not found: {path_str}")

        try:
            # Load model using torch.load
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            # Initialize model with configuration from checkpoint
            config = checkpoint.get("config", {})
            self.model = SynthesizerTrn(
                spec_channels=config.get("spec_channels", 513),
                segment_size=config.get("segment_size", 32),
                inter_channels=config.get("inter_channels", 192),
                hidden_channels=config.get("hidden_channels", 192),
                filter_channels=config.get("filter_channels", 768),
                n_heads=config.get("n_heads", 2),
                n_layers=config.get("n_layers", 6),
                kernel_size=config.get("kernel_size", 3),
                p_dropout=config.get("p_dropout", 0.1),
                resblock=config.get("resblock", "1"),
                resblock_kernel_sizes=config.get("resblock_kernel_sizes", [3, 7, 11]),
                resblock_dilation_sizes=config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
                upsample_rates=config.get("upsample_rates", [8, 8, 2, 2]),
                upsample_initial_channel=config.get("upsample_initial_channel", 512),
                upsample_kernel_sizes=config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
                version="v2",  # Using v2 for Japanese support
                semantic_frame_rate="25hz"  # Based on model name
            )

            # Remove 'model.' prefix from state dict keys
            state_dict = checkpoint["weight"]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove "model." prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            # Load state dict
            self.model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to handle missing keys
            self.model.eval()  # Set to evaluation mode
            
            # Store model info if available
            self.model_info = checkpoint.get("info", {})

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
