"""Service for handling Text-to-Speech operations."""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import faster_whisper
import librosa
import torch
import torch.cuda
import torch.jit
import torch.nn as nn

from ..config import Settings
from ..models.tts import Language
from ..models.utils import load_checkpoint

# Define supported languages
JAPANESE = Language.JAPANESE

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TTSService:
    """Service for handling Text-to-Speech operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = settings.MODEL_DIR
        self.temp_dir = settings.TEMP_DIR

        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load all required models."""
        try:
            # Load GPT-SoVITS models
            gpt_sovits_path = self.model_dir / "gpt_sovits"
            logger.info(f"Loading GPT-SoVITS models from {gpt_sovits_path}")

            checkpoint_paths = {
                "s1": gpt_sovits_path
                / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "s2_D": gpt_sovits_path / "s2D488k.pth",
                "s2_G": gpt_sovits_path / "s2G488k.pth",
            }

            for model_key, model_path in checkpoint_paths.items():
                if not model_path.exists():
                    msg = f"Model file not found: {model_path}"
                    raise FileNotFoundError(msg)
                logger.info(f"Loading {model_key} from {model_path}")
                # Create placeholder module for checkpoint loading
                model = nn.Module()
                self.models[model_key], _, _, _ = load_checkpoint(
                    str(model_path), model, skip_optimizer=True
                )

            # Load UVR5 model for voice separation
            uvr5_path = (
                self.model_dir
                / "uvr5"
                / "uvr5_weights"
                / "vits_vc_gpu_train"
                / "uvr5_weights"
            )
            uvr5_files = list(uvr5_path.glob("*.pth"))
            if not uvr5_files:
                msg = f"No UVR5 model files found in {uvr5_path}"
                raise FileNotFoundError(msg)
            uvr5_model_path = uvr5_files[0]  # Use first .pth file found
            logger.info(f"Loading UVR5 model from {uvr5_model_path}")
            self.models["uvr5"] = torch.jit.load(
                str(uvr5_model_path), map_location=self.device
            )

            # Load Faster Whisper ASR model for Japanese
            asr_path = self.model_dir / "asr/models"
            logger.info(f"Loading Faster Whisper model from {asr_path}")
            compute_type = "float16" if self.device == "cuda" else "float32"
            self.models["asr"] = faster_whisper.WhisperModel(
                model_size_or_path=str(asr_path),
                device=self.device,
                compute_type=compute_type,
            )

            logger.info("All models loaded successfully")
            self._validate_models()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to load required models: {str(e)}")

    def _validate_models(self) -> None:
        """Validate that all required models are loaded."""
        required_models = ["s1", "s2_D", "s2_G", "uvr5", "asr"]
        missing_models = [
            model for model in required_models if model not in self.models
        ]
        if missing_models:
            raise RuntimeError(
                f"Missing required models: " f"{', '.join(missing_models)}"
            )

    def _validate_audio_length(
        self, audio_path: Path, min_length: int, max_length: int
    ) -> bool:
        """Validate audio file length."""
        try:
            y, sr = librosa.load(str(audio_path))
            duration = librosa.get_duration(y=y, sr=sr)
            return min_length <= duration <= max_length
        except Exception as e:
            raise ValueError(f"Error validating audio length: {str(e)}")

    def _save_base64_audio(self, base64_audio: str, prefix: str) -> Path:
        """Save base64 encoded audio to a temporary file."""
        try:
            audio_data = base64.b64decode(base64_audio)
            temp_path = self.temp_dir / f"{prefix}_{os.urandom(8).hex()}.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_data)
            return temp_path
        except Exception as e:
            raise ValueError(f"Error saving audio file: {str(e)}")

    def _encode_audio_base64(self, audio_path: Path) -> str:
        """Encode audio file to base64."""
        try:
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            raise ValueError(f"Error encoding audio file: {str(e)}")

    async def process_zero_shot(
        self,
        text: str,
        reference_audio: str,
        source_lang: Language,
        target_lang: Language,
    ) -> Tuple[str, float]:
        """
        Process zero-shot TTS request.
        Returns: (base64_audio, duration)
        """
        # Validate language support (Japanese only)
        if source_lang != JAPANESE or target_lang != JAPANESE:
            raise ValueError(
                "このサービスは日本語のみをサポートしています。"
                "\nThis service only supports Japanese language "
                "input and output for zero-shot TTS."
            )

        self._validate_models()  # Ensure models are loaded

        try:
            # Save reference audio
            ref_path = self._save_base64_audio(reference_audio, "ref")

            # Validate reference audio length (2-10s, recommended: 5s)
            if not self._validate_audio_length(ref_path, 2, 10):
                raise ValueError(
                    "リファレンス音声は2〜10秒の長さが必要です。推奨は5秒です。"
                    "\nReference audio must be between 2 and 10 seconds long. "
                    "Recommended length is 5 seconds."
                )

            # Initialize GPT-SoVITS model if not already initialized
            if "gpt_sovits" not in self.models:
                from ..models.gpt_sovits import GPTSoVITSModel

                model = GPTSoVITSModel(
                    self.models,
                    self.device,
                )
                self.models["gpt_sovits"] = model

            # Generate output path
            output_path = self.temp_dir / f"output_{os.urandom(8).hex()}.wav"

            # Perform inference
            self.models["gpt_sovits"].infer_zero_shot(
                text=text, reference_path=ref_path, output_path=output_path
            )

            # Encode result to base64
            output_audio = self._encode_audio_base64(output_path)

            # Get duration
            y, sr = librosa.load(str(output_path))
            duration = librosa.get_duration(y=y, sr=sr)

            # Clean up
            ref_path.unlink()
            output_path.unlink()

            return output_audio, duration
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise RuntimeError(f"Zero-shot TTS processing failed: {str(e)}")

    async def process_few_shot(
        self,
        text: str,
        training_audio: str,
        source_lang: Language,
        target_lang: Language,
    ) -> Tuple[str, float]:
        """
        Process few-shot TTS request.
        Returns: (base64_audio, duration)
        """
        # Validate language support (Japanese only)
        if source_lang != JAPANESE or target_lang != JAPANESE:
            raise ValueError(
                "このサービスは日本語のみをサポートしています。"
                "\nThis service only supports Japanese language "
                "input and output for few-shot TTS."
            )

        self._validate_models()  # Ensure models are loaded
        try:
            # Save training audio
            train_path = self._save_base64_audio(training_audio, "train")

            # Validate training audio length (3-120s, recommended: 60s)
            if not self._validate_audio_length(train_path, 3, 120):
                raise ValueError(
                    "トレーニング音声は3〜120秒の長さが必要です。推奨は60秒です。"
                    "\nTraining audio must be between 3 and 120 seconds long. "
                    "Recommended length is 60 seconds."
                )

            # Initialize GPT-SoVITS model if not already initialized
            if "gpt_sovits" not in self.models:
                from ..models.gpt_sovits import GPTSoVITSModel

                model = GPTSoVITSModel(
                    self.models,
                    self.device,
                )
                self.models["gpt_sovits"] = model

            # Train model on user's voice
            self.models["gpt_sovits"].few_shot_train(train_path)

            # Generate output path and perform inference
            output_path = self.temp_dir / f"output_{os.urandom(8).hex()}.wav"
            self.models["gpt_sovits"].infer_few_shot(text, output_path)

            # Get audio and duration
            output_audio = self._encode_audio_base64(output_path)
            y, sr = librosa.load(str(output_path))
            duration = librosa.get_duration(y=y, sr=sr)

            # Clean up
            train_path.unlink()
            output_path.unlink()

            return output_audio, duration
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise RuntimeError(f"Few-shot TTS processing failed: {str(e)}")
