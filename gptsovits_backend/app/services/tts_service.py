import base64
import os
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..models.tts import Language, TTSMode
from ..config import Settings
import librosa
import soundfile as sf
import torch
import torch.cuda
import torch.nn as nn
import torch.utils.data
import torch.utils.model_zoo
import faster_whisper
import logging

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
                "s1": gpt_sovits_path / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "s2_D": gpt_sovits_path / "s2D488k.pth",
                "s2_G": gpt_sovits_path / "s2G488k.pth"
            }
            
            for model_key, model_path in checkpoint_paths.items():
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                logger.info(f"Loading {model_key} from {model_path}")
                self.models[model_key] = torch.load(
                    model_path,
                    map_location=self.device
                )
            
            # Load UVR5 model for voice separation
            uvr5_path = self.model_dir / "uvr5/uvr5_weights/vits_vc_gpu_train/uvr5_weights"
            uvr5_files = list(uvr5_path.glob("*.pth"))
            if not uvr5_files:
                raise FileNotFoundError(f"No UVR5 model files found in {uvr5_path}")
            uvr5_model_path = uvr5_files[0]  # Use first .pth file found
            logger.info(f"Loading UVR5 model from {uvr5_model_path}")
            self.models["uvr5"] = torch.load(
                uvr5_model_path,
                map_location=self.device
            )
            
            # Load Faster Whisper ASR model for Japanese
            asr_path = self.model_dir / "asr/models"
            logger.info(f"Loading Faster Whisper model from {asr_path}")
            self.models["asr"] = faster_whisper.WhisperModel(
                model_size_or_path=str(asr_path),
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "float32"
            )
            
            logger.info("All models loaded successfully")
            self._validate_models()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to load required models: {str(e)}")
    
    def _validate_models(self) -> None:
        """Validate that all required models are loaded."""
        required_models = ["s1", "s2_D", "s2_G", "uvr5", "asr"]
        missing_models = [model for model in required_models if model not in self.models]
        if missing_models:
            raise RuntimeError(f"Missing required models: {', '.join(missing_models)}")

    def _validate_audio_length(self, audio_path: Path, min_length: int, max_length: int) -> bool:
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
        self._validate_models()  # Ensure models are loaded
        try:
            # Save reference audio
            ref_path = self._save_base64_audio(reference_audio, "ref")
            
            # Validate reference audio length (2-10 seconds)
            if not self._validate_audio_length(ref_path, 2, 10):
                raise ValueError("Reference audio must be between 2 and 10 seconds long")
            
            # TODO: Implement actual GPT-SoVITS inference
            # For now, return the reference audio as a placeholder
            output_audio = self._encode_audio_base64(ref_path)
            y, sr = librosa.load(str(ref_path))
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Clean up
            ref_path.unlink()
            
            return output_audio, duration
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
        self._validate_models()  # Ensure models are loaded
        try:
            # Save training audio
            train_path = self._save_base64_audio(training_audio, "train")
            
            # Validate training audio length (3-120 seconds)
            if not self._validate_audio_length(train_path, 3, 120):
                raise ValueError("Training audio must be between 3 and 120 seconds long")
            
            # TODO: Implement actual GPT-SoVITS training and inference
            # For now, return a portion of the training audio as a placeholder
            y, sr = librosa.load(str(train_path), duration=5)
            output_path = self.temp_dir / f"output_{os.urandom(8).hex()}.wav"
            sf.write(output_path, y, sr)
            
            output_audio = self._encode_audio_base64(output_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Clean up
            train_path.unlink()
            output_path.unlink()
            
            return output_audio, duration
        except Exception as e:
            raise RuntimeError(f"Few-shot TTS processing failed: {str(e)}")
