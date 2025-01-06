import base64
import os
from pathlib import Path
from typing import Tuple
from ..models.tts import Language
from ..config import Settings
from ..models.gpt_sovits import GPTSoVITSModel
import librosa
import soundfile as sf


class TTSService:
    """Service for handling Text-to-Speech operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = settings.MODEL_DIR
        self.temp_dir = settings.TEMP_DIR

        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GPT-SoVITS model
        self.gpt_sovits = GPTSoVITSModel(settings)
        try:
            self.gpt_sovits.load_model()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GPT-SoVITS model: {str(e)}"
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
        try:
            # Save reference audio
            ref_path = self._save_base64_audio(reference_audio, "ref")

            # Validate reference audio length (2-10 seconds)
            if not self._validate_audio_length(ref_path, 2, 10):
                raise ValueError(
                    "Reference audio must be between 2 and 10 seconds long"
                )

            # Load reference audio for GPT-SoVITS
            ref_y, ref_sr = librosa.load(str(ref_path))

            # TODO: Implement GPT-SoVITS inference using loaded model
            # Return reference audio as placeholder
            # Will be replaced with inference implementation
            output_audio = self._encode_audio_base64(ref_path)
            # Get duration of reference audio
            duration = librosa.get_duration(y=ref_y, sr=ref_sr)

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
        try:
            # Save training audio
            train_path = self._save_base64_audio(training_audio, "train")

            # Validate training audio length (3-120 seconds)
            if not self._validate_audio_length(train_path, 3, 120):
                raise ValueError(
                    "Training audio must be between 3 and 120 seconds long"
                )

            # Load training audio for GPT-SoVITS
            train_y, train_sr = librosa.load(str(train_path))

            # TODO: Implement GPT-SoVITS training and inference using
            # loaded model
            # For now, return a portion of training audio as placeholder until
            # training is implemented
            y = train_y[: int(5 * train_sr)]  # 5 seconds of audio
            output_path = self.temp_dir / f"output_{os.urandom(8).hex()}.wav"
            sf.write(output_path, y, train_sr)

            output_audio = self._encode_audio_base64(output_path)
            duration = librosa.get_duration(y=y, sr=train_sr)

            # Clean up
            train_path.unlink()
            output_path.unlink()

            return output_audio, duration
        except Exception as e:
            raise RuntimeError(f"Few-shot TTS processing failed: {str(e)}")
