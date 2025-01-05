"""Mock models for testing."""
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa

class MockGPTSoVITSModel(nn.Module):
    """Mock GPT-SoVITS model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # Dummy layer
        
    def forward(self, x):
        return self.linear(x)
        
    def state_dict(self):
        return {
            "linear.weight": np.zeros((10, 10)),
            "linear.bias": np.zeros(10)
        }
        
    def load_state_dict(self, state_dict, strict=True):
        pass  # Mock loading, do nothing
        
    def _validate_audio_length(self, audio_path: str, min_length: float, max_length: float) -> None:
        """Validate audio file length."""
        try:
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            if not (min_length <= duration <= max_length):
                raise ValueError(
                    f"Audio length must be between {min_length} and {max_length} seconds. "
                    f"Current length: {duration:.1f} seconds."
                )
        except Exception as e:
            raise ValueError(f"Error validating audio: {str(e)}")
        
    def infer_zero_shot(self, text: str, reference_path: str, output_path: str) -> None:
        """Mock zero-shot inference."""
        # Validate reference audio (2-10 seconds)
        self._validate_audio_length(reference_path, 2, 10)
        
        # Create dummy output audio
        samples = np.zeros(44100 * 3)  # 3 seconds of silence
        sf.write(output_path, samples, 44100)
        
    def few_shot_train(self, training_path: str) -> None:
        """Mock few-shot training."""
        # Validate training audio (3-120 seconds)
        self._validate_audio_length(training_path, 3, 120)

    def infer_few_shot(self, text: str, output_path: str) -> None:
        """Mock few-shot inference."""
        # Create dummy output audio
        samples = np.zeros(44100 * 3)  # 3 seconds of silence
        sf.write(output_path, samples, 44100)

class MockUVR5Model(nn.Module):
    """Mock UVR5 model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.linear(x)

class MockWhisperModel:
    """Mock Whisper model for testing."""
    def transcribe(self, audio: str, language: str = "ja") -> dict: 
        return {"text": "こんにちは、世界"}
