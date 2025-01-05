"""Test configuration and fixtures."""
import base64
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.tts_service import TTSService
from .mock_models import MockGPTSoVITSModel, MockUVR5Model, MockWhisperModel


@pytest.fixture(autouse=True)
def mock_models():
    """Mock model loading for all tests."""
    def mock_init(self, settings):
        self.settings = settings
        self.model_dir = settings.MODEL_DIR
        self.temp_dir = settings.TEMP_DIR
        self.device = "cpu"
        self.models = {
            's1': MockGPTSoVITSModel(),
            's2_D': MockGPTSoVITSModel(),
            's2_G': MockGPTSoVITSModel(),
            'uvr5': MockUVR5Model(),
            'asr': MockWhisperModel(),
            'gpt_sovits': MockGPTSoVITSModel(),
        }

    with patch.object(TTSService, '__init__', mock_init), \
         patch.object(TTSService, '_load_models') as mock_load:
        mock_load.return_value = None
        yield mock_load


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def reference_audio(test_data_dir):
    """Get the reference audio file."""
    audio_path = test_data_dir / "reference.wav"
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@pytest.fixture
def training_audio(test_data_dir):
    """Get the training audio file."""
    audio_path = test_data_dir / "training.wav"
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
