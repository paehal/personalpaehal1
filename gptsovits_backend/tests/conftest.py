"""Test configuration and fixtures."""
import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


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
