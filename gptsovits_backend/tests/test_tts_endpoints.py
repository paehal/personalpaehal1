"""Test TTS endpoints."""
import base64
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi import status

from app.models.tts import Language


def test_zero_shot_tts_success(client, reference_audio):
    """Test successful zero-shot TTS request."""
    response = client.post(
        "/tts/zero-shot",
        json={
            "text": "こんにちは、世界！",
            "reference_audio": reference_audio,
            "source_lang": Language.JAPANESE,
            "target_lang": Language.JAPANESE,
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "audio" in data
    assert isinstance(data["audio"], str)
    assert "duration" in data
    assert isinstance(data["duration"], float)
    assert data["duration"] > 0


def test_zero_shot_tts_invalid_language(client, reference_audio):
    """Test zero-shot TTS with invalid language."""
    response = client.post(
        "/tts/zero-shot",
        json={
            "text": "Hello, World!",
            "reference_audio": reference_audio,
            "source_lang": Language.ENGLISH,  # Invalid language
            "target_lang": Language.JAPANESE,
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Japanese" in response.json()["detail"]


def test_few_shot_tts_success(client, training_audio):
    """Test successful few-shot TTS request."""
    response = client.post(
        "/tts/few-shot",
        json={
            "text": "こんにちは、世界！",
            "training_audio": training_audio,
            "source_lang": Language.JAPANESE,
            "target_lang": Language.JAPANESE,
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "audio" in data
    assert isinstance(data["audio"], str)
    assert "duration" in data
    assert isinstance(data["duration"], float)
    assert data["duration"] > 0


def test_few_shot_tts_invalid_language(client, training_audio):
    """Test few-shot TTS with invalid language."""
    response = client.post(
        "/tts/few-shot",
        json={
            "text": "Hello, World!",
            "training_audio": training_audio,
            "source_lang": Language.ENGLISH,  # Invalid language
            "target_lang": Language.JAPANESE,
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Japanese" in response.json()["detail"]


def test_zero_shot_tts_invalid_audio_length(client):
    """Test zero-shot TTS with invalid audio length."""
    # Create a very short audio file (1 second)
    sample_rate = 44100
    duration = 1  # 1 second
    samples = np.zeros(sample_rate * duration)
    temp_path = "temp_short.wav"
    sf.write(temp_path, samples, sample_rate)

    with open(temp_path, "rb") as f:
        short_audio = base64.b64encode(f.read()).decode()

    response = client.post(
        "/tts/zero-shot",
        json={
            "text": "こんにちは、世界！",
            "reference_audio": short_audio,
            "source_lang": Language.JAPANESE,
            "target_lang": Language.JAPANESE,
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "2" in response.json()["detail"]  # Mentions minimum length
    assert "10" in response.json()["detail"]  # Mentions maximum length

    # Cleanup
    os.remove(temp_path)


def test_few_shot_tts_invalid_audio_length(client):
    """Test few-shot TTS with invalid audio length."""
    # Create a very short audio file (2 seconds)
    sample_rate = 44100
    duration = 2  # 2 seconds
    samples = np.zeros(sample_rate * duration)
    temp_path = "temp_short.wav"
    sf.write(temp_path, samples, sample_rate)

    with open(temp_path, "rb") as f:
        short_audio = base64.b64encode(f.read()).decode()

    response = client.post(
        "/tts/few-shot",
        json={
            "text": "こんにちは、世界！",
            "training_audio": short_audio,
            "source_lang": Language.JAPANESE,
            "target_lang": Language.JAPANESE,
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "3" in response.json()["detail"]  # Mentions minimum length
    assert "120" in response.json()["detail"]  # Mentions maximum length

    # Cleanup
    os.remove(temp_path)
