"""Test script for TTS endpoints."""

import base64
import json
import os
from pathlib import Path
import wave
import requests


def read_audio_file(path: str) -> str:
    """Read audio file and encode as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def validate_audio_output(
    audio_base64: str,
    min_duration: float = 1.0
) -> bool:
    """Validate audio output format and duration requirements.

    Args:
        audio_base64: Base64 encoded WAV audio data
        min_duration: Minimum required duration in seconds
    """
    try:
        # Decode base64 to WAV
        audio_data = base64.b64decode(audio_base64)
        temp_path = f"temp_output_{os.urandom(4).hex()}.wav"

        with open(temp_path, "wb") as f:
            f.write(audio_data)

        # Check WAV format and duration
        with wave.open(temp_path, "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)

            # Clean up temp file
            os.remove(temp_path)

            return duration >= min_duration
    except Exception as e:
        print(f"Error validating audio output: {str(e)}")
        return False


def test_zero_shot() -> bool:
    """Test zero-shot TTS endpoint."""
    print("\nTesting zero-shot TTS endpoint...")
    url = "http://localhost:8000/tts/zero-shot"

    # Use Japanese text
    text = "こんにちは、音声合成のテストです。"

    try:
        audio = read_audio_file("test_data/reference.wav")
    except FileNotFoundError:
        print("Error: test_data/reference.wav not found")
        return False

    data = {
        "text": text,
        "reference_audio": audio,
        "source_lang": "JA",
        "target_lang": "JA",
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        if "audio" not in result or "duration" not in result:
            print("Error: Invalid response format")
            return False

        if not validate_audio_output(result["audio"]):
            print("Error: Invalid audio output")
            return False

        print("✓ Zero-shot test successful!")
        print(f"Audio duration: {result['duration']} seconds")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error testing zero-shot endpoint: {str(e)}")
        return False


def test_few_shot() -> bool:
    """Test few-shot TTS endpoint."""
    print("\nTesting few-shot TTS endpoint...")
    url = "http://localhost:8000/tts/few-shot"

    # Use Japanese text
    text = "今日は良い天気ですね。"

    try:
        audio = read_audio_file("test_data/training.wav")
    except FileNotFoundError:
        print("Error: test_data/training.wav not found")
        return False

    data = {
        "text": text,
        "training_audio": audio,
        "source_lang": "JA",
        "target_lang": "JA",
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        if "audio" not in result or "duration" not in result:
            print("Error: Invalid response format")
            return False

        if not validate_audio_output(result["audio"]):
            print("Error: Invalid audio output")
            return False

        print("✓ Few-shot test successful!")
        print(f"Audio duration: {result['duration']} seconds")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error testing few-shot endpoint: {str(e)}")
        return False


def test_invalid_input() -> bool:
    """Test invalid input handling."""
    print("\nTesting invalid input handling...")
    url = "http://localhost:8000/tts/zero-shot"

    data = {
        "text": "",  # Empty text should fail
        "reference_audio": "invalid_base64",  # Invalid base64
        "source_lang": "EN",  # Unsupported language
        "target_lang": "JA",
    }

    try:
        response = requests.post(url, json=data)

        # Should return 400 Bad Request
        if response.status_code != 400:
            print(f"Error: Expected status 400, got {response.status_code}")
            return False

        print("✓ Invalid input handling test successful!")
        print(json.dumps(response.json(), indent=2))
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error testing invalid input: {str(e)}")
        return False


if __name__ == "__main__":
    # Create test data directory
    Path("test_data").mkdir(exist_ok=True)

    # Check if test files exist
    if not Path("test_data/reference.wav").exists():
        print("Warning: Please add a reference.wav file (2-10s) in test_data/")
    if not Path("test_data/training.wav").exists():
        print("Warning: Please add a training.wav file (3-120s) in test_data/")

    success = all([test_zero_shot(), test_few_shot(), test_invalid_input()])

    if success:
        print("\n✨ All endpoint tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit(1)
