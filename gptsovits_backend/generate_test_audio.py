"""Generate test audio files for TTS endpoint testing."""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sine_wave(duration, frequency=440, sample_rate=44100):
    """Generate a sine wave of given duration and frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return tone


def create_test_file(filename, duration):
    """Create a test WAV file with the given duration."""
    sample_rate = 44100

    # Generate a simple melody using multiple frequencies
    frequencies = [440, 550, 660]  # A4, C#5, E5
    audio = np.zeros(int(sample_rate * duration))

    # Add each frequency component
    for i, freq in enumerate(frequencies):
        start = i * duration / len(frequencies)
        end = (i + 1) * duration / len(frequencies)
        samples = int((end - start) * sample_rate)
        t = np.linspace(0, end - start, samples, False)
        tone = np.sin(2 * np.pi * freq * t)

        # Apply fade in/out
        fade_samples = int(0.1 * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

        start_idx = int(start * sample_rate)
        end_idx = start_idx + len(tone)
        audio[start_idx:end_idx] += tone

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Save as WAV
    sf.write(filename, audio, sample_rate)
    print(f"Created {filename} ({duration}s)")


def main():
    """Create test audio files."""
    # Create test_data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    # Create reference audio (5 seconds)
    create_test_file(test_dir / "reference.wav", duration=5.0)

    # Create training audio (10 seconds)
    create_test_file(test_dir / "training.wav", duration=10.0)


if __name__ == "__main__":
    main()
