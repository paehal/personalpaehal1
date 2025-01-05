"""Generate test audio files for TTS testing."""
import numpy as np
import soundfile as sf
from pathlib import Path

def generate_test_files():
    """Generate test audio files."""
    # Create test_data directory
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Parameters
    sample_rate = 44100
    
    # Create reference audio (5 seconds)
    ref_duration = 5
    ref_samples = np.zeros(sample_rate * ref_duration)
    sf.write(test_data_dir / "reference.wav", ref_samples, sample_rate)
    
    # Create training audio (60 seconds)
    train_duration = 60
    train_samples = np.zeros(sample_rate * train_duration)
    sf.write(test_data_dir / "training.wav", train_samples, sample_rate)

if __name__ == "__main__":
    generate_test_files()
