"""Script to prepare and verify reference audio for Few-shot TTS testing.

Few-shot TTS用の参照音声を準備・検証するスクリプト。
"""
import os
import logging
import torchaudio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
MIN_DURATION = 3  # seconds
MAX_DURATION = 10  # seconds
REFERENCE_DIR = Path("test_data/reference_audio")

def verify_audio_file(file_path: Path) -> bool:
    """Verify if the audio file meets the requirements.
    
    音声ファイルが要件を満たしているか確認します。
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Check sample rate
        if sample_rate != SAMPLE_RATE:
            logger.warning(f"{file_path}: Invalid sample rate. Expected {SAMPLE_RATE}, got {sample_rate}")
            return False
            
        # Check duration
        duration = waveform.shape[1] / sample_rate
        if duration < MIN_DURATION or duration > MAX_DURATION:
            logger.warning(f"{file_path}: Invalid duration. Expected {MIN_DURATION}-{MAX_DURATION}s, got {duration:.2f}s")
            return False
            
        # Check channels (mono)
        if waveform.shape[0] != 1:
            logger.warning(f"{file_path}: File must be mono. Got {waveform.shape[0]} channels")
            return False
            
        logger.info(f"{file_path}: Valid reference audio file")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def prepare_reference_directory():
    """Prepare reference audio directory and verify existing files.
    
    参照音声ディレクトリを準備し、既存のファイルを検証します。
    """
    # Create directory if it doesn't exist
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reference audio directory: {REFERENCE_DIR}")
    
    # Verify existing files
    wav_files = list(REFERENCE_DIR.glob("*.wav"))
    if not wav_files:
        logger.warning("No reference audio files found")
        logger.info("Please add WAV files with the following specifications:")
        logger.info(f"- Sample rate: {SAMPLE_RATE} Hz")
        logger.info(f"- Duration: {MIN_DURATION}-{MAX_DURATION} seconds")
        logger.info("- Channels: Mono")
        logger.info("- Content: Clear Japanese speech")
        return
        
    valid_files = 0
    for file_path in wav_files:
        if verify_audio_file(file_path):
            valid_files += 1
            
    logger.info(f"Found {valid_files}/{len(wav_files)} valid reference audio files")

def main():
    """Main function to prepare and verify reference audio.
    
    参照音声の準備と検証を行うメイン関数。
    """
    logger.info("Starting reference audio preparation")
    prepare_reference_directory()
    logger.info("Reference audio preparation complete")

if __name__ == "__main__":
    main()
