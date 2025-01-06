"""Script to verify Zero-shot and Few-shot TTS functionality.

TTSの出力を検証するスクリプト。
"""
import os
import sys
import logging
import torch
import torch.cuda
import torchaudio
from pathlib import Path
from app.models.gpt_sovits import GPTSoVITSModel
from app.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure settings
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/gpt_sovits"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
OUTPUT_DIR = Path("test_outputs")

# Create necessary directories
for directory in [MODEL_DIR, TEMP_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {directory}")

def setup_model():
    """Initialize and set up the GPT-SoVITS model.
    
    GPT-SoVITSモデルを初期化してセットアップします。
    """
    try:
        settings = Settings(
            MODEL_DIR=MODEL_DIR,
            TEMP_DIR=TEMP_DIR
        )
        
        # Check CUDA availability
        device = "cpu"
        try:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA is available and will be used")
                # Log CUDA device information
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            else:
                logger.warning("CUDA is not available, using CPU")
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {str(e)}, defaulting to CPU")
        logger.info(f"Using device: {device}")
        
        model = GPTSoVITSModel(
            model_dir=settings.MODEL_DIR,
            device=device
        )
        logger.info("Model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}")
        sys.exit(1)

def verify_zero_shot(model: GPTSoVITSModel):
    """Test Zero-shot TTS functionality.
    
    Zero-shot TTSの機能をテストします。
    """
    test_text = "こんにちは、音声合成のテストです。"
    logger.info(f"Testing Zero-shot TTS with text: {test_text}")
    
    try:
        audio = model.generate_speech(
            text=test_text,
            speaker_id=1,  # Default speaker ID
            mode="zero-shot"
        )
        output_path = OUTPUT_DIR / "test_output_zero_shot.wav"
        torchaudio.save(output_path, audio.unsqueeze(0), sample_rate=24000)
        logger.info(f"Zero-shot audio saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error in Zero-shot TTS: {str(e)}")
        return False

def verify_few_shot(model: GPTSoVITSModel):
    """Test Few-shot TTS functionality.
    
    Few-shot TTSの機能をテストします。
    """
    test_text = "これはFew-shot TTSのテストです。"
    logger.info(f"Testing Few-shot TTS with text: {test_text}")
    
    try:
        # Assuming reference audio is available in the test data directory
        ref_audio_path = Path("test_data/reference_audio.wav")
        if not ref_audio_path.exists():
            logger.error(f"Reference audio not found at: {ref_audio_path}")
            return False
            
        audio = model.generate_speech(
            text=test_text,
            reference_audio=str(ref_audio_path),
            mode="few-shot"
        )
        output_path = OUTPUT_DIR / "test_output_few_shot.wav"
        torchaudio.save(output_path, audio.unsqueeze(0), sample_rate=24000)
        logger.info(f"Few-shot audio saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error in Few-shot TTS: {str(e)}")
        return False

def main():
    """Main function to run TTS verification.
    
    TTS検証を実行するメイン関数。
    """
    logger.info("Starting TTS verification...")
    
    try:
        model = setup_model()
        
        logger.info("Testing Zero-shot TTS...")
        zero_shot_success = verify_zero_shot(model)
        
        logger.info("Testing Few-shot TTS...")
        few_shot_success = verify_few_shot(model)
        
        if zero_shot_success and few_shot_success:
            logger.info("All verifications completed successfully")
            return 0
        else:
            logger.error("Some verifications failed")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error during verification: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    main()
