import pytest
import pytest_asyncio
from pathlib import Path
import base64
import numpy as np
from app.services.tts_service import TTSService
from app.models.tts import Language, TTSMode
import soundfile as sf

def test_tts_service_initialization(test_settings, mock_gpt_sovits_model):
    service = TTSService(test_settings)
    assert service is not None
    assert service.gpt_sovits is not None
    assert service.gpt_sovits.is_model_loaded()

def create_test_audio(path: Path, duration: float = 5.0, sr: int = 22050):
    """Create a test audio file with specified duration"""
    samples = np.random.randn(int(duration * sr))
    sf.write(path, samples, sr)
    return path

@pytest.mark.asyncio
async def test_process_zero_shot(test_settings, mock_gpt_sovits_model):
    service = TTSService(test_settings)
    
    # Create test reference audio
    ref_path = test_settings.TEMP_DIR / "test_ref.wav"
    create_test_audio(ref_path, duration=5.0)
    
    # Convert audio to base64
    with open(ref_path, "rb") as f:
        ref_audio = base64.b64encode(f.read()).decode()
    
    # Test zero-shot processing
    result = await service.process_zero_shot(
        text="テストです",
        reference_audio=ref_audio,
        source_lang=Language.JAPANESE,
        target_lang=Language.JAPANESE
    )
    
    assert result[0] is not None  # audio
    assert isinstance(result[1], float)  # duration
    assert result[1] > 0

@pytest.mark.asyncio
async def test_process_few_shot(test_settings, mock_gpt_sovits_model):
    service = TTSService(test_settings)
    
    # Create test training audio
    train_path = test_settings.TEMP_DIR / "test_train.wav"
    create_test_audio(train_path, duration=10.0)
    
    # Convert audio to base64
    with open(train_path, "rb") as f:
        train_audio = base64.b64encode(f.read()).decode()
    
    # Test few-shot processing
    result = await service.process_few_shot(
        text="テストです",
        training_audio=train_audio,
        source_lang=Language.JAPANESE,
        target_lang=Language.JAPANESE
    )
    
    assert result[0] is not None  # audio
    assert isinstance(result[1], float)  # duration
    assert result[1] > 0
