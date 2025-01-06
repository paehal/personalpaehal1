"""Test Zero-shot and Few-shot TTS functionality.

Zero-shotとFew-shot TTSの機能をテストします。
"""
import pytest
import torch
from pathlib import Path
from app.models.gpt_sovits import GPTSoVITSModel
from app.models.text_utils import JapaneseTextProcessor

@pytest.fixture
def model():
    """Initialize GPT-SoVITS model for testing.
    
    テスト用のGPT-SoVITSモデルを初期化します。
    """
    return GPTSoVITSModel(
        model_path=None  # Mock model path
    )

def test_zero_shot_tts(model):
    """Test Zero-shot TTS functionality.
    
    Zero-shot TTSの機能をテストします。
    """
    # Test with valid input
    text = "こんにちは"
    speaker_id = 1
    output = model.generate_speech(text, speaker_id=speaker_id, use_zero_shot=True)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # Batch size

    # Test without speaker_id
    with pytest.raises(ValueError, match="Speaker ID required for Zero-shot TTS"):
        model.generate_speech(text, use_zero_shot=True)

def test_few_shot_tts(model):
    """Test Few-shot TTS functionality.
    
    Few-shot TTSの機能をテストします。
    """
    # Test with valid input
    text = "こんにちは"
    ref_audio = "path/to/reference.wav"
    output = model.generate_speech(
        text, 
        reference_audio=ref_audio, 
        use_zero_shot=False
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # Batch size

    # Test without reference audio
    with pytest.raises(ValueError, match="Reference audio required for Few-shot TTS"):
        model.generate_speech(text, use_zero_shot=False)

def test_model_not_loaded_error(model):
    """Test error when model is not loaded.
    
    モデルが読み込まれていない場合のエラーをテストします。
    """
    text = "こんにちは"
    with pytest.raises(RuntimeError, match="Model not loaded"):
        model.generate_speech(text)
