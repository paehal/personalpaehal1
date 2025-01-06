import pytest
from pathlib import Path
import torch
import torch.nn as nn
from app.models.gpt_sovits import GPTSoVITSModel
from app.config import Settings

def test_gpt_sovits_model_initialization(tmp_path):
    # Create test settings with temporary directories
    settings = Settings(
        MODEL_DIR=tmp_path / "models",
        TEMP_DIR=tmp_path / "temp",
        MAX_AUDIO_LENGTH=120
    )
    
    # Create model directory
    model_dir = settings.MODEL_DIR / "gpt_sovits"
    model_dir.mkdir(parents=True)
    
    # Create a dummy model file
    model_path = model_dir / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    test_layer = nn.Linear(10, 10)
    dummy_model = {
        'weight': test_layer.state_dict(),
        'config': {'model_config': 'test'},
        'info': 'test model'
    }
    torch.save(dummy_model, str(model_path))
    
    # Initialize model
    model = GPTSoVITSModel(settings)
    assert model is not None
    assert model.model_dir == model_dir
    assert model.model_path == model_path
    
def test_gpt_sovits_model_loading(tmp_path):
    # Create test settings
    settings = Settings(
        MODEL_DIR=tmp_path / "models",
        TEMP_DIR=tmp_path / "temp",
        MAX_AUDIO_LENGTH=120
    )
    
    # Create model directory and dummy model
    model_dir = settings.MODEL_DIR / "gpt_sovits"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    test_layer = nn.Linear(10, 10)
    dummy_model = {
        'weight': test_layer.state_dict(),
        'config': {'model_config': 'test'},
        'info': 'test model'
    }
    torch.save(dummy_model, str(model_path))
    
    # Test model loading
    model = GPTSoVITSModel(settings)
    model.load_model()
    assert model.is_model_loaded()
    assert isinstance(model.model, dict)
    assert 'weight' in model.model
    assert 'config' in model.model
    assert 'info' in model.model
    
def test_gpt_sovits_model_loading_error(tmp_path):
    # Create test settings without model file
    settings = Settings(
        MODEL_DIR=tmp_path / "models",
        TEMP_DIR=tmp_path / "temp",
        MAX_AUDIO_LENGTH=120
    )
    
    # Test model loading error
    model = GPTSoVITSModel(settings)
    with pytest.raises(FileNotFoundError):
        model.load_model()
