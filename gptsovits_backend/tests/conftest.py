import pytest
from pathlib import Path
import torch
import torch.nn as nn
from app.config import Settings

@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        MODEL_DIR=tmp_path / "models",
        TEMP_DIR=tmp_path / "temp",
        MAX_AUDIO_LENGTH=120
    )

@pytest.fixture
def mock_gpt_sovits_model(test_settings, monkeypatch):
    # Create model directory and dummy model
    model_dir = test_settings.MODEL_DIR / "gpt_sovits"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    
    # Create dummy model
    test_layer = nn.Linear(10, 10)
    dummy_model = {
        'weight': test_layer.state_dict(),
        'config': {'model_config': 'test'},
        'info': 'test model'
    }
    torch.save(dummy_model, str(model_path))
    
    return model_path
