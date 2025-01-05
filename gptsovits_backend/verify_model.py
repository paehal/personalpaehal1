import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model():
    try:
        model_path = Path('models/gpt_sovits/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')
        logger.info(f"Attempting to load model from {model_path}")
        
        # Check if file exists and its size
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            return
        logger.info(f"Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Try loading with torch.load first to inspect contents
        try:
            state_dict = torch.load(str(model_path), map_location='cpu')
            logger.info(f"Model contents (torch.load): {type(state_dict)}")
            logger.info(f"Keys in state_dict: {state_dict.keys() if isinstance(state_dict, dict) else 'Not a dictionary'}")
        except Exception as e:
            logger.error(f"Error with torch.load: {str(e)}")
        
        # Try loading with torch.jit
        try:
            model = torch.jit.load(str(model_path))
            logger.info("Model loaded successfully with torch.jit")
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model attributes: {dir(model)}")
        except Exception as e:
            logger.error(f"Error with torch.jit.load: {str(e)}")
            logger.error(f"Error type: {type(e)}")

if __name__ == "__main__":
    verify_model()
