import os
import sys
from pathlib import Path
from app.config import Settings
from app.models.gpt_sovits import GPTSoVITSModel

def main():
    # Initialize settings with default values
    settings = Settings(
        MODEL_DIR=Path("models"),
        TEMP_DIR=Path("temp"),
        MAX_AUDIO_LENGTH=120
    )
    
    # Create model handler
    model = GPTSoVITSModel(settings)
    
    try:
        print("Attempting to load model...")
        model.load_model()
        print("Model loaded successfully!")
        print(f"Model type: {type(model.model)}")
        print("Model structure:")
        for key, value in model.model.items():
            print(f"- {key}: {type(value)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
