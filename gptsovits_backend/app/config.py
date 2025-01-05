from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    MODEL_DIR: Path
    TEMP_DIR: Path
    MAX_AUDIO_LENGTH: int = 300  # Maximum audio length in seconds (5 minutes)

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()
