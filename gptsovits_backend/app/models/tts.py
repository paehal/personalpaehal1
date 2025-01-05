from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class Language(str, Enum):
    """Supported languages for TTS."""
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"
    CANTONESE = "yue"
    CHINESE = "zh"

class TTSMode(str, Enum):
    """TTS processing modes."""
    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"

class TTSRequest(BaseModel):
    """Base TTS request model."""
    text: str = Field(..., description="Text to convert to speech")
    source_language: Language = Field(..., description="Source text language")
    target_language: Language = Field(..., description="Target speech language")
    mode: TTSMode = Field(..., description="TTS processing mode")

class ZeroShotTTSRequest(TTSRequest):
    """Zero-shot TTS request model."""
    reference_audio: str = Field(..., description="Base64 encoded 5-second reference audio")

class FewShotTTSRequest(TTSRequest):
    """Few-shot TTS request model."""
    training_audio: str = Field(..., description="Base64 encoded 1-minute training audio")

class TTSResponse(BaseModel):
    """TTS response model."""
    audio: str = Field(..., description="Base64 encoded output audio")
    duration: float = Field(..., description="Duration of the output audio in seconds")
    error: Optional[str] = Field(None, description="Error message if processing failed")
