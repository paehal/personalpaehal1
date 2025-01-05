"""FastAPI router for Text-to-Speech endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..config import Settings, get_settings
from ..models.tts import (
    FewShotTTSRequest,
    TTSResponse,
    ZeroShotTTSRequest,
)
from ..services.tts_service import TTSService


router = APIRouter(prefix="/api/tts", tags=["tts"])


@router.post("/zero-shot", response_model=TTSResponse)
async def zero_shot_tts(
    request: ZeroShotTTSRequest, settings: Settings = Depends(get_settings)
) -> TTSResponse:
    """
    Generate speech using zero-shot TTS (5-second sample).
    """
    try:
        service = TTSService(settings)
        audio, duration = await service.process_zero_shot(
            text=request.text,
            reference_audio=request.reference_audio,
            source_lang=request.source_language,
            target_lang=request.target_language,
        )
        return TTSResponse(audio=audio, duration=duration, error=None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/few-shot", response_model=TTSResponse)
async def few_shot_tts(
    request: FewShotTTSRequest, settings: Settings = Depends(get_settings)
) -> TTSResponse:
    """
    Generate speech using few-shot TTS (1-minute training).
    """
    try:
        service = TTSService(settings)
        audio, duration = await service.process_few_shot(
            text=request.text,
            training_audio=request.training_audio,
            source_lang=request.source_language,
            target_lang=request.target_language,
        )
        return TTSResponse(audio=audio, duration=duration, error=None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
