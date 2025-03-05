"""This module contains the speech-to-text router"""

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile

from models.whisper import Whisper

from config import Config

router = APIRouter(prefix="/stt", tags=["speech-to-text"])

whisper = Whisper(
    model_size=Config.whisper_model_size(),
    device=Config.whisper_device(),
    compute_type=Config.whisper_compute_type(),
)


@router.post("/transcribe")
async def transcribe(
    language: str = Query(..., description="Language code for transcription"),
    file: UploadFile = File(...),
):
    """Transcribe audio file"""

    if not whisper.is_language_supported(language):
        raise ValueError(f"Language {language} is not supported")

    temp_audio_path = Path("temp_audio.mp3")

    with temp_audio_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = whisper.transcribe(str(temp_audio_path), language)

    # Delete file after processing
    temp_audio_path.unlink()

    return result


@router.get("/languages")
def get_supported_languages():
    """Get supported languages"""

    return whisper.get_supported_languages()
