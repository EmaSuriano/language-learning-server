"""This module contains the speech-to-text router"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile
from faster_whisper import WhisperModel  # type: ignore

router = APIRouter(prefix="/stt", tags=["speech-to-text"])

MODEL_SIZE = "base"
DEVICE = "cuda"

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")


def transcribe_audio(file_path: str, language: Optional[str] = "en"):
    """Transcribe audio file"""
    segments, info = model.transcribe(file_path, beam_size=5, language=language)
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "transcription": [
            {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments
        ],
    }


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):
    """Transcribe audio file"""
    temp_audio_path = Path("temp_audio.mp3")

    with temp_audio_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = transcribe_audio(str(temp_audio_path), language)

    # Delete file after processing
    temp_audio_path.unlink()

    return result


@router.get("/languages")
def get_supported_languages():
    """Get supported languages"""

    return model.supported_languages
