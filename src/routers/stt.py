"""This module contains the speech-to-text router"""

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile
from faster_whisper import WhisperModel  # type: ignore

router = APIRouter(prefix="/stt", tags=["speech-to-text"])

MODEL_SIZE = "base"
DEVICE = "cuda"

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")


def transcribe_audio(file_path: str, language: str):
    """Transcribe audio file"""
    segments, info = model.transcribe(file_path, beam_size=5, language=language)
    transcription_text = ""
    start_time = None
    end_time = None

    for segment in segments:
        transcription_text += segment.text + " "
        if start_time is None:
            start_time = segment.start
        end_time = segment.end

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "text": transcription_text.strip(),
        "start": start_time,
        "end": end_time,
    }


@router.post("/transcribe")
async def transcribe(
    language: str = Query(..., description="Language code for transcription"),
    file: UploadFile = File(...),
):
    """Transcribe audio file"""

    if language not in model.supported_languages:
        raise ValueError(f"Language {language} is not supported")

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
