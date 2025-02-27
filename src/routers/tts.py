"""This module contains the text-to-speech (TTS) API endpoints"""

import io
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import HfApi
from pydantic import BaseModel

from models.kokoro import Kokoro

router = APIRouter(prefix="/tts", tags=["text-to-speech"])

api = HfApi()


KOKORO_LANGUAGE = os.getenv("KOKORO_LANGUAGE", "en")

kokoro = Kokoro(default_lang=KOKORO_LANGUAGE)


class CreateSpeechRequestBody(BaseModel):
    """Request body for the /v1/audio/speech endpoint"""

    input: str
    voice: str
    language: str
    speed: float = 1.0


def generate_wav_audio(body: CreateSpeechRequestBody) -> io.BytesIO:
    """Generate a WAV audio file from text-to-speech input."""

    if not kokoro.is_language_supported(body.language):
        raise HTTPException(
            status_code=404,
            detail=f"Language '{body.language}' not supported.",
        )

    return kokoro.generate_audio(
        text=body.input,
        language=body.language,
        voice=body.voice,
    )


@router.post("/synthesize")
async def synthesize(
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
    """Synthesize text to speech"""

    wav_buffer = generate_wav_audio(body)

    def iter_audio():
        yield from wav_buffer

    return StreamingResponse(iter_audio(), media_type="audio/wav")


@router.post("/synthesize/no-stream")
async def synthesize_file(body: CreateSpeechRequestBody):
    """Synthesize text to speech and return a file response"""

    wav_buffer = generate_wav_audio(body)

    temp_filename = "output.wav"
    with open(temp_filename, "wb") as temp_file:
        temp_file.write(wav_buffer.getvalue())

    return FileResponse(temp_filename, media_type="audio/wav", filename="output.wav")


@router.get("/languages")
def get_supported_languages():
    """Get supported languages"""
    return kokoro.get_supported_languages()


@router.get("/voices/{language}")
def get_voices_by_language(language: str):
    """Get supported languages"""

    # Define repository and folder path
    repo_id = "hexgrad/Kokoro-82M"
    folder_path = "voices/"

    # Get the list of files
    files = api.list_repo_files(repo_id=repo_id)

    # Validate that the language is supported
    if not kokoro.is_language_supported(language):
        raise HTTPException(
            status_code=404,
            detail=f"Language '{language}' not supported.",
        )

    # Filter only files in the 'voices' directory
    all_voices = [
        file.replace(folder_path, "").replace(".pt", "")
        for file in files
        if file.startswith(folder_path)
    ]

    # Get the voices only for the specified language
    alias = kokoro.get_language_aliases()[language]
    language_voice = [file for file in all_voices if file.startswith(alias)]

    return language_voice
