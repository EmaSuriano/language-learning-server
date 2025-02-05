"""This module contains the text-to-speech (TTS) API endpoints"""

import io
import re
import wave
from typing import cast

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import HfApi
from kokoro import KPipeline  # type: ignore
from kokoro.pipeline import ALIASES  # type: ignore
from pydantic import BaseModel

router = APIRouter(prefix="/tts", tags=["text-to-speech"])

DEFAULT_LANGUAGE = "en-us"
DEFAULT_VOICE = "af_bella"
api = HfApi()


class PipelineManager:
    """Manages the pipeline and updates it only if the language has changed."""

    def __init__(self, default_lang: str):
        """Initialize the pipeline manager with a default language."""
        self.lang_code = default_lang
        self.pipeline = KPipeline(lang_code=default_lang)

    def update_pipeline(self, new_lang: str):
        """Updates the pipeline only if the language has changed."""
        if new_lang != self.lang_code:
            self.pipeline = KPipeline(lang_code=new_lang)
            self.lang_code = new_lang
            print(f"Pipeline updated to {new_lang}")


# Usage
manager = PipelineManager(DEFAULT_LANGUAGE)


def clean_text(text):
    """Clean text by removing unwanted characters and emojis"""

    replacements = {
        "–": " ",  # Replace en-dash with space
        "-": " ",  # Replace hyphen with space
        "**": " ",  # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F]|"  # Emoticons
        r"[\U0001F300-\U0001F5FF]|"  # Miscellaneous symbols and pictographs
        r"[\U0001F680-\U0001F6FF]|"  # Transport and map symbols
        r"[\U0001F700-\U0001F77F]|"  # Alchemical symbols
        r"[\U0001F780-\U0001F7FF]|"  # Geometric shapes extended
        r"[\U0001F800-\U0001F8FF]|"  # Supplemental arrows-C
        r"[\U0001F900-\U0001F9FF]|"  # Supplemental symbols and pictographs
        r"[\U0001FA00-\U0001FA6F]|"  # Chess symbols
        r"[\U0001FA70-\U0001FAFF]|"  # Symbols and pictographs extended-A
        r"[\U00002702-\U000027B0]|"  # Dingbats
        r"[\U0001F1E0-\U0001F1FF]"  # Flags (iOS)
        r"",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub(r"", text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


class CreateSpeechRequestBody(BaseModel):
    """Request body for the /v1/audio/speech endpoint"""

    input: str
    voice: str = DEFAULT_VOICE
    language: str = DEFAULT_LANGUAGE
    speed: float = 1.0


def generate_wav_audio(body: CreateSpeechRequestBody) -> io.BytesIO:
    """Generate a WAV audio file from text-to-speech input."""

    text = clean_text(body.input)
    manager.update_pipeline(body.language)

    generator = manager.pipeline(
        text,
        voice=body.voice,
        speed=body.speed,
        split_pattern=r"\n+",
    )

    wav_buffer = io.BytesIO()
    wav_file: wave.Wave_write = cast(wave.Wave_write, wave.open(wav_buffer, "wb"))

    with wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        for _, _, audio in generator:
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer


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

    languages = list(ALIASES.keys())

    return languages


@router.get("/voices/{language}")
def get_voices_by_language(language: str):
    """Get supported languages"""

    # Define repository and folder path
    repo_id = "hexgrad/Kokoro-82M"
    folder_path = "voices/"

    # Get the list of files
    files = api.list_repo_files(repo_id=repo_id)

    # Validate that the language is supported
    if language not in ALIASES:
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
    alias = ALIASES[language]
    language_voice = [file for file in all_voices if file.startswith(alias)]

    return language_voice
