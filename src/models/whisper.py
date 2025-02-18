"""Whisper transcription module"""

from dataclasses import dataclass
from typing import Optional
from faster_whisper import WhisperModel  # type: ignore


@dataclass
class TranscriptionResult:
    """Structured result from transcription"""

    language: str
    language_probability: float
    text: str
    start: Optional[float]
    end: Optional[float]


class Whisper:
    """Handles audio transcription using Whisper"""

    def __init__(
        self,
        model_size: str,
        device: str = "auto",
        compute_type: str = "default",
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, file_path: str, language: str) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            file_path, beam_size=5, language=language
        )

        transcription_text = ""
        start_time = None
        end_time = None

        for segment in segments:
            transcription_text += segment.text + " "
            if start_time is None:
                start_time = segment.start
            end_time = segment.end

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            text=transcription_text.strip(),
            start=start_time,
            end=end_time,
        )

    def get_supported_languages(self):
        return self.model.supported_languages

    def is_language_supported(self, lang: str):
        return lang in self.model.supported_languages
