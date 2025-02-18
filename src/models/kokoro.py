"""Kokoro TTS module"""

import io
import wave
import re
from typing import cast
import numpy as np
from kokoro import KPipeline  # type: ignore
from kokoro.pipeline import ALIASES  # type: ignore


class Kokoro:
    """Handles text-to-speech using Kokoro"""

    def __init__(self, default_lang: str = "en"):
        """Initialize Kokoro with a default language"""
        # Set up language mappings
        self.supported_languages = set(
            lang.split("-")[0] for lang in list(ALIASES.keys())
        )
        self.language_alias_map = {
            key.split("-")[0]: value for key, value in ALIASES.items()
        }
        self.language_codes = {key.split("-")[0]: key for key in ALIASES.keys()}

        self.lang_code = self.language_codes[default_lang]
        self.pipeline = KPipeline(lang_code=self.lang_code)

    def _clean_text(self, text: str) -> str:
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

        # Remove emojis using regex
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
        return re.sub(r"\s+", " ", text).strip()

    def _update_language(self, new_lang: str):
        """Updates the pipeline only if the language has changed"""
        if new_lang != self.lang_code:
            self.pipeline = KPipeline(lang_code=new_lang)
            self.lang_code = new_lang
            print(f"Pipeline updated to {new_lang}")

    def generate_audio(
        self, text: str, language: str, voice: str, speed: float = 1.0
    ) -> io.BytesIO:
        """Generate audio from text"""
        if language not in self.supported_languages:
            raise ValueError(f"Language '{language}' not supported.")

        # Clean text before processing
        clean_text = self._clean_text(text)

        # Update pipeline if needed
        self._update_language(self.language_codes[language])

        # Generate audio
        generator = self.pipeline(
            clean_text,
            voice=voice,
            speed=speed,
            split_pattern=r"\n+",
        )

        # Convert to WAV
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

    def is_language_supported(self, lang: str) -> bool:
        """Get set of supported languages"""
        return lang in self.supported_languages

    def get_supported_languages(self) -> set:
        """Get set of supported languages"""
        return self.supported_languages

    def get_language_aliases(self) -> dict:
        """Get mapping of language codes to aliases"""
        return self.language_alias_map

    def get_language_codes(self) -> dict:
        """Get mapping of short codes to full language codes"""
        return self.language_codes
