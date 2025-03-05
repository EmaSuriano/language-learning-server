"""Test Kokoro TTS configuration and basic functionality"""

import os
import time
from models.kokoro import Kokoro
import wave

from src.config import Config

current_directory = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_directory, "assets/example-tts.mp3")

TEST_TEXT = "Hello, this is a test of the text to speech system. How does it sound?"


def main():
    # Get configuration from environment
    KOKORO_LANGUAGE = Config.kokoro_language()
    KOKORO_VOICE = Config.kokoro_voice()

    print("\n-----------------------------")
    print("Testing Kokoro configuration:")
    print(f"* Language: {KOKORO_LANGUAGE}")
    print(f"* Voice: {KOKORO_VOICE}")

    # Initialize
    kokoro = Kokoro(default_lang=KOKORO_LANGUAGE)

    # Generate audio with metrics
    start_time = time.time()
    wav_buffer = kokoro.generate_audio(
        text=TEST_TEXT,
        language=KOKORO_LANGUAGE,
        voice=KOKORO_VOICE,
    )
    processing_time = time.time() - start_time

    # Save output for verification
    with open(output_path, "wb") as output_file:
        output_file.write(wav_buffer.getvalue())

    # Calculate metrics
    audio_size = len(wav_buffer.getvalue()) / 1024  # Size in KB
    char_count = len(TEST_TEXT)
    word_count = len(TEST_TEXT.split())

    # Get audio duration (samples / sample_rate)
    with wave.open(output_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        audio_duration = frames / rate
        processing_speed = (
            round(audio_duration / processing_time, 2) if processing_time > 0 else 0
        )

    # Print input text
    print("\n-----------------------------")
    print("Input Text:")
    print(f"* Original: {TEST_TEXT}")
    print(f"* Cleaned: {kokoro._clean_text(TEST_TEXT)}")

    # Print synthesis results
    print("\n-----------------------------")
    print("Synthesis Results:")
    print(f"* Output file: {output_path}")
    print(f"* Audio duration: {round(audio_duration, 2)}s")
    print(f"* File size: {round(audio_size, 2)}KB")

    # Print performance metrics
    print("\n-----------------------------")
    print("Performance Metrics:")
    print(f"* Processing time: {round(processing_time, 2)}s")
    print(f"* Characters processed: {char_count}")
    print(f"* Words processed: {word_count}")
    print(f"* Processing speed: {processing_speed}x realtime")


if __name__ == "__main__":
    main()
