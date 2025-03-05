"""Test Ollama configuration and basic functionality"""

import os
import time
from models.whisper import Whisper
from src.config import Config


current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "assets/example-stt.mp3")


def main():
    WHISPER_DEVICE = Config.whisper_device()
    WHISPER_MODEL_SIZE = Config.whisper_model_size()
    WHISPER_COMPUTE_TYPE = Config.whisper_compute_type()

    print("\n-----------------------------")
    print("Testing Whisper configuration:")
    print(f"* Model: {WHISPER_MODEL_SIZE}")
    print(f"* Device: {WHISPER_DEVICE}")
    print(f"* Compute Type: {WHISPER_COMPUTE_TYPE}")

    # Initialize
    whisper = Whisper(
        model_size=WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    # Transcribe with metrics
    start_time = time.time()
    result = whisper.transcribe(file_path=file_path, language="en")
    processing_time = time.time() - start_time

    # Calculate metrics
    audio_duration = result.end if result.end else 0  # Just use end time as duration
    char_count = len(result.text)
    word_count = len(result.text.split())
    processing_speed = (
        round(audio_duration / processing_time, 2) if processing_time > 0 else 0
    )

    # Print transcription results
    print("\n-----------------------------")
    print("Transcription Results:")
    print(f"* Detected language: {result.language}")
    print(f"* Confidence: {result.language_probability:.2f}")
    print(f"* Text: {result.text}")
    print(f"* Duration: {result.start:.2f}s to {result.end:.2f}s")

    # Print performance metrics
    print("\n-----------------------------")
    print("Performance Metrics:")
    print(f"* Processing time: {round(processing_time, 2)}s")
    print(f"* Audio duration: {round(audio_duration, 2)}s")
    print(f"* Characters transcribed: {char_count}")
    print(f"* Words transcribed: {word_count}")
    print(f"* Processing speed: {processing_speed}x realtime")


if __name__ == "__main__":
    main()
