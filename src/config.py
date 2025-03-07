# config.py
from dotenv import load_dotenv
import os
from typing import Any, Dict


class Config:
    """Configuration handler for application environment variables.

    Loads values from .env file and provides typed access with defaults.
    """

    _loaded = False
    _values: Dict[str, Any] = {}

    @classmethod
    def load(cls, env_path: str = ".env") -> None:
        """Load environment variables from the specified .env file.

        Args:
            env_path: Path to the .env file (default: ".env")
        """
        if not cls._loaded:
            # Load environment variables from .env file
            load_dotenv(env_path)
            cls._loaded = True

    @classmethod
    def get(cls, key: str, default: Any = None, cast_type: Any = None) -> Any:
        """Get an environment variable with optional default and type casting.

        Args:
            key: Environment variable name
            default: Default value if the variable is not set
            cast_type: Type to cast the value to (int, float, bool, etc.)

        Returns:
            The environment variable value or default
        """
        # Ensure environment is loaded
        cls.load()

        # Check if already cached
        if key in cls._values:
            return cls._values[key]

        # Get value from environment
        value = os.getenv(key, default)

        # Handle type casting
        if cast_type is not None and value is not None:
            if cast_type is bool:
                # Handle boolean conversion from string
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
            else:
                # Apply the requested cast type
                value = cast_type(value)

        # Cache the value
        cls._values[key] = value
        return value

    # Ollama settings
    @classmethod
    def ollama_model(cls) -> str:
        """Get the Ollama model name."""
        return cls.get("OLLAMA_MODEL", default="phi3-mini")

    @classmethod
    def ollama_url(cls) -> str:
        """Get the Ollama API URL."""
        return cls.get("OLLAMA_URL", default="http://localhost:11434")

    # Whisper settings
    @classmethod
    def whisper_model_size(cls) -> str:
        """Get the Whisper model size."""
        return cls.get("WHISPER_MODEL_SIZE", default="base")

    @classmethod
    def whisper_device(cls) -> str:
        """Get the Whisper compute device."""
        return cls.get("WHISPER_DEVICE", default="auto")

    @classmethod
    def whisper_compute_type(cls) -> str:
        """Get the Whisper compute type."""
        return cls.get("WHISPER_COMPUTE_TYPE", default="float32")

    # Kokoro settings
    @classmethod
    def kokoro_language(cls) -> str:
        """Get the Kokoro language."""
        return cls.get("KOKORO_LANGUAGE", default="en")

    @classmethod
    def kokoro_voice(cls) -> str:
        """Get the Kokoro voice."""
        return cls.get("KOKORO_VOICE", default="af_heart")

    # Level manager settings
    @classmethod
    def level_manager_path(cls) -> str:
        """Get the level manager path."""
        return cls.get("LEVEL_MANAGER_PATH", default="src/level_manager/model")

    @classmethod
    def print_all(cls):
        print(f"Ollama Model: {cls.ollama_model()}")
        print(f"Ollama URL: {cls.ollama_url()}")
        print(f"Whisper Model Size: {cls.whisper_model_size()}")
        print(f"Whisper Device: {cls.whisper_device()}")
        print(f"Whisper Compute Type: {cls.whisper_compute_type()}")
        print(f"Kokoro Language: {cls.kokoro_language()}")
        print(f"Kokoro Voice: {cls.kokoro_voice()}")
        print(f"Level Manager Path: {cls.level_manager_path()}")


# Usage example:
if __name__ == "__main__":
    # Print all configuration values
    print(f"Ollama Model: {Config.ollama_model()}")
    print(f"Ollama URL: {Config.ollama_url()}")
    print(f"Whisper Model Size: {Config.whisper_model_size()}")
    print(f"Whisper Device: {Config.whisper_device()}")
    print(f"Whisper Compute Type: {Config.whisper_compute_type()}")
    print(f"Kokoro Language: {Config.kokoro_language()}")
    print(f"Kokoro Voice: {Config.kokoro_voice()}")
    print(f"Level Manager Path: {Config.level_manager_path()}")
