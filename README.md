# Language Learning Server

This repository contains the backend server for an AI-powered language learning application. It provides a comprehensive set of APIs that enable interactive language learning conversations, speech recognition, text-to-speech, performance evaluation, and adaptive difficulty management.

## Related Projects

This project is part of a larger language learning ecosystem:

- **Research Paper**: [language-learning-paper](https://github.com/EmaSuriano/language-learning-paper) - The academic research behind this project
- **Client Application**: [language-learning-client](https://github.com/EmaSuriano/language-learning-client) - The frontend interface for this language learning system

## Features

- **Conversational Language Learning**: Engage with AI language tutors in contextual scenarios
- **Speech-to-Text**: Convert user speech to text for analysis using Whisper
- **Text-to-Speech**: Generate natural sounding audio responses with Kokoro TTS
- **Performance Evaluation**: Assess grammar, vocabulary, fluency, and goal achievement
- **Adaptive Difficulty**: Automatically adjust language learning levels based on user performance
- **Multi-language Support**: Practice multiple languages with proper CEFR leveling

## System Architecture

The server is built using FastAPI and integrates several key components:

- **Ollama**: Local LLM integration for conversational agents
- **Whisper**: Speech recognition for audio input
- **Kokoro**: Text-to-speech generation for natural audio responses
- **Chroma DB**: Vector database for language examples at different CEFR levels
- **Level Manager**: ML-based system for adjusting difficulty based on user performance

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running locally
- GPU support recommended for Whisper and LLM inference

## Installation

This project uses the `uv` package manager for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/language-learning-server.git
cd language-learning-server

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Initialize database and vector stores
python src/reset_db.py
python src/reset_rag.py
```

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit the `.env` file to configure:

- `OLLAMA_MODEL`: The model to use (default: phi4)
- `OLLAMA_URL`: URL for Ollama API (default: http://localhost:11434)
- `WHISPER_MODEL_SIZE`: Size of Whisper model (base, small, medium, large)
- `WHISPER_DEVICE`: Computing device (cuda, cpu, auto)
- `WHISPER_COMPUTE_TYPE`: Compute precision (float16, float32)
- `KOKORO_LANGUAGE`: Default TTS language
- `KOKORO_VOICE`: Default TTS voice

## Running the Server

Start the FastAPI server with:

```bash
uv run src/start_server.py
```

The server will be available at `http://0.0.0.0:8000` with API documentation at `http://0.0.0.0:8000/docs`.

## API Endpoints

The server provides several API endpoints:

- `/users`: User management
- `/languages`: Language information
- `/situations`: Learning scenarios
- `/assistant`: Conversation with AI language tutors
- `/evaluator`: Performance evaluation
- `/tts`: Text-to-speech conversion
- `/stt`: Speech-to-text conversion
- `/translator`: Translation services
- `/learning-history`: Learning progress tracking

## Windows and CUDA Support

For Windows users with GPU support, it's recommended to run the project through WSL:

```bash
# After starting WSL, expose the port to the host network
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<wsl-ip-address>
```

## Testing

The repository includes several test scripts in the root directory to verify functionality of different components:

```bash
# Test Ollama integration
python src/test_llm.py

# Test speech-to-text
python src/test_stt.py

# Test text-to-speech
python src/test_tts.py

# Test level management system
python src/test_level_manager.py
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
