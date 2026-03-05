"""Configuration centralisée via variables d'environnement."""

import os

# Whisper
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "auto")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "./models")

# Database
DB_PATH = os.getenv("DB_PATH", "./data/transcriptions.db")

# Audio storage
STORE_AUDIO = os.getenv("STORE_AUDIO", "false").lower() in ("true", "1", "yes")
AUDIO_DIR = os.getenv("AUDIO_DIR", "./data/audio")

# Auth
API_KEY = os.getenv("API_KEY", "")

# VAD pre-processing
ENABLE_VAD = os.getenv("ENABLE_VAD", "true").lower() in ("true", "1", "yes")

# LLM (Ollama)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-nemo")
