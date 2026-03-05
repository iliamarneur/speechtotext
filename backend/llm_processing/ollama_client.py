"""Client HTTP pour l'API Ollama (génération de texte en streaming)."""

import json
import requests

from backend.config import OLLAMA_URL


def is_available() -> bool:
    """Vérifie si Ollama est accessible."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """Liste les modèles disponibles sur Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def generate_stream(prompt: str, model: str, system: str = None):
    """
    Génère du texte en streaming via Ollama.

    Yields:
        dict avec 'response' (token) et 'done' (bool).
        Le dernier chunk contient les stats (total_duration, etc.).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if line:
            yield json.loads(line)
