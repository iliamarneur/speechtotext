"""
Service de transcription audio via faster-whisper.
Gère le chargement du modèle et la transcription avec streaming.
"""

import time

from faster_whisper import WhisperModel

from backend.config import MODEL_SIZE, DEVICE, COMPUTE_TYPE, MODEL_DIR

_model: WhisperModel = None
_ready = False
_device_used = ""
_compute_used = ""


def detect_device() -> tuple[str, str]:
    """Détecte automatiquement si CUDA est disponible, sinon fallback CPU."""
    if DEVICE != "auto":
        return DEVICE, COMPUTE_TYPE if COMPUTE_TYPE != "auto" else "float32"
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            print("[Whisper] GPU CUDA détecté — utilisation du GPU")
            ct = "float16" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
            return "cuda", ct
    except Exception:
        pass
    print("[Whisper] Pas de GPU CUDA — utilisation du CPU")
    ct = "int8" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
    return "cpu", ct


def load_model():
    """Charge le modèle Whisper (appelé au démarrage)."""
    global _model, _ready, _device_used, _compute_used

    _device_used, _compute_used = detect_device()
    print(f"\n{'='*60}")
    print(f"  Chargement du modèle Whisper '{MODEL_SIZE}'")
    print(f"  Device: {_device_used} | Compute: {_compute_used}")
    print(f"  Cela peut prendre plusieurs minutes au 1er lancement...")
    print(f"{'='*60}\n")

    start = time.time()
    _model = WhisperModel(
        MODEL_SIZE,
        device=_device_used,
        compute_type=_compute_used,
        download_root=MODEL_DIR,
    )
    elapsed = time.time() - start
    _ready = True

    print(f"\n{'='*60}")
    print(f"  MODÈLE PRÊT ! (chargé en {elapsed:.1f}s)")
    print(f"{'='*60}\n")


def is_ready() -> bool:
    return _ready


def get_model_info() -> dict:
    return {
        "model": MODEL_SIZE,
        "device": _device_used,
        "compute_type": _compute_used,
    }


def transcribe(file_path: str, language: str = None, beam_size: int = 5,
               vad_filter: bool = True):
    """
    Transcrit un fichier audio.

    Returns:
        tuple (segments_generator, info)
        - segments_generator: itérateur de segments (start, end, text, ...)
        - info: TranscriptionInfo (duration, language, language_probability)
    """
    if not _ready:
        raise RuntimeError("Le modèle Whisper n'est pas chargé.")

    lang = language.strip() if language else None
    segments_gen, info = _model.transcribe(
        file_path,
        language=lang,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    return segments_gen, info
