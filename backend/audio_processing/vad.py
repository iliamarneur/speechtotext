"""
Pré-traitement audio avec Silero VAD (Voice Activity Detection).
Détecte les segments de parole, calcule des stats sur l'audio
(ratio parole/silence, durée, nombre de segments).
"""

import numpy as np
import torch
import av

SAMPLING_RATE = 16000

_model = None
_get_speech_timestamps = None
_loaded = False


def _read_audio(file_path: str) -> torch.Tensor:
    """
    Lit un fichier audio et le convertit en tensor mono 16kHz float.
    Utilise PyAV (embarqué avec faster-whisper) pour décoder tout format.
    """
    container = av.open(file_path)
    stream = container.streams.audio[0]

    # Resampler vers 16kHz mono
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLING_RATE)

    frames = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for r in resampled:
            arr = r.to_ndarray().flatten()
            frames.append(arr)
    container.close()

    if not frames:
        return torch.FloatTensor([])

    audio = np.concatenate(frames)
    # Normaliser int16 -> float [-1, 1]
    return torch.FloatTensor(audio.astype(np.float32) / 32768.0)


def load_model():
    """Charge le modèle Silero VAD (appelé au démarrage)."""
    global _model, _get_speech_timestamps, _loaded
    if _loaded:
        return
    print("[VAD] Chargement du modèle Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    _model = model
    _get_speech_timestamps = utils[0]
    _loaded = True
    print("[VAD] Modèle VAD prêt.")


def is_loaded() -> bool:
    return _loaded


def analyze(file_path: str) -> dict:
    """
    Analyse un fichier audio avec VAD.

    Returns:
        dict avec:
        - total_duration: durée totale en secondes
        - speech_duration: durée de parole en secondes
        - silence_duration: durée de silence en secondes
        - speech_ratio: ratio parole/total (0.0 à 1.0)
        - num_speech_segments: nombre de segments de parole détectés
        - segments: liste de {start, end, duration} en secondes
    """
    if not _loaded:
        raise RuntimeError("Le modèle VAD n'est pas chargé. Appeler load_model() d'abord.")

    # Lecture audio via PyAV (compatible tous formats)
    wav = _read_audio(file_path)
    total_samples = len(wav)
    total_duration = total_samples / SAMPLING_RATE

    # Détection des segments de parole
    speech_timestamps = _get_speech_timestamps(
        wav,
        _model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )

    # Conversion en secondes et calcul des stats
    segments = []
    speech_samples = 0
    for ts in speech_timestamps:
        start_sec = ts["start"] / SAMPLING_RATE
        end_sec = ts["end"] / SAMPLING_RATE
        segments.append({
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
        })
        speech_samples += ts["end"] - ts["start"]

    speech_duration = speech_samples / SAMPLING_RATE
    speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

    return {
        "total_duration": round(total_duration, 3),
        "speech_duration": round(speech_duration, 3),
        "silence_duration": round(total_duration - speech_duration, 3),
        "speech_ratio": round(speech_ratio, 3),
        "num_speech_segments": len(segments),
        "segments": segments,
    }
