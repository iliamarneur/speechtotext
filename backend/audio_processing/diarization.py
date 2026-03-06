"""Diarisation des locuteurs via pyannote.audio (optionnel)."""

import numpy as np
import torch
import av

from backend.config import HF_TOKEN, ENABLE_DIARIZATION

SAMPLING_RATE = 16000

_pipeline = None
_available = None


def is_available() -> bool:
    """Verifie si la diarisation est disponible (pyannote installe + HF_TOKEN configure)."""
    global _available
    if _available is not None:
        return _available
    if not ENABLE_DIARIZATION or not HF_TOKEN:
        _available = False
        return False
    try:
        import pyannote.audio  # noqa: F401
        _available = True
    except ImportError:
        _available = False
        print("[DIARIZATION] pyannote.audio non installe. Diarisation desactivee.")
    return _available


def _read_audio(file_path: str) -> torch.Tensor:
    """Lit un fichier audio via PyAV et retourne un tensor mono 16kHz float."""
    container = av.open(file_path)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLING_RATE)

    frames = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for r in resampled:
            frames.append(r.to_ndarray().flatten())
    container.close()

    if not frames:
        return torch.FloatTensor([])

    audio = np.concatenate(frames)
    return torch.FloatTensor(audio.astype(np.float32) / 32768.0)


def _load_pipeline():
    """Charge le pipeline pyannote (lazy, cache)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from pyannote.audio import Pipeline

    print("[DIARIZATION] Chargement du pipeline pyannote...")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    if torch.cuda.is_available():
        _pipeline.to(torch.device("cuda"))
    print("[DIARIZATION] Pipeline pret.")
    return _pipeline


def diarize(audio_path: str, num_speakers: int = None) -> dict:
    """
    Lance la diarisation sur un fichier audio.

    Returns:
        dict avec:
        - turns: list de {start, end, speaker}
        - num_speakers: nombre de locuteurs detectes
        - speakers: liste des labels uniques
    """
    pipeline = _load_pipeline()

    # Charger l'audio avec PyAV (evite la dependance torchcodec/ffmpeg)
    waveform = _read_audio(audio_path)
    audio_input = {"waveform": waveform.unsqueeze(0), "sample_rate": SAMPLING_RATE}

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization_result = pipeline(audio_input, **kwargs)

    turns = []
    speakers_set = set()
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })
        speakers_set.add(speaker)

    return {
        "turns": turns,
        "num_speakers": len(speakers_set),
        "speakers": sorted(speakers_set),
    }


def assign_speakers(segments: list, turns: list) -> list:
    """
    Assigne un locuteur a chaque segment de transcription par chevauchement temporel.

    Args:
        segments: list de {start, end, text} (start/end en secondes)
        turns: list de {start, end, speaker} depuis diarize()

    Returns:
        Les memes segments avec un champ 'speaker' ajoute.
    """
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        best_speaker = None
        best_overlap = 0

        for turn in turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        seg["speaker"] = best_speaker or "SPEAKER_00"

    return segments
