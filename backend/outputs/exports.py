"""
Générateurs d'export pour les transcriptions.
Formats supportés : TXT, JSON, SRT, VTT, Markdown.
"""

import json
from datetime import datetime


def _ms_to_srt_time(ms: int) -> str:
    """Convertit des millisecondes en format SRT (HH:MM:SS,mmm)."""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _ms_to_vtt_time(ms: int) -> str:
    """Convertit des millisecondes en format VTT (HH:MM:SS.mmm)."""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _format_duration(seconds: float) -> str:
    """Formate une durée en secondes en HH:MM:SS."""
    if not seconds:
        return "00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def export_txt(transcription: dict, segments: list) -> str:
    """Export en texte brut avec métadonnées en header."""
    lines = []
    lines.append(f"Transcription : {transcription.get('filename', 'audio')}")
    lines.append(f"Date : {transcription.get('created_at', '')}")
    lines.append(f"Langue : {transcription.get('language', 'N/A')}")
    lines.append(f"Duree : {_format_duration(transcription.get('duration_sec'))}")
    if transcription.get('model_name'):
        lines.append(f"Modele : {transcription['model_name']}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    full_text = " ".join(seg["text"] for seg in segments if seg.get("text"))
    lines.append(full_text)
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Segments horodates :")
    lines.append("")

    for seg in segments:
        start = _format_duration(seg["start_ms"] / 1000)
        end = _format_duration(seg["end_ms"] / 1000)
        lines.append(f"[{start} -> {end}] {seg['text']}")

    return "\n".join(lines)


def export_json(transcription: dict, segments: list) -> str:
    """Export JSON complet."""
    data = {
        "transcription": {
            "id": transcription.get("id"),
            "filename": transcription.get("filename"),
            "duration_sec": transcription.get("duration_sec"),
            "language": transcription.get("language"),
            "language_probability": transcription.get("language_detected"),
            "model": transcription.get("model_name"),
            "device": transcription.get("device"),
            "compute_type": transcription.get("compute_type"),
            "created_at": transcription.get("created_at"),
            "processing_ms": transcription.get("processing_ms"),
            "word_count": transcription.get("word_count"),
            "quality_score": transcription.get("quality_score"),
            "notes": transcription.get("notes"),
        },
        "text": " ".join(seg["text"] for seg in segments if seg.get("text")),
        "segments": [
            {
                "index": seg.get("idx", i),
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg["text"],
            }
            for i, seg in enumerate(segments)
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_srt(segments: list) -> str:
    """Export au format SubRip (SRT)."""
    lines = []
    for i, seg in enumerate(segments):
        lines.append(str(i + 1))
        start = _ms_to_srt_time(seg["start_ms"])
        end = _ms_to_srt_time(seg["end_ms"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def export_vtt(segments: list) -> str:
    """Export au format WebVTT."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments):
        start = _ms_to_vtt_time(seg["start_ms"])
        end = _ms_to_vtt_time(seg["end_ms"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def export_md(transcription: dict, segments: list) -> str:
    """Export Markdown avec tableau de métadonnées + texte + segments."""
    lines = []
    lines.append(f"# Transcription : {transcription.get('filename', 'audio')}")
    lines.append("")

    lines.append("| Propriete | Valeur |")
    lines.append("|-----------|--------|")
    lines.append(f"| Fichier | {transcription.get('filename', 'N/A')} |")
    lines.append(f"| Date | {transcription.get('created_at', 'N/A')} |")
    lines.append(f"| Langue | {transcription.get('language', 'N/A')} |")
    lines.append(f"| Duree | {_format_duration(transcription.get('duration_sec'))} |")
    if transcription.get('model_name'):
        lines.append(f"| Modele | {transcription['model_name']} |")
    if transcription.get('device'):
        lines.append(f"| Device | {transcription['device']} |")
    if transcription.get('processing_ms'):
        lines.append(f"| Temps traitement | {transcription['processing_ms']}ms |")
    if transcription.get('word_count'):
        lines.append(f"| Nombre de mots | {transcription['word_count']} |")
    if transcription.get('quality_score'):
        lines.append(f"| Score qualite | {transcription['quality_score']}/5 |")
    if transcription.get('notes'):
        lines.append(f"| Notes | {transcription['notes']} |")

    lines.append("")
    lines.append("## Texte complet")
    lines.append("")
    full_text = " ".join(seg["text"] for seg in segments if seg.get("text"))
    lines.append(full_text)
    lines.append("")
    lines.append("## Segments horodates")
    lines.append("")

    for seg in segments:
        start = _format_duration(seg["start_ms"] / 1000)
        end = _format_duration(seg["end_ms"] / 1000)
        lines.append(f"**[{start} -> {end}]** {seg['text']}")
        lines.append("")

    return "\n".join(lines)
