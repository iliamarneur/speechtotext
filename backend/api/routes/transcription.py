"""Routes pour la transcription et la gestion des transcriptions."""

import json
import os
import shutil
import tempfile
import time

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse, Response, FileResponse

from backend import config
from backend.api.deps import verify_api_key
from backend.database import (
    get_connection, create_transcription, get_transcription,
    get_transcription_with_segments, list_transcriptions,
    update_transcription_meta, update_segment_text,
    save_result_sync, mark_error_sync, save_analysis_sync,
)
from backend.transcription import whisper_service
from backend.audio_processing import vad
from backend.outputs.exports import export_txt, export_json, export_srt, export_vtt, export_md
from backend.llm_processing import ollama_client
from backend.llm_processing.summarizer import summarize_stream
from backend.llm_processing.key_points import extract_key_points_stream
from backend.llm_processing.actions import extract_actions_stream

router = APIRouter()


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("fr"),
):
    """Transcrit un fichier audio avec streaming SSE (+ pré-traitement VAD)."""
    if not whisper_service.is_ready():
        raise HTTPException(503, "Le modèle est encore en cours de chargement.")

    if not file.filename:
        raise HTTPException(400, "Aucun fichier fourni.")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # Lecture par chunks pour supporter les gros fichiers (évite surcharge mémoire)
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            tmp.write(chunk)
        tmp_path = tmp.name

    # Créer l'entrée DB
    model_info = whisper_service.get_model_info()
    db = await get_connection(config.DB_PATH)
    try:
        tid = await create_transcription(
            db, file.filename,
            model_info["model"], model_info["device"], model_info["compute_type"],
        )
    finally:
        await db.close()

    # Stocker l'audio si configuré
    audio_path = None
    if config.STORE_AUDIO:
        audio_dest = os.path.join(config.AUDIO_DIR, f"{tid}{suffix}")
        shutil.copy2(tmp_path, audio_dest)
        audio_path = audio_dest

    def generate():
        start_time = time.time()
        vad_json = None
        try:
            # Heartbeat pour maintenir la connexion pendant le traitement
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyse audio...'}, ensure_ascii=False)}\n\n"

            # --- Étape 1 : Pré-traitement VAD ---
            if config.ENABLE_VAD and vad.is_loaded():
                try:
                    vad_result = vad.analyze(tmp_path)
                    # Envoyer les stats VAD au client (sans la liste des segments)
                    vad_summary = {
                        "type": "vad",
                        "speech_ratio": vad_result["speech_ratio"],
                        "speech_duration": vad_result["speech_duration"],
                        "silence_duration": vad_result["silence_duration"],
                        "num_speech_segments": vad_result["num_speech_segments"],
                        "total_duration": vad_result["total_duration"],
                    }
                    yield f"data: {json.dumps(vad_summary, ensure_ascii=False)}\n\n"
                    # Sérialiser pour stockage DB
                    vad_json = json.dumps(vad_result, ensure_ascii=False)
                except Exception as e:
                    print(f"[VAD] Erreur pré-traitement (ignorée): {e}")

            # --- Étape 2 : Transcription Whisper ---
            lang = language.strip() or None
            segments_gen, info = whisper_service.transcribe(
                tmp_path, language=lang,
            )
            duration = info.duration or 1.0

            segments = []
            full_text_parts = []

            for seg in segments_gen:
                seg_data = {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                }
                segments.append(seg_data)
                full_text_parts.append(seg.text.strip())

                progress = min(round(seg.end / duration * 100, 1), 99.9)
                event = {
                    "type": "progress",
                    "progress": progress,
                    "segment": seg_data,
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            processing_ms = int((time.time() - start_time) * 1000)
            full_text = " ".join(full_text_parts)
            word_count = len(full_text.split())

            # Sauvegarder en DB
            db_segments = [
                {"start_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000), "text": s["text"]}
                for s in segments
            ]
            save_result_sync(
                config.DB_PATH, tid, info.duration, info.language,
                round(info.language_probability, 3), word_count,
                processing_ms, db_segments, audio_path, vad_json,
            )

            result = {
                "type": "result",
                "progress": 100,
                "transcription_id": tid,
                "text": full_text,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": segments,
            }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

            # --- Étape 3 : Analyse LLM automatique (résumé) ---
            if full_text.strip() and ollama_client.is_available():
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'summary'}, ensure_ascii=False)}\n\n"
                    llm_start = time.time()
                    use_model = config.LLM_MODEL

                    for chunk in summarize_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            llm_ms = int((time.time() - llm_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "summary",
                                chunk["full_text"], use_model, llm_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'summary', 'processing_ms': llm_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'summary', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur résumé automatique (ignorée): {e}")

                # --- Étape 4 : Analyse LLM automatique (points clés) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'key_points'}, ensure_ascii=False)}\n\n"
                    kp_start = time.time()

                    for chunk in extract_key_points_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            kp_ms = int((time.time() - kp_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "key_points",
                                chunk["full_text"], use_model, kp_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'key_points', 'processing_ms': kp_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'key_points', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur points clés automatique (ignorée): {e}")

                # --- Étape 5 : Analyse LLM automatique (actions) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'actions'}, ensure_ascii=False)}\n\n"
                    act_start = time.time()

                    for chunk in extract_actions_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            act_ms = int((time.time() - act_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "actions",
                                chunk["full_text"], use_model, act_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'actions', 'processing_ms': act_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'actions', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur actions automatique (ignorée): {e}")

        except Exception as e:
            mark_error_sync(config.DB_PATH, tid, str(e))
            error = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/api/transcriptions", dependencies=[Depends(verify_api_key)])
async def api_list_transcriptions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str = Query(None),
    period: str = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
):
    """Liste paginée des transcriptions."""
    db = await get_connection(config.DB_PATH)
    try:
        return await list_transcriptions(db, page, per_page, status, period, sort, order)
    finally:
        await db.close()


@router.get("/api/transcriptions/{tid}", dependencies=[Depends(verify_api_key)])
async def api_get_transcription(tid: int):
    """Détail d'une transcription + segments."""
    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvée.")
        return t
    finally:
        await db.close()


@router.patch("/api/transcriptions/{tid}", dependencies=[Depends(verify_api_key)])
async def api_update_transcription(tid: int, request: Request):
    """Met à jour quality_score et/ou notes."""
    body = await request.json()
    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvée.")
        await update_transcription_meta(
            db, tid,
            quality_score=body.get("quality_score"),
            notes=body.get("notes"),
        )
        return {"ok": True}
    finally:
        await db.close()


@router.patch("/api/transcriptions/{tid}/segments/{segment_id}", dependencies=[Depends(verify_api_key)])
async def api_update_segment(tid: int, segment_id: int, request: Request):
    """Édite le texte d'un segment."""
    body = await request.json()
    text = body.get("text")
    if text is None:
        raise HTTPException(400, "Le champ 'text' est requis.")
    db = await get_connection(config.DB_PATH)
    try:
        await update_segment_text(db, segment_id, text)
        return {"ok": True}
    finally:
        await db.close()


@router.get("/api/transcriptions/{tid}/export", dependencies=[Depends(verify_api_key)])
async def api_export(tid: int, format: str = Query("txt")):
    """Exporte une transcription dans le format demandé."""
    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvée.")

        segments = t.pop("segments", [])
        fmt = format.lower()
        filename_base = os.path.splitext(t.get("filename", "transcription"))[0]

        exporters = {
            "txt": (export_txt, "text/plain", "txt"),
            "json": (export_json, "application/json", "json"),
            "srt": (export_srt, "text/plain", "srt"),
            "vtt": (export_vtt, "text/vtt", "vtt"),
            "md": (export_md, "text/markdown", "md"),
        }

        if fmt not in exporters:
            raise HTTPException(400, f"Format '{fmt}' non supporté. Formats: {', '.join(exporters)}")

        export_fn, media_type, ext = exporters[fmt]
        # srt et vtt ne prennent que segments
        if fmt in ("srt", "vtt"):
            content = export_fn(segments)
        else:
            content = export_fn(t, segments)

        return Response(
            content, media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.{ext}"'},
        )
    finally:
        await db.close()


@router.get("/api/audio/{tid}", dependencies=[Depends(verify_api_key)])
async def api_audio(tid: int):
    """Sert le fichier audio stocké."""
    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvée.")
        audio_path = t.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(404, "Fichier audio non disponible.")
        return FileResponse(audio_path)
    finally:
        await db.close()
