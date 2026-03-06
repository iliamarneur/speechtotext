"""Routes pour la transcription et la gestion des transcriptions."""

import json
import os
import shutil
import tempfile
import threading
import time

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse, Response, FileResponse

from backend import config
from backend.api.deps import verify_api_key
from backend.database import (
    get_connection, get_sync_connection, create_transcription, get_transcription,
    get_transcription_with_segments, list_transcriptions,
    update_transcription_meta, update_segment_text,
    save_result_sync, mark_error_sync, save_analysis_sync,
)
from backend.transcription import whisper_service
from backend.audio_processing import vad
from backend.audio_processing import diarization
from backend.outputs.exports import export_txt, export_json, export_srt, export_vtt, export_md
from backend.llm_processing import ollama_client
from backend.llm_processing.summarizer import summarize_stream
from backend.llm_processing.key_points import extract_key_points_stream
from backend.llm_processing.actions import extract_actions_stream
from backend.llm_processing.study_cards import generate_study_cards_stream
from backend.llm_processing.quiz import generate_quiz_stream
from backend.llm_processing.mindmap import generate_mindmap_stream
from backend.llm_processing.slides import generate_slides_stream
from backend.llm_processing.infographic import generate_infographic_stream
from backend.llm_processing.data_table import extract_data_table_stream

router = APIRouter()


def _cleanup_old_audio():
    """Supprime les fichiers audio les plus anciens pour ne garder que AUDIO_MAX_FILES."""
    try:
        audio_dir = config.AUDIO_DIR
        if not os.path.isdir(audio_dir):
            return
        files = []
        for f in os.listdir(audio_dir):
            fp = os.path.join(audio_dir, f)
            if os.path.isfile(fp):
                files.append((os.path.getmtime(fp), fp))
        files.sort(reverse=True)  # Plus récent en premier
        to_delete = files[config.AUDIO_MAX_FILES:]
        if not to_delete:
            return
        for _, fp in to_delete:
            os.unlink(fp)
        # Nettoyer audio_path en DB pour les fichiers supprimés
        conn = get_sync_connection(config.DB_PATH)
        try:
            conn.execute("UPDATE transcriptions SET audio_path=NULL WHERE audio_path IS NOT NULL AND audio_path NOT IN (SELECT audio_path FROM transcriptions WHERE audio_path IS NOT NULL ORDER BY created_at DESC LIMIT ?)", (config.AUDIO_MAX_FILES,))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[AUDIO] Erreur nettoyage audio (ignorée): {e}")


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
        os.makedirs(config.AUDIO_DIR, exist_ok=True)
        audio_dest = os.path.join(config.AUDIO_DIR, f"{tid}{suffix}")
        shutil.copy2(tmp_path, audio_dest)
        audio_path = audio_dest
        # Nettoyage : garder seulement les N derniers fichiers
        _cleanup_old_audio()

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

            # --- Lancer la diarisation en parallèle (analyse audio, pas texte) ---
            diar_thread = None
            diar_result_holder = {}
            if diarization.is_available():
                yield f"data: {json.dumps({'type': 'diarization_start'}, ensure_ascii=False)}\n\n"
                def _run_diarization():
                    try:
                        diar_result_holder["start"] = time.time()
                        diar_result_holder["result"] = diarization.diarize(tmp_path)
                        diar_result_holder["ms"] = int((time.time() - diar_result_holder["start"]) * 1000)
                    except Exception as e:
                        diar_result_holder["error"] = str(e)
                        print(f"[DIARIZATION] Erreur diarisation (ignoree): {e}")
                diar_thread = threading.Thread(target=_run_diarization, daemon=True)
                diar_thread.start()

            # --- Étape 2 : Transcription Whisper (en parallèle avec diarisation) ---
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

            # --- Attendre la fin de la diarisation si lancée ---
            num_speakers = None
            if diar_thread is not None:
                diar_thread.join()
                if "result" in diar_result_holder:
                    diar_result = diar_result_holder["result"]
                    segments = diarization.assign_speakers(segments, diar_result["turns"])
                    num_speakers = diar_result["num_speakers"]
                    diar_ms = diar_result_holder.get("ms", 0)
                    yield f"data: {json.dumps({'type': 'diarization_done', 'num_speakers': num_speakers, 'speakers': diar_result['speakers'], 'processing_ms': diar_ms}, ensure_ascii=False)}\n\n"

            # Sauvegarder en DB
            db_segments = [
                {"start_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000), "text": s["text"], "speaker": s.get("speaker")}
                for s in segments
            ]
            save_result_sync(
                config.DB_PATH, tid, info.duration, info.language,
                round(info.language_probability, 3), word_count,
                processing_ms, db_segments, audio_path, vad_json,
                num_speakers,
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
                "num_speakers": num_speakers,
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

                # --- Étape 6 : Analyse LLM automatique (fiches d'apprentissage) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'study_cards'}, ensure_ascii=False)}\n\n"
                    sc_start = time.time()

                    for chunk in generate_study_cards_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            sc_ms = int((time.time() - sc_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "study_cards",
                                chunk["full_text"], use_model, sc_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'study_cards', 'processing_ms': sc_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'study_cards', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur fiches apprentissage automatique (ignorée): {e}")

                # --- Étape 7 : Analyse LLM automatique (quiz) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'quiz'}, ensure_ascii=False)}\n\n"
                    qz_start = time.time()

                    for chunk in generate_quiz_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            qz_ms = int((time.time() - qz_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "quiz",
                                chunk["full_text"], use_model, qz_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'quiz', 'processing_ms': qz_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'quiz', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur quiz automatique (ignorée): {e}")

                # --- Étape 8 : Analyse LLM automatique (carte mentale) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'mindmap'}, ensure_ascii=False)}\n\n"
                    mm_start = time.time()

                    for chunk in generate_mindmap_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            mm_ms = int((time.time() - mm_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "mindmap",
                                chunk["full_text"], use_model, mm_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'mindmap', 'processing_ms': mm_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'mindmap', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur carte mentale automatique (ignorée): {e}")

                # --- Étape 9 : Analyse LLM automatique (slides) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'slides'}, ensure_ascii=False)}\n\n"
                    sl_start = time.time()

                    for chunk in generate_slides_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            sl_ms = int((time.time() - sl_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "slides",
                                chunk["full_text"], use_model, sl_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'slides', 'processing_ms': sl_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'slides', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur slides automatique (ignorée): {e}")

                # --- Étape 10 : Analyse LLM automatique (infographie) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'infographic'}, ensure_ascii=False)}\n\n"
                    ig_start = time.time()

                    for chunk in generate_infographic_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            ig_ms = int((time.time() - ig_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "infographic",
                                chunk["full_text"], use_model, ig_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'infographic', 'processing_ms': ig_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'infographic', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur infographie automatique (ignorée): {e}")

                # --- Étape 11 : Analyse LLM automatique (tableaux de données) ---
                try:
                    yield f"data: {json.dumps({'type': 'analysis_start', 'analysis': 'data_table'}, ensure_ascii=False)}\n\n"
                    dt_start = time.time()

                    for chunk in extract_data_table_stream(full_text, filename=file.filename, model=use_model):
                        if chunk["done"]:
                            dt_ms = int((time.time() - dt_start) * 1000)
                            save_analysis_sync(
                                config.DB_PATH, tid, "data_table",
                                chunk["full_text"], use_model, dt_ms,
                            )
                            yield f"data: {json.dumps({'type': 'analysis_done', 'analysis': 'data_table', 'processing_ms': dt_ms}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'analysis_token', 'analysis': 'data_table', 'token': chunk['token']}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[LLM] Erreur tableaux données automatique (ignorée): {e}")

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


@router.get("/api/diarization/status", dependencies=[Depends(verify_api_key)])
async def api_diarization_status():
    """Vérifie si la diarisation est disponible."""
    return {"available": diarization.is_available()}


@router.post("/api/transcriptions/{tid}/diarize", dependencies=[Depends(verify_api_key)])
async def api_diarize(tid: int):
    """Lance/relance la diarisation sur une transcription existante (nécessite audio stocké)."""
    if not diarization.is_available():
        raise HTTPException(503, "Diarisation non disponible (pyannote non installé ou HF_TOKEN manquant).")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvée.")
        audio_path = t.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(400, "Fichier audio non stocké. Activez STORE_AUDIO=true et retranscrivez.")
    finally:
        await db.close()

    segments = t.get("segments", [])

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'diarization_start'}, ensure_ascii=False)}\n\n"
            diar_start = time.time()
            diar_result = diarization.diarize(audio_path)
            diar_ms = int((time.time() - diar_start) * 1000)

            # Convertir segments DB en format compatible
            seg_list = [{"start": s["start_ms"] / 1000, "end": s["end_ms"] / 1000, "text": s["text"]} for s in segments]
            seg_list = diarization.assign_speakers(seg_list, diar_result["turns"])

            # Mettre à jour les speakers en DB
            conn = get_sync_connection(config.DB_PATH)
            try:
                for seg_db, seg_upd in zip(segments, seg_list):
                    conn.execute("UPDATE segments SET speaker=? WHERE id=?", (seg_upd.get("speaker"), seg_db["id"]))
                conn.execute("UPDATE transcriptions SET num_speakers=?, updated_at=datetime('now') WHERE id=?",
                             (diar_result["num_speakers"], tid))
                conn.commit()
            finally:
                conn.close()

            yield f"data: {json.dumps({'type': 'diarization_done', 'num_speakers': diar_result['num_speakers'], 'speakers': diar_result['speakers'], 'processing_ms': diar_ms}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'diarization_error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
