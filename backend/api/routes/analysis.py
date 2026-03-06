"""Routes pour les analyses LLM (résumé, points clés, etc.)."""

import json
import time

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from backend import config
from backend.api.deps import verify_api_key
from backend.database import (
    get_connection, get_transcription_with_segments,
    get_analysis, get_analyses, save_analysis_sync,
)
from backend.llm_processing import ollama_client
from backend.llm_processing.summarizer import summarize_stream
from backend.llm_processing.key_points import extract_key_points_stream
from backend.llm_processing.actions import extract_actions_stream
from backend.llm_processing.study_cards import generate_study_cards_stream
from backend.llm_processing.quiz import generate_quiz_stream
from backend.llm_processing.mindmap import generate_mindmap_stream
from backend.llm_processing.slides import generate_slides_stream
from backend.llm_processing.infographic import generate_infographic_stream

router = APIRouter()


@router.get("/api/llm/status")
async def llm_status():
    """Vérifie si le LLM (Ollama) est disponible."""
    available = ollama_client.is_available()
    models = ollama_client.list_models() if available else []
    return {
        "available": available,
        "models": models,
        "current_model": config.LLM_MODEL,
    }


@router.post("/api/transcriptions/{tid}/summarize", dependencies=[Depends(verify_api_key)])
async def summarize(tid: int, model: str = Query(None)):
    """Génère un résumé via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Vérifiez qu'Ollama est lancé.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvée.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment à résumer.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Génération du résumé...'}, ensure_ascii=False)}\n\n"

            for chunk in summarize_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    # Sauvegarder en DB
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "summary",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/extract-key-points", dependencies=[Depends(verify_api_key)])
async def extract_key_points(tid: int, model: str = Query(None)):
    """Extrait les points cles via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extraction des points cles...'}, ensure_ascii=False)}\n\n"

            for chunk in extract_key_points_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "key_points",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/extract-actions", dependencies=[Depends(verify_api_key)])
async def extract_actions(tid: int, model: str = Query(None)):
    """Extrait les actions via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extraction des actions...'}, ensure_ascii=False)}\n\n"

            for chunk in extract_actions_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "actions",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/generate-study-cards", dependencies=[Depends(verify_api_key)])
async def study_cards(tid: int, model: str = Query(None)):
    """Genere des fiches d'apprentissage via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generation des fiches...'}, ensure_ascii=False)}\n\n"

            for chunk in generate_study_cards_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "study_cards",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/generate-quiz", dependencies=[Depends(verify_api_key)])
async def quiz(tid: int, model: str = Query(None)):
    """Genere un quiz via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generation du quiz...'}, ensure_ascii=False)}\n\n"

            for chunk in generate_quiz_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "quiz",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/generate-mindmap", dependencies=[Depends(verify_api_key)])
async def mindmap(tid: int, model: str = Query(None)):
    """Genere une carte mentale via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generation de la carte mentale...'}, ensure_ascii=False)}\n\n"

            for chunk in generate_mindmap_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "mindmap",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/generate-slides", dependencies=[Depends(verify_api_key)])
async def slides(tid: int, model: str = Query(None)):
    """Genere des slides via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generation des slides...'}, ensure_ascii=False)}\n\n"

            for chunk in generate_slides_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "slides",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/transcriptions/{tid}/generate-infographic", dependencies=[Depends(verify_api_key)])
async def infographic(tid: int, model: str = Query(None)):
    """Genere une infographie via LLM en streaming SSE."""
    if not ollama_client.is_available():
        raise HTTPException(503, "LLM non disponible. Verifiez qu'Ollama est lance.")

    db = await get_connection(config.DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
    finally:
        await db.close()

    segments = t.get("segments", [])
    if not segments:
        raise HTTPException(400, "Aucun segment a analyser.")

    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text.strip():
        raise HTTPException(400, "Le texte de la transcription est vide.")

    filename = t.get("filename", "audio")
    use_model = model or config.LLM_MODEL

    def generate():
        start_time = time.time()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generation de l infographie...'}, ensure_ascii=False)}\n\n"

            for chunk in generate_infographic_stream(full_text, filename=filename, model=use_model):
                if chunk["done"]:
                    processing_ms = int((time.time() - start_time) * 1000)
                    analysis_id = save_analysis_sync(
                        config.DB_PATH, tid, "infographic",
                        chunk["full_text"], use_model, processing_ms,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id, 'processing_ms': processing_ms}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk['token']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/api/transcriptions/{tid}/analyses", dependencies=[Depends(verify_api_key)])
async def list_analyses(tid: int):
    """Liste toutes les analyses d'une transcription."""
    db = await get_connection(config.DB_PATH)
    try:
        analyses = await get_analyses(db, tid)
        return {"analyses": analyses}
    finally:
        await db.close()


@router.get("/api/transcriptions/{tid}/analysis/{analysis_type}", dependencies=[Depends(verify_api_key)])
async def get_analysis_by_type(tid: int, analysis_type: str):
    """Récupère la dernière analyse d'un type donné."""
    db = await get_connection(config.DB_PATH)
    try:
        analysis = await get_analysis(db, tid, analysis_type)
        if not analysis:
            raise HTTPException(404, f"Aucune analyse de type '{analysis_type}' trouvée.")
        return analysis
    finally:
        await db.close()
