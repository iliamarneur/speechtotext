"""
Serveur de transcription audio -> texte base sur Faster-Whisper.
Inspire de whisper-asr-webservice (https://github.com/ahmetoner/whisper-asr-webservice)
mais simplifie pour un usage mono-endpoint avec UI integree.

Dashboard avec historique, recherche, exports multi-formats.
"""

import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

from database import (
    init_db, get_connection, create_transcription,
    get_transcription, get_transcription_with_segments,
    list_transcriptions, update_transcription_meta, update_segment_text,
    search, get_dashboard_stats, save_result_sync, mark_error_sync,
)
from exports import export_txt, export_json, export_srt, export_vtt, export_md

# --- Configuration via variables d'environnement ---
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "auto")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "./models")
DB_PATH = os.getenv("DB_PATH", "./data/transcriptions.db")
STORE_AUDIO = os.getenv("STORE_AUDIO", "false").lower() in ("true", "1", "yes")
AUDIO_DIR = os.getenv("AUDIO_DIR", "./data/audio")
API_KEY = os.getenv("API_KEY", "")

model: WhisperModel = None
model_ready = False
device_used = ""
compute_used = ""


def detect_device():
    """Detecte automatiquement si CUDA est disponible, sinon fallback CPU."""
    if DEVICE != "auto":
        return DEVICE, COMPUTE_TYPE if COMPUTE_TYPE != "auto" else "float32"
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            print("GPU CUDA detecte -- utilisation du GPU")
            return "cuda", "float16" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
    except Exception:
        pass
    print("Pas de GPU CUDA disponible -- utilisation du CPU")
    return "cpu", "int8" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele Whisper au demarrage et initialise la DB."""
    global model, model_ready, device_used, compute_used

    # Init DB
    await init_db(DB_PATH)
    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

    device_used, compute_used = detect_device()
    print(f"")
    print(f"{'='*60}")
    print(f"  Telechargement/chargement du modele '{MODEL_SIZE}'")
    print(f"  Device: {device_used} | Compute: {compute_used}")
    print(f"  Cela peut prendre plusieurs minutes au 1er lancement...")
    print(f"{'='*60}")
    print(f"")
    start = time.time()
    model = WhisperModel(
        MODEL_SIZE,
        device=device_used,
        compute_type=compute_used,
        download_root=MODEL_DIR,
    )
    elapsed = time.time() - start
    model_ready = True
    print(f"")
    print(f"{'='*60}")
    print(f"  MODELE PRET ! (charge en {elapsed:.1f}s)")
    print(f"  Ouvrez http://localhost:8000/ pour transcrire")
    print(f"{'='*60}")
    print(f"")
    yield


app = FastAPI(title="Transcription Audio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Auth dependency ---

async def verify_api_key(request: Request):
    """Verifie la cle API si configuree."""
    if not API_KEY:
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(401, "Cle API invalide ou manquante.")


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def index():
    """Sert la page HTML du dashboard."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """Endpoint pour verifier si le modele est pret."""
    return {"ready": model_ready, "model": MODEL_SIZE}


# --- Transcription SSE ---

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("fr"),
):
    """
    Transcrit un fichier audio avec streaming SSE de la progression.
    Envoie des events SSE au fur et a mesure puis le resultat final.
    """
    if not model_ready:
        raise HTTPException(503, "Le modele est encore en cours de chargement.")

    if not file.filename:
        raise HTTPException(400, "Aucun fichier fourni.")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Create DB entry
    db = await get_connection(DB_PATH)
    try:
        tid = await create_transcription(db, file.filename, MODEL_SIZE, device_used, compute_used)
    finally:
        await db.close()

    # Store audio if configured
    audio_path = None
    if STORE_AUDIO:
        audio_dest = os.path.join(AUDIO_DIR, f"{tid}{suffix}")
        shutil.copy2(tmp_path, audio_dest)
        audio_path = audio_dest

    def generate():
        start_time = time.time()
        try:
            lang = language.strip() or None
            segments_gen, info = model.transcribe(
                tmp_path,
                language=lang,
                beam_size=5,
                vad_filter=True,
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

            # Save to DB
            db_segments = [
                {"start_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000), "text": s["text"]}
                for s in segments
            ]
            save_result_sync(
                DB_PATH, tid, info.duration, info.language,
                round(info.language_probability, 3), word_count,
                processing_ms, db_segments, audio_path
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

        except Exception as e:
            mark_error_sync(DB_PATH, tid, str(e))
            error = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- API Endpoints ---

@app.get("/api/transcriptions", dependencies=[Depends(verify_api_key)])
async def api_list_transcriptions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str = Query(None),
    period: str = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
):
    """Liste paginee des transcriptions."""
    db = await get_connection(DB_PATH)
    try:
        return await list_transcriptions(db, page, per_page, status, period, sort, order)
    finally:
        await db.close()


@app.get("/api/transcriptions/{tid}", dependencies=[Depends(verify_api_key)])
async def api_get_transcription(tid: int):
    """Detail d'une transcription + segments."""
    db = await get_connection(DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
        return t
    finally:
        await db.close()


@app.patch("/api/transcriptions/{tid}", dependencies=[Depends(verify_api_key)])
async def api_update_transcription(tid: int, request: Request):
    """Met a jour quality_score et/ou notes."""
    body = await request.json()
    db = await get_connection(DB_PATH)
    try:
        t = await get_transcription(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
        await update_transcription_meta(
            db, tid,
            quality_score=body.get("quality_score"),
            notes=body.get("notes"),
        )
        return {"ok": True}
    finally:
        await db.close()


@app.patch("/api/transcriptions/{tid}/segments/{segment_id}", dependencies=[Depends(verify_api_key)])
async def api_update_segment(tid: int, segment_id: int, request: Request):
    """Edite le texte d'un segment."""
    body = await request.json()
    text = body.get("text")
    if text is None:
        raise HTTPException(400, "Le champ 'text' est requis.")
    db = await get_connection(DB_PATH)
    try:
        await update_segment_text(db, segment_id, text)
        return {"ok": True}
    finally:
        await db.close()


@app.get("/api/transcriptions/{tid}/export", dependencies=[Depends(verify_api_key)])
async def api_export(tid: int, format: str = Query("txt")):
    """Exporte une transcription dans le format demande."""
    db = await get_connection(DB_PATH)
    try:
        t = await get_transcription_with_segments(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")

        segments = t.pop("segments", [])
        fmt = format.lower()
        filename_base = os.path.splitext(t.get("filename", "transcription"))[0]

        if fmt == "txt":
            content = export_txt(t, segments)
            return Response(content, media_type="text/plain",
                          headers={"Content-Disposition": f'attachment; filename="{filename_base}.txt"'})
        elif fmt == "json":
            content = export_json(t, segments)
            return Response(content, media_type="application/json",
                          headers={"Content-Disposition": f'attachment; filename="{filename_base}.json"'})
        elif fmt == "srt":
            content = export_srt(segments)
            return Response(content, media_type="text/plain",
                          headers={"Content-Disposition": f'attachment; filename="{filename_base}.srt"'})
        elif fmt == "vtt":
            content = export_vtt(segments)
            return Response(content, media_type="text/vtt",
                          headers={"Content-Disposition": f'attachment; filename="{filename_base}.vtt"'})
        elif fmt == "md":
            content = export_md(t, segments)
            return Response(content, media_type="text/markdown",
                          headers={"Content-Disposition": f'attachment; filename="{filename_base}.md"'})
        else:
            raise HTTPException(400, f"Format '{fmt}' non supporte. Formats: txt, json, srt, vtt, md")
    finally:
        await db.close()


@app.get("/api/search", dependencies=[Depends(verify_api_key)])
async def api_search(q: str = Query(..., min_length=1)):
    """Recherche FTS5 dans les segments + LIKE filename."""
    db = await get_connection(DB_PATH)
    try:
        results = await search(db, q)
        return {"results": results, "query": q}
    finally:
        await db.close()


@app.get("/api/stats", dependencies=[Depends(verify_api_key)])
async def api_stats(period: str = Query("30d")):
    """Stats agregees pour le dashboard."""
    db = await get_connection(DB_PATH)
    try:
        return await get_dashboard_stats(db, period)
    finally:
        await db.close()


@app.get("/api/audio/{tid}", dependencies=[Depends(verify_api_key)])
async def api_audio(tid: int):
    """Sert le fichier audio stocke."""
    db = await get_connection(DB_PATH)
    try:
        t = await get_transcription(db, tid)
        if not t:
            raise HTTPException(404, "Transcription non trouvee.")
        audio_path = t.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(404, "Fichier audio non disponible.")
        return FileResponse(audio_path)
    finally:
        await db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
