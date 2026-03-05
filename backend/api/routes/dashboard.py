"""Routes pour le dashboard, health check, recherche et stats."""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from backend.api.deps import verify_api_key
from backend.config import DB_PATH
from backend.database import get_connection, search, get_dashboard_stats
from backend.transcription import whisper_service

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index():
    """Sert la page HTML du dashboard."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get("/health")
async def health():
    """Endpoint pour vérifier si le modèle est prêt."""
    info = whisper_service.get_model_info()
    return {"ready": whisper_service.is_ready(), "model": info["model"]}


@router.get("/api/search", dependencies=[Depends(verify_api_key)])
async def api_search(q: str = Query(..., min_length=1)):
    """Recherche FTS5 dans les segments + LIKE filename."""
    db = await get_connection(DB_PATH)
    try:
        results = await search(db, q)
        return {"results": results, "query": q}
    finally:
        await db.close()


@router.get("/api/stats", dependencies=[Depends(verify_api_key)])
async def api_stats(period: str = Query("30d")):
    """Stats agrégées pour le dashboard."""
    db = await get_connection(DB_PATH)
    try:
        return await get_dashboard_stats(db, period)
    finally:
        await db.close()
