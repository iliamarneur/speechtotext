"""
Point d'entrée FastAPI — Audio-to-Knowledge.
Charge les modèles au démarrage et monte les routes.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.database import init_db
from backend.transcription import whisper_service
from backend.audio_processing import vad
from backend.api.routes.transcription import router as transcription_router
from backend.api.routes.dashboard import router as dashboard_router
from backend.api.routes.analysis import router as analysis_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise la DB et charge les modèles au démarrage."""
    # Init DB et dossiers
    await init_db(config.DB_PATH)
    Path(config.AUDIO_DIR).mkdir(parents=True, exist_ok=True)

    # Charger le modèle VAD (léger, ~1s)
    if config.ENABLE_VAD:
        vad.load_model()

    # Charger le modèle Whisper (peut prendre plusieurs minutes)
    whisper_service.load_model()

    print(f"\n  Ouvrez http://localhost:8000/ pour transcrire\n")
    yield


app = FastAPI(title="Audio-to-Knowledge", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(dashboard_router)
app.include_router(transcription_router)
app.include_router(analysis_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
