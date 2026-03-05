"""
Serveur de transcription audio → texte basé sur Faster-Whisper.
Inspiré de whisper-asr-webservice (https://github.com/ahmetoner/whisper-asr-webservice)
mais simplifié pour un usage mono-endpoint avec UI intégrée.
"""

import json
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

# --- Configuration via variables d'environnement ---
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto", "cpu", "cuda"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # "auto", "float16", "int8", "float32"
MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "./models")

model: WhisperModel = None
model_ready = False


def detect_device():
    """Détecte automatiquement si CUDA est disponible, sinon fallback CPU."""
    if DEVICE != "auto":
        return DEVICE, COMPUTE_TYPE if COMPUTE_TYPE != "auto" else "float32"
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            print("GPU CUDA détecté — utilisation du GPU")
            return "cuda", "float16" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
    except Exception:
        pass
    print("Pas de GPU CUDA disponible — utilisation du CPU")
    return "cpu", "int8" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle Whisper au démarrage."""
    global model, model_ready
    device, compute_type = detect_device()
    print(f"")
    print(f"{'='*60}")
    print(f"  Téléchargement/chargement du modèle '{MODEL_SIZE}'")
    print(f"  Device: {device} | Compute: {compute_type}")
    print(f"  Cela peut prendre plusieurs minutes au 1er lancement...")
    print(f"{'='*60}")
    print(f"")
    start = time.time()
    model = WhisperModel(
        MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        download_root=MODEL_DIR,
    )
    elapsed = time.time() - start
    model_ready = True
    print(f"")
    print(f"{'='*60}")
    print(f"  MODÈLE PRÊT ! (chargé en {elapsed:.1f}s)")
    print(f"  Ouvrez http://localhost:8000/ pour transcrire")
    print(f"{'='*60}")
    print(f"")
    yield


app = FastAPI(title="Transcription Audio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Sert la page HTML d'upload."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    """Endpoint pour vérifier si le modèle est prêt."""
    return {"ready": model_ready, "model": MODEL_SIZE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("fr"),
):
    """
    Transcrit un fichier audio avec streaming SSE de la progression.
    Envoie des events SSE au fur et à mesure puis le résultat final.
    """
    if not model_ready:
        raise HTTPException(503, "Le modèle est encore en cours de chargement.")

    if not file.filename:
        raise HTTPException(400, "Aucun fichier fourni.")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def generate():
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

                # Progression basée sur la position dans l'audio
                progress = min(round(seg.end / duration * 100, 1), 99.9)
                event = {
                    "type": "progress",
                    "progress": progress,
                    "segment": seg_data,
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Résultat final
            result = {
                "type": "result",
                "progress": 100,
                "text": " ".join(full_text_parts),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": segments,
            }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except Exception as e:
            error = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
