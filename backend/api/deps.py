"""Dépendances partagées pour les routes API."""

from fastapi import HTTPException, Request

from backend.config import API_KEY


async def verify_api_key(request: Request):
    """Vérifie la clé API si configurée."""
    if not API_KEY:
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(401, "Clé API invalide ou manquante.")
