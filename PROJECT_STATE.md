# PROJECT STATE — Audio-to-Knowledge

## Description

Application locale et open source de transformation audio en knowledge.
Pipeline : audio -> pré-traitement VAD -> transcription Whisper -> (futur: analyse LLM -> génération multi-formats).

## Architecture actuelle

```
speechtotext/
├── main.py                              — Point d'entrée FastAPI (minimal, wiring uniquement)
├── backend/
│   ├── config.py                        — Configuration centralisée (env vars)
│   ├── database.py                      — SQLite async/sync (transcriptions + segments + FTS5)
│   ├── audio_processing/
│   │   └── vad.py                       — Pré-traitement Silero VAD (détection parole/silence)
│   ├── transcription/
│   │   └── whisper_service.py           — Chargement modèle + transcription faster-whisper
│   ├── llm_processing/                  — (vide, préparé pour Phase 2)
│   ├── outputs/
│   │   └── exports.py                   — Export TXT, JSON, SRT, VTT, Markdown
│   └── api/
│       ├── deps.py                      — Auth (API key)
│       └── routes/
│           ├── transcription.py         — POST /transcribe, GET/PATCH /api/transcriptions/*
│           └── dashboard.py             — GET /, /health, /api/stats, /api/search
├── static/index.html                    — Frontend SPA (dashboard, upload, historique)
├── requirements.txt                     — deps Python
├── Dockerfile / Dockerfile.cpu          — Images Docker GPU / CPU
├── docker-compose.yml / .gpu.yml        — Déploiement Docker
└── data/                                — SQLite DB + fichiers audio
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Pré-traitement audio | Silero VAD (torch) |
| Transcription | faster-whisper (large-v3) |
| Backend | FastAPI + uvicorn |
| Base de données | SQLite (aiosqlite) + FTS5 |
| Audio I/O | PyAV (embarqué avec faster-whisper) |
| Frontend | HTML/CSS/JS vanilla (SPA) |
| Déploiement | Docker (CPU + GPU) |

## Features implémentées

- [x] **Architecture modulaire** (backend/ avec modules séparés)
- [x] **Pré-traitement VAD** (Silero VAD) — détection parole/silence, stats envoyées en SSE
- [x] Transcription audio via faster-whisper (modèles tiny/medium/large-v3)
- [x] Streaming SSE de la progression en temps réel (events: vad → progress → result)
- [x] Détection automatique GPU/CPU
- [x] Dashboard avec statistiques
- [x] Historique des transcriptions avec pagination et filtres
- [x] Recherche full-text (FTS5) dans les segments
- [x] Édition inline des segments
- [x] Exports multi-formats (TXT, JSON, SRT, VTT, Markdown)
- [x] Stockage audio optionnel
- [x] Auth par API key (optionnelle)
- [x] Stockage stats VAD en DB (colonne vad_stats JSON)
- [x] Déploiement Docker (CPU + GPU)

## Feature en cours

Aucune — Feature 1 terminée. Prêt pour Feature 2.

## Roadmap

### PHASE 1 — Amélioration transcription
- [x] Feature 1 : Restructuration modulaire + pré-traitement audio avec silero-vad
- [ ] Feature 2 : Segmentation propre + timestamps
- [ ] Feature 3 : Stockage structuré du transcript

### PHASE 2 — Analyse texte
- [ ] Feature 4 : Résumé automatique (LLM local)
- [ ] Feature 5 : Extraction de points clés
- [ ] Feature 6 : Extraction d'actions

### PHASE 3 — Outputs éducatifs
- [ ] Feature 7 : Fiches d'apprentissage
- [ ] Feature 8 : Quiz automatique

### PHASE 4 — Visualisations
- [ ] Feature 9 : Carte mentale (mermaid/markmap)
- [ ] Feature 10 : Slides (marp)
- [ ] Feature 11 : Infographie (vega-lite)

### PHASE 5 — Data
- [ ] Feature 12 : Extraction de tableaux de données

## Décisions techniques

| Date | Décision | Raison |
|------|----------|--------|
| 2024-03-04 | faster-whisper comme moteur | Performance et compatibilité CPU/GPU |
| 2024-03-04 | SQLite + FTS5 | Simple, local, recherche full-text native |
| 2024-03-04 | SSE pour le streaming | Progression temps réel sans WebSocket |
| 2024-03-04 | Frontend vanilla (pas de framework) | Simplicité, pas de build step |
| 2026-03-05 | Architecture modulaire backend/ | Séparation audio_processing/transcription/llm/outputs/api |
| 2026-03-05 | Silero VAD via torch.hub + PyAV | VAD léger, PyAV déjà dispo via faster-whisper (pas de ffmpeg CLI) |
| 2026-03-05 | VAD stats en JSON dans DB | Flexible, évite les migrations de colonnes multiples |

## Déploiement

- **Local** : Windows, venv Python, `uvicorn main:app`
- **Serveur** : root@77.37.51.185, Docker, `/opt/whisper-stt/`
- **Workflow** : dev local -> commit/push GitHub -> deploy via SSH/Docker
