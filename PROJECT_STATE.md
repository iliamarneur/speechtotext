# PROJECT STATE — Audio-to-Knowledge

## Description

Application locale et open source de transformation audio en knowledge.
Pipeline : audio -> pré-traitement VAD -> transcription Whisper -> analyse LLM -> génération multi-formats.

## Architecture actuelle

```
speechtotext/
├── main.py                              — Point d'entrée FastAPI (minimal, wiring uniquement)
├── backend/
│   ├── config.py                        — Configuration centralisée (env vars)
│   ├── database.py                      — SQLite async/sync (transcriptions + segments + analyses + FTS5)
│   ├── audio_processing/
│   │   └── vad.py                       — Pré-traitement Silero VAD
│   ├── transcription/
│   │   └── whisper_service.py           — Service faster-whisper
│   ├── llm_processing/
│   │   ├── ollama_client.py             — Client HTTP Ollama (streaming)
│   │   ├── summarizer.py               — Résumé automatique via LLM
│   │   ├── key_points.py              — Extraction de points clés via LLM
│   │   ├── actions.py                 — Extraction d'actions via LLM
│   │   └── study_cards.py            — Fiches d'apprentissage via LLM
│   ├── outputs/
│   │   └── exports.py                   — Export TXT, JSON, SRT, VTT, Markdown
│   └── api/
│       ├── deps.py                      — Auth (API key)
│       └── routes/
│           ├── transcription.py         — POST /transcribe, CRUD transcriptions
│           ├── dashboard.py             — /, /health, /api/stats, /api/search
│           └── analysis.py              — /api/llm/status, summarize, analyses CRUD
├── static/index.html                    — Frontend SPA
├── requirements.txt                     — deps Python
├── Dockerfile / Dockerfile.cpu          — Images Docker
├── docker-compose.yml / .gpu.yml        — Déploiement Docker
└── data/                                — SQLite DB + fichiers audio
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Pré-traitement audio | Silero VAD (torch) |
| Transcription | faster-whisper (large-v3) |
| LLM / Analyse | Ollama + mistral-nemo (12B) |
| Backend | FastAPI + uvicorn |
| Base de données | SQLite (aiosqlite) + FTS5 |
| Audio I/O | PyAV (embarqué avec faster-whisper) |
| Frontend | HTML/CSS/JS vanilla (SPA) |
| Déploiement | Docker (CPU + GPU) |

## Features implémentées

- [x] **Architecture modulaire** (backend/ avec modules séparés)
- [x] **Pré-traitement VAD** (Silero VAD) — détection parole/silence, stats SSE + DB
- [x] **Résumé automatique** (Ollama/mistral-nemo) — streaming SSE, sauvegarde DB, UI intégrée
- [x] **Extraction de points clés** — points thématiques structurés, streaming SSE, UI intégrée
- [x] **Extraction d'actions** — tâches, décisions, questions en suspens, streaming SSE, UI intégrée
- [x] **Fiches d'apprentissage** — fiches structurées (définition, exemple, à retenir), streaming SSE, UI intégrée
- [x] **Quiz automatique** — QCM généré par LLM, streaming SSE, UI intégrée
- [x] **Carte mentale** — Markmap (d3 + markmap-view), rendu SVG, streaming SSE, UI intégrée
- [x] **Slides** — Présentation Markdown avec viewer interactif, streaming SSE, UI intégrée
- [x] **Infographie** — Graphique Vega-Lite généré par LLM, rendu SVG, streaming SSE, UI intégrée
- [x] Transcription audio via faster-whisper (modèles tiny/medium/large-v3)
- [x] Streaming SSE (vad → progress → result | token → done)
- [x] Détection automatique GPU/CPU
- [x] Dashboard avec statistiques
- [x] Historique des transcriptions avec pagination et filtres
- [x] Recherche full-text (FTS5)
- [x] Édition inline des segments
- [x] Exports multi-formats (TXT, JSON, SRT, VTT, Markdown)
- [x] Stockage audio optionnel
- [x] Auth par API key (optionnelle)
- [x] Upload par chunks (gros fichiers)
- [x] Déploiement Docker (CPU + GPU)

## Feature en cours

Feature 12 — Extraction de tableaux de données.

## Roadmap

### PHASE 1 — Amélioration transcription
- [x] Feature 1 : Restructuration modulaire + pré-traitement audio avec silero-vad
- [ ] ~~Feature 2~~ : Segmentation (déjà fonctionnelle via faster-whisper)
- [ ] ~~Feature 3~~ : Stockage structuré (déjà en place dans DB)

### PHASE 2 — Analyse texte
- [x] Feature 4 : Résumé automatique (LLM local via Ollama)
- [x] Feature 5 : Extraction de points clés
- [x] Feature 6 : Extraction d'actions

### PHASE 3 — Outputs éducatifs
- [x] Feature 7 : Fiches d'apprentissage
- [x] Feature 8 : Quiz automatique

### PHASE 4 — Visualisations
- [x] Feature 9 : Carte mentale (markmap)
- [x] Feature 10 : Slides (viewer interactif)
- [x] Feature 11 : Infographie (vega-lite)

### PHASE 5 — Data
- [ ] Feature 12 : Extraction de tableaux de données (en cours)

## Décisions techniques

| Date | Décision | Raison |
|------|----------|--------|
| 2024-03-04 | faster-whisper comme moteur | Performance et compatibilité CPU/GPU |
| 2024-03-04 | SQLite + FTS5 | Simple, local, recherche full-text native |
| 2024-03-04 | SSE pour le streaming | Progression temps réel sans WebSocket |
| 2024-03-04 | Frontend vanilla (pas de framework) | Simplicité, pas de build step |
| 2026-03-05 | Architecture modulaire backend/ | Séparation claire des responsabilités |
| 2026-03-05 | Silero VAD via torch.hub + PyAV | VAD léger, PyAV déjà dispo |
| 2026-03-05 | Ollama comme runtime LLM | Local, simple, supporte tous les modèles GGUF |
| 2026-03-05 | Table `analyses` générique | type + content = réutilisable pour résumé, points clés, quiz, etc. |
| 2026-03-05 | LLM optionnel (graceful degradation) | Bouton grisé si Ollama non dispo (serveur sans LLM) |

## Déploiement

- **Local** : Windows, venv Python, Ollama pour le LLM
- **Serveur** : root@77.37.51.185, Docker, `/opt/whisper-stt/` (pas de LLM pour l'instant)
- **Workflow** : dev local -> commit/push GitHub -> deploy via SSH/Docker
