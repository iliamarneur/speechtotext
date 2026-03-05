# CHANGELOG

## [0.4.0] - 2026-03-05

### Added
- **Résumé automatique via LLM (Feature 4)**
  - Intégration Ollama (mistral-nemo / llama3.2 / tout modèle compatible)
  - `backend/llm_processing/ollama_client.py` — client HTTP streaming Ollama
  - `backend/llm_processing/summarizer.py` — prompt de résumé structuré (FR)
  - `backend/api/routes/analysis.py` — endpoints résumé + analyses CRUD
  - Streaming SSE des tokens en temps réel
  - Sauvegarde automatique du résumé en DB (table `analyses`)
  - UI : bouton "Résumer" dans la vue détail, affichage streaming markdown
  - Détection automatique de la disponibilité d'Ollama (bouton grisé si absent)
  - `GET /api/llm/status` — vérification statut LLM + liste des modèles
- Table `analyses` générique (type, content, model_name) — réutilisable pour futures features
- Fonction `markdownToHtml()` dans le frontend pour le rendu des résumés
- Upload par chunks (1MB) pour supporter les gros fichiers audio
- Heartbeat SSE pour maintenir la connexion pendant le traitement long
- Timeout keep-alive uvicorn augmenté à 300s

### Changed
- Page par défaut : transcription au lieu du dashboard

## [0.3.0] - 2026-03-05

### Added
- **Architecture modulaire** : restructuration complète du projet
  - `backend/config.py` — configuration centralisée
  - `backend/audio_processing/vad.py` — pré-traitement Silero VAD
  - `backend/transcription/whisper_service.py` — service Whisper isolé
  - `backend/outputs/exports.py` — exports (déplacé)
  - `backend/database.py` — database (déplacé)
  - `backend/api/routes/` — routes séparées (transcription + dashboard)
  - `backend/api/deps.py` — auth partagée
  - `backend/llm_processing/` — préparé pour Phase 2
- **Pré-traitement VAD (Feature 1)** : Silero VAD analyse l'audio avant transcription
  - Détection segments de parole / silence
  - Stats : speech_ratio, speech_duration, silence_duration, num_speech_segments
  - Événement SSE `vad` envoyé au client avant la transcription
  - Stats VAD stockées en DB (colonne `vad_stats` JSON)
- Lecteur audio via PyAV (pas de dépendance ffmpeg CLI)
- `main.py` réduit à un point d'entrée minimal (~40 lignes)
- Dépendances : torch, torchaudio ajoutés
- Dockerfile.cpu mis à jour avec torch CPU-only

### Changed
- `main.py` de monolithique (376 lignes) à point d'entrée minimal
- Titre app : "Transcription Audio" → "Audio-to-Knowledge"

### Removed
- `database.py` et `exports.py` à la racine (déplacés dans backend/)

## [0.2.0] - 2024-03-04

### Added
- Dashboard avec statistiques agrégées
- Historique paginé avec filtres (statut, période, tri)
- Recherche full-text FTS5 dans les segments et noms de fichiers
- Édition inline des segments de transcription
- Exports multi-formats : TXT, JSON, SRT, VTT, Markdown
- Module database.py (SQLite async/sync)
- Module exports.py
- Stockage audio optionnel
- Auth API key optionnelle
- Stats API endpoint

## [0.1.0] - 2024-03-04

### Added
- Transcription audio via faster-whisper
- Streaming SSE de la progression
- Détection automatique GPU/CPU
- Interface web d'upload
- Support Docker CPU et GPU
- Configuration par variables d'environnement
