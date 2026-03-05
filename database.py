"""
Module de gestion de la base de données SQLite pour l'historique des transcriptions.
Utilise aiosqlite pour les opérations async et sqlite3 pour les opérations sync
(nécessaires dans le générateur SSE).
"""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

DB_PATH = "./data/transcriptions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    duration_sec REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    processing_ms INTEGER,
    language TEXT,
    language_detected REAL,
    word_count INTEGER,
    model_name TEXT,
    device TEXT,
    compute_type TEXT,
    audio_path TEXT,
    quality_score INTEGER,
    notes TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE INDEX IF NOT EXISTS idx_segments_transcription ON segments(transcription_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_status ON transcriptions(status);
CREATE INDEX IF NOT EXISTS idx_transcriptions_created ON transcriptions(created_at);
"""


async def init_db(db_path: str = None):
    """Initialise la base de données et crée les tables."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def get_connection(db_path: str = None):
    """Retourne une connexion async à la DB."""
    path = db_path or DB_PATH
    db = await aiosqlite.connect(path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


def get_sync_connection(db_path: str = None):
    """Retourne une connexion sync à la DB (pour le générateur SSE)."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# --- Helpers async ---

async def create_transcription(db, filename: str, model_name: str = None,
                                device: str = None, compute_type: str = None) -> int:
    """Crée une entrée de transcription avec status=processing."""
    cursor = await db.execute(
        """INSERT INTO transcriptions (filename, status, model_name, device, compute_type)
           VALUES (?, 'processing', ?, ?, ?)""",
        (filename, model_name, device, compute_type)
    )
    await db.commit()
    return cursor.lastrowid


async def update_transcription_status(db, tid: int, status: str, **kwargs):
    """Met à jour le statut et d'autres champs d'une transcription."""
    fields = ["status=?", "updated_at=datetime('now')"]
    values = [status]
    for key, val in kwargs.items():
        fields.append(f"{key}=?")
        values.append(val)
    values.append(tid)
    await db.execute(
        f"UPDATE transcriptions SET {', '.join(fields)} WHERE id=?",
        values
    )
    await db.commit()


async def insert_segments(db, tid: int, segments: list):
    """Insère les segments d'une transcription."""
    await db.executemany(
        "INSERT INTO segments (transcription_id, idx, start_ms, end_ms, text) VALUES (?, ?, ?, ?, ?)",
        [(tid, i, seg["start_ms"], seg["end_ms"], seg["text"]) for i, seg in enumerate(segments)]
    )
    await db.commit()


async def get_transcription(db, tid: int) -> dict | None:
    """Récupère une transcription par ID."""
    cursor = await db.execute("SELECT * FROM transcriptions WHERE id=?", (tid,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def get_transcription_with_segments(db, tid: int) -> dict | None:
    """Récupère une transcription avec ses segments."""
    t = await get_transcription(db, tid)
    if t is None:
        return None
    cursor = await db.execute(
        "SELECT * FROM segments WHERE transcription_id=? ORDER BY idx", (tid,)
    )
    rows = await cursor.fetchall()
    t["segments"] = [dict(r) for r in rows]
    return t


async def list_transcriptions(db, page: int = 1, per_page: int = 20,
                               status: str = None, period: str = None,
                               sort: str = "created_at", order: str = "desc") -> dict:
    """Liste paginée des transcriptions avec filtres."""
    conditions = []
    params = []

    if status:
        conditions.append("status=?")
        params.append(status)

    if period:
        days = _parse_period(period)
        if days:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{days} days")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Whitelist sort columns
    allowed_sorts = {"created_at", "filename", "duration_sec", "status", "language"}
    if sort not in allowed_sorts:
        sort = "created_at"
    order = "DESC" if order.lower() == "desc" else "ASC"

    # Count
    cursor = await db.execute(f"SELECT COUNT(*) FROM transcriptions {where}", params)
    row = await cursor.fetchone()
    total = row[0]

    offset = (page - 1) * per_page
    cursor = await db.execute(
        f"SELECT * FROM transcriptions {where} ORDER BY {sort} {order} LIMIT ? OFFSET ?",
        params + [per_page, offset]
    )
    rows = await cursor.fetchall()

    return {
        "items": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
    }


async def update_transcription_meta(db, tid: int, quality_score: int = None, notes: str = None):
    """Met à jour les métadonnées utilisateur (notes, score qualité)."""
    fields = ["updated_at=datetime('now')"]
    values = []
    if quality_score is not None:
        fields.append("quality_score=?")
        values.append(quality_score)
    if notes is not None:
        fields.append("notes=?")
        values.append(notes)
    values.append(tid)
    await db.execute(
        f"UPDATE transcriptions SET {', '.join(fields)} WHERE id=?",
        values
    )
    await db.commit()


async def update_segment_text(db, segment_id: int, text: str):
    """Met à jour le texte d'un segment (édition inline)."""
    await db.execute("UPDATE segments SET text=? WHERE id=?", (text, segment_id))
    await db.commit()


async def search(db, query: str, limit: int = 50) -> list:
    """Recherche FTS5 dans les segments + LIKE sur filename."""
    results = []

    # FTS search in segments
    cursor = await db.execute(
        """SELECT s.id as segment_id, s.transcription_id, s.idx, s.start_ms, s.end_ms,
                  s.text, t.filename
           FROM segments_fts fts
           JOIN segments s ON s.id = fts.rowid
           JOIN transcriptions t ON t.id = s.transcription_id
           WHERE segments_fts MATCH ?
           LIMIT ?""",
        (query, limit)
    )
    rows = await cursor.fetchall()
    for r in rows:
        results.append({**dict(r), "match_type": "segment"})

    # LIKE search on filename
    cursor = await db.execute(
        """SELECT id, filename, status, created_at, duration_sec, language
           FROM transcriptions
           WHERE filename LIKE ?
           LIMIT ?""",
        (f"%{query}%", limit)
    )
    rows = await cursor.fetchall()
    for r in rows:
        results.append({**dict(r), "match_type": "filename"})

    return results


async def get_dashboard_stats(db, period: str = "30d") -> dict:
    """Stats agrégées pour le dashboard."""
    days = _parse_period(period) or 30
    date_filter = f"-{days} days"

    # Stats de la période
    cursor = await db.execute(
        """SELECT
            COUNT(*) as count,
            COALESCE(SUM(duration_sec), 0) as total_duration,
            COALESCE(AVG(processing_ms), 0) as avg_processing_ms
           FROM transcriptions
           WHERE status='completed' AND created_at >= datetime('now', ?)""",
        (date_filter,)
    )
    row = await cursor.fetchone()
    period_stats = dict(row)

    # Stats totales
    cursor = await db.execute(
        """SELECT
            COUNT(*) as total_count,
            COALESCE(SUM(duration_sec), 0) as total_duration_all
           FROM transcriptions WHERE status='completed'"""
    )
    row = await cursor.fetchone()
    total_stats = dict(row)

    # Répartition par statut
    cursor = await db.execute(
        "SELECT status, COUNT(*) as count FROM transcriptions GROUP BY status"
    )
    rows = await cursor.fetchall()
    by_status = {r["status"]: r["count"] for r in rows}

    return {
        "period_count": period_stats["count"],
        "period_duration_sec": period_stats["total_duration"],
        "period_avg_processing_ms": round(period_stats["avg_processing_ms"]),
        "total_count": total_stats["total_count"],
        "total_duration_sec": total_stats["total_duration_all"],
        "by_status": by_status,
    }


# --- Helpers sync (pour le générateur SSE) ---

def save_result_sync(db_path: str, tid: int, duration_sec: float, language: str,
                     language_prob: float, word_count: int, processing_ms: int,
                     segments: list, audio_path: str = None):
    """Sauvegarde le résultat d'une transcription (appelé depuis le générateur sync)."""
    conn = get_sync_connection(db_path)
    try:
        conn.execute(
            """UPDATE transcriptions SET
                status='completed', duration_sec=?, language=?, language_detected=?,
                word_count=?, processing_ms=?, audio_path=?, updated_at=datetime('now')
               WHERE id=?""",
            (duration_sec, language, language_prob, word_count, processing_ms, audio_path, tid)
        )
        conn.executemany(
            "INSERT INTO segments (transcription_id, idx, start_ms, end_ms, text) VALUES (?, ?, ?, ?, ?)",
            [(tid, i, seg["start_ms"], seg["end_ms"], seg["text"]) for i, seg in enumerate(segments)]
        )
        conn.commit()
    finally:
        conn.close()


def mark_error_sync(db_path: str, tid: int, error_message: str):
    """Marque une transcription en erreur (appelé depuis le générateur sync)."""
    conn = get_sync_connection(db_path)
    try:
        conn.execute(
            """UPDATE transcriptions SET status='error', error_message=?, updated_at=datetime('now')
               WHERE id=?""",
            (error_message, tid)
        )
        conn.commit()
    finally:
        conn.close()


def _parse_period(period: str) -> int | None:
    """Parse une période comme '7d', '30d', 'all' en nombre de jours."""
    if not period or period == "all":
        return None
    period = period.strip().lower()
    if period.endswith("d"):
        try:
            return int(period[:-1])
        except ValueError:
            return None
    return None
