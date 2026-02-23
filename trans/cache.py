"""SQLite transcript cache with TTL support."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir


def _cache_dir() -> Path:
    return Path(user_cache_dir("trans"))


def _cache_db() -> Path:
    return _cache_dir() / "transcripts.db"


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            video_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            transcript TEXT,
            format TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


class CacheManager:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db = db_path or _cache_db()

    def get(self, video_id: str, fmt: str = 'txt', ttl_days: int = 30) -> tuple[str, str] | None:
        """Return (transcript, title) if cached and within TTL, else None."""
        if not self._db.exists():
            return None
        conn = sqlite3.connect(self._db)
        cursor = conn.execute(
            '''SELECT transcript, title FROM transcripts
               WHERE video_id = ? AND format = ?
               AND created_at > datetime('now', ?)''',
            (video_id, fmt, f'-{ttl_days} days'),
        )
        row = cursor.fetchone()
        conn.close()
        return row if row else None

    def put(
        self,
        video_id: str,
        url: str,
        title: str,
        transcript: str,
        fmt: str = 'txt',
        model: str | None = None,
    ) -> None:
        """Store a transcript in the cache."""
        _init_db(self._db)
        conn = sqlite3.connect(self._db)
        conn.execute(
            '''INSERT OR REPLACE INTO transcripts
               (video_id, url, title, transcript, format, model)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (video_id, url, title, transcript, fmt, model),
        )
        conn.commit()
        conn.close()

    def clear(self) -> int:
        """Delete all cached entries. Returns number of rows deleted."""
        if not self._db.exists():
            return 0
        conn = sqlite3.connect(self._db)
        cursor = conn.execute('DELETE FROM transcripts')
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        if not self._db.exists():
            return {'count': 0, 'size_mb': 0.0, 'oldest': None, 'newest': None}
        conn = sqlite3.connect(self._db)
        cursor = conn.execute(
            'SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM transcripts'
        )
        row = cursor.fetchone()
        conn.close()
        size_mb = self._db.stat().st_size / (1024 * 1024) if self._db.exists() else 0.0
        return {
            'count': row[0] or 0,
            'size_mb': round(size_mb, 2),
            'oldest': row[1],
            'newest': row[2],
        }
