"""SQLite transcript cache with TTL support.

Schema v2 contract: rows store the canonical Whisper segment list + info dict
keyed by ``video_id`` alone. Rendering to txt/srt/vtt/json is a render-time
concern handled by ``trans.formatter.write_output`` at hit time. Before v2 the
cache stored a rendered-text blob keyed by ``(video_id, format)`` which
silently downgraded ``--format all`` cache hits to txt-only output.

Upgrade policy is destructive: a schema mismatch drops the existing table.
Re-transcription repopulates on next use. Pre-1.0; acceptable.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

SCHEMA_VERSION = 2

CacheHit = namedtuple("CacheHit", ["segments", "info", "title", "source"])


def _cache_dir() -> Path:
    return Path(user_cache_dir("trans"))


def _cache_db() -> Path:
    return _cache_dir() / "transcripts.db"


def _json_safe_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # faster-whisper / CTranslate2 sometimes return numpy scalars for
    # start/end; json.dumps raises TypeError on them. Cast at the cache
    # boundary so the rest of the pipeline can keep using whatever floats
    # the engine returns.
    out: list[dict[str, Any]] = []
    for s in segments:
        item: dict[str, Any] = {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": str(s["text"]),
        }
        if "speaker" in s:
            item["speaker"] = str(s["speaker"])
        out.append(item)
    return out


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    # Whitelisted to the three keys trans.transcriber currently emits. New
    # keys added upstream are silently dropped from the cache until this
    # function is updated — keep in sync with transcriber.TranscriptionEngine.
    out: dict[str, Any] = {}
    if "language" in info:
        out["language"] = str(info["language"])
    if "language_probability" in info:
        out["language_probability"] = float(info["language_probability"])
    if "duration" in info:
        out["duration"] = float(info["duration"])
    return out


def _connect(db_path: Path) -> sqlite3.Connection:
    # 5s timeout protects the inter-process migration race; without this a
    # second process starting against a fresh DB during the first process's
    # CREATE TABLE can fail with "database is locked" instead of waiting.
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


class CacheManager:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db = db_path or _cache_db()
        # Lazy schema-ensure preserves the invariant that --no-cache touches
        # no files. Construction is free; first public method call is what
        # actually creates/migrates the DB.
        self._schema_ensured = False

    def _ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        self._db.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect(self._db)
        try:
            # BEGIN IMMEDIATE serializes the check-then-act across processes:
            # if two trans invocations race to migrate a fresh DB, the second
            # waits on the writer lock, then re-reads user_version and skips
            # the drop.
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                discarded = 0
                try:
                    discarded = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
                except sqlite3.OperationalError:
                    pass
                conn.execute("DROP TABLE IF EXISTS transcripts")
                # `source` is reserved for the native-captions follow-up
                # (task-cache-native-captions); whisper rows tag it 'whisper'
                # by default so a future native row can be filtered apart.
                conn.execute(
                    "CREATE TABLE transcripts ("
                    "video_id TEXT PRIMARY KEY, "
                    "url TEXT, "
                    "title TEXT, "
                    "segments_json TEXT NOT NULL, "
                    "info_json TEXT, "
                    "model TEXT, "
                    "source TEXT NOT NULL DEFAULT 'whisper', "
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.commit()
                if discarded:
                    # Only warn when the migration actually destroyed user data.
                    # First-run init on a fresh box is invisible by design.
                    print(
                        f"trans: cache schema upgraded to v{SCHEMA_VERSION} — "
                        f"discarded {discarded} existing entries. "
                        "Re-transcription will repopulate on next use.",
                        file=sys.stderr,
                    )
            else:
                conn.commit()
        finally:
            conn.close()
        self._schema_ensured = True

    def get(self, video_id: str, ttl_days: int = 30) -> CacheHit | None:
        """Return cached segments + info if within TTL, else None."""
        self._ensure_schema()
        conn = _connect(self._db)
        try:
            cursor = conn.execute(
                "SELECT segments_json, info_json, title, source "
                "FROM transcripts "
                "WHERE video_id = ? "
                "AND created_at > datetime('now', ?)",
                (video_id, f"-{ttl_days} days"),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        segments_json, info_json, title, source = row
        segments = json.loads(segments_json)
        info = json.loads(info_json) if info_json is not None else None
        return CacheHit(segments=segments, info=info, title=title, source=source)

    def put(
        self,
        video_id: str,
        url: str,
        title: str,
        segments: list[dict[str, Any]],
        info: dict[str, Any] | None = None,
        model: str | None = None,
        source: str = "whisper",
        ttl_days: int = 30,
    ) -> None:
        """Store segments + info for a video.

        Replaces any prior row for the same ``video_id``. Also opportunistically
        evicts rows older than ``ttl_days`` in the same transaction — auto-prune
        on write keeps the SQLite file bounded without requiring users to run
        ``trans cache prune`` by hand. Cheap operation; runs at write rate, which
        is exactly when growth happens.
        """
        self._ensure_schema()
        segments_json = json.dumps(_json_safe_segments(segments))
        info_json = json.dumps(_json_safe_info(info)) if info is not None else None
        conn = _connect(self._db)
        try:
            conn.execute(
                "DELETE FROM transcripts WHERE created_at < datetime('now', ?)",
                (f"-{ttl_days} days",),
            )
            conn.execute(
                "INSERT OR REPLACE INTO transcripts "
                "(video_id, url, title, segments_json, info_json, model, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (video_id, url, title, segments_json, info_json, model, source),
            )
            conn.commit()
        finally:
            conn.close()

    def clear(self) -> int:
        """Delete all cached entries. Returns number of rows deleted."""
        if not self._db.exists():
            return 0
        self._ensure_schema()
        conn = _connect(self._db)
        try:
            cursor = conn.execute("DELETE FROM transcripts")
            count = cursor.rowcount
            conn.commit()
        finally:
            conn.close()
        return count

    def prune(self, ttl_days: int = 30) -> int:
        """Delete entries whose ``created_at`` is older than ``ttl_days``.

        Returns the number of rows removed. Cheap and idempotent — safe to
        invoke from a manual ``trans cache prune`` subcommand. Pre-1.0 the
        TTL is whatever the caller passes; the CLI defaults to
        ``cfg.cache.ttl_days``.
        """
        if not self._db.exists():
            return 0
        self._ensure_schema()
        conn = _connect(self._db)
        try:
            cursor = conn.execute(
                "DELETE FROM transcripts WHERE created_at < datetime('now', ?)",
                (f"-{ttl_days} days",),
            )
            count = cursor.rowcount
            conn.commit()
        finally:
            conn.close()
        return count

    def stats(self, ttl_days: int | None = None) -> dict[str, Any]:
        """Return cache statistics.

        Without ``ttl_days``, the result has the historical shape
        (``count`` / ``size_mb`` / ``oldest`` / ``newest``) plus
        ``count_active`` (= ``count``) and ``count_expired`` (= 0) so
        callers can always read the active/expired split without
        branching on the call shape.

        With ``ttl_days``, ``count_active`` reflects rows newer than the
        cutoff and ``count_expired`` reflects rows older.
        """
        if not self._db.exists():
            return {
                "count": 0,
                "count_active": 0,
                "count_expired": 0,
                "size_mb": 0.0,
                "oldest": None,
                "newest": None,
            }
        self._ensure_schema()
        conn = _connect(self._db)
        try:
            total_row = conn.execute(
                "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM transcripts"
            ).fetchone()
            if ttl_days is None:
                active = total_row[0] or 0
                expired = 0
            else:
                active = conn.execute(
                    "SELECT COUNT(*) FROM transcripts WHERE created_at > datetime('now', ?)",
                    (f"-{ttl_days} days",),
                ).fetchone()[0]
                expired = (total_row[0] or 0) - active
        finally:
            conn.close()
        size_mb = self._db.stat().st_size / (1024 * 1024)
        return {
            "count": total_row[0] or 0,
            "count_active": active,
            "count_expired": expired,
            "size_mb": round(size_mb, 2),
            "oldest": total_row[1],
            "newest": total_row[2],
        }
