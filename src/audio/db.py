"""SQLite data access layer for data/audio.sqlite.

Conventions:
- All connections use WAL mode and FK enforcement.
- AUDIO_ROOT is the canonical filesystem root for audio paths,
  resolved via data/audio/local symlink (external, not in Git).
- segments.path is stored relative to AUDIO_ROOT
  (e.g. processed/common_voice/...).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DB_PATH = REPO_ROOT / "data" / "audio.sqlite"
AUDIO_ROOT = Path(os.environ.get("TELAFFUZ_AUDIO_ROOT", REPO_ROOT / "data" / "audio" / "local"))

EXPECTED_SCHEMA_VERSION = "3.0"


def connect(db_path: Path = AUDIO_DB_PATH, *, check_schema: bool = True) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"audio.sqlite not found at {db_path}; run `uv run python scripts/init_audio_db.py`"
        )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if check_schema:
        v = schema_version(conn)
        if v != EXPECTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"audio.sqlite schema_version={v}, expected {EXPECTED_SCHEMA_VERSION}"
            )
    return conn


def schema_version(conn: sqlite3.Connection) -> str | None:
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return row["value"] if row else None


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    entity: str,
    entity_id: str | None = None,
    payload_json: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO audit_log(ts, actor, action, entity, entity_id, payload_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (now_iso(), actor, action, entity, entity_id, payload_json),
    )
