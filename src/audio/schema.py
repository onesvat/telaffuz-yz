"""DDL and initialization for data/audio.sqlite (schema v3).

This module owns the canonical schema definition. scripts/init_audio_db.py
is a thin CLI wrapper around init_db.

Schema v3 changes from v2:
- Removed recordings table; single segments table only.
- segments.id changed from TEXT to INTEGER AUTOINCREMENT.
- segments.path is now the unique key (audio file path).
- Simplified FK references throughout.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "3.0"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    path           TEXT NOT NULL UNIQUE,
    provider       TEXT NOT NULL,
    duration_s     REAL NOT NULL CHECK (duration_s > 0),
    sample_rate    INTEGER NOT NULL CHECK (sample_rate > 0),
    channels       INTEGER NOT NULL CHECK (channels > 0),
    speaker_id     TEXT NOT NULL,
    speaker_source TEXT NOT NULL,
    checksum       TEXT,
    publisher_split TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_segments_provider ON segments(provider);
CREATE INDEX IF NOT EXISTS ix_segments_speaker  ON segments(speaker_id);
CREATE INDEX IF NOT EXISTS ix_segments_pub_split ON segments(publisher_split);

CREATE TABLE IF NOT EXISTS asr_configs (
    config_hash  TEXT PRIMARY KEY,
    engine       TEXT NOT NULL,
    model_version TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    code_version TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id      INTEGER NOT NULL REFERENCES segments(id),
    source          TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    run_hash        TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL,
    language        TEXT NOT NULL DEFAULT 'tr',
    confidence      REAL,
    is_primary      INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
    created_at      TEXT NOT NULL,
    UNIQUE(segment_id, source, model_version, run_hash)
);
CREATE INDEX IF NOT EXISTS ix_transcript_candidates_segment_source
    ON transcript_candidates(segment_id, source);
CREATE INDEX IF NOT EXISTS ix_transcript_candidates_source
    ON transcript_candidates(source);
CREATE UNIQUE INDEX IF NOT EXISTS ix_transcript_candidates_one_primary
    ON transcript_candidates(segment_id)
    WHERE is_primary = 1;

CREATE TABLE IF NOT EXISTS word_occurrences (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id               INTEGER NOT NULL REFERENCES segments(id),
    transcript_candidate_id  INTEGER NOT NULL REFERENCES transcript_candidates(id),
    word_index               INTEGER NOT NULL CHECK (word_index >= 0),
    word_surface             TEXT NOT NULL,
    word_norm                TEXT NOT NULL,
    start_ms                 INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms                   INTEGER NOT NULL CHECK (end_ms > start_ms),
    confidence               REAL,
    source                   TEXT NOT NULL,
    phone_alignment_json     TEXT,
    created_at               TEXT NOT NULL,
    UNIQUE(transcript_candidate_id, word_index)
);
CREATE INDEX IF NOT EXISTS ix_word_occurrences_segment
    ON word_occurrences(segment_id);
CREATE INDEX IF NOT EXISTS ix_word_occurrences_norm
    ON word_occurrences(word_norm);

CREATE TABLE IF NOT EXISTS phoneme_sequences (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scope          TEXT NOT NULL CHECK (scope IN ('segment', 'word')),
    segment_id     INTEGER NOT NULL REFERENCES segments(id),
    word_occurrence_id INTEGER REFERENCES word_occurrences(id),
    source         TEXT NOT NULL,
    phonemes_json  TEXT NOT NULL,
    g2p_version    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    CHECK (
        (scope = 'segment' AND word_occurrence_id IS NULL) OR
        (scope = 'word' AND word_occurrence_id IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS ix_phonseq_segment_source
    ON phoneme_sequences(segment_id, source);
CREATE UNIQUE INDEX IF NOT EXISTS ix_phonseq_segment_unique
    ON phoneme_sequences(segment_id, source, g2p_version)
    WHERE scope = 'segment';
CREATE UNIQUE INDEX IF NOT EXISTS ix_phonseq_word_unique
    ON phoneme_sequences(word_occurrence_id, source, g2p_version)
    WHERE scope = 'word';

CREATE TABLE IF NOT EXISTS segment_alignments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id          INTEGER NOT NULL REFERENCES segments(id),
    source_sequence_id  INTEGER NOT NULL REFERENCES phoneme_sequences(id),
    aligner_version     TEXT NOT NULL,
    alignment_json      TEXT NOT NULL,
    summary_json        TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL,
    UNIQUE(segment_id, source_sequence_id, aligner_version)
);
CREATE INDEX IF NOT EXISTS ix_segment_alignments_segment
    ON segment_alignments(segment_id);

CREATE TABLE IF NOT EXISTS segment_assessments (
    segment_id            INTEGER NOT NULL REFERENCES segments(id),
    assessment_version    TEXT NOT NULL,
    metrics_json          TEXT NOT NULL,
    trust_tier            TEXT NOT NULL CHECK (trust_tier IN (
                            'high', 'medium', 'low', 'reject'
                          )),
    reject_reasons_json   TEXT NOT NULL DEFAULT '[]',
    computed_at           TEXT NOT NULL,
    PRIMARY KEY (segment_id, assessment_version)
);
CREATE INDEX IF NOT EXISTS ix_segment_assessments_tier
    ON segment_assessments(trust_tier);

CREATE TABLE IF NOT EXISTS manual_annotations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type   TEXT NOT NULL CHECK (entity_type IN (
                    'segment', 'word_occurrence',
                    'transcript_candidate', 'segment_alignment'
                  )),
    entity_id     TEXT NOT NULL,
    label         TEXT NOT NULL,
    value_json    TEXT NOT NULL,
    actor         TEXT NOT NULL,
    notes         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_manual_annotations_entity
    ON manual_annotations(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_manual_annotations_label
    ON manual_annotations(label);

CREATE TABLE IF NOT EXISTS datasets (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    version      TEXT NOT NULL,
    unit         TEXT NOT NULL CHECK (unit IN (
                 'segment', 'word_occurrence'
               )),
    spec_json    TEXT NOT NULL,
    description  TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS dataset_memberships (
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    entity_type   TEXT NOT NULL CHECK (entity_type IN (
                  'segment', 'word_occurrence'
                )),
    entity_id     TEXT NOT NULL,
    split         TEXT NOT NULL,
    hash_bucket   INTEGER NOT NULL CHECK (0 <= hash_bucket AND hash_bucket <= 99),
    added_at      TEXT NOT NULL,
    PRIMARY KEY (dataset_id, entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS ix_memb_dataset_split ON dataset_memberships(dataset_id, split);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    actor         TEXT NOT NULL,
    action        TEXT NOT NULL,
    entity        TEXT NOT NULL,
    entity_id     TEXT,
    payload_json  TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_entity ON audit_log(entity, entity_id);
"""

EXPECTED_TABLES = frozenset({
    "schema_meta",
    "segments",
    "asr_configs",
    "transcript_candidates",
    "word_occurrences",
    "phoneme_sequences",
    "segment_alignments",
    "segment_assessments",
    "manual_annotations",
    "datasets",
    "dataset_memberships",
    "audit_log",
})


def init_db(db_path: Path, *, force: bool = False) -> None:
    if db_path.exists() and force:
        db_path.unlink()
        for sfx in ("-wal", "-shm"):
            sibling = db_path.with_name(db_path.name + sfx)
            if sibling.exists():
                sibling.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("schema_version", SCHEMA_VERSION, now),
        )
        conn.commit()

        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        missing = EXPECTED_TABLES - tables
        if missing:
            raise RuntimeError(f"DB init incomplete; missing tables: {missing}")
    finally:
        conn.close()
