"""Smoke tests for data/audio.sqlite schema v3.

Each test uses a temp DB; never touches the real data/audio.sqlite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from audio.db import EXPECTED_SCHEMA_VERSION, connect, now_iso, schema_version, transaction
from audio.datasets import speaker_bucket, split_for_bucket
from audio.schema import EXPECTED_TABLES, init_db


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "audio.sqlite"
    init_db(db)
    return db


def test_init_creates_all_tables(tmp_db: Path) -> None:
    conn = sqlite3.connect(str(tmp_db))
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = {r[0] for r in rows}
    assert EXPECTED_TABLES.issubset(tables), f"missing: {EXPECTED_TABLES - tables}"
    conn.close()


def test_schema_version_recorded(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        assert schema_version(conn) == EXPECTED_SCHEMA_VERSION
    finally:
        conn.close()


def _insert_segment(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        """INSERT INTO segments(path, provider, duration_s, sample_rate, channels,
                                speaker_id, speaker_source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "processed/common_voice/common_voice_tr_001.wav",
            "common_voice",
            3.42,
            16000,
            1,
            "cv_client_xyz",
            "validated.tsv.client_id",
            now_iso(),
        ),
    )
    return cur.lastrowid


def test_segment_insert_roundtrip(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            seg_id = _insert_segment(conn)
        row = conn.execute("SELECT * FROM segments WHERE id=?", (seg_id,)).fetchone()
        assert row is not None
        assert row["id"] == 1  # AUTOINCREMENT starts at 1
        assert row["duration_s"] == 3.42
        assert row["speaker_id"] == "cv_client_xyz"
        assert row["provider"] == "common_voice"
    finally:
        conn.close()


def test_unique_path_constraint(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            _insert_segment(conn)
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                conn.execute(
                    """INSERT INTO segments(path, provider, duration_s, sample_rate, channels,
                                            speaker_id, speaker_source, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "processed/common_voice/common_voice_tr_001.wav",  # duplicate path
                        "common_voice",
                        5.0,
                        48000,
                        1,
                        "cv_other",
                        "placeholder",
                        now_iso(),
                    ),
                )
    finally:
        conn.close()


def test_foreign_key_enforcement(tmp_db: Path) -> None:
    """Inserting a transcript for a non-existent segment_id must fail."""
    conn = connect(tmp_db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                conn.execute(
                    """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                                         text, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (999, "publisher_gt", "v1", "test", now_iso()),  # non-existent segment_id
                )
    finally:
        conn.close()


def test_transcript_candidate_unique_constraint(tmp_db: Path) -> None:
    """(segment_id, source, model_version, run_hash) must be unique."""
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            seg_id = _insert_segment(conn)
        with transaction(conn):
            conn.execute(
                """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                                     text, is_primary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (seg_id, "publisher_gt", "orig", "first", 1, now_iso()),
            )
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                conn.execute(
                    """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                                         text, is_primary, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (seg_id, "publisher_gt", "orig", "duplicate", 0, now_iso()),
                )
    finally:
        conn.close()


def test_only_one_primary_transcript_per_segment(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            seg_id = _insert_segment(conn)
            conn.execute(
                """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                                     text, is_primary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (seg_id, "publisher_gt", "orig", "Anne geldi.", 1, now_iso()),
            )
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                conn.execute(
                    """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                                         text, is_primary, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (seg_id, "manual", "v1", "Anne geldi.", 1, now_iso()),
                )
    finally:
        conn.close()


def test_word_occurrence_and_json_alignment_roundtrip(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            seg_id = _insert_segment(conn)
            cur = conn.execute(
                """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                                     text, is_primary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (seg_id, "publisher_gt", "orig", "Anne geldi.", 1, now_iso()),
            )
            transcript_id = cur.lastrowid
            cur = conn.execute(
                """INSERT INTO word_occurrences(segment_id, transcript_candidate_id,
                                                word_index, word_surface, word_norm,
                                                start_ms, end_ms, confidence, source,
                                                phone_alignment_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    seg_id,
                    transcript_id,
                    0,
                    "Anne",
                    "anne",
                    120,
                    610,
                    0.91,
                    "whisper_timestamp_on_gt",
                    '{"phones":[["a",120,245,0.87],["n",245,330,0.82]]}',
                    now_iso(),
                ),
            )
            word_id = cur.lastrowid
            cur = conn.execute(
                """INSERT INTO phoneme_sequences(scope, segment_id, word_occurrence_id,
                                                 source, phonemes_json, g2p_version,
                                                 created_at)
                   VALUES ('word', ?, ?, 'g2p_from_gt',
                           '["a","n","n","e"]', 'test-g2p', ?)""",
                (seg_id, word_id, now_iso()),
            )
            sequence_id = cur.lastrowid
            conn.execute(
                """INSERT INTO segment_alignments(segment_id, source_sequence_id,
                                                 aligner_version, alignment_json,
                                                 summary_json, created_at)
                   VALUES (?, ?, 'mms-test',
                           '{"phones":[{"p":"a","s":120,"e":245,"c":0.87}]}',
                           '{"mean_confidence":0.87}', ?)""",
                (seg_id, sequence_id, now_iso()),
            )

        row = conn.execute(
            "SELECT * FROM word_occurrences WHERE word_norm='anne'"
        ).fetchone()
        assert row is not None
        assert row["start_ms"] == 120
        assert '"phones"' in row["phone_alignment_json"]
    finally:
        conn.close()


def test_segment_assessment_roundtrip(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            seg_id = _insert_segment(conn)
            conn.execute(
                """INSERT INTO segment_assessments(segment_id, assessment_version,
                                                   metrics_json, trust_tier,
                                                   reject_reasons_json, computed_at)
                   VALUES (?, 'v1',
                           '{"snr":0.82,"wer_vs_gt":0.03}', 'high', '[]', ?)""",
                (seg_id, now_iso()),
            )
        row = conn.execute(
            "SELECT trust_tier, metrics_json FROM segment_assessments WHERE segment_id=?",
            (seg_id,)
        ).fetchone()
        assert row["trust_tier"] == "high"
        assert '"snr":0.82' in row["metrics_json"]
    finally:
        conn.close()


def test_manual_annotation_accepts_segment_and_word_entities(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        with transaction(conn):
            seg_id = _insert_segment(conn)
            conn.execute(
                """INSERT INTO manual_annotations(entity_type, entity_id, label, value_json,
                                                  actor, notes, created_at)
                   VALUES ('segment', ?, 'clean_standard_speech', 'true',
                           'onur', 'net ve dogal', ?)""",
                (str(seg_id), now_iso()),
            )
        row = conn.execute("SELECT * FROM manual_annotations").fetchone()
        assert row["entity_type"] == "segment"
        assert row["label"] == "clean_standard_speech"
    finally:
        conn.close()


def test_audit_log_roundtrip(tmp_db: Path) -> None:
    from audio.db import audit

    conn = connect(tmp_db)
    try:
        with transaction(conn):
            audit(
                conn,
                actor="script:test",
                action="create",
                entity="segment",
                entity_id="1",
                payload_json='{"foo": "bar"}',
            )
        rows = conn.execute("SELECT * FROM audit_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["action"] == "create"
        assert rows[0]["entity_id"] == "1"
    finally:
        conn.close()


# Speaker-aware split determinism


def test_speaker_bucket_deterministic() -> None:
    assert speaker_bucket("alice") == speaker_bucket("alice")
    assert speaker_bucket("alice") != speaker_bucket("bob")
    for s in ["user_001", "audiobooks/Cesur Yeni Dünya", "cv_client_a1b2c3"]:
        bucket = speaker_bucket(s)
        assert 0 <= bucket <= 99


def test_split_assignment() -> None:
    assert split_for_bucket(0) == "train"
    assert split_for_bucket(89) == "train"
    assert split_for_bucket(90) == "val"
    assert split_for_bucket(94) == "val"
    assert split_for_bucket(95) == "test"
    assert split_for_bucket(99) == "test"
    with pytest.raises(ValueError):
        split_for_bucket(100)
    with pytest.raises(ValueError):
        split_for_bucket(-1)


def test_split_proportions_at_scale() -> None:
    """Across 10k synthetic speakers, splits should approximate 90/5/5."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for i in range(10_000):
        counts[split_for_bucket(speaker_bucket(f"speaker_{i:06d}"))] += 1
    assert 8800 <= counts["train"] <= 9200
    assert 400 <= counts["val"] <= 600
    assert 400 <= counts["test"] <= 600