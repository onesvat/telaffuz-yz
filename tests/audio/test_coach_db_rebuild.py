"""Smoke tests for audio.coach_db.rebuild_reference (Faz 3)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, *, sample_rate: int = 16_000, duration_s: float = 1.0) -> None:
    import soundfile as sf

    rng = np.random.default_rng(0)
    n = int(duration_s * sample_rate)
    silence_head = np.zeros(int(0.08 * sample_rate), dtype=np.float32)
    voiced = (
        0.3
        * rng.normal(0, 1, size=n - 2 * int(0.08 * sample_rate)).astype(np.float32)
    )
    silence_tail = np.zeros(int(0.08 * sample_rate), dtype=np.float32)
    audio = np.concatenate([silence_head, voiced, silence_tail])
    sf.write(str(path), audio, sample_rate)


def _populate_audio_db(db_path: Path, *, audio_path: str, duration_s: float = 1.0) -> int:
    from audio.schema import init_db

    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO segments(path, provider, duration_s, sample_rate, channels,
                                speaker_id, speaker_source, created_at)
           VALUES (?, 'common_voice', ?, 16000, 1, 'spk_1', 'fixture', ?)""",
        (audio_path, duration_s, now),
    )
    segment_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        """INSERT INTO phoneme_sequences(scope, segment_id, source, phonemes_json,
                                         g2p_version, created_at)
           VALUES ('segment', ?, 'expected_49', ?, 'fixture_v1', ?)""",
        (segment_id, json.dumps(["m", "a", "s"]), now),
    )
    sequence_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    alignment_json = json.dumps(
        {
            "schema_version": 1,
            "phones": [
                {
                    "expected_phone": "m",
                    "alignment_phone": "m",
                    "observed_model_phone": "m",
                    "start_ms": 80,
                    "end_ms": 180,
                    "confidence": 0.92,
                },
                {
                    "expected_phone": "a",
                    "alignment_phone": "a",
                    "observed_model_phone": "a",
                    "start_ms": 180,
                    "end_ms": 700,
                    "confidence": 0.95,
                },
                {
                    "expected_phone": "s",
                    "alignment_phone": "s",
                    "observed_model_phone": "s",
                    "start_ms": 700,
                    "end_ms": 900,
                    "confidence": 0.83,
                },
            ],
        }
    )
    conn.execute(
        """INSERT INTO segment_alignments(segment_id, source_sequence_id, aligner_version,
                                         alignment_json, summary_json, created_at)
           VALUES (?, ?, 'mms-fa-test', ?, '{}', ?)""",
        (segment_id, sequence_id, alignment_json, now),
    )
    conn.commit()
    conn.close()
    return segment_id


@pytest.mark.parametrize("workers", [1, 2])
def test_rebuild_reference_writes_new_schema_with_class_features(
    tmp_path: Path, workers: int
) -> None:
    from audio.coach_db import (
        COACH_REFERENCE_DERIVATION_POLICY,
        COACH_REFERENCE_FEATURE_VERSION,
        rebuild_reference,
    )

    audio_root = tmp_path / "audio"
    audio_root.mkdir(parents=True)
    audio_path = "fixture.wav"
    _write_wav(audio_root / audio_path, duration_s=1.0)

    audio_db = tmp_path / "audio.sqlite"
    _populate_audio_db(audio_db, audio_path=audio_path, duration_s=1.0)

    output_db = tmp_path / "reference.sqlite"
    report = rebuild_reference(
        audio_db=audio_db,
        audio_root=audio_root,
        output_db=output_db,
        providers=("common_voice",),
        workers=workers,
    )

    assert report.feature_version == COACH_REFERENCE_FEATURE_VERSION
    assert report.derivation_policy == COACH_REFERENCE_DERIVATION_POLICY
    assert report.workers == workers
    assert report.inserted_rows >= 1
    assert output_db.exists()
    assert not list(tmp_path.glob("reference.building.shard*.sqlite"))

    out_conn = sqlite3.connect(str(output_db))
    out_conn.row_factory = sqlite3.Row
    cols = {row["name"] for row in out_conn.execute("PRAGMA table_info(phone_features)")}
    expected_new_cols = {
        "active_duration_ms",
        "anchor_confidence",
        "duration_reliability",
        "voiced_start_ms",
        "voiced_end_ms",
        "spectral_skew",
        "spectral_kurtosis",
        "vot_ms",
        "burst_centroid_hz",
        "burst_spectral_skew",
        "frication_rise_db_per_ms",
        "f2_transition_slope_hz_per_ms",
        "derivation_policy",
    }
    assert expected_new_cols.issubset(cols)

    rows = list(out_conn.execute("SELECT * FROM phone_features ORDER BY phone_index"))
    out_conn.close()
    assert len(rows) >= 1
    # At least one row must carry the CV-only provider tag and feature version.
    assert all(row["provider"] == "common_voice" for row in rows)
    assert all(row["feature_version"] == COACH_REFERENCE_FEATURE_VERSION for row in rows)
    # Anchor confidence should be propagated from the alignment JSON.
    confidences = [row["anchor_confidence"] for row in rows]
    assert any(c is not None and c > 0.5 for c in confidences)


def test_rebuild_reference_validates_audio_root(tmp_path: Path) -> None:
    from audio.coach_db import rebuild_reference

    # Existing audio_db, missing audio_root → should reject audio_root.
    audio_db = tmp_path / "audio.sqlite"
    _populate_audio_db(audio_db, audio_path="phantom.wav", duration_s=0.5)

    with pytest.raises(FileNotFoundError, match="audio_root"):
        rebuild_reference(
            audio_db=audio_db,
            audio_root=tmp_path / "missing_root",
            output_db=tmp_path / "out.sqlite",
        )


def test_rebuild_reference_validates_workers(tmp_path: Path) -> None:
    from audio.coach_db import rebuild_reference

    audio_root = tmp_path / "audio"
    audio_root.mkdir(parents=True)
    audio_path = "fixture.wav"
    _write_wav(audio_root / audio_path, duration_s=0.5)

    audio_db = tmp_path / "audio.sqlite"
    _populate_audio_db(audio_db, audio_path=audio_path, duration_s=0.5)

    with pytest.raises(ValueError, match="workers"):
        rebuild_reference(
            audio_db=audio_db,
            audio_root=audio_root,
            output_db=tmp_path / "out.sqlite",
            workers=0,
        )
