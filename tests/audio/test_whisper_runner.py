"""Smoke tests for whisper_runner.

Does not load a real WhisperModel; validates insert/audit/resume logic with
mock result dicts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from audio.db import connect, now_iso
from audio.schema import init_db
from audio.whisper_runner import (
    GT_PROVIDERS,
    WHISPER_TRANSCRIPT_SOURCE,
    WHISPER_WORD_SOURCE,
    WhisperConfig,
    _insert_result,
    _segment_confidence,
    ensure_asr_config,
    iter_pending_segments,
    load_manifest_filter,
    normalize_word,
    parse_shard_spec,
    segment_in_shard,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "audio.sqlite"
    init_db(db)
    return db


def _insert_seg(conn: sqlite3.Connection, *, path: str, provider: str) -> int:
    cur = conn.execute(
        """INSERT INTO segments(path, provider, duration_s, sample_rate, channels,
                                speaker_id, speaker_source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (path, provider, 3.0, 16000, 1, "spk_x", "fixed", now_iso()),
    )
    return int(cur.lastrowid)


def _mock_result(*, segment_id: int, provider: str, text: str = "merhaba dünya",
                  words: list[dict] | None = None, language: str = "tr",
                  duration: float = 3.0, ok: bool = True) -> dict:
    if words is None:
        words = [
            {"start": 0.10, "end": 0.65, "word": "merhaba", "probability": 0.95},
            {"start": 0.70, "end": 1.20, "word": "dünya", "probability": 0.88},
        ]
    return {
        "ok": ok,
        "worker_id": 0,
        "segment_id": segment_id,
        "provider": provider,
        "text": text,
        "language": language,
        "duration": duration,
        "words": words,
        "elapsed": 0.1,
    }


# ─── config hash & normalization ─────────────────────────────────────────────


def test_config_hash_stable_across_field_order() -> None:
    cfg1 = WhisperConfig(model="large-v3-turbo", beam_size=5)
    cfg2 = WhisperConfig(beam_size=5, model="large-v3-turbo")
    assert cfg1.config_hash() == cfg2.config_hash()


def test_config_hash_changes_with_meaningful_field() -> None:
    cfg1 = WhisperConfig(beam_size=5)
    cfg2 = WhisperConfig(beam_size=1)
    assert cfg1.config_hash() != cfg2.config_hash()


def test_config_hash_short_16_hex() -> None:
    h = WhisperConfig().config_hash()
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_normalize_word_turkish_locale() -> None:
    assert normalize_word("İstanbul") == "istanbul"
    assert normalize_word("IRAK") == "ırak"
    assert normalize_word("Merhaba,") == "merhaba"
    assert normalize_word("DÜNYA.") == "dünya"
    assert normalize_word("'evet'") == "'evet'"


def test_parse_shard_spec_accepts_single_and_multiple_1_based() -> None:
    assert parse_shard_spec("6/10") == ({5}, 10)
    assert parse_shard_spec("6,7/10") == ({5, 6}, 10)


@pytest.mark.parametrize("spec", ["0/10", "11/10", "1/0", "x/10", "1,12/10"])
def test_parse_shard_spec_rejects_invalid(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_shard_spec(spec)


def test_segment_in_shard_uses_stable_1_based_segment_ids() -> None:
    shard = parse_shard_spec("2/3")
    assert [i for i in range(1, 8) if segment_in_shard(i, *shard)] == [2, 5]


def test_segment_confidence_avg() -> None:
    words = [{"probability": 0.9}, {"probability": 0.8}, {"probability": 1.0}]
    assert _segment_confidence(words) == pytest.approx(0.9)


def test_segment_confidence_empty() -> None:
    assert _segment_confidence([]) is None


# ─── ensure_asr_config ───────────────────────────────────────────────────────


def test_ensure_asr_config_inserts_once(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h1 = ensure_asr_config(conn, cfg)
    h2 = ensure_asr_config(conn, cfg)
    rows = conn.execute("SELECT COUNT(*) AS n FROM asr_configs").fetchone()
    conn.close()
    assert h1 == h2
    assert rows["n"] == 1


def test_ensure_asr_config_records_full_json(tmp_db: Path) -> None:
    cfg = WhisperConfig(beam_size=3)
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    row = conn.execute(
        "SELECT engine, model_version, config_json FROM asr_configs WHERE config_hash=?",
        (h,),
    ).fetchone()
    conn.close()
    assert row["engine"] == "faster_whisper"
    assert row["model_version"] == cfg.model
    payload = json.loads(row["config_json"])
    assert payload["beam_size"] == 3
    assert payload["language"] == "tr"


# ─── insert behaviour: trust model + word_occurrences ────────────────────────


def test_insert_sets_is_primary_for_non_gt_provider(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path="processed/audiobooks/studio/x.wav", provider="audiobooks")
    _insert_result(conn, cfg, h, _mock_result(segment_id=seg_id, provider="audiobooks"))
    conn.commit()

    row = conn.execute(
        "SELECT is_primary, source, run_hash, confidence FROM transcript_candidates "
        "WHERE segment_id=?", (seg_id,)
    ).fetchone()
    conn.close()
    assert row["is_primary"] == 1
    assert row["source"] == WHISPER_TRANSCRIPT_SOURCE
    assert row["run_hash"] == h
    assert row["confidence"] == pytest.approx(0.915, abs=0.001)


@pytest.mark.parametrize("provider", sorted(GT_PROVIDERS))
def test_insert_keeps_whisper_secondary_for_gt(provider: str, tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path=f"processed/{provider}/x.wav", provider=provider)
    conn.execute(
        """INSERT INTO transcript_candidates
             (segment_id, source, model_version, run_hash, text, language,
              is_primary, created_at)
             VALUES (?, 'publisher_gt', 'legacy_original', '', 'gt text', 'tr', 1, ?)""",
        (seg_id, now_iso()),
    )
    _insert_result(conn, cfg, h, _mock_result(segment_id=seg_id, provider=provider))
    conn.commit()

    rows = conn.execute(
        "SELECT source, is_primary FROM transcript_candidates WHERE segment_id=?", (seg_id,)
    ).fetchall()
    conn.close()
    by_src = {r["source"]: r["is_primary"] for r in rows}
    assert by_src["publisher_gt"] == 1
    assert by_src[WHISPER_TRANSCRIPT_SOURCE] == 0


def test_insert_writes_word_occurrences_with_normalization(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path="processed/podcasts/x.wav", provider="podcasts")
    words = [
        {"start": 0.0, "end": 0.4, "word": " İstanbul", "probability": 0.9},
        {"start": 0.4, "end": 0.8, "word": " büyük.", "probability": 0.7},
    ]
    _insert_result(conn, cfg, h, _mock_result(segment_id=seg_id, provider="podcasts", words=words))
    conn.commit()

    rows = conn.execute(
        "SELECT word_index, word_surface, word_norm, start_ms, end_ms, confidence, source "
        "FROM word_occurrences WHERE segment_id=? ORDER BY word_index", (seg_id,)
    ).fetchall()
    conn.close()
    assert [r["word_index"] for r in rows] == [0, 1]
    assert rows[0]["word_surface"].strip() == "İstanbul"
    assert rows[0]["word_norm"] == "istanbul"
    assert rows[0]["start_ms"] == 0 and rows[0]["end_ms"] == 400
    assert rows[1]["word_norm"] == "büyük"
    assert rows[1]["source"] == WHISPER_WORD_SOURCE


def test_insert_handles_zero_duration_word(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path="processed/youtube/x.wav", provider="youtube")
    words = [{"start": 1.0, "end": 1.0, "word": "ek", "probability": 0.6}]
    _insert_result(conn, cfg, h, _mock_result(segment_id=seg_id, provider="youtube", words=words))
    conn.commit()
    row = conn.execute(
        "SELECT start_ms, end_ms FROM word_occurrences WHERE segment_id=?", (seg_id,)
    ).fetchone()
    conn.close()
    assert row["start_ms"] == 1000
    assert row["end_ms"] == 1001  # bumped past CHECK end_ms > start_ms


def test_insert_empty_text_logs_audit(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path="processed/audiobooks/x.wav", provider="audiobooks")
    _insert_result(conn, cfg, h,
                   _mock_result(segment_id=seg_id, provider="audiobooks",
                                text="", words=[]))
    conn.commit()
    audit_rows = conn.execute(
        "SELECT action, entity_id FROM audit_log WHERE action='whisper_empty'"
    ).fetchall()
    conn.close()
    assert len(audit_rows) == 1
    assert audit_rows[0]["entity_id"] == str(seg_id)


def test_insert_non_tr_language_logs_audit(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path="processed/youtube/x.wav", provider="youtube")
    _insert_result(conn, cfg, h,
                   _mock_result(segment_id=seg_id, provider="youtube", language="en"))
    conn.commit()
    audit_rows = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action='whisper_non_tr'"
    ).fetchall()
    conn.close()
    assert len(audit_rows) == 1
    payload = json.loads(audit_rows[0]["payload_json"])
    assert payload["detected"] == "en"


# ─── resume / idempotence ────────────────────────────────────────────────────


def test_iter_pending_skips_already_transcribed(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    s1 = _insert_seg(conn, path="processed/audiobooks/studio/a.wav", provider="audiobooks")
    s2 = _insert_seg(conn, path="processed/audiobooks/studio/b.wav", provider="audiobooks")
    _insert_result(conn, cfg, h, _mock_result(segment_id=s1, provider="audiobooks"))
    conn.commit()
    pending = list(iter_pending_segments(conn, config_hash=h))
    conn.close()
    pending_ids = {p[0] for p in pending}
    assert s1 not in pending_ids
    assert s2 in pending_ids


def test_iter_pending_respects_providers_and_limit(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    _insert_seg(conn, path="processed/audiobooks/studio/a.wav", provider="audiobooks")
    _insert_seg(conn, path="processed/podcasts/b.wav", provider="podcasts")
    _insert_seg(conn, path="processed/audiobooks/drama/c.wav", provider="audiobooks")
    pending = list(iter_pending_segments(
        conn, config_hash=h, providers=["audiobooks", "podcasts"], limit=10
    ))
    conn.close()
    providers = {p[2] for p in pending}
    assert providers == {"audiobooks", "podcasts"}
    assert len(pending) == 3


def test_load_manifest_filter_keeps_only_needs_whisper_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "atlas_audio_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "segment_id,path,provider,duration_s,speaker_id,transcript_source,needs_whisper,role,selection_reason",
                "10,processed/audiobooks/studio/a.wav,audiobooks,3,spk,missing_whisper,true,atlas_silver,fixture",
                "11,processed/audiobooks/studio/b.wav,audiobooks,3,spk,whisper_large_v3_turbo,false,atlas_silver,fixture",
                "12,processed/audiobooks/drama/c.wav,audiobooks,3,spk,missing_whisper,TRUE,atlas_silver,fixture",
            ]
        ),
        encoding="utf-8",
    )

    manifest_filter = load_manifest_filter(manifest, only_needs_whisper=True)

    assert manifest_filter.segment_ids == frozenset({10, 12})
    assert manifest_filter.providers == ("audiobooks",)


def test_iter_pending_respects_manifest_filter(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    s1 = _insert_seg(conn, path="processed/audiobooks/studio/a.wav", provider="audiobooks")
    s2 = _insert_seg(conn, path="processed/audiobooks/studio/b.wav", provider="audiobooks")
    s3 = _insert_seg(conn, path="processed/audiobooks/drama/c.wav", provider="audiobooks")
    _insert_result(conn, cfg, h, _mock_result(segment_id=s2, provider="audiobooks"))
    conn.commit()

    pending = list(
        iter_pending_segments(
            conn,
            config_hash=h,
            segment_ids={s1, s2},
        )
    )
    conn.close()

    assert [p[0] for p in pending] == [s1]
    assert s3 not in {p[0] for p in pending}


def test_iter_pending_respects_shard_before_limit(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    ids = [
        _insert_seg(conn, path=f"processed/audiobooks/studio/{i}.wav", provider="audiobooks")
        for i in range(1, 8)
    ]
    pending = list(iter_pending_segments(
        conn, config_hash=h, shard=parse_shard_spec("2/3"), limit=2
    ))
    conn.close()
    assert [p[0] for p in pending] == [ids[1], ids[4]]


def test_insert_or_ignore_prevents_duplicate(tmp_db: Path) -> None:
    cfg = WhisperConfig()
    conn = connect(tmp_db)
    h = ensure_asr_config(conn, cfg)
    seg_id = _insert_seg(conn, path="processed/audiobooks/studio/a.wav", provider="audiobooks")
    _insert_result(conn, cfg, h, _mock_result(segment_id=seg_id, provider="audiobooks"))
    _insert_result(conn, cfg, h, _mock_result(segment_id=seg_id, provider="audiobooks"))
    conn.commit()
    tc = conn.execute(
        "SELECT COUNT(*) AS n FROM transcript_candidates WHERE segment_id=?", (seg_id,)
    ).fetchone()
    wo = conn.execute(
        "SELECT COUNT(*) AS n FROM word_occurrences WHERE segment_id=?", (seg_id,)
    ).fetchone()
    conn.close()
    assert tc["n"] == 1
    assert wo["n"] == 2  # 2 mock words, only one insertion (second skipped via rowcount==0)
