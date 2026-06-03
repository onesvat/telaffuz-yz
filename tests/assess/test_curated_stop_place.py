from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest


def test_kar_context_buckets_only_high_confidence_plain_kar() -> None:
    from assess.curated_stop_place import classify_kar_context

    snow = classify_kar_context(["bugün", "kar", "yağdı"], 1)
    assert snow.meaning == "snow"
    assert snow.phone == "kʰ"
    assert snow.status == "trusted_auto"

    profit = classify_kar_context(["şirket", "kar", "etti"], 1)
    assert profit.meaning == "profit"
    assert profit.phone == "cʰ"
    assert profit.status == "trusted_auto"

    circumflex = classify_kar_context(["şirket", "kâr", "açıkladı"], 1, surface="kâr")
    assert circumflex.meaning == "profit"
    assert circumflex.phone == "cʰ"

    uncertain = classify_kar_context(["bugün", "kar", "gördüm"], 1)
    assert uncertain.status == "uncertain"
    assert uncertain.phone is None


def test_parse_curation_csv_validates_phone_and_status(tmp_path: Path) -> None:
    from assess.curated_stop_place import CSV_FIELDS, CurationError, parse_curation_csv

    path = tmp_path / "curation.csv"
    row = {
        "segment_id": "1",
        "audio_path": "/tmp/a.wav",
        "start_ms": "10",
        "end_ms": "30",
        "phone": "x",
        "word": "kar",
        "meaning": "",
        "prev_phone": "",
        "next_phone": "",
        "speaker_id": "spk1",
        "provider": "fixture",
        "split": "train",
        "label_source": "manual",
        "status": "accepted",
        "notes": "",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(CurationError, match="phone must be one of"):
        parse_curation_csv(path)

    row["phone"] = "kʰ"
    row["status"] = "maybe"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(CurationError, match="status must be one of"):
        parse_curation_csv(path)


def test_build_curated_reference_excludes_uncertain_and_reject_rows(tmp_path: Path) -> None:
    from assess.curated_stop_place import (
        CSV_FIELDS,
        build_curated_reference_db,
    )

    reference_db = tmp_path / "reference.sqlite"
    _write_reference_db(reference_db, phones=("kʰ",), rows_per_phone=4)
    audio_db = tmp_path / "audio.sqlite"
    _write_audio_db(audio_db)

    curation = tmp_path / "curation.csv"
    rows = [
        _curation_row(1, "kʰ", "trusted_auto"),
        _curation_row(2, "cʰ", "accepted"),
        _curation_row(3, "kʰ", "uncertain"),
        _curation_row(4, "kʰ", "reject"),
    ]
    with curation.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    out = tmp_path / "curated.sqlite"
    report = build_curated_reference_db(
        curation_csv=curation,
        reference_db=reference_db,
        audio_db=audio_db,
        output_db=out,
    )

    assert report.training_rows == 2
    assert report.inserted_rows == 2
    conn = sqlite3.connect(str(out))
    try:
        assert conn.execute("SELECT COUNT(*) FROM curation_manifest").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM phone_features").fetchone()[0] == 2
        phones = [
            row[0]
            for row in conn.execute("SELECT expected_phone FROM phone_features ORDER BY id")
        ]
        assert phones == ["kʰ", "cʰ"]
        included = conn.execute(
            "SELECT COUNT(*) FROM curation_manifest WHERE included_in_training = 1"
        ).fetchone()[0]
        assert included == 2
    finally:
        conn.close()


def test_mine_candidates_uses_timed_non_primary_words_when_primary_has_none(
    tmp_path: Path,
) -> None:
    from assess.curated_stop_place import mine_stop_place_candidates

    reference_db = tmp_path / "reference.sqlite"
    _write_reference_db(reference_db, phones=("kʰ",), rows_per_phone=1)
    audio_db = tmp_path / "audio.sqlite"
    _write_audio_db_for_mining(audio_db)

    rows = mine_stop_place_candidates(
        audio_db=audio_db,
        reference_db=reference_db,
        audio_root=tmp_path / "audio",
        providers=("fixture",),
    )

    assert len(rows) == 1
    assert rows[0].word == "kar"
    assert rows[0].meaning == "snow"
    assert rows[0].label_source == "kar_context_snow"


def test_overlay_merge_replaces_only_target_phone_payloads() -> None:
    from assess.curated_stop_place import (
        merge_authority_overlay,
        merge_calibration_overlay,
        merge_gmm_overlay,
    )

    base_gmm = {
        "feature_version": "base",
        "phones": {"a": {"phone": "a", "n": 10}, "kʰ": {"phone": "kʰ", "n": 10}},
    }
    repl_gmm = {"feature_version": "curated", "phones": {"kʰ": {"phone": "kʰ", "n": 99}}}
    merged_gmm = merge_gmm_overlay(base_gmm, repl_gmm, phones=("kʰ",), strict=True)
    assert merged_gmm["phones"]["a"]["n"] == 10
    assert merged_gmm["phones"]["kʰ"]["n"] == 99

    base_cal = {
        "feature_version": "base",
        "gmm_signature": "stale",
        "phones": {"a": {"phone": "a", "n": 10}, "kʰ": {"phone": "kʰ", "n": 10}},
    }
    repl_cal = {"feature_version": "curated", "phones": {"kʰ": {"phone": "kʰ", "n": 88}}}
    merged_cal = merge_calibration_overlay(base_cal, repl_cal, phones=("kʰ",), strict=True)
    assert merged_cal["phones"]["a"]["n"] == 10
    assert merged_cal["phones"]["kʰ"]["n"] == 88
    assert merged_cal["gmm_signature"] is None

    base_auth = {
        "fp_ceiling": 0.02,
        "contrasts": [
            _authority_row("a", "e", "base"),
            _authority_row("kʰ", "cʰ", "base"),
            _authority_row("kʰ", "e", "base"),
        ],
    }
    repl_auth = {
        "fp_ceiling": 0.02,
        "contrasts": [
            _authority_row("kʰ", "cʰ", "curated", n_target=3, n_alt=3),
            _authority_row("kʰ", "e", "empty", n_target=3, n_alt=0),
        ],
    }
    merged_auth = merge_authority_overlay(
        base_auth,
        repl_auth,
        phones=("kʰ", "cʰ"),
        scope="involving-target",
        skip_zero_sample_rows=True,
    )
    by_pair = {
        (row["target"], row["alternative"]): row["reason"]
        for row in merged_auth["contrasts"]
    }
    assert by_pair[("a", "e")] == "base"
    assert by_pair[("kʰ", "cʰ")] == "curated"
    assert by_pair[("kʰ", "e")] == "base"


def test_tiny_curated_reference_builds_four_phone_gmm(tmp_path: Path) -> None:
    from assess.curated_stop_place import CSV_FIELDS, build_curated_reference_db
    from assess.stats import BuildConfig, build_atlas_gmm

    reference_db = tmp_path / "reference.sqlite"
    _write_reference_db(reference_db, phones=("kʰ", "cʰ", "k", "c"), rows_per_phone=8)
    audio_db = tmp_path / "audio.sqlite"
    _write_audio_db(audio_db)

    curation = tmp_path / "curation.csv"
    rows = []
    ref_id = 1
    for phone in ("kʰ", "cʰ", "k", "c"):
        for _ in range(8):
            rows.append(_curation_row(ref_id, phone, "accepted"))
            ref_id += 1
    with curation.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    curated = tmp_path / "curated.sqlite"
    build_curated_reference_db(
        curation_csv=curation,
        reference_db=reference_db,
        audio_db=audio_db,
        output_db=curated,
    )
    config = BuildConfig(
        providers=("fixture",),
        min_samples=4,
        confidence_floor=0.5,
        per_phone_cap=0,
        train_buckets="0..99",
        test_buckets="",
        d_quantile=0.95,
        seed=0,
        min_feature_coverage=0.6,
    )
    atlas = build_atlas_gmm(curated, "tiny_curated_v1", config)
    assert set(atlas.phones) == {"kʰ", "cʰ", "k", "c"}
    assert all(phone.n >= 8 for phone in atlas.phones.values())


def _curation_row(ref_id: int, phone: str, status: str) -> dict[str, str]:
    return {
        "segment_id": str(ref_id),
        "audio_path": f"/tmp/{ref_id}.wav",
        "start_ms": "10",
        "end_ms": "40",
        "phone": phone,
        "word": "kar",
        "meaning": "",
        "prev_phone": "",
        "next_phone": "",
        "speaker_id": f"spk{ref_id}",
        "provider": "fixture",
        "split": "train",
        "label_source": "manual",
        "status": status,
        "notes": f"ref_feature_id={ref_id}",
    }


def _authority_row(
    target: str,
    alternative: str,
    reason: str,
    *,
    n_target: int = 1,
    n_alt: int = 1,
) -> dict[str, object]:
    return {
        "target": target,
        "alternative": alternative,
        "reason": reason,
        "override_policy": "quality_only",
        "reliability": {
            "n_target": {"train": n_target, "dev": 0, "test": 0},
            "n_alternative": {"train": n_alt, "dev": 0, "test": 0},
        },
    }


def _write_audio_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE segment_alignments (id INTEGER PRIMARY KEY, segment_id INTEGER)"
    )
    conn.commit()
    conn.close()


def _write_audio_db_for_mining(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE segments (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            provider TEXT NOT NULL,
            speaker_id TEXT NOT NULL
        );
        CREATE TABLE segment_alignments (
            id INTEGER PRIMARY KEY,
            segment_id INTEGER NOT NULL,
            alignment_json TEXT NOT NULL
        );
        CREATE TABLE transcript_candidates (
            id INTEGER PRIMARY KEY,
            segment_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            is_primary INTEGER NOT NULL
        );
        CREATE TABLE word_occurrences (
            id INTEGER PRIMARY KEY,
            segment_id INTEGER NOT NULL,
            transcript_candidate_id INTEGER NOT NULL,
            word_index INTEGER NOT NULL,
            word_surface TEXT NOT NULL,
            word_norm TEXT NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO segments (id, path, provider, speaker_id) VALUES (1, 'a.wav', 'fixture', 'spk1')"
    )
    conn.execute(
        """
        INSERT INTO segment_alignments (id, segment_id, alignment_json)
        VALUES (
            1,
            1,
            '{"phones":[{"expected_phone":"kʰ"},{"expected_phone":"a"}]}'
        )
        """
    )
    conn.execute(
        "INSERT INTO transcript_candidates (id, segment_id, text, is_primary) VALUES (10, 1, 'Kar yağdı.', 1)"
    )
    conn.execute(
        "INSERT INTO transcript_candidates (id, segment_id, text, is_primary) VALUES (11, 1, 'Kar yağdı.', 0)"
    )
    conn.executemany(
        """
        INSERT INTO word_occurrences (
            segment_id, transcript_candidate_id, word_index, word_surface,
            word_norm, start_ms, end_ms
        ) VALUES (1, 11, ?, ?, ?, ?, ?)
        """,
        [(0, "Kar", "kar", 0, 200), (1, "yağdı", "yağdı", 200, 400)],
    )
    conn.commit()
    conn.close()


def _write_reference_db(
    path: Path, *, phones: tuple[str, ...], rows_per_phone: int
) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE phone_features (
            id INTEGER PRIMARY KEY,
            feature_version TEXT NOT NULL,
            expected_phone TEXT NOT NULL,
            provider TEXT,
            speaker_id TEXT,
            start_ms INTEGER,
            end_ms INTEGER,
            duration_ms REAL,
            feature_confidence REAL,
            formant_success INTEGER,
            vot_ms REAL,
            burst_centroid_hz REAL,
            burst_spectral_skew REAL,
            f2_transition_slope_hz_per_ms REAL,
            f2_locus_hz REAL,
            voiced_fraction REAL,
            closure_voicing_ratio REAL,
            segment_alignment_id INTEGER,
            phone_index INTEGER
        )
        """
    )
    rows = []
    ref_id = 1
    for phone_idx, phone in enumerate(phones):
        for item_idx in range(rows_per_phone):
            offset = phone_idx * 100 + item_idx
            rows.append(
                (
                    ref_id,
                    "assess_coach_reference_v1",
                    phone,
                    "fixture",
                    f"spk{ref_id}",
                    10,
                    40,
                    30.0,
                    1.0,
                    0,
                    12.0 + offset,
                    2000.0 + offset,
                    0.2 + item_idx * 0.01,
                    10.0 + phone_idx,
                    1800.0 + offset,
                    0.05 + item_idx * 0.01,
                    0.1 + phone_idx * 0.05,
                    ref_id,
                    0,
                )
            )
            ref_id += 1
    conn.executemany(
        """
        INSERT INTO phone_features (
            id, feature_version, expected_phone, provider, speaker_id,
            start_ms, end_ms, duration_ms, feature_confidence, formant_success,
            vot_ms, burst_centroid_hz, burst_spectral_skew,
            f2_transition_slope_hz_per_ms, f2_locus_hz, voiced_fraction,
            closure_voicing_ratio, segment_alignment_id, phone_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
