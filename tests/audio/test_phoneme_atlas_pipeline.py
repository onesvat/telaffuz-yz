from __future__ import annotations

import csv
import json
import math
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from audio.db import connect, now_iso, transaction
from audio.schema import init_db


def _segment(
    conn: sqlite3.Connection,
    *,
    path: str,
    provider: str,
    duration_s: float = 10.0,
    speaker_id: str | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO segments(path, provider, duration_s, sample_rate, channels,
                                speaker_id, speaker_source, created_at)
           VALUES (?, ?, ?, 16000, 1, ?, 'fixture', ?)""",
        (path, provider, duration_s, speaker_id or f"{provider}_speaker", now_iso()),
    )
    return int(cur.lastrowid)


def _transcript(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    source: str,
    text: str,
    is_primary: int = 1,
) -> None:
    conn.execute(
        """INSERT INTO transcript_candidates(segment_id, source, model_version,
                                             text, is_primary, created_at)
           VALUES (?, ?, 'fixture', ?, ?, ?)""",
        (segment_id, source, text, is_primary, now_iso()),
    )


def _fixture_db(tmp_path: Path) -> Path:
    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            cv = _segment(
                conn,
                path="processed/common_voice/cv_001.wav",
                provider="common_voice",
            )
            _transcript(conn, segment_id=cv, source="publisher_gt", text="Merhaba")

            issai = _segment(
                conn,
                path="processed/issai_tsc/train/issai_001.wav",
                provider="issai_tsc",
            )
            _transcript(conn, segment_id=issai, source="publisher_gt", text="Anne")

            audio = _segment(
                conn,
                path="processed/audiobooks/Book A/Book A/0001.wav",
                provider="audiobooks",
            )
            _transcript(conn, segment_id=audio, source="whisper_large_v3_turbo", text="Baba")

            studio = _segment(
                conn,
                path="processed/audiobooks/studio/ders-1/001.wav",
                provider="audiobooks",
            )
            _transcript(conn, segment_id=studio, source="whisper_large_v3_turbo", text="Tepe")

            drama = _segment(
                conn,
                path="processed/audiobooks/drama/sipsevdi/001/001_0001.wav",
                provider="audiobooks",
            )
            _transcript(conn, segment_id=drama, source="whisper_large_v3_turbo", text="Su")

            drama2 = _segment(
                conn,
                path="processed/audiobooks/drama/tarih-soylesileri/001/001_0001.wav",
                provider="audiobooks",
            )
            _transcript(
                conn,
                segment_id=drama2,
                source="whisper_large_v3_turbo",
                text="Ders",
            )
            youtube = _segment(
                conn,
                path="processed/youtube/show/001.wav",
                provider="youtube",
            )
            _transcript(conn, segment_id=youtube, source="whisper_large_v3_turbo", text="Para")
    finally:
        conn.close()
    return db


def test_pilot_manifest_uses_ready_segments_and_quota_seed(tmp_path: Path) -> None:
    from audio.atlas_pilot import PilotSpec, build_pilot_manifest_rows

    conn = connect(_fixture_db(tmp_path))
    try:
        rows = build_pilot_manifest_rows(
            conn,
            PilotSpec(
                provider_seconds={
                    "common_voice": 10.0,
                    "issai_tsc": 10.0,
                    "audiobooks": 10.0,
                },
                rare_seconds=0.0,
                seed=42,
            ),
        )
        rows_again = build_pilot_manifest_rows(
            conn,
            PilotSpec(
                provider_seconds={
                    "common_voice": 10.0,
                    "issai_tsc": 10.0,
                    "audiobooks": 10.0,
                },
                rare_seconds=0.0,
                seed=42,
            ),
        )
    finally:
        conn.close()

    assert [row.as_dict() for row in rows] == [row.as_dict() for row in rows_again]
    assert {row.provider for row in rows} == {
        "common_voice",
        "issai_tsc",
        "audiobooks",
    }
    assert all(row.needs_whisper is False for row in rows)
    assert not any("youtube" in row.path for row in rows)
    assert {row.selection_reason for row in rows} == {
        "pilot_common_voice_seed42",
        "pilot_issai_tsc_seed42",
        "pilot_audiobooks_seed42",
    }


def test_pilot_manifest_adds_rare_allophone_supplement(tmp_path: Path) -> None:
    from audio.atlas_pilot import PilotSpec, build_pilot_manifest_rows

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            rare = _segment(
                conn,
                path="processed/common_voice/rare_para.wav",
                provider="common_voice",
                duration_s=8.0,
            )
            _transcript(conn, segment_id=rare, source="publisher_gt", text="Para")
            plain = _segment(
                conn,
                path="processed/common_voice/plain_su.wav",
                provider="common_voice",
                duration_s=8.0,
            )
            _transcript(conn, segment_id=plain, source="publisher_gt", text="Su")

        rows = build_pilot_manifest_rows(
            conn,
            PilotSpec(provider_seconds={}, rare_seconds=8.0, seed=42),
        )
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0].path == "processed/common_voice/rare_para.wav"
    assert rows[0].selection_reason == "pilot_rare_allophone_seed42"


def test_allophant_mapping_reports_unknown_inventory_symbols() -> None:
    from audio.atlas_pilot import map_allophant_inventory

    mapped = map_allophant_inventory(["p", "ø", "g", "x"])

    assert mapped.phones == ("p", "œ", "ɡ")
    assert mapped.unknown == ("x",)


def test_mms_emission_plan_respects_manifest_cache_limit_and_shard(tmp_path: Path) -> None:
    from audio.mms_runner import plan_mms_emission_jobs

    manifest = tmp_path / "pilot_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "segment_id,path,provider,duration_s,speaker_id,transcript_source,needs_whisper,role,selection_reason",
                "10,processed/common_voice/a.wav,common_voice,3,spk,publisher_gt,false,atlas_pilot,pilot",
                "11,processed/common_voice/b.wav,common_voice,3,spk,publisher_gt,false,atlas_pilot,pilot",
                "12,processed/common_voice/c.wav,common_voice,3,spk,publisher_gt,false,atlas_pilot,pilot",
            ]
        ),
        encoding="utf-8",
    )
    emission_dir = tmp_path / "emissions"
    emission_dir.mkdir()
    (emission_dir / "10.npz").write_bytes(b"cached")

    jobs = plan_mms_emission_jobs(
        manifest,
        emission_dir=emission_dir,
        shard=(1, 2),
        limit=1,
    )

    assert [job.segment_id for job in jobs] == [11]
    assert jobs[0].audio_path == Path("processed/common_voice/b.wav")


def test_alignment_upsert_is_idempotent_and_keeps_phone_layers(tmp_path: Path) -> None:
    from audio.alignment import AlignmentItem, load_alignment_phone_instances
    from audio.alignment import upsert_segment_alignment

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            seg_id = _segment(conn, path="processed/common_voice/para.wav", provider="common_voice")

        items = (
            AlignmentItem(
                expected_phone="pʰ",
                alignment_phone="p",
                observed_model_phone="p",
                start_ms=0,
                end_ms=55,
                confidence=0.93,
            ),
            AlignmentItem(
                expected_phone="a",
                alignment_phone="a",
                observed_model_phone="a",
                start_ms=55,
                end_ms=120,
                confidence=0.91,
            ),
        )
        first_id = upsert_segment_alignment(
            conn,
            segment_id=seg_id,
            expected_phones=("pʰ", "a"),
            items=items,
            aligner_version="mms-fixture",
        )
        second_id = upsert_segment_alignment(
            conn,
            segment_id=seg_id,
            expected_phones=("pʰ", "a"),
            items=items,
            aligner_version="mms-fixture",
        )
        instances = load_alignment_phone_instances(conn, aligner_version="mms-fixture")
    finally:
        conn.close()

    assert first_id == second_id
    assert len(instances) == 2
    assert instances[0]["expected_phone"] == "pʰ"
    assert instances[0]["alignment_phone"] == "p"
    assert instances[0]["observed_model_phone"] == "p"
    assert instances[0]["manual_phone"] is None
    assert instances[0]["evidence_type"] == "manual-only"
    assert json.loads(instances[0]["feature_json"])["duration_ms"] == 55


def test_mms_fa_projection_is_lossy_but_keeps_expected_layers() -> None:
    from audio.alignment import project_expected_phones_for_mms_fa

    projected = project_expected_phones_for_mms_fa(
        ("ˈ", "pʰ", "æ", "ɾ̞̊", "β", "t͡ʃ", "d͡ʒ", "ɯ", "œ", "y", "aː")
    )

    assert [item.expected_phone for item in projected] == [
        "pʰ",
        "æ",
        "ɾ̞̊",
        "β",
        "t͡ʃ",
        "d͡ʒ",
        "ɯ",
        "œ",
        "y",
        "aː",
    ]
    assert [item.alignment_phone for item in projected] == [
        "p",
        "e",
        "ɾ",
        "v",
        "t͡ʃ",
        "d͡ʒ",
        "ɯ",
        "œ",
        "y",
        "aː",
    ]
    assert [item.model_token for item in projected] == [
        "p",
        "e",
        "r",
        "v",
        "c",
        "j",
        "i",
        "o",
        "u",
        "a",
    ]


def test_mms_fa_projection_covers_inventory_except_stress() -> None:
    from g2p.constants import ALL_PHONEMES, STRESS

    from audio.alignment import project_expected_phones_for_mms_fa

    projected = project_expected_phones_for_mms_fa(sorted(ALL_PHONEMES))

    assert {item.expected_phone for item in projected} == set(ALL_PHONEMES - {STRESS})
    assert all(item.model_token is not None for item in projected)


def test_mms_fa_projection_drops_word_separator() -> None:
    from g2p.constants import CTC_SPC

    from audio.alignment import project_expected_phones_for_mms_fa

    projected = project_expected_phones_for_mms_fa(("a", CTC_SPC, "b"))

    assert [item.expected_phone for item in projected] == ["a", "b"]
    assert [item.model_token for item in projected] == ["a", "b"]


def test_alignment_json_includes_model_projection_token() -> None:
    from audio.alignment import AlignmentItem, parse_alignment_json

    payload = AlignmentItem(
        expected_phone="ɾ̞̊",
        alignment_phone="ɾ",
        observed_model_phone="r",
        start_ms=0,
        end_ms=30,
        confidence=0.8,
        alignment_model_token="r",
    ).as_json()

    assert payload["alignment_model_token"] == "r"

    parsed = parse_alignment_json(
        json.dumps(
            {
                "schema_version": 1,
                "phones": [payload],
            },
            ensure_ascii=False,
        )
    )

    assert parsed[0].alignment_model_token == "r"


def test_mms_fa_spans_convert_to_layered_alignment_items() -> None:
    from audio.alignment import alignment_items_from_mms_fa_spans
    from audio.alignment import project_expected_phones_for_mms_fa

    projected = project_expected_phones_for_mms_fa(("pʰ", "æ", "ɾ̞̊"))
    spans = [
        SimpleNamespace(start=0, end=5, score=0.9),
        SimpleNamespace(start=5, end=9, score=0.8),
        SimpleNamespace(start=9, end=10, score=0.7),
    ]

    items = alignment_items_from_mms_fa_spans(
        projected,
        spans,
        frame_count=10,
        duration_s=1.0,
    )

    assert [item.expected_phone for item in items] == ["pʰ", "æ", "ɾ̞̊"]
    assert [item.alignment_phone for item in items] == ["p", "e", "ɾ"]
    assert [item.observed_model_phone for item in items] == ["p", "e", "r"]
    assert [item.alignment_model_token for item in items] == ["p", "e", "r"]
    assert [(item.start_ms, item.end_ms) for item in items] == [
        (0, 500),
        (500, 900),
        (900, 1000),
    ]


def test_alignment_fixture_builder_drops_stress_without_shifting_targets() -> None:
    from audio.alignment import alignment_items_from_expected_fixture

    items = alignment_items_from_expected_fixture(("ˈ", "pʰ", "a"), frame_ms=25)

    assert [item.expected_phone for item in items] == ["pʰ", "a"]
    assert [item.alignment_phone for item in items] == ["p", "a"]
    assert [item.start_ms for item in items] == [0, 25]


def test_mms_fa_spans_clamp_edges_to_signal_bounds() -> None:
    from audio.alignment import alignment_items_from_mms_fa_spans
    from audio.alignment import project_expected_phones_for_mms_fa

    projected = project_expected_phones_for_mms_fa(("pʰ", "æ", "ɾ̞̊"))
    spans = [
        SimpleNamespace(start=2, end=4, score=0.9),
        SimpleNamespace(start=5, end=7, score=0.8),
        SimpleNamespace(start=8, end=9, score=0.7),
    ]

    items = alignment_items_from_mms_fa_spans(
        projected,
        spans,
        frame_count=10,
        duration_s=1.0,
        signal_bounds_frames=(1, 9),
    )

    # Edge spans should clamp to the supplied signal bounds, not 0 / frame_count.
    assert items[0].start_ms == 100
    assert items[-1].end_ms == 900


def test_mms_fa_spans_default_behavior_unchanged() -> None:
    from audio.alignment import alignment_items_from_mms_fa_spans
    from audio.alignment import project_expected_phones_for_mms_fa

    projected = project_expected_phones_for_mms_fa(("pʰ", "æ"))
    spans = [
        SimpleNamespace(start=2, end=4, score=0.9),
        SimpleNamespace(start=6, end=8, score=0.8),
    ]
    items = alignment_items_from_mms_fa_spans(
        projected, spans, frame_count=10, duration_s=1.0
    )

    assert items[0].start_ms == 0
    assert items[-1].end_ms == 1000


def test_feature_extraction_schema_and_null_formants() -> None:
    from audio.features import extract_phone_feature, feature_to_json

    sample_rate = 16_000
    t = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * math.pi * 220 * t)

    feature = extract_phone_feature(
        audio,
        sample_rate=sample_rate,
        start_s=0.0,
        end_s=0.05,
        phone="p",
    )
    payload = json.loads(feature_to_json(feature))

    assert payload["duration_ms"] == 50
    assert payload["rms"] > 0
    assert payload["f1_hz"] is None
    assert payload["f2_hz"] is None
    assert payload["f3_hz"] is None
    assert payload["vot_ms"] is None
    assert len(payload["mfcc"]) == 13


def test_textgrid_manual_import_and_coverage_summary(tmp_path: Path) -> None:
    from audio.manual import manual_coverage_summary, parse_textgrid_phone_intervals

    textgrid = tmp_path / "manual.TextGrid"
    textgrid.write_text(
        """
File type = "ooTextFile"
Object class = "TextGrid"

item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.04
        text = "pʰ|accepted"
    intervals [2]:
        xmin = 0.04
        xmax = 0.08
        text = "tʰ|uncertain"
    intervals [3]:
        xmin = 0.08
        xmax = 0.12
        text = "kʰ|reject"
""",
        encoding="utf-8",
    )

    intervals = parse_textgrid_phone_intervals(textgrid, segment_id=7)
    coverage = manual_coverage_summary(intervals, minimum_per_phone=1)

    assert [item.status for item in intervals] == ["accepted", "uncertain", "reject"]
    assert coverage["pʰ"]["accepted"] == 1
    assert coverage["tʰ"]["accepted"] == 0
    assert coverage["kʰ"]["accepted"] == 0
    assert coverage["pʰ"]["meets_minimum"] is True
    assert coverage["tʰ"]["meets_minimum"] is False


def test_reviewed_textgrid_summary_counts_files_labels_and_malformed(
    tmp_path: Path,
) -> None:
    from audio.manual import summarize_textgrid_paths

    reviewed_dir = tmp_path / "reviewed_textgrids"
    reviewed_dir.mkdir()
    (reviewed_dir / "7.TextGrid").write_text(
        """
File type = "ooTextFile"
Object class = "TextGrid"

item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.04
        text = "pʰ|accepted"
    intervals [2]:
        xmin = 0.04
        xmax = 0.08
        text = "β̞|reject"
""",
        encoding="utf-8",
    )
    (reviewed_dir / "8.TextGrid").write_text(
        """
File type = "ooTextFile"
Object class = "TextGrid"

item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.04
        text = "kʰ|maybe"
""",
        encoding="utf-8",
    )

    summary = summarize_textgrid_paths(sorted(reviewed_dir.glob("*.TextGrid")))

    assert summary.file_count == 2
    assert summary.label_count == 3
    assert summary.malformed_label_count == 1
    assert summary.status_counts == {"accepted": 1, "reject": 1, "uncertain": 1}
    assert summary.phone_status_counts[("pʰ", "accepted")] == 1
    assert summary.phone_status_counts[("β̞", "reject")] == 1


def test_import_reviewed_textgrids_inserts_manual_phone_annotations(
    tmp_path: Path,
) -> None:
    from audio.manual import import_reviewed_textgrids

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    textgrid = tmp_path / "7.TextGrid"
    textgrid.write_text(
        """
File type = "ooTextFile"
Object class = "TextGrid"

item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.04
        text = "pʰ|accepted"
    intervals [2]:
        xmin = 0.04
        xmax = 0.08
        text = "β̞|reject"
""",
        encoding="utf-8",
    )
    try:
        result = import_reviewed_textgrids(conn, [textgrid], actor="manual:praat")
        conn.commit()
        rows = conn.execute(
            """SELECT label, actor, value_json
               FROM manual_annotations
               ORDER BY id"""
        ).fetchall()
    finally:
        conn.close()

    assert result.inserted_count == 2
    assert result.summary.file_count == 1
    assert result.summary.label_count == 2
    assert result.summary.malformed_label_count == 0
    assert [(row["label"], row["actor"]) for row in rows] == [
        ("manual_phone", "manual:praat"),
        ("manual_phone", "manual:praat"),
    ]
    assert json.loads(rows[0]["value_json"])["status"] == "accepted"
    assert json.loads(rows[1]["value_json"])["status"] == "reject"


def test_manual_queue_candidates_select_target_per_phone(tmp_path: Path) -> None:
    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.manual import build_manual_queue_candidates

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            seg_a = _segment(
                conn,
                path="processed/common_voice/a.wav",
                provider="common_voice",
                duration_s=2.0,
            )
            seg_b = _segment(
                conn,
                path="processed/common_voice/b.wav",
                provider="common_voice",
                duration_s=2.0,
            )
            upsert_segment_alignment(
                conn,
                segment_id=seg_a,
                expected_phones=("pʰ", "a", "tʰ"),
                items=(
                    AlignmentItem("pʰ", "p", "p", 10, 70, 0.9, "p"),
                    AlignmentItem("a", "a", "a", 70, 130, 0.8, "a"),
                    AlignmentItem("tʰ", "t", "t", 130, 180, 0.7, "t"),
                ),
                aligner_version="mms-fixture",
            )
            upsert_segment_alignment(
                conn,
                segment_id=seg_b,
                expected_phones=("pʰ",),
                items=(
                    AlignmentItem("pʰ", "p", "p", 20, 80, 0.5, "p"),
                ),
                aligner_version="mms-fixture",
            )

        candidates = build_manual_queue_candidates(
            conn,
            aligner_version="mms-fixture",
            target_per_phone=1,
            target_phones=("pʰ", "tʰ"),
        )
    finally:
        conn.close()

    assert [(row.expected_phone, row.segment_id) for row in candidates] == [
        ("pʰ", seg_a),
        ("tʰ", seg_a),
    ]
    assert candidates[0].path == "processed/common_voice/a.wav"
    assert candidates[0].candidate_reason == "manual-only_top_confidence"


def test_manual_queue_candidates_balance_provider_quotas(tmp_path: Path) -> None:
    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.manual import build_manual_queue_candidates

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            segments: list[tuple[int, str]] = []
            fixtures = [
                ("processed/audiobooks/drama/a.wav", "audiobooks", 0.99),
                ("processed/audiobooks/drama/b.wav", "audiobooks", 0.98),
                ("processed/audiobooks/a.wav", "audiobooks", 0.97),
                ("processed/common_voice/a.wav", "common_voice", 0.96),
                ("processed/audiobooks/studio/a.wav", "audiobooks", 0.95),
                ("processed/issai_tsc/a.wav", "issai_tsc", 0.94),
            ]
            for index, (path, provider, confidence) in enumerate(fixtures, start=1):
                segment_id = _segment(
                    conn,
                    path=path,
                    provider=provider,
                    duration_s=2.0,
                )
                segments.append((segment_id, provider))
                upsert_segment_alignment(
                    conn,
                    segment_id=segment_id,
                    expected_phones=("pʰ",),
                    items=(
                        AlignmentItem(
                            "pʰ",
                            "p",
                            "p",
                            index * 10,
                            index * 10 + 20,
                            confidence,
                            "p",
                        ),
                    ),
                    aligner_version="mms-fixture",
                )

        candidates = build_manual_queue_candidates(
            conn,
            aligner_version="mms-fixture",
            target_per_phone=3,
            target_phones=("pʰ",),
            provider_quotas={
                "audiobooks": 1,
                "common_voice": 1,
                "issai_tsc": 1,
            },
            provider_priority=(
                "audiobooks",
                "common_voice",
                "issai_tsc",
            ),
            confidence_floor=0.9,
        )
    finally:
        conn.close()

    by_segment = {segment_id: provider for segment_id, provider in segments}
    assert [by_segment[row.segment_id] for row in candidates] == [
        "audiobooks",
        "common_voice",
        "issai_tsc",
    ]
    assert {row.candidate_reason for row in candidates} == {
        "manual-only_provider_balanced"
    }


def test_write_manual_queue_textgrids_make_praat_intervals(tmp_path: Path) -> None:
    from audio.manual import ManualQueueCandidate, write_manual_queue_textgrids

    candidates = [
        ManualQueueCandidate(
            segment_id=7,
            path="processed/common_voice/a.wav",
            expected_phone="pʰ",
            alignment_phone="p",
            start_ms=100,
            end_ms=180,
            confidence=0.9,
            candidate_reason="manual-only_top_confidence",
        ),
        ManualQueueCandidate(
            segment_id=7,
            path="processed/common_voice/a.wav",
            expected_phone="tʰ",
            alignment_phone="t",
            start_ms=300,
            end_ms=360,
            confidence=0.8,
            candidate_reason="manual-only_top_confidence",
        ),
    ]

    written = write_manual_queue_textgrids(
        candidates,
        out_dir=tmp_path,
        segment_durations={7: 0.5},
    )

    text = written[0].read_text(encoding="utf-8")
    assert 'name = "phones"' in text
    assert 'text = "pʰ|uncertain"' in text
    assert 'text = "tʰ|uncertain"' in text
    assert "xmax = 0.5" in text




def test_export_writes_required_columns_and_evidence_csv(tmp_path: Path) -> None:
    from audio.atlas_export import PhoneInstance, coverage_by_phone
    from audio.atlas_export import write_g2p_audio_evidence_csv, write_phone_instances

    rows = [
        PhoneInstance(
            segment_id=1,
            word_occurrence_id=None,
            expected_phone="pʰ",
            alignment_phone="p",
            observed_model_phone="p",
            manual_phone="pʰ",
            evidence_type="manual-only",
            feature_json='{"duration_ms":55}',
        ),
        PhoneInstance(
            segment_id=1,
            word_occurrence_id=None,
            expected_phone="a",
            alignment_phone="a",
            observed_model_phone="a",
            manual_phone=None,
            evidence_type="model-observable",
            feature_json='{"duration_ms":65}',
        ),
    ]

    parquet_path = tmp_path / "phone_instances.parquet"
    csv_path = tmp_path / "g2p_audio_evidence.csv"
    write_phone_instances(rows, parquet_path)
    write_g2p_audio_evidence_csv(rows, csv_path)

    import pandas as pd

    table = pd.read_parquet(parquet_path)
    assert list(table.columns) == [
        "segment_id",
        "word_occurrence_id",
        "expected_phone",
        "alignment_phone",
        "observed_model_phone",
        "manual_phone",
        "evidence_type",
        "feature_json",
    ]
    assert coverage_by_phone(rows)["pʰ"]["manual_phone_count"] == 1
    with csv_path.open(encoding="utf-8", newline="") as fh:
        evidence = list(csv.DictReader(fh))
    assert evidence[0]["evidence_class"] == "supported"


def test_export_overlays_accepted_manual_annotations(tmp_path: Path) -> None:
    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.atlas_export import instances_from_alignment_db
    from audio.manual import ManualPhoneInterval, import_textgrid_intervals

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            seg_id = _segment(
                conn,
                path="processed/common_voice/manual.wav",
                provider="common_voice",
                duration_s=1.0,
            )
            upsert_segment_alignment(
                conn,
                segment_id=seg_id,
                expected_phones=("pʰ", "β̞"),
                items=(
                    AlignmentItem("pʰ", "p", "p", 100, 180, 0.9, "p"),
                    AlignmentItem("β̞", "v", "v", 200, 260, 0.8, "v"),
                ),
                aligner_version="mms-fixture",
            )
            import_textgrid_intervals(
                conn,
                [
                    ManualPhoneInterval(seg_id, 0.100, 0.180, "pʰ", "accepted"),
                    ManualPhoneInterval(seg_id, 0.200, 0.260, "β̞", "reject"),
                ],
            )

        rows = instances_from_alignment_db(conn, aligner_version="mms-fixture")
    finally:
        conn.close()

    by_phone = {row.expected_phone: row for row in rows}
    assert by_phone["pʰ"].manual_phone == "pʰ"
    assert by_phone["β̞"].manual_phone is None


def test_manual_review_report_mentions_beta_decision_and_layer_scope() -> None:
    from audio.manual import ManualTextGridSummary, render_manual_review_report

    summary = ManualTextGridSummary(
        file_count=121,
        label_count=140,
        malformed_label_count=0,
        status_counts={"accepted": 120, "uncertain": 12, "reject": 8},
        phone_status_counts={
            ("pʰ", "accepted"): 19,
            ("pʰ", "uncertain"): 1,
            ("tʰ", "accepted"): 19,
            ("tʰ", "reject"): 1,
            ("kʰ", "accepted"): 17,
            ("kʰ", "uncertain"): 3,
            ("cʰ", "accepted"): 17,
            ("cʰ", "uncertain"): 2,
            ("cʰ", "reject"): 1,
            ("ɾ̞̊", "accepted"): 17,
            ("ɾ̞̊", "uncertain"): 2,
            ("ɾ̞̊", "reject"): 1,
            ("β", "accepted"): 17,
            ("β", "uncertain"): 3,
            ("β̞", "accepted"): 14,
            ("β̞", "uncertain"): 1,
            ("β̞", "reject"): 5,
        },
    )

    report = render_manual_review_report(summary)

    assert "# Manual Phoneme Atlas Review" in report
    assert "| `β̞` | 14 | 1 | 5 | Accepted |" in report
    assert "`β̞` was accepted into the manual atlas with 14 accepted, 1 uncertain," in report
    assert "`manual_phone` evidence" in report
    assert "do not replace model output" in report


def test_pilot_report_summarizes_alignment_and_manual_evidence(tmp_path: Path) -> None:
    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.atlas_export import render_pilot_report
    from audio.manual import ManualPhoneInterval, import_textgrid_intervals

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            seg_id = _segment(
                conn,
                path="processed/common_voice/pilot.wav",
                provider="common_voice",
                duration_s=1.0,
            )
            upsert_segment_alignment(
                conn,
                segment_id=seg_id,
                expected_phones=("pʰ", "a"),
                items=(
                    AlignmentItem("pʰ", "p", "p", 0, 50, 0.9, "p"),
                    AlignmentItem("a", "a", "a", 50, 120, 0.8, "a"),
                ),
                aligner_version="mms-fixture",
            )
            import_textgrid_intervals(
                conn,
                [ManualPhoneInterval(seg_id, 0.0, 0.05, "pʰ", "accepted")],
            )

        report = render_pilot_report(conn, aligner_version="mms-fixture")
    finally:
        conn.close()

    assert "# Pilot Atlas Evidence Report" in report
    assert "- Aligned segments: 1" in report
    assert "- Phone windows: 2" in report
    assert "| `pʰ` | manual-only | 1 | 1 | supported |" in report
    assert "Missing Acoustic Feature" in report


def test_full_atlas_summary_includes_all_inventory_and_beta_supported(
    tmp_path: Path,
) -> None:
    from g2p.constants import ALL_PHONEMES

    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.atlas_export import build_phone_summary_rows, write_phone_summary_csv
    from audio.manual import ManualPhoneInterval, import_textgrid_intervals

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            seg_id = _segment(
                conn,
                path="processed/common_voice/full.wav",
                provider="common_voice",
                duration_s=1.0,
            )
            upsert_segment_alignment(
                conn,
                segment_id=seg_id,
                expected_phones=("β̞", "a"),
                items=(
                    AlignmentItem("β̞", "v", "v", 0, 40, 0.9, "v"),
                    AlignmentItem("a", "a", "a", 40, 100, 0.95, "a"),
                ),
                aligner_version="mms-fixture",
            )
            import_textgrid_intervals(
                conn,
                [ManualPhoneInterval(seg_id, 0.0, 0.04, "β̞", "accepted")],
            )

        rows = build_phone_summary_rows(conn, aligner_version="mms-fixture")
        out_path = tmp_path / "phone_summary.csv"
        write_phone_summary_csv(rows, out_path)
    finally:
        conn.close()

    assert len(rows) == len(ALL_PHONEMES)
    by_phone = {row["expected_phone"]: row for row in rows}
    assert by_phone["β̞"]["manual_accepted"] == 1
    assert by_phone["β̞"]["risk_class"] == "supported"
    assert by_phone["β̞"]["evidence_class"] == "supported"
    assert by_phone["ˈ"]["risk_class"] == "low_count"

    with out_path.open(encoding="utf-8", newline="") as fh:
        exported = list(csv.DictReader(fh))
    assert len(exported) == len(ALL_PHONEMES)
    assert exported[0]["expected_phone"]


def test_stress_syllable_instances_map_word_chunks_and_single_syllable(
    tmp_path: Path,
) -> None:
    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.atlas_export import (
        build_phone_summary_rows,
        build_stress_syllable_instances,
    )

    db = tmp_path / "audio.sqlite"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            seg_id = _segment(
                conn,
                path="processed/common_voice/stress.wav",
                provider="common_voice",
                duration_s=1.0,
            )
            upsert_segment_alignment(
                conn,
                segment_id=seg_id,
                expected_phones=("ˈ", "a", "n", "<spc>", "m", "a", "ˈ", "s", "a"),
                items=(
                    AlignmentItem("a", "a", "a", 0, 100, 0.9, "a"),
                    AlignmentItem("n", "n", "n", 100, 200, 0.9, "n"),
                    AlignmentItem("m", "m", "m", 300, 360, 0.9, "m"),
                    AlignmentItem("a", "a", "a", 360, 440, 0.9, "a"),
                    AlignmentItem("s", "s", "s", 440, 500, 0.9, "s"),
                    AlignmentItem("a", "a", "a", 500, 620, 0.9, "a"),
                ),
                aligner_version="mms-fixture",
            )

        rows = build_stress_syllable_instances(conn, aligner_version="mms-fixture")
        summary = build_phone_summary_rows(
            conn,
            aligner_version="mms-fixture",
            stress_rows=rows,
        )
    finally:
        conn.close()

    assert len(rows) == 2
    first, second = rows
    assert first.word_index == 0
    assert first.syllable_count == 1
    assert first.verdict == "not_applicable"
    assert second.word_index == 1
    assert second.predicted_stress_syllable == 1
    assert second.acoustic_peak_syllable == 1
    assert second.verdict == "ok"

    by_phone = {row["expected_phone"]: row for row in summary}
    assert by_phone["ˈ"]["instance_count"] == 2
    assert by_phone["ˈ"]["risk_class"] == "supported"
    assert by_phone["ˈ"]["stress_event_count"] == 2
    assert by_phone["ˈ"]["stress_evaluable_count"] == 1
    assert by_phone["ˈ"]["stress_match_count"] == 1
    assert by_phone["ˈ"]["stress_not_applicable_count"] == 1


def test_stress_peak_verdict_separates_match_mismatch_and_low_confidence() -> None:
    from audio.atlas_export import stress_peak_verdict

    ok, peak, cues = stress_peak_verdict(
        predicted_stress_syllable=1,
        syllables=[
            {"f0_hz": 150.0, "intensity_db": 55.0, "duration_ms": 80.0},
            {"f0_hz": 230.0, "intensity_db": 72.0, "duration_ms": 140.0},
        ],
    )
    assert ok == "ok"
    assert peak == 1
    assert "stress_match" in cues["cues"]

    mismatch, peak, cues = stress_peak_verdict(
        predicted_stress_syllable=1,
        syllables=[
            {"f0_hz": 240.0, "intensity_db": 75.0, "duration_ms": 150.0},
            {"f0_hz": 150.0, "intensity_db": 55.0, "duration_ms": 80.0},
        ],
    )
    assert mismatch == "mispronounced"
    assert peak == 0
    assert "stress_mismatch" in cues["cues"]

    low, peak, cues = stress_peak_verdict(
        predicted_stress_syllable=1,
        syllables=[
            {"f0_hz": None, "intensity_db": None, "duration_ms": 100.0},
            {"f0_hz": None, "intensity_db": None, "duration_ms": 100.0},
        ],
    )
    assert low == "low_confidence"
    assert peak is None
    assert "flat_acoustic_profile" in cues["cues"]


def test_full_atlas_report_summarizes_manifest_coverage(tmp_path: Path) -> None:
    from audio.alignment import AlignmentItem, upsert_segment_alignment
    from audio.atlas_export import render_full_atlas_report

    db = tmp_path / "audio.sqlite"
    manifest = tmp_path / "manifest.csv"
    init_db(db)
    conn = connect(db)
    try:
        with transaction(conn):
            aligned_seg = _segment(
                conn,
                path="processed/common_voice/aligned.wav",
                provider="common_voice",
                duration_s=1.0,
            )
            missing_seg = _segment(
                conn,
                path="processed/common_voice/missing.wav",
                provider="common_voice",
                duration_s=1.0,
            )
            upsert_segment_alignment(
                conn,
                segment_id=aligned_seg,
                expected_phones=("a",),
                items=(AlignmentItem("a", "a", "a", 0, 100, 0.9, "a"),),
                aligner_version="mms-fixture",
            )
        manifest.write_text(
            "segment_id,path,provider,duration_s,speaker_id,transcript_source,"
            "needs_whisper,role,selection_reason\n"
            f"{aligned_seg},processed/common_voice/aligned.wav,common_voice,1,s,"
            "publisher_gt,false,atlas_gold,test\n"
            f"{missing_seg},processed/common_voice/missing.wav,common_voice,1,s,"
            "publisher_gt,false,atlas_gold,test\n",
            encoding="utf-8",
        )

        report = render_full_atlas_report(
            conn,
            aligner_version="mms-fixture",
            manifest_path=manifest,
        )
    finally:
        conn.close()

    assert "# Phone Atlas Results" in report
    assert "- Manifest segments: 2" in report
    assert "- Aligned manifest segments: 1" in report
    assert "- Missing manifest segments: 1" in report
    assert "`expected_49`, `observed_model`, `observed_acoustic`, and `manual_phone` are separate evidence layers" in report
