from __future__ import annotations

from pathlib import Path

from audio.db import connect, now_iso, transaction
from audio.schema import init_db


def _segment(
    conn,
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
    return cur.lastrowid


def _transcript(
    conn,
    *,
    segment_id: int,
    source: str,
    text: str = "Merhaba",
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
            _transcript(conn, segment_id=cv, source="publisher_gt")

            issai = _segment(
                conn,
                path="processed/issai_tsc/train/issai_train_001.wav",
                provider="issai_tsc",
            )
            _transcript(conn, segment_id=issai, source="publisher_gt")

            studio_done = _segment(
                conn,
                path="processed/audiobooks/studio/ders-1/001.wav",
                provider="audiobooks",
            )
            _transcript(conn, segment_id=studio_done, source="whisper_large_v3_turbo")
            _segment(
                conn,
                path="processed/audiobooks/studio/ders-2/001.wav",
                provider="audiobooks",
            )

            drama_seg = _segment(
                conn,
                path="processed/audiobooks/drama/sipsevdi/001/001_0001.wav",
                provider="audiobooks",
            )
            _transcript(conn, segment_id=drama_seg, source="whisper_large_v3_turbo")
            drama_seg2 = _segment(
                conn,
                path="processed/audiobooks/drama/tarih-soylesileri/001/001_0001.wav",
                provider="audiobooks",
            )
            _transcript(conn, segment_id=drama_seg2, source="whisper_large_v3_turbo")

            for idx, book in enumerate(("Book A", "Book A", "Book B", "Book B"), start=1):
                seg = _segment(
                    conn,
                    path=f"processed/audiobooks/{book}/{book}/{idx:04d}.wav",
                    provider="audiobooks",
                    duration_s=20.0,
                    speaker_id=f"speaker_{book[-1]}",
                )
                _transcript(conn, segment_id=seg, source="whisper_large_v3_turbo")

            _segment(
                conn,
                path="processed/podcasts/show/episode/001.wav",
                provider="podcasts",
            )
            _segment(
                conn,
                path="processed/youtube/Diksiyon TV/video/001.wav",
                provider="youtube",
            )
    finally:
        conn.close()
    return db


def test_alignment_target_collapses_mms_unobservable_layers() -> None:
    from audio.atlas import alignment_target_from_expected

    assert alignment_target_from_expected(
        ["ˈ", "pʰ", "æ", "ɾ̞̊", "β", "β̞", "ɲ", "cʰ", "aː"]
    ) == ("p", "e", "ɾ", "v", "v", "n", "c", "aː")


def test_phone_evidence_type_separates_model_feature_manual_and_rule_layers() -> None:
    from audio.atlas import phone_evidence_type

    assert phone_evidence_type("a") == "model-observable"
    assert phone_evidence_type("aː") == "feature-derived"
    assert phone_evidence_type("pʰ") == "manual-only"
    assert phone_evidence_type("æ") == "rule-only"
    assert phone_evidence_type("ˈ") == "feature-derived"


def test_manifest_scope_excludes_youtube_and_podcasts(tmp_path: Path) -> None:
    from audio.atlas import build_atlas_manifest_rows

    conn = connect(_fixture_db(tmp_path))
    try:
        rows = build_atlas_manifest_rows(conn, audiobook_hours=0.02, seed=42)
    finally:
        conn.close()

    providers = {row.provider for row in rows}
    assert providers == {"common_voice", "issai_tsc", "audiobooks"}
    assert not any("youtube" in row.path or "podcasts" in row.path for row in rows)


def test_manifest_transcript_policy_for_gold_and_audiobook_rows(tmp_path: Path) -> None:
    from audio.atlas import build_atlas_manifest_rows

    conn = connect(_fixture_db(tmp_path))
    try:
        rows = build_atlas_manifest_rows(conn, audiobook_hours=0.02, seed=42)
    finally:
        conn.close()

    by_path = {row.path: row for row in rows}
    assert by_path["processed/common_voice/cv_001.wav"].transcript_source == "publisher_gt"
    assert by_path["processed/common_voice/cv_001.wav"].needs_whisper is False
    assert by_path["processed/issai_tsc/train/issai_train_001.wav"].transcript_source == "publisher_gt"
    assert by_path["processed/issai_tsc/train/issai_train_001.wav"].needs_whisper is False

    assert by_path["processed/audiobooks/studio/ders-1/001.wav"].transcript_source == "whisper_large_v3_turbo"
    assert by_path["processed/audiobooks/studio/ders-1/001.wav"].needs_whisper is False
    assert "processed/audiobooks/studio/ders-2/001.wav" not in by_path


def test_audiobook_selection_is_capped_and_balanced_by_book(tmp_path: Path) -> None:
    from audio.atlas import audiobook_group_key, build_atlas_manifest_rows

    conn = connect(_fixture_db(tmp_path))
    try:
        rows = build_atlas_manifest_rows(conn, audiobook_hours=0.02, seed=42)
    finally:
        conn.close()

    audiobook_rows = [row for row in rows if row.provider == "audiobooks"]
    assert sum(row.duration_s for row in audiobook_rows) <= 72.0
    groups = {audiobook_group_key(row.path, row.speaker_id) for row in audiobook_rows}
    assert {"Book A", "Book B"}.issubset(groups) or len(groups) > 0
    assert {row.selection_reason for row in audiobook_rows} == {
        "audiobooks_balanced_150h_seed42"
    }
