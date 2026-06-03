"""Smoke tests for the coach length channel — pure logic, no I/O."""

from __future__ import annotations


def test_build_duration_stats_drops_thin_phones() -> None:
    from assess.length import build_duration_stats

    rows = [{"phone": "a", "dur_ms": 60 + i} for i in range(80)]
    rows += [{"phone": "aː", "dur_ms": 90} for _ in range(10)]  # too thin
    stats = build_duration_stats(rows, min_samples=50)
    assert "a" in stats.phones
    assert "aː" not in stats.phones  # below min_samples
    assert stats.phones["a"].scale_ms > 0.0


def test_length_cue_notes_and_score() -> None:
    from assess.length import build_duration_stats, length_cue

    rows = [{"phone": "a", "dur_ms": d} for d in list(range(40, 90)) * 2]
    stats = build_duration_stats(rows, min_samples=10)

    at_median = length_cue("a", stats.phones["a"].median_ms, stats, tau=2.0)
    assert at_median.note == "expected"
    assert at_median.score is not None and at_median.score > 0.99

    long_cue = length_cue("a", 500.0, stats, tau=2.0)
    assert long_cue.note == "long"
    assert long_cue.score is not None and long_cue.score < at_median.score


def test_length_cue_flags_clear_short_long_vowel() -> None:
    from assess.length import DurationStats, PhoneDuration, length_cue

    stats = DurationStats(
        phones={
            "a": PhoneDuration("a", median_ms=71.0, scale_ms=22.0, n=1000),
            "aː": PhoneDuration("aː", median_ms=85.0, scale_ms=25.0, n=500),
        }
    )

    clear_short = length_cue("aː", 55.0, stats)
    assert clear_short.note == "short"

    expected = length_cue("aː", 70.0, stats)
    assert expected.note == "expected"


def test_length_cue_unscored_when_missing_or_degenerate() -> None:
    from assess.length import DurationStats, PhoneDuration, length_cue

    assert length_cue("a", 80.0, None).note is None
    degenerate = DurationStats(phones={"a": PhoneDuration("a", 80.0, 0.0, 100)})
    cue = length_cue("a", 200.0, degenerate)
    assert cue.note is None and cue.score is None


def test_feedback_note_for_length_maps_vocabulary() -> None:
    from assess.length import feedback_note_for_length

    assert feedback_note_for_length("short") == "too_short"
    assert feedback_note_for_length("long") == "too_long"
    assert feedback_note_for_length("expected") is None
    assert feedback_note_for_length(None) is None
