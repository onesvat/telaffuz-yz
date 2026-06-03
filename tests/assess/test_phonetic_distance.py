from __future__ import annotations

import pytest


def test_exact_phone_scores_full_credit() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone("a", "a")

    assert result.score == 1.0
    assert result.reason == "exact"


def test_analysis_corrected_phone_scores_full_credit() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone("aː", "aː", raw_observed="a", analysis_observed="aː")

    assert result.score == 1.0
    assert result.reason == "analysis_corrected"


def test_missing_phone_scores_zero() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone("a", None)

    assert result.score == 0.0
    assert result.reason == "missing"


def test_vowel_consonant_mismatch_is_near_zero() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone("a", "kʰ")

    assert result.score <= 0.10
    assert result.distance_features["class"]["match"] is False


def test_long_short_vowel_pair_gets_partial_credit() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone("aː", "a")

    assert 0.5 < result.score < 1.0
    assert "length" in result.distance_features["differences"]


def test_duration_can_raise_same_quality_length_mismatch() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone(
        "aː",
        "a",
        duration_evidence={"note": "expected", "score": 0.91, "reliable": True},
    )

    assert result.score == pytest.approx(0.90)
    assert result.distance_features["modifiers"][0]["effect"] == "supports_target_length"


def test_close_same_family_consonants_score_above_unrelated_stops() -> None:
    from assess.phonetic_distance import score_phone

    close = score_phone("cʰ", "kʰ")
    far = score_phone("cʰ", "p")

    assert close.score > far.score
    assert close.score > 0.70
    assert "place" in close.distance_features["differences"]


def test_same_family_fricatives_get_partial_credit() -> None:
    from assess.phonetic_distance import score_phone

    result = score_phone("s", "ʃ")

    assert 0.5 < result.score < 1.0
    assert result.distance_features["manner"]["match"] is True
    assert result.distance_features["place"]["match"] is False
