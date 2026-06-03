"""Smoke tests for assess.contracts public dataclasses."""

from __future__ import annotations

import pytest


def test_phone_interval_serializes() -> None:
    from assess.contracts import PhoneInterval

    interval = PhoneInterval(
        target_phone="a",
        start_ms=120,
        end_ms=210,
        speech_start_ms_original=80,
        speech_end_ms_original=420,
    )
    d = interval.as_dict()
    assert d["target_phone"] == "a"
    assert d["start_ms"] == 120
    assert d["end_ms"] == 210
    assert d["speech_start_ms_original"] == 80
    assert d["speech_end_ms_original"] == 420


def test_coach_request_requires_exactly_one_target() -> None:
    from assess.contracts import CoachRequest

    CoachRequest(audio_path="a.wav", model_alias="mms-1b", target_text="merhaba")
    CoachRequest(audio_path="a.wav", model_alias="mms-1b", target_phones=["m", "e"])

    with pytest.raises(ValueError, match="exactly one"):
        CoachRequest(audio_path="a.wav", model_alias="mms-1b")

    with pytest.raises(ValueError, match="exactly one"):
        CoachRequest(
            audio_path="a.wav",
            model_alias="mms-1b",
            target_text="merhaba",
            target_phones=["m"],
        )


def test_coach_request_requires_audio_path() -> None:
    from assess.contracts import CoachRequest

    with pytest.raises(ValueError, match="audio_path"):
        CoachRequest(audio_path="", model_alias="mms-1b", target_text="merhaba")


def test_wav2vec_evidence_serializes() -> None:
    from assess.contracts import Wav2VecEvidence

    ev = Wav2VecEvidence(
        target_phone="i",
        observed_phone="e",
        target_logprob=-1.5,
        best_logprob=-0.8,
        gop_raw=-0.7,
        observed_matches_target=False,
        frame_count=4,
        confidence=1.0,
        reasons=[],
    )
    d = ev.as_dict()
    assert d["target_phone"] == "i"
    assert d["observed_phone"] == "e"
    assert d["gop_raw"] == pytest.approx(-0.7)
    assert d["confidence"] == 1.0
    assert d["observed_source"] == "peak_frame"


def test_coach_result_serializes() -> None:
    from assess.contracts import CoachResult

    result = CoachResult(
        model_alias="mms-1b",
        status="ok",
        speech_start_ms_original=0,
        speech_end_ms_original=500,
        phones=[],
        free_decode=[],
    )
    d = result.as_dict()
    assert d["model_alias"] == "mms-1b"
    assert d["status"] == "ok"
    assert d["phones"] == []


def test_coach_phone_result_serializes_top_level_scores() -> None:
    from assess.contracts import CoachPhoneResult

    result = CoachPhoneResult(
        target_phone="a",
        start_ms=10,
        end_ms=80,
        status="correct",
        score=1.0,
        atlas={"quality_score": 0.82},
        wav2vec={"identity_score": 0.91},
        identity_score=0.91,
        quality_score=0.82,
        length_score=0.76,
        analysis_source="raw",
        reasons=[],
    )

    d = result.as_dict()
    assert d["status"] == "correct"
    assert d["score"] == pytest.approx(1.0)
    assert d["identity_score"] == pytest.approx(0.91)
    assert d["quality_score"] == pytest.approx(0.82)
    assert d["length_score"] == pytest.approx(0.76)
    assert d["analysis_source"] == "raw"
