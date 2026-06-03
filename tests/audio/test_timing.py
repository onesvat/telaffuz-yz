"""Smoke tests for audio.timing — speech region, spans, endpointing."""

from __future__ import annotations

import numpy as np
import pytest


def test_speech_region_rejects_invalid_bounds() -> None:
    from audio.timing import SpeechRegion

    with pytest.raises(ValueError, match="non-negative"):
        SpeechRegion(start_ms_original=-1, end_ms_original=100)

    with pytest.raises(ValueError, match="greater than"):
        SpeechRegion(start_ms_original=200, end_ms_original=100)


def test_speech_region_duration_ms() -> None:
    from audio.timing import SpeechRegion

    region = SpeechRegion(start_ms_original=100, end_ms_original=400)
    assert region.duration_ms == 300


def test_speech_region_as_dict() -> None:
    from audio.timing import SpeechRegion

    region = SpeechRegion(start_ms_original=50, end_ms_original=300, status="ok")
    d = region.as_dict()
    assert d["speech_start_ms_original"] == 50
    assert d["speech_end_ms_original"] == 300
    assert d["endpoint_status"] == "ok"


def test_token_anchor_center() -> None:
    from audio.timing import TokenAnchor

    anchor = TokenAnchor(
        target_phone="a",
        start_ms_original=100,
        end_ms_original=200,
        confidence=0.9,
    )
    assert anchor.center_ms_original == 150


def test_phone_span_requires_anchor_center_inside() -> None:
    from audio.timing import PhoneSpan

    with pytest.raises(ValueError, match="anchor center must be inside"):
        PhoneSpan(
            target_phone="a",
            start_ms_original=100,
            end_ms_original=300,
            anchor_start_ms_original=50,   # outside span
            anchor_end_ms_original=80,
            confidence=None,
        )


def test_phone_span_as_dict() -> None:
    from audio.timing import PhoneSpan

    span = PhoneSpan(
        target_phone="e",
        start_ms_original=0,
        end_ms_original=200,
        anchor_start_ms_original=80,
        anchor_end_ms_original=120,
        confidence=0.85,
    )
    d = span.as_dict()
    assert d["target_phone"] == "e"
    assert d["duration_ms"] == 200
    assert d["confidence"] == pytest.approx(0.85)


def test_detect_speech_region_pure_silence() -> None:
    from audio.timing import detect_speech_region

    silence = np.zeros(16000, dtype=np.float32)
    region = detect_speech_region(silence, 16000)
    # Pure silence falls back to full clip
    assert region.status == "no_speech_fallback"
    assert region.start_ms_original == 0
    assert region.duration_ms > 0


def test_detect_speech_region_has_activity() -> None:
    from audio.timing import detect_speech_region

    rng = np.random.default_rng(42)
    # 0.5s silence, 0.5s speech-like noise, 0.5s silence at 16kHz
    silence = np.zeros(8000, dtype=np.float32)
    speech = rng.normal(0, 0.3, size=8000).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])
    region = detect_speech_region(audio, 16000)
    assert region.status == "ok"
    # Speech region should be somewhere in the middle
    assert region.start_ms_original < 600
    assert region.end_ms_original > 400


def test_spans_from_token_anchors_one_phone() -> None:
    from audio.timing import SpeechRegion, TokenAnchor, spans_from_token_anchors

    region = SpeechRegion(start_ms_original=0, end_ms_original=500)
    anchor = TokenAnchor(
        target_phone="a", start_ms_original=200, end_ms_original=300, confidence=0.9
    )
    spans = spans_from_token_anchors([anchor], region)
    assert len(spans) == 1
    span = spans[0]
    assert span.target_phone == "a"
    assert span.start_ms_original == 0
    assert span.end_ms_original == 500


def test_spans_from_token_anchors_two_phones() -> None:
    from audio.timing import SpeechRegion, TokenAnchor, spans_from_token_anchors

    region = SpeechRegion(start_ms_original=0, end_ms_original=600)
    anchors = [
        TokenAnchor("m", 80, 120, 0.8),
        TokenAnchor("a", 200, 280, 0.9),
    ]
    spans = spans_from_token_anchors(anchors, region)
    assert len(spans) == 2
    # Boundary between spans is the midpoint of adjacent centers
    assert spans[0].end_ms_original == spans[1].start_ms_original
    assert spans[1].end_ms_original == 600


def test_detect_voiced_edges_locates_active_region() -> None:
    from audio.timing import detect_voiced_edges

    rng = np.random.default_rng(7)
    silence = np.zeros(8000, dtype=np.float32)
    speech = rng.normal(0, 0.3, size=8000).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])  # 1.5 s @ 16 kHz

    edges = detect_voiced_edges(audio, 16000)
    assert edges is not None
    voiced_start_ms, voiced_end_ms = edges
    # Edges should sit inside the active middle section (500–1000 ms band)
    assert 400 <= voiced_start_ms <= 600
    assert 950 <= voiced_end_ms <= 1100


def test_detect_voiced_edges_returns_none_for_silence() -> None:
    from audio.timing import detect_voiced_edges

    silence = np.zeros(16000, dtype=np.float32)
    assert detect_voiced_edges(silence, 16000) is None


def test_detect_voiced_edges_respects_search_window() -> None:
    from audio.timing import detect_voiced_edges

    rng = np.random.default_rng(11)
    silence = np.zeros(8000, dtype=np.float32)
    speech = rng.normal(0, 0.3, size=8000).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])
    # Search window restricted to the silence at the head — no voicing inside
    assert detect_voiced_edges(audio, 16000, start_ms=0, end_ms=400) is None
