"""Smoke tests for audio.features helpers (measurement_window, dynamics)."""

from __future__ import annotations

import numpy as np
import pytest


def test_measurement_window_centres_on_voiced_region() -> None:
    from audio.features import measurement_window

    start, end = measurement_window(
        span_start_ms=0.0,
        span_end_ms=200.0,
        voiced_start_ms=120.0,
        voiced_end_ms=180.0,
        policy_ms=80,
    )
    # Voiced midpoint is 150 ms; width capped at 60% × 60 ms = 36 ms.
    centre = (start + end) / 2.0
    assert centre == pytest.approx(150.0, abs=0.5)
    assert end - start == pytest.approx(36.0, abs=0.5)


def test_measurement_window_falls_back_to_geometric_centre_without_voiced() -> None:
    from audio.features import measurement_window

    start, end = measurement_window(
        span_start_ms=0.0,
        span_end_ms=100.0,
        voiced_start_ms=None,
        voiced_end_ms=None,
        policy_ms=80,
    )
    # Span midpoint is 50 ms; width capped at 60% × 100 ms = 60 ms.
    assert start == pytest.approx(20.0, abs=0.5)
    assert end == pytest.approx(80.0, abs=0.5)


def test_measurement_window_clamps_to_span_bounds() -> None:
    from audio.features import measurement_window

    # Voiced bounds extend past the span on both sides — must clamp.
    start, end = measurement_window(
        span_start_ms=50.0,
        span_end_ms=120.0,
        voiced_start_ms=10.0,
        voiced_end_ms=200.0,
        policy_ms=80,
    )
    assert start >= 50.0
    assert end <= 120.0


def test_measurement_window_rejects_empty_span() -> None:
    from audio.features import measurement_window

    with pytest.raises(ValueError, match="positive"):
        measurement_window(
            span_start_ms=100.0,
            span_end_ms=100.0,
            voiced_start_ms=None,
            voiced_end_ms=None,
        )


def test_spectral_moments_for_pure_tone() -> None:
    from audio.features import spectral_moments

    sample_rate = 16_000
    t = np.arange(int(0.05 * sample_rate)).astype(np.float32) / sample_rate
    audio = 0.5 * np.sin(2 * np.pi * 1500 * t).astype(np.float32)
    centroid, bandwidth, skew, kurtosis = spectral_moments(audio, sample_rate)
    assert centroid is not None
    assert centroid == pytest.approx(1500.0, rel=0.05)
    # Pure tone has narrow band, skew near zero
    assert bandwidth is not None
    assert skew is not None
    assert kurtosis is not None


def test_spectral_moments_handles_silence() -> None:
    from audio.features import spectral_moments

    silence = np.zeros(1024, dtype=np.float32)
    assert spectral_moments(silence, 16_000) == (None, None, None, None)


def test_detect_burst_returns_offset_in_ms() -> None:
    from audio.features import detect_burst_ms

    sample_rate = 16_000
    pre = np.zeros(int(0.04 * sample_rate), dtype=np.float32)
    burst = 0.7 * np.hanning(int(0.005 * sample_rate)).astype(np.float32)
    tail = (
        0.3
        * np.random.default_rng(0)
        .normal(0, 1, size=int(0.05 * sample_rate))
        .astype(np.float32)
    )
    audio = np.concatenate([pre, burst, tail])
    burst_time = detect_burst_ms(audio, sample_rate)
    assert burst_time is not None
    assert 30.0 <= burst_time <= 60.0


def test_frication_rise_db_per_ms_positive_for_rising_intensity() -> None:
    from audio.features import frication_rise_db_per_ms

    sample_rate = 16_000
    n = int(0.06 * sample_rate)
    ramp = (np.linspace(0.0, 1.0, n) ** 2).astype(np.float32)
    rng = np.random.default_rng(7)
    audio = (ramp * rng.normal(0, 1, size=n).astype(np.float32)).astype(np.float32)
    slope = frication_rise_db_per_ms(audio, sample_rate)
    assert slope is not None
    assert slope > 0  # intensity rising over time


def test_formant_transition_slope_positive_when_value_grows() -> None:
    from audio.features import formant_transition_slope_hz_per_ms

    slope = formant_transition_slope_hz_per_ms(1200.0, 1600.0, duration_ms=80.0)
    assert slope == pytest.approx(5.0)
    assert formant_transition_slope_hz_per_ms(None, 1500.0, 50.0) is None
    assert formant_transition_slope_hz_per_ms(1200.0, 1500.0, 0.0) is None
