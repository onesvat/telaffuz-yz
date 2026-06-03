"""Smoke tests for assess.features — signal-anchored windowing."""

from __future__ import annotations

import numpy as np
import pytest


def _interval(target_phone: str, start_ms: int, end_ms: int):
    from assess.contracts import PhoneInterval

    return PhoneInterval(
        target_phone=target_phone,
        start_ms=start_ms,
        end_ms=end_ms,
        speech_start_ms_original=0,
        speech_end_ms_original=end_ms + 200,
    )


def _audio_with_voiced_middle(
    *,
    pre_silence_ms: int,
    voiced_ms: int,
    post_silence_ms: int,
    sample_rate: int = 16_000,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    pre = np.zeros(int(pre_silence_ms * sample_rate / 1000), dtype=np.float32)
    voiced = (
        0.25
        * rng.normal(0, 1, size=int(voiced_ms * sample_rate / 1000)).astype(np.float32)
    )
    post = np.zeros(int(post_silence_ms * sample_rate / 1000), dtype=np.float32)
    return np.concatenate([pre, voiced, post])


def test_duration_reliability_full_when_voiced_detected() -> None:
    from assess.features import duration_reliability_for

    reliability, notes = duration_reliability_for(
        target_phone="a",
        prev_phone="m",
        next_phone="n",
        voiced_detected=True,
    )
    assert reliability == 1.0
    assert notes == []


def test_duration_reliability_drops_on_vowel_vowel_neighbour() -> None:
    from assess.features import duration_reliability_for

    reliability, notes = duration_reliability_for(
        target_phone="a",
        prev_phone="e",
        next_phone="m",
        voiced_detected=True,
    )
    assert reliability < 1.0
    assert "vowel_vowel_neighbour" in notes


def test_duration_reliability_zero_when_voicing_undetected() -> None:
    from assess.features import duration_reliability_for

    reliability, notes = duration_reliability_for(
        target_phone="a",
        prev_phone=None,
        next_phone=None,
        voiced_detected=False,
    )
    assert reliability == 0.0
    assert "voiced_edges_undetected" in notes


def test_feature_extractor_signal_anchored_duration() -> None:
    from assess.features import FeatureExtractor

    sample_rate = 16_000
    audio = _audio_with_voiced_middle(
        pre_silence_ms=80, voiced_ms=100, post_silence_ms=80, sample_rate=sample_rate
    )
    interval = _interval("a", start_ms=0, end_ms=260)

    extractor = FeatureExtractor()
    fs = extractor.extract(audio, sample_rate, interval)

    # Duration should approximate the voiced portion (≈100 ms), not the
    # aligner span (260 ms). Allow generous tolerance for RMS frame boundaries.
    assert fs.voiced_start_ms is not None
    assert fs.voiced_end_ms is not None
    assert 60.0 <= fs.active_duration_ms <= 160.0
    assert fs.duration_reliability == 1.0


def test_feature_extractor_drops_reliability_for_vowel_vowel() -> None:
    from assess.features import FeatureExtractor

    sample_rate = 16_000
    audio = _audio_with_voiced_middle(
        pre_silence_ms=20, voiced_ms=120, post_silence_ms=20, sample_rate=sample_rate
    )
    interval = _interval("a", start_ms=0, end_ms=160)

    extractor = FeatureExtractor()
    fs = extractor.extract(
        audio, sample_rate, interval, prev_phone="e", next_phone="m"
    )
    assert fs.duration_reliability < 1.0
    assert "vowel_vowel_neighbour" in fs.duration_notes


def test_feature_extractor_voiced_undetected_falls_back_to_span() -> None:
    from assess.features import FeatureExtractor

    sample_rate = 16_000
    audio = np.zeros(sample_rate, dtype=np.float32)  # 1 s pure silence
    interval = _interval("a", start_ms=200, end_ms=500)

    extractor = FeatureExtractor()
    fs = extractor.extract(audio, sample_rate, interval)
    assert fs.voiced_start_ms is None
    assert fs.duration_reliability == 0.0
    # Falls back to the aligner-span duration (≈ 300 ms)
    assert fs.active_duration_ms == pytest.approx(300.0, abs=1.0)


def test_nucleus_window_uses_voiced_centre() -> None:
    from assess.contracts import PhoneInterval
    from assess.features import nucleus_window_samples

    sample_rate = 16_000
    interval = PhoneInterval(
        target_phone="a",
        start_ms=0,
        end_ms=200,
        speech_start_ms_original=0,
        speech_end_ms_original=400,
    )

    geometric_n0, geometric_n1 = nucleus_window_samples(
        sample_rate, sample_rate, interval, 80
    )
    voiced_n0, voiced_n1 = nucleus_window_samples(
        sample_rate,
        sample_rate,
        interval,
        80,
        voiced_start_ms=140.0,
        voiced_end_ms=180.0,
    )
    # Voiced-anchored centre is at 160 ms, far from geometric 100 ms.
    geometric_centre = (geometric_n0 + geometric_n1) / 2.0
    voiced_centre = (voiced_n0 + voiced_n1) / 2.0
    assert voiced_centre > geometric_centre


# ---------------------------------------------------------------------------
# Faz 2: class-specific feature design
# ---------------------------------------------------------------------------

def test_phone_class_categorises_all_inventory_groups() -> None:
    from assess.features import PhoneClass, phone_class

    assert phone_class("a") is PhoneClass.VOWEL
    assert phone_class("aː") is PhoneClass.VOWEL
    assert phone_class("æ") is PhoneClass.VOWEL
    assert phone_class("m") is PhoneClass.SONORANT
    assert phone_class("ɾ") is PhoneClass.SONORANT
    assert phone_class("s") is PhoneClass.FRICATIVE
    assert phone_class("ʃ") is PhoneClass.FRICATIVE
    assert phone_class("t͡ʃ") is PhoneClass.AFFRICATE
    assert phone_class("d͡ʒ") is PhoneClass.AFFRICATE
    assert phone_class("p") is PhoneClass.STOP
    assert phone_class("pʰ") is PhoneClass.STOP


def test_feature_names_for_returns_class_specific_set() -> None:
    from assess.features import feature_names_for

    assert "f1_hz" in feature_names_for("a")
    assert "spectral_skew" in feature_names_for("s")
    assert "vot_ms" in feature_names_for("p")
    assert "frication_rise_db_per_ms" in feature_names_for("t͡ʃ")
    assert "spectral_centroid_hz" in feature_names_for("m")


def test_vowel_feature_set_carries_formants_not_burst_features() -> None:
    from assess.features import FeatureExtractor

    sample_rate = 16_000
    rng = np.random.default_rng(31)
    audio = (
        0.3
        * rng.normal(0, 1, size=int(0.15 * sample_rate)).astype(np.float32)
    )
    pre = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
    post = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
    full = np.concatenate([pre, audio, post])

    extractor = FeatureExtractor()
    fs = extractor.extract(
        full,
        sample_rate,
        _interval("a", start_ms=20, end_ms=240),
    )
    # Vowels: spectral_skew/kurtosis stay None (no fricative dynamics asked).
    assert fs.spectral_skew is None
    assert fs.spectral_kurtosis is None
    assert fs.vot_ms is None
    assert fs.burst_centroid_hz is None


def test_fricative_feature_set_fills_spectral_skew_and_kurtosis() -> None:
    from assess.features import FeatureExtractor

    sample_rate = 16_000
    rng = np.random.default_rng(13)
    # High-frequency noise approximating /s/ frication
    span = (
        0.4
        * rng.normal(0, 1, size=int(0.12 * sample_rate)).astype(np.float32)
    )
    pre = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
    post = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
    audio = np.concatenate([pre, span, post])

    extractor = FeatureExtractor()
    fs = extractor.extract(
        audio,
        sample_rate,
        _interval("s", start_ms=20, end_ms=200),
    )
    assert fs.spectral_skew is not None
    assert fs.spectral_kurtosis is not None
    assert fs.frication_rise_db_per_ms is not None
    assert fs.vot_ms is None  # not a stop
    assert fs.f1_hz is None
    assert fs.f2_hz is None


def test_stop_feature_set_fills_burst_features() -> None:
    from assess.features import FeatureExtractor

    sample_rate = 16_000
    # Pre-burst silence, burst impulse, voicing-like signal
    silence = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
    burst = 0.6 * np.hanning(int(0.005 * sample_rate)).astype(np.float32)
    voicing = (
        0.2
        * np.sin(
            2
            * np.pi
            * 140
            * np.arange(int(0.08 * sample_rate)).astype(np.float32)
            / sample_rate
        )
    ).astype(np.float32)
    post = np.zeros(int(0.02 * sample_rate), dtype=np.float32)
    audio = np.concatenate([silence, burst, voicing, post])

    extractor = FeatureExtractor()
    fs = extractor.extract(
        audio,
        sample_rate,
        _interval("p", start_ms=20, end_ms=200),
    )
    assert fs.burst_centroid_hz is not None
    # /p/ feature design pulls VOT and burst skew. Frication rise stays None.
    assert fs.frication_rise_db_per_ms is None
    assert fs.spectral_skew is None  # not a fricative
    assert "release20_band_1000_1500_log" in fs.raw_release_features
    assert fs.release20_band_1000_1500_log == fs.raw_release_features[
        "release20_band_1000_1500_log"
    ]


def test_release_spectrum_features_tracks_high_frequency_release_energy() -> None:
    from assess.features import release_spectrum_features

    sample_rate = 16_000
    t = np.arange(int(0.04 * sample_rate), dtype=np.float32) / sample_rate
    low = np.sin(2 * np.pi * 800 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 3500 * t).astype(np.float32)

    low_features = release_spectrum_features(low, sample_rate, burst_ms=0.0)
    high_features = release_spectrum_features(high, sample_rate, burst_ms=0.0)

    assert high_features["release20_high_low_2k"] > low_features["release20_high_low_2k"]
    assert (
        high_features["release20_band_3000_4000_log"]
        > low_features["release20_band_3000_4000_log"]
    )


def test_stop_formant_transition_features_tracks_palatal_f2_pull() -> None:
    from assess import features as feature_module
    from assess.features import stop_formant_transition_features

    sample_rate = 16_000
    times = np.array([20.0, 40.0, 60.0, 95.0, 120.0, 140.0])

    class FakeTrack:
        times_ms = times
        f1 = np.full(times.size, 600.0)
        f2 = np.array([2300.0, 2400.0, 2350.0, 1700.0, 1650.0, 1680.0])
        f3 = np.array([3300.0, 3350.0, 3320.0, 2850.0, 2800.0, 2820.0])

    original = feature_module.formant_track
    feature_module.formant_track = lambda _span, _sample_rate: FakeTrack()
    try:
        values = stop_formant_transition_features(
            np.ones(2000, dtype=np.float32),
            sample_rate,
            burst_ms=0.0,
        )
    finally:
        feature_module.formant_track = original

    assert values["transition_f2_onset_hz"] == pytest.approx(2350.0)
    assert values["transition_f2_mid_hz"] == pytest.approx(1680.0)
    assert values["transition_f2_delta_hz"] > 600.0
    assert values["transition_f3_onset_hz"] > values["transition_f3_mid_hz"]


def test_feature_vector_returns_class_specific_dimension() -> None:
    from assess.features import FeatureSet, feature_vector

    fs = FeatureSet(
        duration_ms=100.0,
        rms=0.1,
        spectral_centroid_hz=3500.0,
        spectral_bandwidth_hz=1200.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.1,
        f0_mean_hz=None,
        f1_hz=None,
        f2_hz=None,
        f3_hz=None,
        formant_confidence=0.0,
        feature_confidence=1.0,
        low_confidence_reasons=[],
        spectral_skew=0.5,
        spectral_kurtosis=2.0,
        frication_duration_ms=60.0,
    )
    vec = feature_vector(fs, "s")
    assert vec is not None
    from assess.features import feature_names_for

    assert len(vec) == len(feature_names_for("s"))


def test_feature_vector_returns_none_when_dynamic_feature_missing() -> None:
    from assess.features import FeatureSet, feature_vector

    fs = FeatureSet(
        duration_ms=100.0,
        rms=0.1,
        spectral_centroid_hz=3500.0,
        spectral_bandwidth_hz=1200.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.1,
        f0_mean_hz=None,
        f1_hz=None,
        f2_hz=None,
        f3_hz=None,
        formant_confidence=0.0,
        feature_confidence=0.0,
        low_confidence_reasons=["formant_extraction_failed"],
        spectral_skew=None,  # missing for fricative
        spectral_kurtosis=None,
    )
    assert feature_vector(fs, "s") is None
