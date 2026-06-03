"""Coach feature extraction: FeatureSet, FeatureExtractor, window helpers, confidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from assess.contracts import PhoneInterval
from audio.features import (
    closure_burst_landmarks,
    formant_track,
    frication_interval,
    frication_rise_db_per_ms as _frication_rise_db_per_ms,
    linear_slope,
    low_freq_energy_ratio,
    measurement_window,
    spectral_moments,
    voicing_track,
)
from audio.timing import detect_voiced_edges

# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

_SPAN_MIN_MS = 20.0
_NUCLEUS_MIN_MS = 15.0
_VOICING_MIN_FRACTION = 0.3
_ACTIVE_DURATION_MIN_MS = 15.0
_VOWEL_VOWEL_RELIABILITY = 0.5
# IPA vowel symbols (short and long)
_VOWEL_CHARS: frozenset[str] = frozenset("aeiouâîûAEIOUÂÎÛ")


def _is_vowel_char(phone: str | None) -> bool:
    if not phone:
        return False
    return phone[0] in _VOWEL_CHARS


def assess_confidence(
    duration_ms: float,
    nucleus_duration_ms: float,
    formant_ok: bool,
    voiced_fraction: float,
    target_phone: str,
    *,
    active_duration_ms: float | None = None,
    phone_class_: "PhoneClass | None" = None,
) -> tuple[float, list[str]]:
    """Return (feature_confidence, low_confidence_reasons).

    Confidence reflects *core measurement validity* only — span/active length
    and, for classes that need them, formant extraction and voicing. A missing
    optional dynamic cue (VOT, frication rise, nasal murmur, …) is normal for
    many tokens and is recorded as a NULL column, not penalised here; per-phone
    feature coverage is handled at GMM build time instead.
    """
    reasons: list[str] = []
    # Stops/affricates are inherently brief: their evidence is the transient
    # burst (gated separately by burst_confidence), not sustained signal length.
    # Penalising them for a short span would zero out otherwise-valid bursts.
    transient_class = phone_class_ in {PhoneClass.STOP, PhoneClass.AFFRICATE}
    if duration_ms < _SPAN_MIN_MS and not transient_class:
        reasons.append("span_too_short")
    # Formant failure only blocks classes that need formants.
    formant_required = phone_class_ in {None, PhoneClass.VOWEL, PhoneClass.SONORANT}
    if formant_required and not formant_ok:
        reasons.append("formant_extraction_failed")
    if target_phone in _VOWEL_CHARS and voiced_fraction < _VOICING_MIN_FRACTION:
        reasons.append("insufficient_voicing")
    if (
        active_duration_ms is not None
        and active_duration_ms < _ACTIVE_DURATION_MIN_MS
        and not transient_class
    ):
        reasons.append("active_signal_too_short")
    return (0.0 if reasons else 1.0), reasons


def duration_reliability_for(
    *,
    target_phone: str,
    prev_phone: str | None,
    next_phone: str | None,
    voiced_detected: bool,
) -> tuple[float, list[str]]:
    """Return (duration_reliability, notes) given context.

    Reliability degrades when voiced edges cannot be detected (signal-derived
    duration falls back to span timing) and when both neighbours are vowels,
    making the boundary fuzzy.
    """
    notes: list[str] = []
    reliability = 1.0
    if not voiced_detected:
        notes.append("voiced_edges_undetected")
        reliability = min(reliability, 0.0)
    if _is_vowel_char(target_phone) and (
        _is_vowel_char(prev_phone) or _is_vowel_char(next_phone)
    ):
        notes.append("vowel_vowel_neighbour")
        reliability = min(reliability, _VOWEL_VOWEL_RELIABILITY)
    return reliability, notes


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def span_window_samples(
    audio_len: int,
    sample_rate: int,
    interval: PhoneInterval,
) -> tuple[int, int]:
    """Full aligned phone span as sample indices, clamped to audio_len."""
    s0 = max(0, int(round(interval.start_ms / 1000.0 * sample_rate)))
    s1 = min(audio_len, int(round(interval.end_ms / 1000.0 * sample_rate)))
    return s0, s1


# Class-specific analysis window widths (ms). A raw aligner span can be
# ~20 ms — too short for time-extended consonant cues (VOT, burst moments,
# frication rise, F2 transition). These widen it to span the relevant acoustic
# event without reaching back into the preceding segment.
STOP_WINDOW_MS = 150
STOP_BACK_MS = 30
FRICATIVE_WINDOW_MS = 60
SONORANT_WINDOW_MS = 80
VOWEL_WINDOW_MS = 90


def class_window_samples(
    audio_len: int,
    sample_rate: int,
    interval: PhoneInterval,
    target_class: "PhoneClass | None",
) -> tuple[int, int]:
    """Analysis window for time-extended dynamic consonant features.

    Stops/affricates take a small backward margin (to include the closure
    silence) and extend forward past the aligned start to capture the burst,
    VOT and the voicing onset of the following segment. The burst landmark is
    found relative to the closure floor, so a preceding vowel in the backward
    margin cannot be mistaken for the burst. Fricatives widen symmetrically
    around the span midpoint. Sonorants extend forward to straddle the formant
    transition into the adjacent vowel. Vowels and unknown classes fall back
    to the raw aligned span.
    """
    s0 = max(0, int(round(interval.start_ms / 1000.0 * sample_rate)))
    s1 = min(audio_len, int(round(interval.end_ms / 1000.0 * sample_rate)))
    if target_class in {PhoneClass.STOP, PhoneClass.AFFRICATE}:
        back = int(round(STOP_BACK_MS / 1000.0 * sample_rate))
        width = int(round(STOP_WINDOW_MS / 1000.0 * sample_rate))
        return max(0, s0 - back), min(audio_len, max(s1, s0 + width))
    if target_class == PhoneClass.SONORANT:
        width = int(round(SONORANT_WINDOW_MS / 1000.0 * sample_rate))
        return s0, min(audio_len, max(s1, s0 + width))
    if target_class in {PhoneClass.FRICATIVE, PhoneClass.VOWEL}:
        width = int(
            round(
                (FRICATIVE_WINDOW_MS if target_class == PhoneClass.FRICATIVE
                 else VOWEL_WINDOW_MS) / 1000.0 * sample_rate
            )
        )
        mid = (s0 + s1) // 2
        a1 = min(audio_len, max(s1, mid + width // 2))
        a0 = max(0, a1 - width)
        return a0, a1
    return s0, s1


def nucleus_window_samples(
    audio_len: int,
    sample_rate: int,
    interval: PhoneInterval,
    nucleus_policy_ms: int = 80,
    *,
    voiced_start_ms: float | None = None,
    voiced_end_ms: float | None = None,
) -> tuple[int, int]:
    """Measurement window anchored to the voiced active region.

    Falls back to the geometric centre of the phone span when ``voiced_start_ms``
    or ``voiced_end_ms`` are unavailable.
    """
    span_start_ms = float(interval.start_ms)
    span_end_ms = float(interval.end_ms)
    start_ms, end_ms = measurement_window(
        span_start_ms,
        span_end_ms,
        voiced_start_ms,
        voiced_end_ms,
        policy_ms=nucleus_policy_ms,
    )

    span_s0 = max(0, int(round(span_start_ms / 1000.0 * sample_rate)))
    span_s1 = min(audio_len, int(round(span_end_ms / 1000.0 * sample_rate)))
    n0 = max(span_s0, int(round(start_ms / 1000.0 * sample_rate)))
    n1 = min(span_s1, int(round(end_ms / 1000.0 * sample_rate)))
    if n1 <= n0:
        n0 = span_s0
        n1 = max(span_s0 + 1, span_s1)
    return n0, n1


# ---------------------------------------------------------------------------
# FeatureSet dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSet:
    """All acoustic features for one phone interval."""

    # Span window
    duration_ms: float
    rms: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    mfcc: list[float]
    voiced_fraction: float
    f0_mean_hz: float | None

    # Nucleus window
    f1_hz: float | None
    f2_hz: float | None
    f3_hz: float | None
    formant_confidence: float

    # Confidence metadata
    feature_confidence: float
    low_confidence_reasons: list[str]

    # Signal-anchored timing (added in Faz 1)
    active_duration_ms: float = 0.0
    voiced_start_ms: float | None = None
    voiced_end_ms: float | None = None
    duration_reliability: float = 1.0
    duration_notes: list[str] = field(default_factory=list)

    # Class-specific dynamic features
    spectral_skew: float | None = None
    spectral_kurtosis: float | None = None
    vot_ms: float | None = None
    closure_ms: float | None = None
    closure_duration_ms: float | None = None
    closure_voicing_ratio: float | None = None
    burst_centroid_hz: float | None = None
    burst_spectral_skew: float | None = None
    burst_confidence: float | None = None
    frication_rise_db_per_ms: float | None = None
    frication_duration_ms: float | None = None
    f2_transition_slope_hz_per_ms: float | None = None
    f2_locus_hz: float | None = None
    nasal_murmur_ratio: float | None = None
    vowel_f2_movement_hz_per_ms: float | None = None
    raw_release_features: dict[str, float] = field(default_factory=dict)

    def __getattr__(self, name: str) -> float:
        try:
            return self.raw_release_features[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


# ---------------------------------------------------------------------------
# Formant extraction
# ---------------------------------------------------------------------------


def _formants(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float | None, float | None]:
    """Median (F1, F2, F3) Hz from the continuous Burg formant contour.

    Uses :func:`formant_track`, which samples ``to_formant_burg`` directly
    rather than gating on a pitch track — the pitch gate needs ≥ 3 voiced
    frames and silently fails on the short (~50 ms) class windows.
    """
    track = formant_track(samples, sample_rate)
    if track is None:
        return (None, None, None)

    def _median(values: np.ndarray) -> float | None:
        finite = values[np.isfinite(values) & (values > 0.0)]
        return float(np.median(finite)) if finite.size >= 2 else None

    return _median(track.f1), _median(track.f2), _median(track.f3)


def _fallback_mfcc_like(samples: np.ndarray, sample_rate: int) -> list[float]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size < 2:
        return [0.0] * 13
    mag = np.abs(np.fft.rfft(arr * np.hanning(arr.size)))
    bins = np.array_split(np.log1p(mag), 13)
    return [float(np.mean(chunk)) if chunk.size else 0.0 for chunk in bins]


def _fallback_voiced_fraction(samples: np.ndarray, sample_rate: int) -> float:
    arr = np.asarray(samples, dtype=np.float32)
    frame_length = max(1, int(round(sample_rate * 0.02)))
    if arr.size < frame_length:
        return 0.0
    n = arr.size // frame_length
    frames = arr[: n * frame_length].reshape(n, frame_length)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if rms.size == 0:
        return 0.0
    threshold = max(float(np.percentile(rms, 20)) * 3.0, 1e-4)
    return float(np.mean(rms >= threshold))


# ---------------------------------------------------------------------------
# Class-specific dynamic feature extraction helpers
# ---------------------------------------------------------------------------

def _safe_phone_class(phone: str) -> "PhoneClass | None":
    try:
        return phone_class(phone)
    except ValueError:
        return None


@dataclass(frozen=True)
class DynamicFeatures:
    """Class-specific dynamic acoustic features for one phone."""

    spectral_skew: float | None = None
    spectral_kurtosis: float | None = None
    vot_ms: float | None = None
    closure_duration_ms: float | None = None
    closure_voicing_ratio: float | None = None
    burst_centroid_hz: float | None = None
    burst_spectral_skew: float | None = None
    burst_confidence: float | None = None
    frication_rise_db_per_ms: float | None = None
    frication_duration_ms: float | None = None
    f2_transition_slope_hz_per_ms: float | None = None
    f2_locus_hz: float | None = None
    nasal_murmur_ratio: float | None = None
    vowel_f2_movement_hz_per_ms: float | None = None
    raw_release_features: dict[str, float] = field(default_factory=dict)


_BURST_WINDOW_MS = 20.0
_F2_TRANSITION_MS = 60.0
_STOP_TRANSITION_ONSET_MS: tuple[float, float] = (20.0, 70.0)
_STOP_TRANSITION_MID_MS: tuple[float, float] = (90.0, 140.0)
_RELEASE_WINDOWS_MS: tuple[float, ...] = (20.0, 40.0)
_RELEASE_N_FFT = 2048
_RELEASE_BAND_EDGES_HZ: tuple[float, ...] = (
    0.0,
    500.0,
    1000.0,
    1500.0,
    2000.0,
    2500.0,
    3000.0,
    4000.0,
    5000.0,
    6500.0,
)


def _dct_type2_ortho(values: np.ndarray) -> np.ndarray:
    """Small DCT-II implementation for dependency-free band-shape features."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    index = np.arange(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        out[k] = scale * float(
            np.sum(arr * np.cos(np.pi * (index + 0.5) * k / n))
        )
    return out


def _power_spectrum(segment: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(segment, dtype=np.float64)
    if arr.size < 24:
        return None
    arr = arr - float(np.mean(arr))
    n_fft = _RELEASE_N_FFT
    padded = np.zeros(n_fft, dtype=np.float64)
    usable = min(arr.size, n_fft)
    padded[:usable] = arr[:usable] * np.hanning(usable)
    power = np.abs(np.fft.rfft(padded)) ** 2
    total = float(power.sum())
    if total <= 1e-20:
        return None
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, power


def _band_energy(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(power[mask].sum())


def release_spectrum_features(
    span: np.ndarray,
    sample_rate: int,
    *,
    burst_ms: float,
) -> dict[str, float]:
    """Release-window spectral-band features for dorsal stop place.

    These are intentionally low-level waveform cues: log band-energy fractions,
    high/low energy ratios, and DCT coefficients over coarse spectral bands.
    They do not use target text, following-vowel identity, or any semantic
    context.
    """
    out: dict[str, float] = {}
    start = max(0, int(round(burst_ms / 1000.0 * sample_rate)))
    for window_ms in _RELEASE_WINDOWS_MS:
        end = min(span.size, start + int(round(window_ms / 1000.0 * sample_rate)))
        segment = span[start:end]
        spectrum = _power_spectrum(segment, sample_rate)
        if spectrum is None:
            continue
        freqs, power = spectrum
        total = float(power.sum())
        prefix = f"release{int(window_ms)}"
        nyquist = sample_rate / 2.0
        edges = [edge for edge in _RELEASE_BAND_EDGES_HZ if edge < nyquist]
        if not edges or edges[-1] < nyquist:
            edges.append(nyquist)
        band_fractions: list[float] = []
        for low, high in zip(edges, edges[1:], strict=False):
            energy = _band_energy(power, freqs, low, high)
            fraction = energy / (total + 1e-20)
            band_fractions.append(fraction)
            low_i = int(round(low))
            high_i = int(round(high))
            if (low_i, high_i) in {
                (500, 1000),
                (1000, 1500),
                (2500, 3000),
                (3000, 4000),
            }:
                out[f"{prefix}_band_{low_i}_{high_i}_log"] = float(
                    np.log(fraction + 1e-12)
                )
        low_2k = _band_energy(power, freqs, 0.0, 2000.0)
        low_3k = _band_energy(power, freqs, 0.0, 3000.0)
        out[f"{prefix}_high_low_2k"] = _band_energy(
            power, freqs, 2000.0, nyquist
        ) / (low_2k + 1e-20)
        out[f"{prefix}_high_low_3k"] = _band_energy(
            power, freqs, 3000.0, nyquist
        ) / (low_3k + 1e-20)
        probs = power / (total + 1e-20)
        out[f"{prefix}_centroid_hz"] = float(np.sum(freqs * probs))
        cep = _dct_type2_ortho(np.log(np.asarray(band_fractions) + 1e-12))
        for idx in (1, 3, 4):
            if cep.size > idx:
                out[f"{prefix}_band_dct{idx}"] = float(cep[idx])
    return out


def stop_formant_transition_features(
    span: np.ndarray,
    sample_rate: int,
    *,
    burst_ms: float,
) -> dict[str, float]:
    """Post-release F2/F3 transition features for dorsal stop place.

    Palatal dorsal stops can be cued by how strongly the release pulls the
    following vowel frontward. These features measure robust early and later
    F2/F3 medians after the burst; they remain target-independent.
    """
    track = formant_track(span, sample_rate)
    if track is None:
        return {}

    def median_in(formant: np.ndarray, window: tuple[float, float]) -> float | None:
        lower = burst_ms + window[0]
        upper = burst_ms + window[1]
        selected = (track.times_ms >= lower) & (track.times_ms <= upper)
        values = formant[selected]
        finite = values[np.isfinite(values) & (values > 0.0)]
        return float(np.median(finite)) if finite.size else None

    f2_onset = median_in(track.f2, _STOP_TRANSITION_ONSET_MS)
    f2_mid = median_in(track.f2, _STOP_TRANSITION_MID_MS)
    f3_onset = median_in(track.f3, _STOP_TRANSITION_ONSET_MS)
    f3_mid = median_in(track.f3, _STOP_TRANSITION_MID_MS)

    out: dict[str, float] = {}
    if f2_onset is not None:
        out["transition_f2_onset_hz"] = f2_onset
    if f2_mid is not None:
        out["transition_f2_mid_hz"] = f2_mid
    if f2_onset is not None and f2_mid is not None:
        out["transition_f2_delta_hz"] = f2_onset - f2_mid
    if f3_onset is not None:
        out["transition_f3_onset_hz"] = f3_onset
    if f3_mid is not None:
        out["transition_f3_mid_hz"] = f3_mid
    if f3_onset is not None and f3_mid is not None:
        out["transition_f3_delta_hz"] = f3_onset - f3_mid
    return out


def _stop_affricate_dynamics(span: np.ndarray, sample_rate: int) -> dict[str, object]:
    """Closure/burst-anchored features for stops and affricates.

    Landmarks (closure floor, burst release), voicing track and the formant
    contour are each computed once and shared across VOT, burst moments,
    closure voicing and the post-burst F2 locus/slope.
    """
    out: dict[str, object] = {}
    landmarks = closure_burst_landmarks(span, sample_rate)
    if landmarks.burst_confidence is not None:
        out["burst_confidence"] = landmarks.burst_confidence
    burst_ms = landmarks.burst_ms
    if landmarks.closure_ms is not None and burst_ms is not None:
        out["closure_duration_ms"] = max(0.0, burst_ms - landmarks.closure_ms)

    if burst_ms is not None:
        raw_release_features = release_spectrum_features(
            span, sample_rate, burst_ms=burst_ms
        )
        raw_release_features.update(
            stop_formant_transition_features(span, sample_rate, burst_ms=burst_ms)
        )
        out["raw_release_features"] = raw_release_features
        start = max(0, int(round(burst_ms / 1000.0 * sample_rate)))
        end = min(span.size, start + int(round(_BURST_WINDOW_MS / 1000.0 * sample_rate)))
        segment = span[start:end]
        if segment.size >= 32:
            centroid, _bw, skew, _kurt = spectral_moments(segment, sample_rate)
            if centroid is not None:
                out["burst_centroid_hz"] = centroid
            if skew is not None:
                out["burst_spectral_skew"] = skew

    # Voicing is searched only from the closure onward, so a preceding vowel
    # leaking into the backward margin cannot be read as this stop's voicing.
    if burst_ms is not None:
        floor_ms = landmarks.closure_ms if landmarks.closure_ms is not None else 0.0
        times_v, voiced = voicing_track(span, sample_rate)
        after_closure = times_v >= floor_ms
        times_c = times_v[after_closure]
        voiced_c = voiced[after_closure]
        if voiced_c.size and voiced_c.any():
            onset_ms = float(times_c[int(np.argmax(voiced_c))])
            out["vot_ms"] = onset_ms - burst_ms
        # Closure voicing: periodicity between the closure floor and the burst.
        during_closure = voiced[(times_v >= floor_ms) & (times_v < burst_ms)]
        if during_closure.size:
            out["closure_voicing_ratio"] = float(during_closure.mean())

    if burst_ms is not None:
        track = formant_track(span, sample_rate)
        if track is not None:
            sel = (track.times_ms >= burst_ms) & (
                track.times_ms <= burst_ms + _F2_TRANSITION_MS
            )
            slope = linear_slope(track.times_ms[sel], track.f2[sel])
            if slope is not None:
                out["f2_transition_slope_hz_per_ms"] = slope
            finite_f2 = track.f2[sel][np.isfinite(track.f2[sel])]
            if finite_f2.size:
                out["f2_locus_hz"] = float(finite_f2[0])
    return out


def _fricative_dynamics(span: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Frication-interval-localised spectral moments, rise and duration."""
    out: dict[str, float] = {}
    interval = frication_interval(span, sample_rate)
    if interval is not None:
        start_ms, end_ms = interval
        out["frication_duration_ms"] = end_ms - start_ms
        a = max(0, int(round(start_ms / 1000.0 * sample_rate)))
        b = min(span.size, int(round(end_ms / 1000.0 * sample_rate)))
        segment = span[a:b] if b - a >= 32 else span
    else:
        segment = span
    if segment.size >= 32:
        _c, _bw, skew, kurt = spectral_moments(segment, sample_rate)
        if skew is not None:
            out["spectral_skew"] = skew
        if kurt is not None:
            out["spectral_kurtosis"] = kurt
    rise = _frication_rise_db_per_ms(span, sample_rate)
    if rise is not None:
        out["frication_rise_db_per_ms"] = rise
    return out


def _sonorant_dynamics(span: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Nasal murmur ratio and the F2 transition slope across the sonorant."""
    out: dict[str, float] = {}
    ratio = low_freq_energy_ratio(span, sample_rate, cutoff_hz=500.0)
    if ratio is not None:
        out["nasal_murmur_ratio"] = ratio
    track = formant_track(span, sample_rate)
    if track is not None:
        slope = linear_slope(track.times_ms, track.f2)
        if slope is not None:
            out["f2_transition_slope_hz_per_ms"] = slope
    return out


def _vowel_dynamics(nucleus: np.ndarray, sample_rate: int) -> dict[str, float]:
    """F2 movement across the vowel nucleus (diphthongisation cue)."""
    out: dict[str, float] = {}
    track = formant_track(nucleus, sample_rate)
    if track is not None:
        slope = linear_slope(track.times_ms, track.f2)
        if slope is not None:
            out["vowel_f2_movement_hz_per_ms"] = slope
    return out


def extract_dynamic_features(
    class_span: np.ndarray,
    sample_rate: int,
    target_class: "PhoneClass | None",
) -> DynamicFeatures:
    """Dispatch to the class-appropriate dynamic feature extractor."""
    values: dict[str, float] = {}
    if target_class in {PhoneClass.STOP, PhoneClass.AFFRICATE}:
        values.update(_stop_affricate_dynamics(class_span, sample_rate))
    if target_class in {PhoneClass.FRICATIVE, PhoneClass.AFFRICATE}:
        values.update(_fricative_dynamics(class_span, sample_rate))
    if target_class == PhoneClass.SONORANT:
        values.update(_sonorant_dynamics(class_span, sample_rate))
    if target_class == PhoneClass.VOWEL:
        values.update(_vowel_dynamics(class_span, sample_rate))
    return DynamicFeatures(**values)


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract FeatureSet for a single phone interval."""

    def __init__(self, nucleus_policy_ms: int = 80) -> None:
        self.nucleus_policy_ms = nucleus_policy_ms

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        interval: PhoneInterval,
        *,
        prev_phone: str | None = None,
        next_phone: str | None = None,
    ) -> FeatureSet:
        """Compute FeatureSet for one phone interval (mono float32 audio).

        Duration is measured from the signal voicing/energy edges within the
        aligned span. When voicing edges cannot be detected, falls back to the
        aligner span and lowers ``duration_reliability``. Vowel–vowel
        neighbourhoods further reduce ``duration_reliability``.
        """
        try:
            import librosa
        except ModuleNotFoundError:
            librosa = None  # type: ignore[assignment]

        audio_len = len(audio)

        s0, s1 = span_window_samples(audio_len, sample_rate, interval)
        span = audio[s0:s1].astype(np.float32)
        span_duration_ms = (s1 - s0) / sample_rate * 1000.0

        if len(span) > 0:
            rms = float(np.sqrt(np.mean(span ** 2)))
            if librosa is not None:
                centroid = float(
                    np.mean(librosa.feature.spectral_centroid(y=span, sr=sample_rate))
                )
                bandwidth = float(
                    np.mean(librosa.feature.spectral_bandwidth(y=span, sr=sample_rate))
                )
                mfcc_matrix = librosa.feature.mfcc(y=span, sr=sample_rate, n_mfcc=13)
                mfcc_mean: list[float] = mfcc_matrix.mean(axis=1).tolist()
                f0_arr, voiced_flag, _ = librosa.pyin(
                    span, fmin=75.0, fmax=600.0, sr=sample_rate
                )
                voiced_fraction = (
                    float(np.mean(voiced_flag)) if len(voiced_flag) > 0 else 0.0
                )
                f0_voiced = f0_arr[voiced_flag] if voiced_flag.any() else np.array([])
                f0_mean: float | None = (
                    float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else None
                )
            else:
                centroid_opt, bandwidth_opt, _skew, _kurt = spectral_moments(
                    span, sample_rate
                )
                centroid = 0.0 if centroid_opt is None else float(centroid_opt)
                bandwidth = 0.0 if bandwidth_opt is None else float(bandwidth_opt)
                mfcc_mean = _fallback_mfcc_like(span, sample_rate)
                voiced_fraction = _fallback_voiced_fraction(span, sample_rate)
                f0_mean = None
        else:
            rms = 0.0
            centroid = 0.0
            bandwidth = 0.0
            mfcc_mean = [0.0] * 13
            voiced_fraction = 0.0
            f0_mean = None

        # Signal-anchored timing. Edges are searched within the aligned span.
        voiced_edges = detect_voiced_edges(
            audio,
            sample_rate,
            start_ms=int(interval.start_ms),
            end_ms=int(interval.end_ms),
        )
        if voiced_edges is not None:
            voiced_start_ms = float(voiced_edges[0])
            voiced_end_ms = float(voiced_edges[1])
            active_duration_ms = voiced_end_ms - voiced_start_ms
        else:
            voiced_start_ms = None
            voiced_end_ms = None
            active_duration_ms = span_duration_ms

        reliability, duration_notes = duration_reliability_for(
            target_phone=interval.target_phone,
            prev_phone=prev_phone,
            next_phone=next_phone,
            voiced_detected=voiced_edges is not None,
        )

        n0, n1 = nucleus_window_samples(
            audio_len,
            sample_rate,
            interval,
            self.nucleus_policy_ms,
            voiced_start_ms=voiced_start_ms,
            voiced_end_ms=voiced_end_ms,
        )
        nucleus_duration_ms = (n1 - n0) / sample_rate * 1000.0

        target_class = _safe_phone_class(interval.target_phone)

        # Time-extended cues need a wider window than the ~20 ms aligned span;
        # the class window captures the vowel nucleus, burst, frication or
        # transition that a short raw anchor is too narrow to hold.
        c0, c1 = class_window_samples(audio_len, sample_rate, interval, target_class)
        class_span = audio[c0:c1].astype(np.float64)

        # Formants only for vowels and sonorants. Other classes leave them None.
        if target_class in {PhoneClass.VOWEL, PhoneClass.SONORANT}:
            f1, f2, f3 = _formants(class_span, sample_rate)
        else:
            f1, f2, f3 = None, None, None
        formant_ok = f1 is not None
        formant_confidence = 1.0 if formant_ok else 0.0

        dyn = extract_dynamic_features(class_span, sample_rate, target_class)

        feature_confidence, reasons = assess_confidence(
            duration_ms=active_duration_ms,
            nucleus_duration_ms=nucleus_duration_ms,
            formant_ok=formant_ok,
            voiced_fraction=voiced_fraction,
            target_phone=interval.target_phone,
            active_duration_ms=active_duration_ms,
            phone_class_=target_class,
        )

        return FeatureSet(
            duration_ms=active_duration_ms,
            rms=rms,
            spectral_centroid_hz=centroid,
            spectral_bandwidth_hz=bandwidth,
            mfcc=mfcc_mean,
            voiced_fraction=voiced_fraction,
            f0_mean_hz=f0_mean,
            f1_hz=f1,
            f2_hz=f2,
            f3_hz=f3,
            formant_confidence=formant_confidence,
            feature_confidence=feature_confidence,
            low_confidence_reasons=reasons,
            active_duration_ms=active_duration_ms,
            voiced_start_ms=voiced_start_ms,
            voiced_end_ms=voiced_end_ms,
            duration_reliability=reliability,
            duration_notes=duration_notes,
            spectral_skew=dyn.spectral_skew,
            spectral_kurtosis=dyn.spectral_kurtosis,
            vot_ms=dyn.vot_ms,
            closure_ms=dyn.closure_duration_ms,
            closure_duration_ms=dyn.closure_duration_ms,
            closure_voicing_ratio=dyn.closure_voicing_ratio,
            burst_centroid_hz=dyn.burst_centroid_hz,
            burst_spectral_skew=dyn.burst_spectral_skew,
            burst_confidence=dyn.burst_confidence,
            frication_rise_db_per_ms=dyn.frication_rise_db_per_ms,
            frication_duration_ms=dyn.frication_duration_ms,
            f2_transition_slope_hz_per_ms=dyn.f2_transition_slope_hz_per_ms,
            f2_locus_hz=dyn.f2_locus_hz,
            nasal_murmur_ratio=dyn.nasal_murmur_ratio,
            vowel_f2_movement_hz_per_ms=dyn.vowel_f2_movement_hz_per_ms,
            raw_release_features=dict(dyn.raw_release_features),
        )


# ---------------------------------------------------------------------------
# Phone class membership helpers
# ---------------------------------------------------------------------------

from g2p.constants import (  # noqa: E402
    ALL_CONSONANTS,
    ALL_VOWELS,
    SONORANTS,
)


class PhoneClass(str, Enum):
    """Coach-level phone classes with distinct acoustic signatures."""

    VOWEL = "vowel"
    SONORANT = "sonorant"
    FRICATIVE = "fricative"
    AFFRICATE = "affricate"
    STOP = "stop"


_FRICATIVES: frozenset[str] = frozenset({"f", "v", "s", "z", "ʃ", "ʒ", "h", "β", "β̞"})
_AFFRICATES: frozenset[str] = frozenset({"t͡ʃ", "d͡ʒ"})
_STOPS: frozenset[str] = frozenset(
    {"p", "b", "t", "d", "k", "ɡ", "c", "ɟ", "pʰ", "tʰ", "kʰ", "cʰ"}
)


def short_identity(phone: str) -> str:
    """Long vowels map to their short counterpart."""
    if phone.endswith("ː"):
        return phone[:-1]
    return phone


def is_vowel(phone: str) -> bool:
    return phone in ALL_VOWELS


def is_consonant(phone: str) -> bool:
    return phone in ALL_CONSONANTS


def phone_class(phone: str) -> PhoneClass:
    """Return the coach phone class for an IPA phoneme."""
    base = short_identity(phone)
    if base in ALL_VOWELS:
        return PhoneClass.VOWEL
    if base in SONORANTS:
        return PhoneClass.SONORANT
    if base in _FRICATIVES:
        return PhoneClass.FRICATIVE
    if base in _AFFRICATES:
        return PhoneClass.AFFRICATE
    if base in _STOPS:
        return PhoneClass.STOP
    raise ValueError(f"unknown phone class for {phone!r}")


CLASS_FEATURE_NAMES: dict[PhoneClass, tuple[str, ...]] = {
    PhoneClass.VOWEL: (
        "f1_hz",
        "f2_hz",
        "f3_hz",
        "vowel_f2_movement_hz_per_ms",
    ),
    PhoneClass.SONORANT: (
        "f2_hz",
        "spectral_centroid_hz",
        "voiced_fraction",
        "nasal_murmur_ratio",
        "f2_transition_slope_hz_per_ms",
    ),
    PhoneClass.FRICATIVE: (
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_skew",
        "spectral_kurtosis",
        "voiced_fraction",
        "frication_duration_ms",
    ),
    PhoneClass.AFFRICATE: (
        "frication_rise_db_per_ms",
        "burst_centroid_hz",
        "voiced_fraction",
        "spectral_centroid_hz",
        "closure_duration_ms",
    ),
    PhoneClass.STOP: (
        "vot_ms",
        "burst_centroid_hz",
        "burst_spectral_skew",
        "f2_transition_slope_hz_per_ms",
        "f2_locus_hz",
        "voiced_fraction",
        "closure_voicing_ratio",
    ),
}


def feature_names_for(phone: str) -> tuple[str, ...]:
    return CLASS_FEATURE_NAMES[phone_class(phone)]


def feature_vector_from_names(
    feature_set: FeatureSet, names: tuple[str, ...]
) -> list[float] | None:
    """Pull a feature vector using explicit field names; None if any field is None."""
    values: list[float] = []
    for name in names:
        v = getattr(feature_set, name, None)
        if v is None:
            return None
        values.append(float(v))
    return values


def feature_vector(feature_set: FeatureSet, phone: str) -> list[float] | None:
    """Pull the class-specific feature vector; None if any required field is None."""
    return feature_vector_from_names(feature_set, feature_names_for(phone))
