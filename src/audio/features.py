"""Acoustic feature extraction for phoneme-atlas phone windows."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from g2p.constants import ALL_VOWELS, PLAIN_STOPS


@dataclass(frozen=True)
class PhoneFeature:
    duration_ms: int
    rms: float | None
    intensity_db: float | None
    f0_hz: float | None
    f1_hz: float | None
    f2_hz: float | None
    f3_hz: float | None
    spectral_centroid_hz: float | None
    spectral_bandwidth_hz: float | None
    mfcc: tuple[float, ...]
    vot_ms: float | None
    closure_ms: float | None
    burst_confidence: float | None


def _slice_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    start = max(0, int(round(start_s * sample_rate)))
    end = min(len(audio), int(round(end_s * sample_rate)))
    if end <= start:
        return np.asarray([], dtype=np.float32)
    return np.asarray(audio[start:end], dtype=np.float32)


def spectral_moments(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (centroid_hz, bandwidth_hz, skew, kurtosis_excess) of the FFT spectrum.

    Skew and kurtosis are computed against the magnitude-spectrum distribution
    over rfft frequencies. Returns ``(None, None, None, None)`` when the
    signal is too short or carries no spectral mass.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size < 32:
        return None, None, None, None
    windowed = arr * np.hanning(arr.size)
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / sample_rate)
    total = float(mag.sum())
    if total <= 0:
        return None, None, None, None
    p = mag / total
    mean = float(np.sum(freqs * p))
    var = float(np.sum(((freqs - mean) ** 2) * p))
    if var <= 0:
        return mean, 0.0, None, None
    std = math.sqrt(var)
    centroid = mean
    bandwidth = std
    skew = float(np.sum(((freqs - mean) / std) ** 3 * p))
    kurt_excess = float(np.sum(((freqs - mean) / std) ** 4 * p)) - 3.0
    return centroid, bandwidth, skew, kurt_excess


def _rms_frames(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float,
) -> tuple[np.ndarray, int]:
    """Compute per-frame RMS and return (rms_values, frame_length_samples)."""
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    if samples.size < frame_length:
        return np.empty(0, dtype=np.float64), frame_length
    n = samples.size // frame_length
    if n == 0:
        return np.empty(0, dtype=np.float64), frame_length
    truncated = samples[: n * frame_length].astype(np.float64)
    return (
        np.sqrt(np.mean(truncated.reshape(n, frame_length) ** 2, axis=1)),
        frame_length,
    )


def detect_burst_ms(samples: np.ndarray, sample_rate: int) -> float | None:
    """Locate the burst onset within the given window.

    Picks the first frame whose RMS jumps to at least half the peak RMS.
    Returns the time offset in milliseconds from the start of ``samples``,
    or ``None`` when no clear onset is detectable.
    """
    arr = np.asarray(samples, dtype=np.float32)
    rms_values, frame_length = _rms_frames(arr, sample_rate=sample_rate, frame_ms=2.0)
    if rms_values.size < 4 or float(rms_values.max()) < 1e-4:
        return None
    peak = float(rms_values.max())
    threshold = 0.5 * peak
    above = np.flatnonzero(rms_values >= threshold)
    if above.size == 0:
        return None
    burst_idx = int(above[0])
    return float(burst_idx * frame_length * 1000.0 / sample_rate)


def detect_voicing_onset_ms(samples: np.ndarray, sample_rate: int) -> float | None:
    """Locate voicing onset (first periodic frame) inside the given window."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size < int(sample_rate * 0.02):
        return None
    try:
        import librosa
    except ImportError:
        return None
    try:
        _f0, voiced_flag, _ = librosa.pyin(
            arr, fmin=75.0, fmax=600.0, sr=sample_rate, frame_length=512
        )
    except Exception:
        return None
    if voiced_flag is None or not voiced_flag.any():
        return None
    hop_length = 128  # librosa default
    first_voiced_frame = int(np.argmax(voiced_flag))
    return float(first_voiced_frame * hop_length * 1000.0 / sample_rate)


def vot_ms_from_signal(samples: np.ndarray, sample_rate: int) -> float | None:
    """Voice Onset Time = voicing onset − burst, in milliseconds.

    Returns ``None`` when either the burst or the voicing onset cannot be
    detected. Negative values indicate prevoicing.
    """
    burst = detect_burst_ms(samples, sample_rate)
    voicing = detect_voicing_onset_ms(samples, sample_rate)
    if burst is None or voicing is None:
        return None
    return voicing - burst


def frication_rise_db_per_ms(samples: np.ndarray, sample_rate: int) -> float | None:
    """Slope of RMS-in-dB across the first half of a frication burst.

    Computes the linear least-squares slope of 20·log10(rms+ε) versus
    frame-centre time (in ms) over the first half of the signal. Returns
    ``None`` when too few frames are usable.
    """
    arr = np.asarray(samples, dtype=np.float32)
    rms_values, frame_length = _rms_frames(arr, sample_rate=sample_rate, frame_ms=5.0)
    if rms_values.size < 6:
        return None
    half = max(3, rms_values.size // 2)
    rms_half = rms_values[:half]
    eps = 1e-9
    db = 20.0 * np.log10(rms_half + eps)
    times_ms = (np.arange(half) + 0.5) * frame_length * 1000.0 / sample_rate
    if times_ms.size < 2:
        return None
    slope, _intercept = np.polyfit(times_ms, db, 1)
    return float(slope)


def formant_transition_slope_hz_per_ms(
    f_start: float | None,
    f_end: float | None,
    duration_ms: float,
) -> float | None:
    """Linear slope of a formant value across a span (Hz per ms)."""
    if f_start is None or f_end is None or duration_ms <= 0.0:
        return None
    return float((float(f_end) - float(f_start)) / float(duration_ms))


# ---------------------------------------------------------------------------
# Landmark and contour primitives (generic signal layer; no phone knowledge)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureBurst:
    """Stop/affricate landmarks, in ms from the start of the analysed window."""

    closure_ms: float | None
    burst_ms: float | None
    closure_depth_db: float | None
    burst_confidence: float | None


def _frame_db(
    samples: np.ndarray, sample_rate: int, frame_ms: float
) -> tuple[np.ndarray, int]:
    rms_values, frame_length = _rms_frames(
        samples, sample_rate=sample_rate, frame_ms=frame_ms
    )
    if rms_values.size == 0:
        return np.empty(0, dtype=np.float64), frame_length
    return 20.0 * np.log10(rms_values + 1e-9), frame_length


def closure_burst_landmarks(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 2.0,
    burst_rise_db: float = 9.0,
) -> ClosureBurst:
    """Locate the closure minimum and the burst release inside ``samples``.

    Keys on the rise out of the closure silence: the closure is the
    lowest-energy frame in the leading 80 % of the window, and the burst is
    the first later frame whose broadband energy climbs ``burst_rise_db``
    above the closure floor. Because it anchors on the closure floor — not a
    fraction of the (vowel-dominated) peak — a following vowel inside the
    window is never mistaken for the burst.
    """
    arr = np.asarray(samples, dtype=np.float32)
    db, frame_length = _frame_db(arr, sample_rate, frame_ms)
    if db.size < 5:
        return ClosureBurst(None, None, None, None)

    def _ms(idx: int) -> float:
        return float(idx * frame_length * 1000.0 / sample_rate)

    search_end = max(1, int(db.size * 0.8))
    closure_idx = int(np.argmin(db[:search_end]))
    floor = float(db[closure_idx])
    depth = float(db.max()) - floor
    after = db[closure_idx:]
    candidates = np.flatnonzero(after >= floor + burst_rise_db)
    if candidates.size == 0 or depth < burst_rise_db:
        return ClosureBurst(_ms(closure_idx), None, depth, None)
    burst_idx = closure_idx + int(candidates[0])
    return ClosureBurst(
        closure_ms=_ms(closure_idx),
        burst_ms=_ms(burst_idx),
        closure_depth_db=depth,
        burst_confidence=float(min(1.0, depth / 30.0)),
    )


def voicing_track(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_ms, voiced_bool) per pyin frame, or empty arrays."""
    arr = np.asarray(samples, dtype=np.float32)
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=bool))
    if arr.size < int(sample_rate * 0.02):
        return empty
    try:
        import librosa
    except ImportError:
        return empty
    try:
        _f0, voiced_flag, _ = librosa.pyin(
            arr, fmin=75.0, fmax=600.0, sr=sample_rate, frame_length=512
        )
    except Exception:
        return empty
    if voiced_flag is None or voiced_flag.size == 0:
        return empty
    hop_length = 128  # librosa default
    times = np.arange(voiced_flag.size) * hop_length * 1000.0 / sample_rate
    return times, np.asarray(voiced_flag, dtype=bool)


def _highfreq_energy_frames(
    samples: np.ndarray, sample_rate: int, cutoff_hz: float, frame_ms: float
) -> tuple[np.ndarray, int]:
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    if samples.size < frame_length:
        return np.empty(0, dtype=np.float64), frame_length
    n = samples.size // frame_length
    frames = samples[: n * frame_length].reshape(n, frame_length).astype(np.float64)
    mag = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1))
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    return mag[:, freqs >= cutoff_hz].sum(axis=1), frame_length


def frication_interval(
    samples: np.ndarray,
    sample_rate: int,
    *,
    cutoff_hz: float = 2000.0,
    frame_ms: float = 5.0,
) -> tuple[float, float] | None:
    """Return (start_ms, end_ms) of the longest sustained high-frequency run.

    Frames whose energy above ``cutoff_hz`` exceeds half the peak count as
    frication; the longest contiguous run is returned. ``None`` when there is
    no sustained turbulence.
    """
    arr = np.asarray(samples, dtype=np.float64)
    hi, frame_length = _highfreq_energy_frames(arr, sample_rate, cutoff_hz, frame_ms)
    if hi.size < 3 or float(hi.max()) <= 0.0:
        return None
    mask = hi >= 0.5 * float(hi.max())
    best_start, best_len, run_start = 0, 0, None
    for i, active in enumerate([*mask.tolist(), False]):
        if active and run_start is None:
            run_start = i
        elif not active and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if best_len == 0:
        return None

    def _ms(idx: int) -> float:
        return float(idx * frame_length * 1000.0 / sample_rate)

    return _ms(best_start), _ms(best_start + best_len)


@dataclass(frozen=True)
class FormantTrack:
    """Continuous formant contour; values may be NaN where undefined."""

    times_ms: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


def formant_track(
    samples: np.ndarray,
    sample_rate: int,
    *,
    time_step: float = 0.0075,
    max_formant_hz: float = 5500.0,
) -> FormantTrack | None:
    """Sample a Praat Burg formant contour over ``samples``."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size < int(sample_rate * 0.025):
        return None
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None
    try:
        sound = parselmouth.Sound(arr, sampling_frequency=float(sample_rate))
        formant = sound.to_formant_burg(
            time_step=time_step, maximum_formant=max_formant_hz
        )
        n_frames = int(call(formant, "Get number of frames"))
    except Exception:
        return None
    if n_frames < 2:
        return None
    times, f1s, f2s, f3s = [], [], [], []
    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        times.append(t * 1000.0)
        f1s.append(call(formant, "Get value at time", 1, t, "hertz", "linear"))
        f2s.append(call(formant, "Get value at time", 2, t, "hertz", "linear"))
        f3s.append(call(formant, "Get value at time", 3, t, "hertz", "linear"))
    return FormantTrack(
        np.asarray(times), np.asarray(f1s), np.asarray(f2s), np.asarray(f3s)
    )


def linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope of y over x across finite pairs; None if < 2 usable."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(mask.sum()) < 2:
        return None
    xs, ys = xs[mask], ys[mask]
    if float(np.ptp(xs)) <= 0.0:
        return None
    slope, _intercept = np.polyfit(xs, ys, 1)
    return float(slope)


def low_freq_energy_ratio(
    samples: np.ndarray, sample_rate: int, *, cutoff_hz: float = 500.0
) -> float | None:
    """Fraction of spectral magnitude below ``cutoff_hz`` (nasal murmur cue)."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size < 32:
        return None
    mag = np.abs(np.fft.rfft(arr * np.hanning(arr.size)))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / sample_rate)
    total = float(mag.sum())
    if total <= 0.0:
        return None
    return float(mag[freqs < cutoff_hz].sum()) / total


def measurement_window(
    span_start_ms: float,
    span_end_ms: float,
    voiced_start_ms: float | None,
    voiced_end_ms: float | None,
    *,
    policy_ms: int = 80,
    min_width_ms: float = 1.0,
) -> tuple[float, float]:
    """Build a measurement window anchored to the active signal region.

    Centred on the midpoint of the voiced interval when ``voiced_start_ms`` and
    ``voiced_end_ms`` are both supplied and ordered; otherwise the window
    falls back to the geometric centre of the span. Width is capped at
    ``policy_ms`` and at 60% of the active duration. Result is always clamped
    to the span bounds.
    """
    span_start = float(span_start_ms)
    span_end = float(span_end_ms)
    if span_end <= span_start:
        raise ValueError("span must be positive")

    if (
        voiced_start_ms is not None
        and voiced_end_ms is not None
        and float(voiced_end_ms) > float(voiced_start_ms)
    ):
        active_start = max(span_start, float(voiced_start_ms))
        active_end = min(span_end, float(voiced_end_ms))
        if active_end > active_start:
            active_dur = active_end - active_start
            width = min(float(policy_ms), active_dur * 0.6)
            half = max(width / 2.0, min_width_ms / 2.0)
            centre = (active_start + active_end) / 2.0
            return (
                max(active_start, centre - half),
                min(active_end, centre + half),
            )

    centre = (span_start + span_end) / 2.0
    span_dur = span_end - span_start
    width = min(float(policy_ms), span_dur * 0.6)
    half = max(width / 2.0, min_width_ms / 2.0)
    return (max(span_start, centre - half), min(span_end, centre + half))


def _rms(samples: np.ndarray) -> float | None:
    if samples.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(samples))))


def _spectral_features(
    samples: np.ndarray,
    *,
    sample_rate: int,
) -> tuple[float | None, float | None, tuple[float, ...]]:
    if samples.size < 2:
        return None, None, tuple(0.0 for _ in range(13))
    windowed = samples * np.hanning(samples.size)
    magnitudes = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    total = float(np.sum(magnitudes))
    if total <= 0:
        return None, None, tuple(0.0 for _ in range(13))
    centroid = float(np.sum(freqs * magnitudes) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / total))

    log_bins = [np.log1p(chunk) for chunk in np.array_split(magnitudes, 13)]
    mfcc_like = tuple(float(np.mean(bin_values)) for bin_values in log_bins)
    return centroid, bandwidth, mfcc_like


def _f0_for_vowel(samples: np.ndarray, *, sample_rate: int, phone: str) -> float | None:
    if phone not in ALL_VOWELS or samples.size < sample_rate // 100:
        return None
    centered = samples - float(np.mean(samples))
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if corr.size == 0 or corr[0] <= 0:
        return None
    min_lag = max(1, int(sample_rate / 500))
    max_lag = min(corr.size, int(sample_rate / 60))
    if max_lag <= min_lag:
        return None
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    if lag <= 0:
        return None
    return float(sample_rate / lag)


def extract_phone_feature(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start_s: float,
    end_s: float,
    phone: str,
) -> PhoneFeature:
    """Extract a small, null-tolerant feature JSON payload for one phone span."""
    samples = _slice_audio(audio, sample_rate=sample_rate, start_s=start_s, end_s=end_s)
    duration_ms = int(round(max(0.0, end_s - start_s) * 1000.0))
    rms = _rms(samples)
    intensity = 20.0 * math.log10(rms + 1e-12) if rms is not None else None
    centroid, bandwidth, mfcc = _spectral_features(samples, sample_rate=sample_rate)
    f0 = _f0_for_vowel(samples, sample_rate=sample_rate, phone=phone)

    # VOT/closure/burst need stop-release localization. Keep null unless a
    # downstream detector supplies reliable boundaries.
    is_stop = phone in PLAIN_STOPS or phone in {"pʰ", "tʰ", "kʰ", "cʰ"}
    return PhoneFeature(
        duration_ms=duration_ms,
        rms=rms,
        intensity_db=intensity,
        f0_hz=f0,
        f1_hz=None,
        f2_hz=None,
        f3_hz=None,
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=bandwidth,
        mfcc=mfcc,
        vot_ms=None if is_stop else None,
        closure_ms=None if is_stop else None,
        burst_confidence=None if is_stop else None,
    )


def feature_to_json(feature: PhoneFeature) -> str:
    return json.dumps(asdict(feature), ensure_ascii=False, sort_keys=True)
