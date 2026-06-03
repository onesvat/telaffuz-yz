"""Speech timing contracts, RMS endpointing, and phone span construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Timing contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeechRegion:
    """Detected speech region in original-audio milliseconds."""

    start_ms_original: int
    end_ms_original: int
    status: str = "ok"

    def __post_init__(self) -> None:
        if self.start_ms_original < 0:
            raise ValueError("start_ms_original must be non-negative")
        if self.end_ms_original <= self.start_ms_original:
            raise ValueError("end_ms_original must be greater than start_ms_original")

    @property
    def duration_ms(self) -> int:
        return self.end_ms_original - self.start_ms_original

    def as_dict(self) -> dict[str, object]:
        return {
            "speech_start_ms_original": self.start_ms_original,
            "speech_end_ms_original": self.end_ms_original,
            "speech_duration_ms": self.duration_ms,
            "endpoint_status": self.status,
        }


@dataclass(frozen=True)
class TokenAnchor:
    """Raw aligner token anchor expressed in original-audio milliseconds."""

    target_phone: str
    start_ms_original: int
    end_ms_original: int
    confidence: float | None

    def __post_init__(self) -> None:
        if not self.target_phone:
            raise ValueError("target_phone must not be empty")
        if self.start_ms_original < 0:
            raise ValueError("start_ms_original must be non-negative")
        if self.end_ms_original <= self.start_ms_original:
            raise ValueError("end_ms_original must be greater than start_ms_original")

    @property
    def center_ms_original(self) -> int:
        return int(round((self.start_ms_original + self.end_ms_original) / 2.0))


@dataclass(frozen=True)
class PhoneSpan:
    """Continuous phone span in original-audio milliseconds."""

    target_phone: str
    start_ms_original: int
    end_ms_original: int
    anchor_start_ms_original: int
    anchor_end_ms_original: int
    confidence: float | None

    def __post_init__(self) -> None:
        if not self.target_phone:
            raise ValueError("target_phone must not be empty")
        if self.start_ms_original < 0:
            raise ValueError("start_ms_original must be non-negative")
        if self.end_ms_original <= self.start_ms_original:
            raise ValueError("end_ms_original must be greater than start_ms_original")
        if self.anchor_start_ms_original < 0:
            raise ValueError("anchor_start_ms_original must be non-negative")
        if self.anchor_end_ms_original <= self.anchor_start_ms_original:
            raise ValueError(
                "anchor_end_ms_original must be greater than anchor_start_ms_original"
            )
        anchor_center = int(
            round((self.anchor_start_ms_original + self.anchor_end_ms_original) / 2.0)
        )
        if not (self.start_ms_original <= anchor_center <= self.end_ms_original):
            raise ValueError("anchor center must be inside phone span")

    @property
    def duration_ms(self) -> int:
        return self.end_ms_original - self.start_ms_original

    def as_dict(self) -> dict[str, object]:
        return {
            "target_phone": self.target_phone,
            "start_ms": self.start_ms_original,
            "end_ms": self.end_ms_original,
            "duration_ms": self.duration_ms,
            "anchor_start_ms": self.anchor_start_ms_original,
            "anchor_end_ms": self.anchor_end_ms_original,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# RMS endpointing
# ---------------------------------------------------------------------------

_FRAME_MS = 20
_HOP_MS = 10
_NOISE_PERCENTILE = 20.0
_NOISE_MULTIPLIER = 3.0
_ABSOLUTE_THRESHOLD = 0.01
_MERGE_GAP_MS = 80
_PADDING_MS = 40


def detect_speech_region(samples: np.ndarray, sample_rate_hz: int) -> SpeechRegion:
    """Return the deterministic RMS speech region for mono audio samples."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    if audio.size == 0:
        raise ValueError("audio must contain at least one sample")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    clip_duration_ms = max(1, _samples_to_ms(audio.size, sample_rate_hz))
    frame_length = max(1, int(round(sample_rate_hz * _FRAME_MS / 1000.0)))
    hop_length = max(1, int(round(sample_rate_hz * _HOP_MS / 1000.0)))

    frame_starts = _frame_starts(audio.size, frame_length, hop_length)
    rms_values = np.asarray(
        [
            np.sqrt(np.mean(np.square(audio[start : start + frame_length])))
            for start in frame_starts
        ],
        dtype=np.float64,
    )
    noise_floor = float(np.percentile(rms_values, _NOISE_PERCENTILE))
    threshold = max(noise_floor * _NOISE_MULTIPLIER, _ABSOLUTE_THRESHOLD)
    speech_frame_indexes = np.flatnonzero(rms_values >= threshold)

    if speech_frame_indexes.size == 0:
        return SpeechRegion(0, clip_duration_ms, "no_speech_fallback")

    groups = _speech_groups(
        speech_frame_indexes, frame_starts, frame_length, sample_rate_hz
    )
    start_frame_index = groups[0][0]
    end_frame_index = groups[-1][1]
    start_sample = frame_starts[start_frame_index]
    end_sample = min(audio.size, frame_starts[end_frame_index] + frame_length)

    start_ms = max(0, _samples_to_ms(start_sample, sample_rate_hz) - _PADDING_MS)
    end_ms = min(
        clip_duration_ms,
        _samples_to_ms(end_sample, sample_rate_hz) + _PADDING_MS,
    )
    return SpeechRegion(start_ms, end_ms, "ok")


def _frame_starts(sample_count: int, frame_length: int, hop_length: int) -> list[int]:
    if sample_count <= frame_length:
        return [0]
    starts = list(range(0, sample_count - frame_length + 1, hop_length))
    final_start = sample_count - frame_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _speech_groups(
    speech_frame_indexes: np.ndarray,
    frame_starts: list[int],
    frame_length: int,
    sample_rate_hz: int,
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    group_start = int(speech_frame_indexes[0])
    previous = group_start
    max_gap_samples = sample_rate_hz * _MERGE_GAP_MS / 1000.0

    for raw_index in speech_frame_indexes[1:]:
        current = int(raw_index)
        gap_samples = frame_starts[current] - (frame_starts[previous] + frame_length)
        if max(0, gap_samples) < max_gap_samples:
            previous = current
            continue
        groups.append((group_start, previous))
        group_start = current
        previous = current

    groups.append((group_start, previous))
    return groups


def _samples_to_ms(sample_count: int, sample_rate_hz: int) -> int:
    return int(round(sample_count * 1000.0 / sample_rate_hz))


def detect_voiced_edges(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
) -> tuple[int, int] | None:
    """Locate energy-active edges within ``[start_ms, end_ms)`` (no padding).

    Returns ``(voiced_start_ms, voiced_end_ms)`` in the original audio timeline
    when at least one frame exceeds the RMS threshold; ``None`` otherwise.
    Stricter than :func:`detect_speech_region` — it never pads beyond the
    detected edges and never falls back to the full clip on pure silence.
    """
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    clip_ms = max(0, _samples_to_ms(audio.size, sample_rate_hz))
    start_ms = max(0, int(start_ms))
    end_ms = clip_ms if end_ms is None else max(start_ms, min(clip_ms, int(end_ms)))
    if end_ms - start_ms < _FRAME_MS:
        return None

    start_sample = max(0, int(round(start_ms * sample_rate_hz / 1000.0)))
    end_sample = min(audio.size, int(round(end_ms * sample_rate_hz / 1000.0)))
    frame_length = max(1, int(round(sample_rate_hz * _FRAME_MS / 1000.0)))
    if end_sample - start_sample < frame_length:
        return None

    region = audio[start_sample:end_sample]
    hop_length = max(1, int(round(sample_rate_hz * _HOP_MS / 1000.0)))
    frame_starts = _frame_starts(region.size, frame_length, hop_length)
    rms_values = np.asarray(
        [
            np.sqrt(np.mean(np.square(region[s : s + frame_length])))
            for s in frame_starts
        ],
        dtype=np.float64,
    )
    if rms_values.size == 0:
        return None
    noise_floor = float(np.percentile(rms_values, _NOISE_PERCENTILE))
    threshold = max(noise_floor * _NOISE_MULTIPLIER, _ABSOLUTE_THRESHOLD)
    active = np.flatnonzero(rms_values >= threshold)
    if active.size == 0:
        return None

    voiced_start_in_region = frame_starts[int(active[0])]
    voiced_end_in_region = min(
        region.size, frame_starts[int(active[-1])] + frame_length
    )
    voiced_start_ms = start_ms + _samples_to_ms(voiced_start_in_region, sample_rate_hz)
    voiced_end_ms = start_ms + _samples_to_ms(voiced_end_in_region, sample_rate_hz)
    if voiced_end_ms <= voiced_start_ms:
        return None
    return voiced_start_ms, voiced_end_ms


# ---------------------------------------------------------------------------
# Span construction from token anchors
# ---------------------------------------------------------------------------

def spans_from_token_anchors(
    anchors: Sequence[TokenAnchor],
    speech_region: SpeechRegion,
) -> list[PhoneSpan]:
    """Construct continuous phone spans from ordered token anchors."""
    if not anchors:
        return []

    centers = [anchor.center_ms_original for anchor in anchors]
    for previous_center, center in zip(centers, centers[1:]):
        if center < previous_center:
            raise ValueError("anchors must be ordered by center_ms_original")

    boundaries = [speech_region.start_ms_original]
    boundaries.extend(
        int(round((left_center + right_center) / 2.0))
        for left_center, right_center in zip(centers, centers[1:])
    )
    boundaries.append(speech_region.end_ms_original)

    spans: list[PhoneSpan] = []
    for anchor, start_ms_original, end_ms_original in zip(
        anchors, boundaries, boundaries[1:]
    ):
        if end_ms_original <= start_ms_original:
            raise ValueError("midpoint boundaries must produce positive spans")
        spans.append(
            PhoneSpan(
                target_phone=anchor.target_phone,
                start_ms_original=start_ms_original,
                end_ms_original=end_ms_original,
                anchor_start_ms_original=anchor.start_ms_original,
                anchor_end_ms_original=anchor.end_ms_original,
                confidence=anchor.confidence,
            )
        )

    return spans
