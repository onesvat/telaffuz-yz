"""Length channel: per-phone duration typicality from wav2vec-posterior spans.

This channel is intentionally separate from the quality GMM. Its reference
durations come from the SAME wav2vec-posterior timing the runtime uses, so the
two sides are comparable. The older reference aligner cannot source it: that aligner
floors phone spans at the ~20 ms frame, collapsing every phone to a degenerate
duration (see the duration-channel notes). The cue is soft and non-gating: it
annotates a length note and contributes a length_score, never a verdict.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from g2p.constants import LONG_TO_SHORT

DEFAULT_TAU = 2.0
DEFAULT_MIN_SAMPLES = 50
LONG_VOWEL_SHORT_TAU = 0.5
LONG_VOWEL_SHORT_RATIO = 0.75
LENGTH_NOTE_TO_FEEDBACK = {"short": "too_short", "long": "too_long"}


def feedback_note_for_length(length_note: str | None) -> str | None:
    """Map a length cue note to the legacy duration-note vocabulary."""
    return LENGTH_NOTE_TO_FEEDBACK.get(length_note or "")


def _robust_scale(durations: list[float]) -> tuple[float, float]:
    """Median and IQR/1.349 robust scale; scale 0.0 marks a degenerate spread."""
    ordered = sorted(durations)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0

    def _pct(p: float) -> float:
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    scale = (_pct(0.75) - _pct(0.25)) / 1.349
    if not math.isfinite(scale) or scale <= 0.0:
        return median, 0.0
    return median, scale


@dataclass(frozen=True)
class PhoneDuration:
    phone: str
    median_ms: float
    scale_ms: float
    n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "phone": self.phone,
            "median_ms": self.median_ms,
            "scale_ms": self.scale_ms,
            "n": self.n,
        }


@dataclass(frozen=True)
class DurationStats:
    phones: dict[str, PhoneDuration]

    def get(self, phone: str) -> PhoneDuration | None:
        return self.phones.get(phone)

    def as_dict(self) -> dict[str, object]:
        return {
            "channel_version": "assess_coach_length_v1",
            "phones": {p: d.as_dict() for p, d in self.phones.items()},
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> DurationStats:
        raw = data.get("phones", {})
        phones: dict[str, PhoneDuration] = {}
        for phone, entry in raw.items():  # type: ignore[union-attr]
            phones[phone] = PhoneDuration(
                phone=phone,
                median_ms=float(entry["median_ms"]),
                scale_ms=float(entry["scale_ms"]),
                n=int(entry["n"]),
            )
        return cls(phones=phones)

    @classmethod
    def load(cls, path: Path | str) -> DurationStats:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LengthCue:
    z: float | None
    note: str | None
    score: float | None

    def as_dict(self) -> dict[str, object]:
        return {"z": self.z, "note": self.note, "score": self.score}


_UNSCORED = LengthCue(z=None, note=None, score=None)


def build_duration_stats(
    rows: Iterable[Mapping[str, object]],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> DurationStats:
    """Per-phone duration distribution from (phone, dur_ms) rows.

    Phones with fewer than ``min_samples`` observations are omitted, so the
    runtime cue stays unscored for phones whose reference is too thin to trust
    (e.g. the rarer long vowels).
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        phone = str(row["phone"])
        dur = float(row["dur_ms"])
        if math.isfinite(dur) and dur > 0:
            buckets[phone].append(dur)
    phones: dict[str, PhoneDuration] = {}
    for phone, durs in buckets.items():
        if len(durs) < min_samples:
            continue
        median, scale = _robust_scale(durs)
        phones[phone] = PhoneDuration(phone, median, scale, len(durs))
    return DurationStats(phones=phones)


def length_cue(
    phone: str,
    span_ms: float,
    stats: DurationStats | None,
    *,
    tau: float = DEFAULT_TAU,
) -> LengthCue:
    """Soft length typicality of a runtime span against the phone's reference.

    Returns an unscored cue when no reference distribution exists or its spread
    is degenerate. The score is a Gaussian typicality in (0, 1]: 1.0 at the
    reference median, decaying as the span deviates.
    """
    if stats is None:
        return _UNSCORED
    entry = stats.get(phone)
    if entry is None or entry.scale_ms <= 0.0:
        return _UNSCORED
    z = (float(span_ms) - entry.median_ms) / entry.scale_ms
    if z < -tau:
        note = "short"
    elif z > tau:
        note = "long"
    else:
        note = "expected"
    clear_short_threshold = _long_vowel_clear_short_threshold(phone, entry, stats)
    if clear_short_threshold is not None and float(span_ms) <= clear_short_threshold:
        note = "short"
    score = math.exp(-0.5 * (z / tau) ** 2)
    return LengthCue(z=z, note=note, score=score)


def _long_vowel_clear_short_threshold(
    phone: str,
    long_entry: PhoneDuration,
    stats: DurationStats,
) -> float | None:
    """High-precision shortness threshold for long-vowel targets.

    The long-vowel corpus distributions overlap their short counterparts, so a
    standard z-threshold is too permissive for careful isolated productions.
    This threshold only marks tokens that are short both relative to the long
    median and to the short-vowel reference distribution.
    """
    short_phone = LONG_TO_SHORT.get(phone)
    if short_phone is None:
        return None
    thresholds: list[float] = []
    if math.isfinite(long_entry.median_ms) and long_entry.median_ms > 0.0:
        thresholds.append(long_entry.median_ms * LONG_VOWEL_SHORT_RATIO)
    short_entry = stats.get(short_phone)
    if (
        short_entry is not None
        and short_entry.scale_ms > 0.0
        and math.isfinite(short_entry.median_ms)
        and math.isfinite(short_entry.scale_ms)
    ):
        thresholds.append(
            short_entry.median_ms - LONG_VOWEL_SHORT_TAU * short_entry.scale_ms
        )
    positive = [value for value in thresholds if math.isfinite(value) and value > 0.0]
    return min(positive) if positive else None
