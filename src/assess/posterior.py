"""Posterior frame to phone-span evidence helpers.

Model- and audio-independent helpers for projecting wav2vec frame posteriors
onto phone spans and deriving target-vs-best identity evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PosteriorFrame:
    frame_index: int
    start_ms: int
    end_ms: int
    token_logprobs: Mapping[str, float] = field(default_factory=dict)

    @property
    def best_phone(self) -> str | None:
        if not self.token_logprobs:
            return None
        return max(self.token_logprobs, key=self.token_logprobs.__getitem__)

    @property
    def best_logprob(self) -> float | None:
        phone = self.best_phone
        return None if phone is None else float(self.token_logprobs[phone])


@dataclass(frozen=True)
class PosteriorTargetAnchor:
    target_token: str
    start_ms: int
    end_ms: int
    start_frame_index: int
    end_frame_index: int
    confidence: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "target_token": self.target_token,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "confidence": self.confidence,
        }


def frame_posteriors_from_logits(
    logits: np.ndarray,
    id_to_token: Mapping[int, str],
    *,
    drop_tokens: set[str] | frozenset[str],
    total_ms: int,
) -> tuple[PosteriorFrame, ...]:
    """Logits (T, V) → frame-level log posterior map.

    drop_tokens are excluded from the observed/best phone candidates; the
    canonical GOP backbone operates over real phoneme tokens only.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"logits must be 2-D (T,V), got shape {arr.shape}")
    n_frames = arr.shape[0]
    if n_frames == 0:
        return ()

    shifted = arr - arr.max(axis=1, keepdims=True)
    log_denom = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_denom

    frames: list[PosteriorFrame] = []
    for frame_index in range(n_frames):
        token_logprobs: dict[str, float] = {}
        for token_id in range(arr.shape[1]):
            token = id_to_token.get(token_id)
            if token is None or token in drop_tokens:
                continue
            token_logprobs[token] = float(log_probs[frame_index, token_id])
        frames.append(
            PosteriorFrame(
                frame_index=frame_index,
                start_ms=round(frame_index * total_ms / n_frames),
                end_ms=round((frame_index + 1) * total_ms / n_frames),
                token_logprobs=token_logprobs,
            )
        )
    return tuple(frames)


def force_align_targets_to_posteriors(
    target_tokens: Sequence[str],
    frames: Iterable[PosteriorFrame],
) -> tuple[PosteriorTargetAnchor, ...]:
    """Monotonic Viterbi alignment from frame posteriors to target tokens.

    Every target receives at least one frame. Transitions can either stay on the
    current target or advance by one target per frame; blanks are intentionally
    absent because ``PosteriorFrame`` already carries phoneme-only posteriors.
    Returns an empty tuple when no complete path exists.
    """
    targets = tuple(target_tokens)
    frame_tuple = tuple(frames)
    n_targets = len(targets)
    n_frames = len(frame_tuple)
    if n_targets == 0 or n_frames == 0 or n_frames < n_targets:
        return ()

    scores = np.full((n_targets, n_frames), -np.inf, dtype=np.float64)
    back = np.full((n_targets, n_frames), -1, dtype=np.int64)

    first_lp = _target_logprob(frame_tuple[0], targets[0])
    if first_lp is None:
        return ()
    scores[0, 0] = first_lp

    for frame_index in range(1, n_frames):
        max_target_index = min(n_targets - 1, frame_index)
        for target_index in range(max_target_index + 1):
            lp = _target_logprob(frame_tuple[frame_index], targets[target_index])
            if lp is None:
                continue
            stay_score = scores[target_index, frame_index - 1]
            advance_score = (
                scores[target_index - 1, frame_index - 1]
                if target_index > 0
                else -np.inf
            )
            if advance_score > stay_score:
                previous_target_index = target_index - 1
                previous_score = advance_score
            else:
                previous_target_index = target_index
                previous_score = stay_score
            if not np.isfinite(previous_score):
                continue
            scores[target_index, frame_index] = previous_score + lp
            back[target_index, frame_index] = previous_target_index

    if not np.isfinite(scores[n_targets - 1, n_frames - 1]):
        return ()

    assignment = [0] * n_frames
    target_index = n_targets - 1
    for frame_index in range(n_frames - 1, -1, -1):
        assignment[frame_index] = target_index
        if frame_index > 0:
            target_index = int(back[target_index, frame_index])
            if target_index < 0:
                return ()

    anchors: list[PosteriorTargetAnchor] = []
    for target_index, token in enumerate(targets):
        indexes = [i for i, assigned in enumerate(assignment) if assigned == target_index]
        if not indexes:
            return ()
        first = frame_tuple[indexes[0]]
        last = frame_tuple[indexes[-1]]
        confidences = [
            math.exp(float(frame_tuple[i].token_logprobs[token]))
            for i in indexes
            if token in frame_tuple[i].token_logprobs
        ]
        confidence = (
            float(sum(confidences) / len(confidences)) if confidences else None
        )
        anchors.append(
            PosteriorTargetAnchor(
                target_token=token,
                start_ms=first.start_ms,
                end_ms=last.end_ms,
                start_frame_index=first.frame_index,
                end_frame_index=last.frame_index,
                confidence=confidence,
            )
        )

    return tuple(anchors)


def _target_logprob(frame: PosteriorFrame, target_token: str) -> float | None:
    value = frame.token_logprobs.get(target_token)
    return None if value is None else float(value)


# Raw phone identity margin = log P(target) - log P(argmax) <= 0.
# Sigmoid is shifted so that strong target evidence and clear alternatives land
# on opposite sides of 0.5.
GOP_SHIFT: float = 1.0


@dataclass(frozen=True)
class PeakFrameEvidence:
    """Posterior evidence at the frame with the highest target log-probability."""

    target_logprob: float | None
    best_logprob: float | None
    observed_phone: str | None
    peak_frame_index: int | None
    frame_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "target_logprob": self.target_logprob,
            "best_logprob": self.best_logprob,
            "observed_phone": self.observed_phone,
            "peak_frame_index": self.peak_frame_index,
            "frame_count": self.frame_count,
        }


def peak_frame_posterior(
    span_start_ms: int,
    span_end_ms: int,
    frames: Iterable[PosteriorFrame],
    target_token: str,
) -> PeakFrameEvidence:
    """Locate the frame inside ``[span_start_ms, span_end_ms)`` with the
    highest target-token log probability and return its identity evidence.

    ``best_logprob`` and ``observed_phone`` are read from the same peak frame
    not averaged across the span. This anchors identity evidence to the point
    where the target is most likely, rather than a span mean that blends ramps
    and transitions.
    """
    overlapping: list[PosteriorFrame] = []
    for frame in frames:
        if frame.end_ms <= span_start_ms or frame.start_ms >= span_end_ms:
            continue
        overlapping.append(frame)

    if not overlapping:
        return PeakFrameEvidence(None, None, None, None, 0)

    best_frame: PosteriorFrame | None = None
    best_target_logprob: float | None = None
    for frame in overlapping:
        target_lp = frame.token_logprobs.get(target_token)
        if target_lp is None:
            continue
        if best_target_logprob is None or target_lp > best_target_logprob:
            best_target_logprob = float(target_lp)
            best_frame = frame

    if best_frame is None:
        # Target token missing from vocab over the whole span; emit
        # frame_count so callers can still distinguish OOV from no_speech.
        return PeakFrameEvidence(None, None, None, None, len(overlapping))

    best_phone = best_frame.best_phone
    best_logprob = (
        float(best_frame.token_logprobs[best_phone])
        if best_phone is not None
        else None
    )
    return PeakFrameEvidence(
        target_logprob=best_target_logprob,
        best_logprob=best_logprob,
        observed_phone=best_phone,
        peak_frame_index=best_frame.frame_index,
        frame_count=len(overlapping),
    )


def compute_gop_score(
    canonical_logprob: float | None,
    best_logprob: float | None,
    *,
    fallback: float | None = None,
    shift: float = GOP_SHIFT,
) -> float | None:
    """Target-vs-best posterior margin → [0, 1] continuous identity score.

    raw = canonical_logprob − best_logprob (≤ 0; 0 = perfect match, very
    negative = model prefers a different phoneme). sigmoid(raw + shift)
    compresses the value into [0, 1]. Numerical noise may push raw slightly
    above 0, giving a score near the upper bound but never exceeding 1.

    If either posterior field is missing, returns ``fallback``.
    """
    if canonical_logprob is None or best_logprob is None:
        return fallback
    raw = float(canonical_logprob) - float(best_logprob)
    z = raw + float(shift)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
