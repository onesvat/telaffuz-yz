"""Coach orchestration: default services wiring and assessment runner."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g2p.constants import CTC_SPC, LONG_TO_SHORT, SHORT_TO_LONG, SONORANTS, STRESS
from g2p.pipeline import transcribe_training_text

from assess.contracts import (
    CoachPhoneResult,
    CoachRequest,
    CoachResult,
    PhoneInterval,
    Wav2VecEvidence,
)
from assess.length import (
    DurationStats,
    LengthCue,
    feedback_note_for_length,
    length_cue,
)
from assess.phonetic_distance import score_phone
from assess.posterior import (
    PosteriorFrame,
    compute_gop_score,
    force_align_targets_to_posteriors,
    peak_frame_posterior,
)

NON_TIMED_TOKENS = frozenset({CTC_SPC, STRESS})
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.toml"
DEFAULT_COACH_LENGTH_PATH = REPO_ROOT / "configs" / "coach_duration_v1.json"
DEFAULT_WAV2VEC_MIN_FRAMES = 1
RESULT_EXCLUDED_TOKENS: frozenset[str] = frozenset({CTC_SPC, STRESS})
PAIRWISE_MIN_MARGIN = 0.10
PAIRWISE_MIN_RECALL = 0.15
PAIRWISE_MIN_QUALITY_LIFT = 0.20
PAIRWISE_UNSCORED_MIN_MARGIN = 0.35
PAIRWISE_UNSCORED_MIN_RECALL = 0.20

# Tokens to drop from wav2vec posterior (suprasegmentals + specials)
_WAV2VEC_DROP_TOKENS: frozenset[str] = frozenset(
    {
        "<pad>", "<unk>", "<s>", "</s>", "<sil>", "<blank>", "<spc>",
        "[PAD]", "[UNK]", "|", "spc",
        "ˈ", "ˌ",
    }
)

# Long-vowel colon normalization
_COLON_LENGTH: dict[str, str] = {
    "a:": "aː", "e:": "eː", "i:": "iː", "o:": "oː",
    "u:": "uː", "y:": "yː", "œ:": "œː", "ɯ:": "ɯː",
}


def normalize_token(token: str) -> str:
    """Map a raw model token to its canonical IPA form."""
    return _COLON_LENGTH.get(token, token)


def target_in_vocab(target_phone: str, normalized_vocab: set[str]) -> bool:
    return normalize_token(target_phone) in normalized_vocab


# ---------------------------------------------------------------------------
# Target sequence resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSequence:
    all_phones: list[str]
    timed_phones: list[str]
    source_indices: list[int]


@dataclass(frozen=True)
class TimingResult:
    status: str
    speech_start_ms_original: int
    speech_end_ms_original: int
    intervals: list[PhoneInterval]


# ---------------------------------------------------------------------------
# Wav2Vec evidence helpers (v2-style posterior span binding)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodedPhone:
    ipa: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass(frozen=True)
class QuantitySpan:
    """CTC-neighbour span used for phone duration/quantity evidence."""

    start_ms: int
    end_ms: int
    source: str
    reliability: float
    notes: tuple[str, ...] = ()

    @property
    def duration_ms(self) -> float:
        return float(max(0, self.end_ms - self.start_ms))

    def as_dict(self) -> dict[str, object]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "source": self.source,
            "reliability": self.reliability,
            "notes": list(self.notes),
        }


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    if end_a <= start_a or end_b <= start_b:
        return 0.0
    return float(max(0, min(end_a, end_b) - max(start_a, start_b)))


def _normalized_logprobs(token_logprobs: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for token, logprob in token_logprobs.items():
        nt = normalize_token(token)
        current = normalized.get(nt)
        if current is None or logprob > current:
            normalized[nt] = logprob
    return normalized


def _normalized_posterior_frame(frame: object) -> PosteriorFrame:
    return PosteriorFrame(
        frame_index=int(getattr(frame, "frame_index", 0)),
        start_ms=int(getattr(frame, "start_ms", 0)),
        end_ms=int(getattr(frame, "end_ms", 0)),
        token_logprobs=_normalized_logprobs(
            dict(getattr(frame, "token_logprobs", {}))
        ),
    )


def _unusable_evidence(target_phone: str, frame_count: int, reason: str) -> Wav2VecEvidence:
    return Wav2VecEvidence(
        target_phone=target_phone,
        observed_phone=None,
        target_logprob=None,
        best_logprob=None,
        gop_raw=None,
        observed_matches_target=None,
        frame_count=frame_count,
        confidence=0.0,
        reasons=[reason],
        peak_frame_index=None,
        identity_score=None,
    )


def evidence_for_span(
    interval: PhoneInterval,
    frames: list[object],
    normalized_vocab: set[str],
    min_frames: int,
) -> Wav2VecEvidence:
    target = normalize_token(interval.target_phone)
    if not target_in_vocab(interval.target_phone, normalized_vocab):
        return _unusable_evidence(interval.target_phone, 0, "target_oov_for_model")

    posterior_frames = [_normalized_posterior_frame(frame) for frame in frames]
    peak = peak_frame_posterior(
        interval.start_ms,
        interval.end_ms,
        posterior_frames,
        target,
    )
    if peak.frame_count == 0:
        return _unusable_evidence(interval.target_phone, 0, "no_frames")
    if peak.frame_count < min_frames:
        return _unusable_evidence(
            interval.target_phone,
            peak.frame_count,
            "sparse_posterior",
        )
    if (
        peak.target_logprob is None
        or peak.best_logprob is None
        or peak.observed_phone is None
    ):
        return _unusable_evidence(
            interval.target_phone,
            peak.frame_count,
            "target_posterior_missing",
        )

    observed_phone = peak.observed_phone
    target_logprob = peak.target_logprob
    best_logprob = peak.best_logprob
    gop_raw = target_logprob - best_logprob

    return Wav2VecEvidence(
        target_phone=interval.target_phone,
        observed_phone=observed_phone,
        target_logprob=target_logprob,
        best_logprob=best_logprob,
        gop_raw=gop_raw,
        observed_matches_target=(observed_phone == target),
        frame_count=peak.frame_count,
        confidence=1.0,
        reasons=[],
        peak_frame_index=peak.peak_frame_index,
        identity_score=compute_gop_score(target_logprob, best_logprob),
    )


def evidence_for_spans(
    spans: list[PhoneInterval],
    frames: list[object],
    vocab: set[str],
    min_frames: int,
) -> list[Wav2VecEvidence]:
    normalized_vocab = {normalize_token(t) for t in vocab}
    return [
        evidence_for_span(span, frames, normalized_vocab, min_frames)
        for span in spans
    ]


def timing_from_wav2vec_posteriors(
    target_phones: list[str],
    frames: list[object],
    speech_region: object,
) -> TimingResult:
    """Build coach timing spans from wav2vec posteriors without an external aligner."""
    from audio.timing import TokenAnchor, spans_from_token_anchors  # noqa: PLC0415

    speech_start_ms = int(getattr(speech_region, "start_ms_original"))
    speech_end_ms = int(getattr(speech_region, "end_ms_original"))
    frame_list = list(frames)
    if str(getattr(speech_region, "status", "ok")) != "ok":
        return TimingResult(
            status="no_speech",
            speech_start_ms_original=speech_start_ms,
            speech_end_ms_original=speech_end_ms,
            intervals=[],
        )
    if not target_phones:
        return TimingResult(
            status="alignment_failed",
            speech_start_ms_original=speech_start_ms,
            speech_end_ms_original=speech_end_ms,
            intervals=[],
        )

    posterior_frames = [
        _normalized_posterior_frame(frame)
        for frame in frame_list
        if _overlap_ms(
            speech_start_ms,
            speech_end_ms,
            int(getattr(frame, "start_ms", 0)),
            int(getattr(frame, "end_ms", 0)),
        )
        > 0
    ]
    anchors = force_align_targets_to_posteriors(
        [normalize_token(phone) for phone in target_phones],
        posterior_frames,
    )
    if len(anchors) != len(target_phones):
        return TimingResult(
            status="alignment_failed",
            speech_start_ms_original=speech_start_ms,
            speech_end_ms_original=speech_end_ms,
            intervals=[],
        )

    token_anchors: list[TokenAnchor] = []
    for target_phone, anchor in zip(target_phones, anchors, strict=True):
        anchor_start = max(speech_start_ms, anchor.start_ms)
        anchor_end = min(speech_end_ms, anchor.end_ms)
        if anchor_end <= anchor_start:
            return TimingResult(
                status="alignment_failed",
                speech_start_ms_original=speech_start_ms,
                speech_end_ms_original=speech_end_ms,
                intervals=[],
            )
        token_anchors.append(
            TokenAnchor(
                target_phone=target_phone,
                start_ms_original=anchor_start,
                end_ms_original=anchor_end,
                confidence=anchor.confidence,
            )
        )

    spans = spans_from_token_anchors(token_anchors, speech_region)
    return TimingResult(
        status="ok",
        speech_start_ms_original=speech_start_ms,
        speech_end_ms_original=speech_end_ms,
        intervals=[
            PhoneInterval(
                target_phone=span.target_phone,
                start_ms=span.start_ms_original,
                end_ms=span.end_ms_original,
                speech_start_ms_original=speech_start_ms,
                speech_end_ms_original=speech_end_ms,
            )
            for span in spans
        ],
    )


def _score_attr(evidence: object, *names: str) -> float | None:
    for name in names:
        value = getattr(evidence, name, None)
        if value is not None:
            return float(value)
    return None


def _duration_ms_for_length(feature: object, interval: PhoneInterval) -> float:
    measured = _score_attr(feature, "active_duration_ms", "duration_ms")
    if measured is not None and measured > 0.0:
        return measured
    return float(interval.end_ms - interval.start_ms)


def _duration_cue_reliable(
    feature: object,
    cue: object,
    *,
    quantity_span: QuantitySpan | None = None,
) -> bool:
    if getattr(cue, "score", None) is None:
        return False
    if quantity_span is not None and quantity_span.duration_ms > 0.0:
        return quantity_span.reliability >= 1.0
    reliability = _score_attr(feature, "duration_reliability")
    return reliability is None or reliability >= 1.0


def _object_value(item: object, name: str, default: object = None) -> object:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _object_float(item: object, name: str) -> float | None:
    value = _object_value(item, name)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _scoreable_result_phones(phones: list[str]) -> list[str]:
    return [
        normalize_token(phone)
        for phone in phones
        if phone not in RESULT_EXCLUDED_TOKENS
    ]


def _timed_wav2vec_phones(free_decode: list[object]) -> list[dict[str, object]]:
    phones: list[dict[str, object]] = []
    for item in free_decode:
        raw_ipa = _object_value(item, "ipa")
        if raw_ipa is None:
            continue
        ipa = normalize_token(str(raw_ipa))
        if ipa in RESULT_EXCLUDED_TOKENS or ipa in _WAV2VEC_DROP_TOKENS:
            continue
        start = _object_value(item, "start_ms")
        end = _object_value(item, "end_ms")
        if start is None or end is None:
            continue
        start_ms = int(start)
        end_ms = int(end)
        if end_ms <= start_ms:
            continue
        phones.append(
            {
                "ipa": ipa,
                "raw_ipa": str(raw_ipa),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": _object_float(item, "confidence"),
            }
        )
    return phones


def _target_payload(request: CoachRequest, target: TargetSequence) -> dict[str, object]:
    scoreable = _scoreable_result_phones(target.all_phones)
    return {
        "text": request.target_text,
        "ipa": "".join(scoreable),
        "phonemes": list(target.all_phones),
        "scoreable_phonemes": scoreable,
    }


def _wav2vec_payload(
    *,
    model_alias: str,
    raw_phones: list[dict[str, object]],
) -> dict[str, object]:
    phonemes = [str(phone["ipa"]) for phone in raw_phones]
    return {
        "model": model_alias,
        "ipa": "".join(phonemes),
        "phonemes": phonemes,
        "phones": [dict(phone) for phone in raw_phones],
        "timed_phones": [dict(phone) for phone in raw_phones],
    }


def _posterior_candidate_payload(
    raw_phones: list[dict[str, object]],
    frames: list[object],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    posterior_frames = [_normalized_posterior_frame(frame) for frame in frames]
    out: list[dict[str, object]] = []
    for index, phone in enumerate(raw_phones):
        start_ms = int(phone["start_ms"])
        end_ms = int(phone["end_ms"])
        sums: dict[str, float] = {}
        frame_count = 0
        for frame in posterior_frames:
            if _overlap_ms(start_ms, end_ms, frame.start_ms, frame.end_ms) <= 0.0:
                continue
            frame_count += 1
            for token, logprob in frame.token_logprobs.items():
                if token in RESULT_EXCLUDED_TOKENS or token in _WAV2VEC_DROP_TOKENS:
                    continue
                sums[token] = sums.get(token, 0.0) + math.exp(float(logprob))
        candidates = []
        if frame_count > 0:
            candidates = [
                {
                    "phone": token,
                    "probability": round(value / frame_count, 6),
                }
                for token, value in sorted(
                    sums.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:limit]
            ]
        out.append(
            {
                "index": index,
                "raw_ipa": phone.get("ipa"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "frame_count": frame_count,
                "candidates": candidates,
            }
        )
    return out


def _intervals_from_raw_phones(
    raw_phones: list[dict[str, object]],
    *,
    speech_start_ms: int,
    speech_end_ms: int,
) -> list[PhoneInterval]:
    return [
        PhoneInterval(
            target_phone=str(phone["ipa"]),
            start_ms=int(phone["start_ms"]),
            end_ms=int(phone["end_ms"]),
            speech_start_ms_original=speech_start_ms,
            speech_end_ms_original=speech_end_ms,
        )
        for phone in raw_phones
    ]


def _quantity_spans_from_raw_phones(
    raw_phones: list[dict[str, object]],
    *,
    speech_start_ms: int,
    speech_end_ms: int,
) -> list[QuantitySpan]:
    """Build continuous duration spans from neighbouring CTC token anchors.

    CTC free-decode token spans are confidence peaks, often a single ~20 ms
    frame. They are good identity anchors but poor phone durations. For
    duration/quantity evidence, each phone gets the region between adjacent
    token boundaries: previous token end -> next token start. This is
    target-independent and applies uniformly to every decoded phone.
    """
    spans: list[QuantitySpan] = []
    for index, phone in enumerate(raw_phones):
        raw_start = int(phone["start_ms"])
        raw_end = int(phone["end_ms"])
        notes: list[str] = []
        source = "ctc_neighbor_span"
        reliability = 1.0

        if index > 0:
            start = int(raw_phones[index - 1]["end_ms"])
        else:
            start = speech_start_ms
            notes.append("left_edge")

        if index + 1 < len(raw_phones):
            end = int(raw_phones[index + 1]["start_ms"])
        else:
            end = speech_end_ms
            notes.append("right_edge")

        start = max(speech_start_ms, min(raw_start, start))
        end = min(speech_end_ms, max(raw_end, end))
        if end <= start:
            start = raw_start
            end = raw_end
            source = "raw_token_span_fallback"
            reliability = 0.0
            notes.append("invalid_neighbor_span")

        spans.append(
            QuantitySpan(
                start_ms=start,
                end_ms=end,
                source=source,
                reliability=reliability,
                notes=tuple(notes),
            )
        )
    return spans


_LONG_QUANTITY_MIN_MS = 320.0
_LONG_QUANTITY_MIN_MS_BEFORE_SONORANT = 175.0
_SHORT_TO_LONG_Z = 4.0
_LONG_TO_SHORT_Z = 3.0


def _duration_entry(stats: object | None, phone: str) -> object | None:
    if isinstance(stats, DurationStats):
        return stats.get(phone)
    getter = getattr(stats, "get", None)
    if callable(getter):
        return getter(phone)
    return None


def _entry_float(entry: object | None, name: str) -> float | None:
    value = getattr(entry, name, None)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _short_to_long_threshold(
    short_phone: str,
    stats: object | None,
    *,
    next_phone: str | None = None,
) -> float:
    min_ms = (
        _LONG_QUANTITY_MIN_MS_BEFORE_SONORANT
        if next_phone in SONORANTS
        else _LONG_QUANTITY_MIN_MS
    )
    entry = _duration_entry(stats, short_phone)
    median = _entry_float(entry, "median_ms")
    scale = _entry_float(entry, "scale_ms")
    if median is None or scale is None or scale <= 0.0:
        return min_ms
    return max(min_ms, median + _SHORT_TO_LONG_Z * scale)


def _long_to_short_threshold(long_phone: str, stats: object | None) -> float:
    short_phone = LONG_TO_SHORT.get(long_phone, long_phone)
    entry = _duration_entry(stats, short_phone)
    median = _entry_float(entry, "median_ms")
    scale = _entry_float(entry, "scale_ms")
    if median is None or scale is None or scale <= 0.0:
        return _LONG_QUANTITY_MIN_MS
    return median + _LONG_TO_SHORT_Z * scale


def _quantity_length_cue(
    phone: str,
    span_ms: float,
    stats: object | None,
    *,
    next_phone: str | None = None,
) -> LengthCue:
    cue = length_cue(phone, span_ms, stats)  # type: ignore[arg-type]
    if cue.note == "long" and phone in SHORT_TO_LONG:
        threshold = _short_to_long_threshold(phone, stats, next_phone=next_phone)
        if span_ms < threshold:
            return LengthCue(z=cue.z, note="expected", score=cue.score)
    if cue.note == "short" and phone in LONG_TO_SHORT:
        threshold = _long_to_short_threshold(phone, stats)
        if span_ms > threshold:
            return LengthCue(z=cue.z, note="expected", score=cue.score)
    return cue


def _duration_ms_for_quantity(
    feature: object,
    interval: PhoneInterval,
    quantity_span: QuantitySpan | None,
) -> float:
    if quantity_span is not None and quantity_span.duration_ms > 0.0:
        return quantity_span.duration_ms
    return _duration_ms_for_length(feature, interval)


def _duration_analysis_candidate(
    *,
    index: int,
    raw_phone: str,
    feature: object,
    interval: PhoneInterval,
    stats: object | None,
    quantity_span: QuantitySpan | None = None,
    next_phone: str | None = None,
) -> dict[str, object] | None:
    span_ms = _duration_ms_for_quantity(feature, interval, quantity_span)
    cue = _quantity_length_cue(
        raw_phone,
        span_ms,
        stats,
        next_phone=next_phone,
    )
    to_phone: str | None = None
    if raw_phone in SHORT_TO_LONG and cue.note == "long":
        to_phone = SHORT_TO_LONG[raw_phone]
    elif raw_phone in LONG_TO_SHORT and cue.note == "short":
        to_phone = LONG_TO_SHORT[raw_phone]
    if to_phone is None:
        return None

    reliable = _duration_cue_reliable(
        feature,
        cue,
        quantity_span=quantity_span,
    )
    reasons: list[str] = []
    if cue.score is None:
        reasons.append("duration_unscored")
    if not reliable:
        reasons.append("duration_evidence_unreliable")
    return {
        "index": index,
        "source": "duration",
        "from": raw_phone,
        "to": to_phone,
        "passed": cue.score is not None and reliable,
        "applied": False,
        "confidence": 1.0 if reliable else 0.0,
        "evidence": {
            "duration_ms": span_ms,
            "z": cue.z,
            "note": cue.note,
            "score": cue.score,
            "reliable": reliable,
            "quantity_span": (
                quantity_span.as_dict() if quantity_span is not None else None
            ),
        },
        "reasons": reasons,
    }


def _analysis_thresholds(atlas: object) -> tuple[bool, float, float]:
    from assess.stats import (  # noqa: PLC0415
        _CALIBRATED_DEFAULT_DELTA,
        _CALIBRATED_INSIDE_FIT_FLOOR,
        _DEFAULT_DELTA,
        _INSIDE_FIT_FLOOR,
    )

    calibrated = bool(getattr(atlas, "calibrated", False))
    floor_value = getattr(atlas, "fit_floor", None)
    delta_value = getattr(atlas, "delta", None)
    floor = (
        float(floor_value)
        if isinstance(floor_value, (int, float)) and np.isfinite(float(floor_value))
        else (_CALIBRATED_INSIDE_FIT_FLOOR if calibrated else _INSIDE_FIT_FLOOR)
    )
    delta = (
        float(delta_value)
        if isinstance(delta_value, (int, float)) and np.isfinite(float(delta_value))
        else (_CALIBRATED_DEFAULT_DELTA if calibrated else _DEFAULT_DELTA)
    )
    return calibrated, floor, delta


def _same_phone_class(first: str, second: str) -> bool:
    from assess.features import phone_class  # noqa: PLC0415

    try:
        return phone_class(first) is phone_class(second)
    except ValueError:
        return False


def _nearest_analysis_candidate(
    *,
    index: int,
    raw_phone: str,
    atlas: object,
) -> dict[str, object] | None:
    best_match = getattr(atlas, "best_match", None)
    if best_match is None:
        return None
    to_phone = normalize_token(str(best_match))
    if to_phone == raw_phone:
        return None

    confidence = _score_attr(atlas, "confidence") or 0.0
    target_quality = _score_attr(
        atlas,
        "quality_score",
        "target_typicality",
        "target_fit",
    )
    alternative_quality = _score_attr(
        atlas,
        "best_match_typicality",
        "best_match_fit",
    )
    margin = _score_attr(atlas, "margin")
    calibrated, floor, delta = _analysis_thresholds(atlas)

    reasons: list[str] = []
    if confidence <= 0.0:
        reasons.append("acoustic_evidence_unusable")
    if target_quality is None:
        reasons.append("target_quality_missing")
    elif target_quality >= floor:
        reasons.append("target_quality_above_floor")
    if alternative_quality is None:
        reasons.append("alternative_quality_missing")
    elif alternative_quality < floor:
        reasons.append("alternative_quality_below_floor")
    if margin is None:
        reasons.append("nearest_margin_missing")
    elif margin > -delta:
        reasons.append("nearest_margin_below_threshold")
    if not _same_phone_class(raw_phone, to_phone):
        reasons.append("nearest_cross_class")

    return {
        "index": index,
        "source": "gmm_nearest",
        "from": raw_phone,
        "to": to_phone,
        "passed": not reasons,
        "applied": False,
        "confidence": 1.0 if not reasons else 0.0,
        "evidence": {
            "target_quality": target_quality,
            "alternative_quality": alternative_quality,
            "margin": margin,
            "calibrated": calibrated,
            "floor": floor,
            "margin_threshold": delta,
            "atlas_verdict": getattr(atlas, "atlas_verdict", None),
        },
        "reasons": reasons,
    }


def _pairwise_float(candidate: dict[str, object], name: str) -> float | None:
    evidence = candidate.get("evidence")
    if isinstance(evidence, dict):
        value = evidence.get(name)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def _pairwise_rank_key(candidate: dict[str, object]) -> tuple[bool, float, float, float]:
    margin = _pairwise_float(candidate, "margin")
    fp_estimate = _pairwise_float(candidate, "fp_estimate")
    recall_estimate = _pairwise_float(candidate, "recall_estimate")
    return (
        candidate.get("passed") is True,
        margin if margin is not None else float("-inf"),
        -(fp_estimate if fp_estimate is not None else float("inf")),
        recall_estimate if recall_estimate is not None else float("-inf"),
    )


def _pairwise_analysis_candidates(
    *,
    index: int,
    raw_phone: str,
    contrast_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in contrast_rows:
        alternative = row.get("alternative")
        if alternative is None:
            continue
        to_phone = normalize_token(str(alternative))
        if to_phone == raw_phone:
            continue
        passed = row.get("passed") is True
        reasons = [
            str(reason)
            for reason in row.get("reasons", [])
            if isinstance(reason, str)
        ]
        if not passed and not reasons:
            reasons.append("pairwise_detector_not_passed")
        evidence = {
            "score": row.get("score"),
            "threshold": row.get("threshold"),
            "margin": row.get("margin"),
            "detector_kind": row.get("detector_kind"),
            "calibration_independent": row.get("calibration_independent"),
            "features": row.get("features", []),
            "feature_values": row.get("feature_values", {}),
            "fp_estimate": row.get("fp_estimate"),
            "recall_estimate": row.get("recall_estimate"),
            "feature_confidence": row.get("feature_confidence"),
            "target_quality": row.get("target_quality"),
            "target_quality_ceiling": row.get("target_quality_ceiling"),
            "reason": row.get("reason"),
        }
        candidates.append(
            {
                "index": index,
                "source": "pairwise_contrast",
                "from": raw_phone,
                "to": to_phone,
                "passed": passed,
                "applied": False,
                "confidence": 1.0 if passed else 0.0,
                "evidence": evidence,
                "reasons": reasons,
            }
        )
    candidates.sort(key=_pairwise_rank_key, reverse=True)
    return candidates


def _quality_score_for_candidate(atlas: object) -> float | None:
    return _score_attr(atlas, "quality_score", "target_typicality", "target_fit")


def _with_pairwise_runtime_calibration(
    candidates: list[dict[str, object]],
    *,
    feature: object,
    assess_atlas: Callable[[object, str], object],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.get("source") != "pairwise_contrast":
            out.append(candidate)
            continue

        item = dict(candidate)
        evidence = dict(item.get("evidence") or {})
        reasons = [
            str(reason)
            for reason in item.get("reasons", [])
            if isinstance(reason, str)
        ]
        to_phone = str(item.get("to"))
        alternative_atlas = assess_atlas(feature, to_phone)
        alternative_quality = _quality_score_for_candidate(alternative_atlas)
        target_quality = _pairwise_float(item, "target_quality")
        margin = _pairwise_float(item, "margin")
        recall = _pairwise_float(item, "recall_estimate")
        quality_lift: float | None = None
        calibration_independent = evidence.get("calibration_independent") is True

        if margin is None:
            reasons.append("pairwise_margin_missing")
        elif margin < PAIRWISE_MIN_MARGIN:
            reasons.append("pairwise_margin_below_runtime_floor")
        if recall is None:
            reasons.append("pairwise_recall_missing")
        elif recall < PAIRWISE_MIN_RECALL:
            reasons.append("pairwise_recall_below_runtime_floor")

        if calibration_independent:
            evidence["quality_lift_skipped"] = "calibration_independent_detector"
        elif target_quality is not None and alternative_quality is not None:
            quality_lift = alternative_quality - target_quality
            if quality_lift < PAIRWISE_MIN_QUALITY_LIFT:
                reasons.append("pairwise_quality_lift_below_runtime_floor")
        elif (
            margin is not None
            and recall is not None
            and (
                margin < PAIRWISE_UNSCORED_MIN_MARGIN
                or recall < PAIRWISE_UNSCORED_MIN_RECALL
            )
        ):
            reasons.append("pairwise_unscored_evidence_below_runtime_floor")

        evidence["alternative_quality"] = alternative_quality
        evidence["quality_lift"] = quality_lift
        evidence["runtime_margin_floor"] = PAIRWISE_MIN_MARGIN
        evidence["runtime_recall_floor"] = PAIRWISE_MIN_RECALL
        evidence["runtime_quality_lift_floor"] = PAIRWISE_MIN_QUALITY_LIFT
        item["evidence"] = evidence
        if reasons:
            item["passed"] = False
            item["confidence"] = 0.0
            item["reasons"] = reasons
        out.append(item)
    return out


def _with_candidates_applied(
    candidates: list[dict[str, object]],
    applied: dict[str, object] | None,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["applied"] = candidate is applied
        out.append(item)
    return out


def _analysis_change_payload(
    *,
    change_id: str,
    index: int,
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        "id": change_id,
        "index": index,
        "from": candidate["from"],
        "to": candidate["to"],
        "source": candidate["source"],
        "confidence": candidate.get("confidence"),
        "evidence": candidate.get("evidence", {}),
        "reasons": list(candidate.get("reasons") or []),
    }


def _quality_for_analysis_phone(*, atlas: object) -> float | None:
    """Production-quality typicality from the atlas/GMM channel only.

    The soft length cue is intentionally *not* folded in here. Length is a
    symmetric duration typicality against connected-speech medians; careful
    isolated speech runs systematically longer, so an earlier
    ``min(atlas, length)`` combination collapsed quality for
    well-articulated-but-naturally-long vowels (a native /e/ at ~150 ms landed
    several robust scales above the ~70 ms corpus median and dragged quality to
    ~0.1). Length stays an independent, non-gating channel (note + z + score)
    surfaced alongside quality; genuine length *errors* are scored separately by
    the phonetic-distance channel (aː↔a).
    """
    score = _score_attr(atlas, "quality_score", "target_typicality", "target_fit")
    if score is None:
        return None
    return max(0.0, min(1.0, float(score)))


def _analysis_payload(
    phones: list[dict[str, object]],
    changes: list[dict[str, object]],
) -> dict[str, object]:
    phonemes = [str(phone["ipa"]) for phone in phones]
    return {
        "ipa": "".join(phonemes),
        "phonemes": phonemes,
        "phones": [dict(phone) for phone in phones],
        "timed_phones": [dict(phone) for phone in phones],
        "changes": [dict(change) for change in changes],
    }


def _edit_alignment(
    expected: list[str],
    observed: list[str],
) -> list[dict[str, object]]:
    n = len(expected)
    m = len(observed)
    dp: list[list[tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0)
        back[i][0] = "missing"
    for j in range(1, m + 1):
        dp[0][j] = (j, 0)
        back[0][j] = "extra"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == observed[j - 1]:
                prev_cost, prev_matches = dp[i - 1][j - 1]
                best = ((prev_cost, prev_matches + 1), "correct")
            else:
                prev_cost, prev_matches = dp[i - 1][j - 1]
                best = ((prev_cost + 1, prev_matches), "incorrect")
            missing_cost, missing_matches = dp[i - 1][j]
            extra_cost, extra_matches = dp[i][j - 1]
            candidates = [
                best,
                ((missing_cost + 1, missing_matches), "missing"),
                ((extra_cost + 1, extra_matches), "extra"),
            ]
            dp[i][j], back[i][j] = min(
                candidates,
                key=lambda item: (item[0][0], -item[0][1]),
            )

    rows: list[dict[str, object]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"correct", "incorrect"}:
            rows.append(
                {
                    "status": op,
                    "expected": expected[i - 1],
                    "observed": observed[j - 1],
                    "index_expected": i - 1,
                    "index_observed": j - 1,
                }
            )
            i -= 1
            j -= 1
        elif op == "missing":
            rows.append(
                {
                    "status": "missing",
                    "expected": expected[i - 1],
                    "observed": None,
                    "index_expected": i - 1,
                    "index_observed": None,
                }
            )
            i -= 1
        else:
            rows.append(
                {
                    "status": "extra",
                    "expected": None,
                    "observed": observed[j - 1],
                    "index_expected": None,
                    "index_observed": j - 1,
                }
            )
            j -= 1
    rows.reverse()
    return rows


def _result_payload(
    *,
    target_phones: list[str],
    analysis_phones: list[dict[str, object]],
    length_stats: object | None = None,
    analysis_candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    observed = [str(phone["ipa"]) for phone in analysis_phones]
    alignment = _edit_alignment(target_phones, observed)
    candidates_by_index = _analysis_candidates_by_index(analysis_candidates or [])
    rows: list[dict[str, object]] = []
    correct_qualities: list[float] = []
    for index, row in enumerate(alignment):
        observed_index = row["index_observed"]
        analysis_phone = (
            analysis_phones[observed_index]
            if isinstance(observed_index, int) and observed_index < len(analysis_phones)
            else None
        )
        quality = None
        if row["status"] == "correct" and analysis_phone is not None:
            quality_value = analysis_phone.get("quality")
            if isinstance(quality_value, (int, float)):
                quality = float(quality_value)
                correct_qualities.append(quality)
        changed_summary = None
        if analysis_phone is not None and analysis_phone.get("changed_by_analysis"):
            changed_summary = {
                "from": analysis_phone.get("raw_ipa"),
                "to": analysis_phone.get("ipa"),
                "source": analysis_phone.get("analysis_source"),
                "change_id": analysis_phone.get("change_id"),
            }
        # Length is an independent, non-gating channel: it never folds into
        # `quality` (see `_quality_for_analysis_phone`). Surfaced per observed
        # phone so the UI can annotate "kısa/uzun/normal" without penalising the
        # quality of a well-articulated but naturally long/short production.
        length = None
        if analysis_phone is not None:
            length = {
                "score": analysis_phone.get("length_score"),
                "note": analysis_phone.get("length_note"),
                "z": analysis_phone.get("length_z"),
                "duration_ms": analysis_phone.get("duration_ms"),
            }
        raw_observed = (
            analysis_phone.get("raw_ipa") if analysis_phone is not None else None
        )
        analysis_observed = (
            analysis_phone.get("ipa") if analysis_phone is not None else row["observed"]
        )
        duration_evidence = _target_duration_evidence(
            expected=row["expected"],
            analysis_phone=analysis_phone,
            length_stats=length_stats,
            target_phones=target_phones,
            expected_index=row["index_expected"],
        )
        pairwise_candidates = (
            candidates_by_index.get(observed_index, [])
            if isinstance(observed_index, int)
            else []
        )
        phone_score = score_phone(
            str(row["expected"]) if row["expected"] is not None else None,
            str(row["observed"]) if row["observed"] is not None else None,
            raw_observed=str(raw_observed) if raw_observed is not None else None,
            analysis_observed=(
                str(analysis_observed) if analysis_observed is not None else None
            ),
            duration_evidence=duration_evidence,
            pairwise_candidates=pairwise_candidates,
        )
        rows.append(
            {
                "index": index,
                "index_expected": row["index_expected"],
                "index_observed": row["index_observed"],
                "expected": row["expected"],
                "observed": row["observed"],
                "status": row["status"],
                "strict_status": row["status"],
                "score": phone_score.score,
                "score_percent": round(phone_score.score_percent, 6),
                "reason": _result_reason(
                    str(row["status"]),
                    row["expected"],
                    row["observed"],
                ),
                "score_reason": phone_score.reason,
                "raw_observed": raw_observed,
                "analysis_observed": analysis_observed,
                "distance_features": phone_score.distance_features,
                "quality": quality,
                "length": length,
                "changed_by_analysis": changed_summary,
                "timing": None
                if analysis_phone is None
                else {
                    "start_ms": analysis_phone.get("start_ms"),
                    "end_ms": analysis_phone.get("end_ms"),
                },
            }
        )

    target_count = len(target_phones)
    correct_count = sum(1 for row in rows if row["status"] == "correct")
    extra_count = sum(1 for row in rows if row["status"] == "extra")
    incorrect_count = sum(1 for row in rows if row["status"] == "incorrect")
    missing_count = sum(1 for row in rows if row["status"] == "missing")
    accuracy = correct_count / target_count if target_count else 0.0
    target_scores = [
        float(row["score"])
        for row in rows
        if row.get("index_expected") is not None
    ]
    mean_phone_score = (
        sum(target_scores) / len(target_scores)
        if target_scores
        else 0.0
    )
    extra_penalty = (
        target_count / (target_count + extra_count)
        if target_count + extra_count > 0
        else 1.0
    )
    quality_score = (
        sum(correct_qualities) / len(correct_qualities)
        if correct_qualities
        else None
    )
    word_score = mean_phone_score * extra_penalty
    overall_score = 100.0 * word_score
    phone_score_rows = [
        {
            "index": row["index"],
            "index_expected": row["index_expected"],
            "expected": row["expected"],
            "observed": row["observed"],
            "score": row["score"],
            "score_percent": row["score_percent"],
            "strict_status": row["strict_status"],
            "score_reason": row["score_reason"],
        }
        for row in rows
    ]
    return {
        "phones": rows,
        "score": {
            "item_score": round(overall_score, 6),
            "max_item_score": 100.0,
            "word_score": round(word_score, 6),
            "phoneme_core": round(mean_phone_score, 6),
            "extra_penalty": round(extra_penalty, 6),
            "score_version": "phonetic-distance-v1",
            "phone_scores": phone_score_rows,
        },
        "summary": {
            "target_count": target_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "missing_count": missing_count,
            "extra_count": extra_count,
            "accuracy": round(accuracy, 6),
            "mean_phone_score": round(mean_phone_score, 6),
            "word_score": round(word_score, 6),
            "quality_score": None if quality_score is None else round(quality_score, 6),
            "extra_penalty": round(extra_penalty, 6),
            "quality_factor": 1.0,
            "overall_score": round(overall_score, 6),
            "score_version": "phonetic-distance-v1",
        },
    }


def _analysis_candidates_by_index(
    candidates: list[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {}
    for candidate in candidates:
        index = candidate.get("index")
        if isinstance(index, int):
            out.setdefault(index, []).append(candidate)
    return out


def _target_duration_evidence(
    *,
    expected: object,
    analysis_phone: dict[str, object] | None,
    length_stats: object | None,
    target_phones: list[str],
    expected_index: object,
) -> dict[str, object] | None:
    if expected is None or analysis_phone is None or length_stats is None:
        return None
    duration = analysis_phone.get("duration_ms")
    if not isinstance(duration, (int, float)) or not np.isfinite(float(duration)):
        return None
    next_phone = (
        target_phones[expected_index + 1]
        if isinstance(expected_index, int)
        and expected_index + 1 < len(target_phones)
        else None
    )
    cue = _quantity_length_cue(
        str(expected),
        float(duration),
        length_stats,
        next_phone=next_phone,
    )
    quantity_span = analysis_phone.get("quantity_span")
    reliability = None
    if isinstance(quantity_span, dict):
        value = quantity_span.get("reliability")
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            reliability = float(value)
    reliable = reliability is None or reliability >= 1.0
    return {
        "duration_ms": float(duration),
        "z": cue.z,
        "note": cue.note,
        "score": cue.score,
        "reliable": reliable,
    }


def _result_reason(status: str, expected: object, observed: object) -> str | None:
    if status == "correct":
        return None
    if status == "missing":
        return "phone_missing"
    if status == "extra":
        return "phone_extra"
    expected_phone = str(expected) if expected is not None else None
    observed_phone = str(observed) if observed is not None else None
    if (
        expected_phone is not None
        and observed_phone is not None
        and LONG_TO_SHORT.get(expected_phone) == observed_phone
    ):
        return "long_vowel_heard_short"
    if (
        expected_phone is not None
        and observed_phone is not None
        and LONG_TO_SHORT.get(observed_phone) == expected_phone
    ):
        return "short_vowel_heard_long"
    return "phone_mismatch"


def _free_decode_bounds(free_decode: list[object]) -> tuple[int, int] | None:
    """Return (min start_ms, max end_ms) across timed free-decode tokens.

    Returns ``None`` when no token carries usable timestamps.
    """
    starts: list[int] = []
    ends: list[int] = []
    for item in free_decode:
        start = (
            item.get("start_ms") if isinstance(item, dict)
            else getattr(item, "start_ms", None)
        )
        end = (
            item.get("end_ms") if isinstance(item, dict)
            else getattr(item, "end_ms", None)
        )
        if start is None or end is None:
            continue
        starts.append(int(start))
        ends.append(int(end))
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _reconcile_speech_region(
    speech_region: object,
    free_decode: list[object],
    sample_count: int,
    sample_rate: int,
) -> object:
    """Widen the RMS speech region to cover wav2vec's free-decode extent.

    ``detect_speech_region`` keys off frame energy, so a quietly produced
    syllable whose amplitude dips below the relative threshold is dropped even
    when wav2vec decodes it confidently. The forced aligner then never sees
    those frames and squeezes the remaining targets onto the wrong ones.
    wav2vec is the more sensitive content detector, so the eligible alignment
    window is the union of the RMS region and the free-decode span.
    """
    from audio.timing import SpeechRegion  # noqa: PLC0415

    if not free_decode:
        return speech_region

    region_start = int(getattr(speech_region, "start_ms_original"))
    region_end = int(getattr(speech_region, "end_ms_original"))
    status_ok = str(getattr(speech_region, "status", "ok")) == "ok"
    bounds = _free_decode_bounds(free_decode)
    if bounds is None:
        # No timestamps to reconcile against; only repair a total VAD fallback.
        if status_ok:
            return speech_region
        return SpeechRegion(region_start, region_end, "ok")

    clip_ms = max(1, int(round(sample_count * 1000.0 / sample_rate)))
    free_start, free_end = bounds
    new_start = max(0, min(region_start, free_start))
    new_end = min(clip_ms, max(region_end, free_end))
    if new_end <= new_start:
        return speech_region
    return SpeechRegion(new_start, new_end, "ok")

# ---------------------------------------------------------------------------
# Services wiring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoachRuntimeServices:
    load_audio: Callable[[Path], tuple[np.ndarray, int]]
    align: Callable[[np.ndarray, int, list[str]], TimingResult]
    extract_feature: Callable[..., object]
    assess_atlas: Callable[[object, str], object]
    run_wav2vec: Callable[[np.ndarray, int, list[PhoneInterval], str], object]
    length_stats: object | None = None
    assess_contrasts: Callable[[object, str, object], list[dict[str, object]]] | None = None


# ---------------------------------------------------------------------------
# Target sequence
# ---------------------------------------------------------------------------

def resolve_target_sequence(request: CoachRequest) -> TargetSequence:
    if request.target_phones is not None:
        all_phones = list(request.target_phones)
    else:
        g2p_result = transcribe_training_text(request.target_text or "")
        drop_reason = getattr(g2p_result, "drop_reason", None)
        if drop_reason:
            raise ValueError(f"G2P could not produce target phones: {drop_reason}")
        all_phones = list(g2p_result.tokens)

    timed_phones: list[str] = []
    source_indices: list[int] = []
    for index, phone in enumerate(all_phones):
        if phone in NON_TIMED_TOKENS:
            continue
        timed_phones.append(phone)
        source_indices.append(index)

    if not timed_phones:
        raise ValueError("G2P could not produce target phones: no timed phones")

    return TargetSequence(
        all_phones=all_phones,
        timed_phones=timed_phones,
        source_indices=source_indices,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_coach(
    request: CoachRequest,
    *,
    services: CoachRuntimeServices | None = None,
) -> CoachResult:
    runtime_services = services or build_default_runtime_services(request)
    target = resolve_target_sequence(request)
    audio, sample_rate = runtime_services.load_audio(Path(request.audio_path))
    wav_run = runtime_services.run_wav2vec(audio, sample_rate, [], request.model_alias)
    status = (
        "model_unavailable"
        if getattr(wav_run, "status", None) == "model_unavailable"
        else "ok"
    )
    raw_phones = _timed_wav2vec_phones(list(getattr(wav_run, "free_decode", [])))
    audio_end_ms = max(0, int(round(audio.size * 1000.0 / sample_rate)))
    if raw_phones:
        speech_start_ms = min(int(phone["start_ms"]) for phone in raw_phones)
        speech_end_ms = max(int(phone["end_ms"]) for phone in raw_phones)
    else:
        speech_start_ms = 0
        speech_end_ms = audio_end_ms

    target_block = _target_payload(request, target)
    wav2vec_block = _wav2vec_payload(
        model_alias=request.model_alias,
        raw_phones=raw_phones,
    )
    posterior_candidates = _posterior_candidate_payload(
        raw_phones,
        list(getattr(wav_run, "frame_posteriors", ()) or ()),
    )
    if posterior_candidates:
        wav2vec_block["posterior_candidates"] = posterior_candidates

    if status != "ok" or not raw_phones:
        final_status = status if status != "ok" else "no_speech"
        empty_analysis = _analysis_payload([], [])
        empty_result = _result_payload(
            target_phones=list(target_block["scoreable_phonemes"]),  # type: ignore[arg-type]
            analysis_phones=[],
            length_stats=runtime_services.length_stats,
        )
        return CoachResult(
            model_alias=request.model_alias,
            status=final_status,
            speech_start_ms_original=speech_start_ms,
            speech_end_ms_original=speech_end_ms,
            phones=[],
            free_decode=list(getattr(wav_run, "free_decode", [])),
            target=target_block,
            wav2vec=wav2vec_block,
            analysis=empty_analysis,
            result=empty_result,
            debug={
                "analysis_candidates": [],
                "posterior_candidates": posterior_candidates,
                "reason": final_status,
            },
        )

    intervals = _intervals_from_raw_phones(
        raw_phones,
        speech_start_ms=speech_start_ms,
        speech_end_ms=speech_end_ms,
    )
    quantity_spans = _quantity_spans_from_raw_phones(
        raw_phones,
        speech_start_ms=speech_start_ms,
        speech_end_ms=speech_end_ms,
    )
    features = []
    for index, interval in enumerate(intervals):
        prev_phone = intervals[index - 1].target_phone if index > 0 else None
        next_phone = (
            intervals[index + 1].target_phone
            if index + 1 < len(intervals)
            else None
        )
        features.append(
            runtime_services.extract_feature(
                audio,
                sample_rate,
                interval,
                prev_phone=prev_phone,
                next_phone=next_phone,
            )
        )

    raw_atlas_items = [
        runtime_services.assess_atlas(feature, interval.target_phone)
        for feature, interval in zip(features, intervals, strict=True)
    ]
    raw_contrast_items: list[list[dict[str, object]]] = [
        [] for _ in raw_atlas_items
    ]
    if runtime_services.assess_contrasts is not None:
        updated_atlas_items: list[object] = []
        raw_contrast_items = []
        for feature, interval, atlas in zip(
            features,
            intervals,
            raw_atlas_items,
            strict=True,
        ):
            # The interval phone is the raw wav2vec hypothesis, so detector
            # evaluation stays independent of the requested target sequence.
            contrast_candidates = list(
                runtime_services.assess_contrasts(
                    feature,
                    interval.target_phone,
                    atlas,
                )
            )
            raw_contrast_items.append(contrast_candidates)
            updated_atlas_items.append(
                _with_contrast_candidates(atlas, contrast_candidates)
            )
        raw_atlas_items = updated_atlas_items

    analysis_phones: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    debug_candidates: list[dict[str, object]] = []
    legacy_phones: list[CoachPhoneResult] = []

    for index, (
        raw_phone,
        interval,
        quantity_span,
        feature,
        raw_atlas,
        raw_contrasts,
    ) in enumerate(
        zip(
            raw_phones,
            intervals,
            quantity_spans,
            features,
            raw_atlas_items,
            raw_contrast_items,
            strict=True,
        )
    ):
        raw_ipa = str(raw_phone["ipa"])
        next_phone = (
            intervals[index + 1].target_phone
            if index + 1 < len(intervals)
            else None
        )
        candidates: list[dict[str, object]] = []
        duration_candidate = _duration_analysis_candidate(
            index=index,
            raw_phone=raw_ipa,
            feature=feature,
            interval=interval,
            stats=runtime_services.length_stats,
            quantity_span=quantity_span,
            next_phone=next_phone,
        )
        if duration_candidate is not None:
            candidates.append(duration_candidate)
        pairwise_candidates = _pairwise_analysis_candidates(
            index=index,
            raw_phone=raw_ipa,
            contrast_rows=raw_contrasts,
        )
        candidates.extend(
            _with_pairwise_runtime_calibration(
                pairwise_candidates,
                feature=feature,
                assess_atlas=runtime_services.assess_atlas,
            )
        )
        nearest_candidate = _nearest_analysis_candidate(
            index=index,
            raw_phone=raw_ipa,
            atlas=raw_atlas,
        )
        if nearest_candidate is not None:
            candidates.append(nearest_candidate)
        applied = next(
            (candidate for candidate in candidates if candidate.get("passed") is True),
            None,
        )
        final_phone = str(applied["to"]) if applied is not None else raw_ipa
        change_id: str | None = None
        if applied is not None and final_phone != raw_ipa:
            change_id = f"analysis-{index}"
            changes.append(
                _analysis_change_payload(
                    change_id=change_id,
                    index=index,
                    candidate=applied,
                )
            )
        debug_candidates.extend(_with_candidates_applied(candidates, applied))

        final_atlas = raw_atlas
        if final_phone != raw_ipa:
            final_atlas = runtime_services.assess_atlas(feature, final_phone)
        duration_ms = _duration_ms_for_quantity(feature, interval, quantity_span)
        final_length = _quantity_length_cue(
            final_phone,
            duration_ms,
            runtime_services.length_stats,
            next_phone=next_phone,
        )
        quality = _quality_for_analysis_phone(atlas=final_atlas)
        analysis_phone = {
            "ipa": final_phone,
            "raw_ipa": raw_ipa,
            "start_ms": int(raw_phone["start_ms"]),
            "end_ms": int(raw_phone["end_ms"]),
            "confidence": raw_phone.get("confidence"),
            "quality": quality,
            "quality_score": quality,
            "length_score": final_length.score,
            "length_note": final_length.note,
            "length_z": final_length.z,
            "duration_ms": duration_ms,
            "quantity_span": quantity_span.as_dict(),
            "changed_by_analysis": change_id is not None,
            "change_id": change_id,
            "analysis_source": applied.get("source") if applied is not None else "raw",
        }
        analysis_phones.append(analysis_phone)
        legacy_phones.append(
            CoachPhoneResult(
                target_phone=final_phone,
                start_ms=interval.start_ms,
                end_ms=interval.end_ms,
                status="correct",
                score=1.0,
                atlas=final_atlas,
                wav2vec={
                    "observed_phone": raw_ipa,
                    "confidence": raw_phone.get("confidence"),
                    "observed_matches_target": final_phone == raw_ipa,
                    "observed_source": "free_decode",
                    "reasons": [],
                },
                identity_score=_object_float(raw_phone, "confidence"),
                quality_score=quality,
                length_score=final_length.score,
                length_note=feedback_note_for_length(final_length.note),
                analysis_source=applied.get("source") if applied else "raw",
                changed_by_analysis=change_id is not None,
                reasons=list(applied.get("reasons") or []) if applied else [],
            )
        )

    analysis_block = _analysis_payload(analysis_phones, changes)
    result_block = _result_payload(
        target_phones=list(target_block["scoreable_phonemes"]),  # type: ignore[arg-type]
        analysis_phones=analysis_phones,
        length_stats=runtime_services.length_stats,
        analysis_candidates=debug_candidates,
    )
    return CoachResult(
        model_alias=request.model_alias,
        status=status,
        speech_start_ms_original=speech_start_ms,
        speech_end_ms_original=speech_end_ms,
        phones=legacy_phones,
        free_decode=list(getattr(wav_run, "free_decode", [])),
        target=target_block,
        wav2vec=wav2vec_block,
        analysis=analysis_block,
        result=result_block,
        debug={
            "analysis_candidates": debug_candidates,
            "posterior_candidates": posterior_candidates,
            "speech_start_ms_original": speech_start_ms,
            "speech_end_ms_original": speech_end_ms,
        },
    )


# ---------------------------------------------------------------------------
# Default services factory
# ---------------------------------------------------------------------------

def _load_acoustic_scorer(
    path: Path, *, calibration_path: Path | None = None
) -> Callable[[object, str], object]:
    from assess.stats import (  # noqa: PLC0415
        AtlasGMM,
        AtlasStats,
        GMMCalibration,
        assess_atlas,
        assess_atlas_gmm,
    )

    if _is_gmm_artifact(path):
        gmm_atlas = AtlasGMM.load(path)
        calibration = (
            GMMCalibration.load(calibration_path)
            if calibration_path is not None and calibration_path.exists()
            else None
        )

        def gmm_call(feature: object, target_phone: str) -> object:
            return assess_atlas_gmm(  # type: ignore[arg-type]
                feature,
                target_phone,
                gmm_atlas,
                calibration=calibration,
            )

        return gmm_call

    atlas_stats = AtlasStats.load(path)

    def stats_call(feature: object, target_phone: str) -> object:
        return assess_atlas(feature, target_phone, atlas_stats)  # type: ignore[arg-type]

    return stats_call


def _is_gmm_artifact(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False
    phones = data.get("phones")
    if not isinstance(phones, dict):
        return False
    return any(
        isinstance(payload, dict) and "components" in payload
        for payload in phones.values()
    )


def _with_contrast_candidates(
    atlas: object,
    candidates: list[dict[str, object]],
) -> object:
    method = getattr(atlas, "with_contrast_candidates", None)
    if callable(method):
        return method(candidates)
    try:
        setattr(atlas, "contrast_candidates", candidates)
    except Exception:
        return atlas
    return atlas


def build_default_runtime_services(request: CoachRequest) -> CoachRuntimeServices:
    from assess.authority import AuthorityTable  # noqa: PLC0415
    from assess.features import FeatureExtractor  # noqa: PLC0415
    from assess.inference import Wav2VecInferenceService  # noqa: PLC0415
    from assess.registry import ModelRegistry  # noqa: PLC0415
    from audio.timing import detect_speech_region  # noqa: PLC0415

    if not request.stats_path:
        raise ValueError("stats_path is required for default runtime services")
    if not request.authority_path:
        raise ValueError("authority_path is required for default runtime services")

    calibration_path = (
        Path(request.calibration_path) if request.calibration_path else None
    )
    acoustic_scorer = _load_acoustic_scorer(
        Path(request.stats_path), calibration_path=calibration_path
    )
    authority = AuthorityTable.load(Path(request.authority_path))
    length_stats = (
        DurationStats.load(DEFAULT_COACH_LENGTH_PATH)
        if DEFAULT_COACH_LENGTH_PATH.exists()
        else None
    )
    extractor = FeatureExtractor()
    registry = ModelRegistry.from_config(DEFAULT_MODELS_CONFIG)
    wav2vec_service = Wav2VecInferenceService(registry)
    wav2vec_cache: dict[str, object | None] = {}

    def load_audio(path: Path) -> tuple[np.ndarray, int]:
        import soundfile as sf  # noqa: PLC0415

        samples, sample_rate = sf.read(str(path), dtype="float32")
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        return audio, int(sample_rate)

    def wav_output(audio: np.ndarray, sample_rate: int) -> object | None:
        if (
            wav2vec_cache.get("audio") is audio
            and wav2vec_cache.get("sample_rate") == int(sample_rate)
        ):
            return wav2vec_cache.get("output")
        output = wav2vec_service.run(
            request.model_alias,
            audio,
            sample_rate,
        )
        wav2vec_cache.clear()
        wav2vec_cache.update(
            {
                "audio": audio,
                "sample_rate": int(sample_rate),
                "output": output,
            }
        )
        return output

    def align_fn(
        audio: np.ndarray,
        sample_rate: int,
        target_phones: list[str],
    ) -> TimingResult:
        speech_region = detect_speech_region(audio, sample_rate)
        output = wav_output(audio, sample_rate)
        if output is None:
            return TimingResult(
                status="model_unavailable",
                speech_start_ms_original=speech_region.start_ms_original,
                speech_end_ms_original=speech_region.end_ms_original,
                intervals=[],
            )
        frame_posteriors = getattr(output, "frame_posteriors", ()) or ()
        free_decode = getattr(output, "phones", ()) or ()
        speech_region = _reconcile_speech_region(
            speech_region, list(free_decode), int(audio.size), int(sample_rate)
        )
        return timing_from_wav2vec_posteriors(
            target_phones,
            list(frame_posteriors),
            speech_region,
        )

    def atlas_call(feature: object, target_phone: str) -> object:
        return acoustic_scorer(feature, target_phone)

    def wav_call(
        audio: np.ndarray,
        sample_rate: int,
        intervals: list[PhoneInterval],
        model_alias: str,
    ) -> object:
        return _wav2vec_result_from_output(
            wav_output(audio, sample_rate),
            intervals,
            model_alias,
        )

    def contrast_call(feature: object, target_phone: str, atlas: object) -> list[dict[str, object]]:
        return authority.contrast_candidates(
            feature_set=feature,
            target_phone=target_phone,
            atlas=atlas,
        )

    return CoachRuntimeServices(
        load_audio=load_audio,
        align=align_fn,
        extract_feature=extractor.extract,
        assess_atlas=atlas_call,
        run_wav2vec=wav_call,
        length_stats=length_stats,
        assess_contrasts=contrast_call,
    )


# ---------------------------------------------------------------------------
# Heavy backend adapters (lazy imports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Wav2VecRunResult:
    model_alias: str
    status: str
    free_decode: list[object]
    spans: list[Wav2VecEvidence]
    frame_posteriors: tuple[object, ...] = ()


def _wav2vec_result_from_output(
    output: object | None,
    intervals: list[PhoneInterval],
    model_alias: str,
) -> _Wav2VecRunResult:
    if output is None:
        return _Wav2VecRunResult(
            model_alias=model_alias,
            status="model_unavailable",
            free_decode=[],
            spans=[
                _unusable_evidence(span.target_phone, 0, "model_unavailable")
                for span in intervals
            ],
            frame_posteriors=(),
        )
    if getattr(output, "status", "ok") == "model_unavailable":
        return _Wav2VecRunResult(
            model_alias=model_alias,
            status="model_unavailable",
            free_decode=[],
            spans=[
                _unusable_evidence(span.target_phone, 0, "model_unavailable")
                for span in intervals
            ],
            frame_posteriors=(),
        )

    # assess.inference returns InferenceResult, not the tuple format
    # Extract posterior frames from the inference result
    free_decode_phones = getattr(output, "phones", []) or []
    frame_posteriors = getattr(output, "frame_posteriors", ()) or ()

    # Build vocab from posterior frames
    vocab: set[str] = set()
    for frame in frame_posteriors:
        vocab.update(getattr(frame, "token_logprobs", {}).keys())

    spans = evidence_for_spans(
        intervals,
        list(frame_posteriors),
        vocab,
        DEFAULT_WAV2VEC_MIN_FRAMES,
    )
    return _Wav2VecRunResult(
        model_alias=model_alias,
        status="ok",
        free_decode=list(free_decode_phones),
        spans=spans,
        frame_posteriors=tuple(frame_posteriors),
    )
