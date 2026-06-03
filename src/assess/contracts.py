"""Public JSON-stable contracts for the coach runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _plain(value: object) -> object:
    if hasattr(value, "as_dict"):
        return value.as_dict()  # type: ignore[no-any-return, attr-defined]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class PhoneInterval:
    """Phone interval expressed in original-audio milliseconds."""

    target_phone: str
    start_ms: int
    end_ms: int
    speech_start_ms_original: int
    speech_end_ms_original: int

    def as_dict(self) -> dict[str, object]:
        return {
            "target_phone": self.target_phone,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speech_start_ms_original": self.speech_start_ms_original,
            "speech_end_ms_original": self.speech_end_ms_original,
        }


@dataclass(frozen=True)
class Wav2VecEvidence:
    """Raw wav2vec posterior evidence for one phone span."""

    target_phone: str
    observed_phone: str | None
    target_logprob: float | None
    best_logprob: float | None
    gop_raw: float | None
    observed_matches_target: bool | None
    frame_count: int
    confidence: float
    reasons: list[str]
    peak_frame_index: int | None = None
    identity_score: float | None = None
    observed_source: str = "peak_frame"

    def as_dict(self) -> dict[str, object]:
        return {
            "target_phone": self.target_phone,
            "observed_phone": self.observed_phone,
            "target_logprob": self.target_logprob,
            "best_logprob": self.best_logprob,
            "gop_raw": self.gop_raw,
            "observed_matches_target": self.observed_matches_target,
            "frame_count": self.frame_count,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "peak_frame_index": self.peak_frame_index,
            "identity_score": self.identity_score,
            "observed_source": self.observed_source,
        }


@dataclass(frozen=True)
class CoachRequest:
    """Input contract for a single coach assessment."""

    audio_path: str
    model_alias: str
    target_text: str | None = None
    target_phones: list[str] | None = None
    stats_path: str | None = None
    authority_path: str | None = None
    calibration_path: str | None = None

    def __post_init__(self) -> None:
        has_text = bool(self.target_text)
        has_phones = bool(self.target_phones)
        if has_text == has_phones:
            raise ValueError("exactly one of target_text or target_phones is required")
        if not self.audio_path:
            raise ValueError("audio_path must not be empty")
        if not self.model_alias:
            raise ValueError("model_alias must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "model_alias": self.model_alias,
            "target_text": self.target_text,
            "target_phones": list(self.target_phones) if self.target_phones else None,
            "stats_path": self.stats_path,
            "authority_path": self.authority_path,
            "calibration_path": self.calibration_path,
        }


@dataclass(frozen=True)
class CoachPhoneResult:
    """Per-phone result carrying both identity and quality channels."""

    target_phone: str
    start_ms: int
    end_ms: int
    status: str
    score: float | None
    atlas: Any
    wav2vec: Any
    identity_score: float | None = None
    quality_score: float | None = None
    length_score: float | None = None
    length_note: str | None = None
    analysis_source: str | None = None
    changed_by_analysis: bool = False
    reasons: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "target_phone": self.target_phone,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "status": self.status,
            "score": self.score,
            "identity_score": self.identity_score,
            "quality_score": self.quality_score,
            "length_score": self.length_score,
            "length_note": self.length_note,
            "analysis_source": self.analysis_source,
            "changed_by_analysis": self.changed_by_analysis,
            "reasons": list(self.reasons or []),
            "atlas": _plain(self.atlas),
            "wav2vec": _plain(self.wav2vec),
        }


@dataclass(frozen=True)
class CoachResult:
    """Top-level coach assessment result."""

    model_alias: str
    status: str
    speech_start_ms_original: int
    speech_end_ms_original: int
    phones: list[CoachPhoneResult]
    free_decode: list[Any]
    target: dict[str, object] | None = None
    wav2vec: dict[str, object] | None = None
    analysis: dict[str, object] | None = None
    result: dict[str, object] | None = None
    debug: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "model_alias": self.model_alias,
            "status": self.status,
            "speech_start_ms_original": self.speech_start_ms_original,
            "speech_end_ms_original": self.speech_end_ms_original,
            "phones": [_plain(phone) for phone in self.phones],
            "free_decode": [_plain(phone) for phone in self.free_decode],
            "target": _plain(self.target) if self.target is not None else None,
            "wav2vec": _plain(self.wav2vec) if self.wav2vec is not None else None,
            "analysis": _plain(self.analysis) if self.analysis is not None else None,
            "result": _plain(self.result) if self.result is not None else None,
            "debug": _plain(self.debug) if self.debug is not None else None,
        }
