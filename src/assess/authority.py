"""Pairwise contrast authority for acoustic analysis candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContrastAuthority:
    target: str
    alternative: str
    wav2vec_reliable: bool
    atlas_separable: bool
    atlas_override_allowed: bool
    native_fp_rate: float
    model_scope: str = "mms-1b"
    acoustic_separability: float | None = None
    threshold: float | None = None
    fp_estimate: float | None = None
    override_policy: str = ""
    reliability: dict[str, object] = field(default_factory=dict)
    sanity: dict[str, object] = field(default_factory=dict)
    detector: dict[str, object] = field(default_factory=dict)
    reason: str = ""


def _default_authority(target: str, alternative: str) -> ContrastAuthority:
    return ContrastAuthority(
        target=target,
        alternative=alternative,
        wav2vec_reliable=False,
        atlas_separable=False,
        atlas_override_allowed=False,
        native_fp_rate=1.0,
        fp_estimate=1.0,
        override_policy="no_override",
        reason="missing_authority_row",
    )


@dataclass(frozen=True)
class AuthorityTable:
    contrasts: dict[tuple[str, str], ContrastAuthority]
    fp_ceiling: float

    def get(self, target: str, alternative: str) -> ContrastAuthority:
        return self.contrasts.get(
            (target, alternative),
            _default_authority(target, alternative),
        )

    def override_allowed(self, target: str, alternative: str) -> bool:
        rec = self.get(target, alternative)
        return (
            _policy_for(rec) == "hard_override"
            and rec.atlas_override_allowed
            and rec.atlas_separable
            and _fp_rate_for(rec) <= self.fp_ceiling
        )

    def wav2vec_reliable(self, target: str, alternative: str) -> bool:
        return self.get(target, alternative).wav2vec_reliable

    def hard_rows_for_target(self, target: str) -> list[ContrastAuthority]:
        return [
            rec
            for (row_target, _), rec in self.contrasts.items()
            if row_target == target and self.override_allowed(rec.target, rec.alternative)
        ]

    def contrast_candidates(
        self,
        *,
        feature_set: object,
        target_phone: str,
        atlas: object | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for rec in self.hard_rows_for_target(target_phone):
            detector = rec.detector
            if not detector:
                continue
            evidence = _evaluate_detector(
                feature_set=feature_set,
                atlas=atlas,
                rec=rec,
                detector=detector,
            )
            if evidence is not None:
                candidates.append(evidence)
        candidates.sort(key=_detector_rank_key, reverse=True)
        return candidates[:limit]

    @classmethod
    def load(cls, path: Path | str) -> AuthorityTable:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("authority artifact must be a JSON object")
        contrasts_payload = data.get("contrasts", [])
        if not isinstance(contrasts_payload, list):
            raise ValueError("authority contrasts must be a list")
        contrasts: dict[tuple[str, str], ContrastAuthority] = {}
        for raw_row in contrasts_payload:
            row = _as_mapping(raw_row)
            fp_estimate = _optional_float(
                row.get("fp_estimate", row.get("native_fp_rate", 1.0))
            )
            acoustic_separability = _parse_acoustic_separability(
                row.get("acoustic_separability", row.get("median_margin"))
            )
            atlas_override_allowed = bool(row.get("atlas_override_allowed", False))
            override_policy = str(row.get("override_policy") or "")
            if not override_policy:
                override_policy = (
                    "hard_override" if atlas_override_allowed else "quality_only"
                )
            rec = ContrastAuthority(
                target=str(row["target"]),
                alternative=str(row["alternative"]),
                wav2vec_reliable=bool(row.get("wav2vec_reliable", False)),
                atlas_separable=bool(row.get("atlas_separable", False)),
                atlas_override_allowed=atlas_override_allowed,
                native_fp_rate=float(row.get("native_fp_rate", fp_estimate or 1.0)),
                model_scope=str(row.get("model_scope") or "mms-1b"),
                acoustic_separability=acoustic_separability,
                threshold=_optional_float(row.get("threshold")),
                fp_estimate=fp_estimate,
                override_policy=override_policy,
                reliability=dict(row.get("reliability") or {}),
                sanity=dict(row.get("sanity") or {}),
                detector=dict(row.get("detector") or {}),
                reason=str(row.get("reason") or ""),
            )
            contrasts[(rec.target, rec.alternative)] = rec
        return cls(
            contrasts=contrasts,
            fp_ceiling=float(data.get("fp_ceiling", 0.05)),
        )


def _as_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("authority contrast rows must be JSON objects")
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _parse_acoustic_separability(value: object) -> float | None:
    if isinstance(value, dict):
        return _optional_float(value.get("score"))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _optional_float(value)


def _policy_for(rec: ContrastAuthority) -> str:
    if rec.override_policy:
        return rec.override_policy
    return "hard_override" if rec.atlas_override_allowed else "quality_only"


def _fp_rate_for(rec: ContrastAuthority) -> float:
    if rec.fp_estimate is not None:
        return rec.fp_estimate
    return rec.native_fp_rate


def _evaluate_detector(
    *,
    feature_set: object,
    atlas: object | None,
    rec: ContrastAuthority,
    detector: dict[str, object],
) -> dict[str, object] | None:
    features = [
        str(name)
        for name in detector.get("features", [])
        if isinstance(name, str) and name
    ]
    weights = _float_list(detector.get("weights"))
    center = _float_list(detector.get("center"))
    scale = _float_list(detector.get("scale"))
    threshold = _optional_float(detector.get("threshold"))
    if (
        not features
        or not weights
        or threshold is None
        or len(weights) != len(features)
    ):
        return None
    if len(center) != len(features):
        center = [0.0] * len(features)
    if len(scale) != len(features):
        scale = [1.0] * len(features)

    values: list[float] = []
    for feature in features:
        value = _optional_float(getattr(feature_set, feature, None))
        if value is None:
            return None
        values.append(value)

    score = 0.0
    for value, mean, denom, weight in zip(values, center, scale, weights, strict=True):
        safe_scale = denom if math.isfinite(denom) and abs(denom) > 1e-9 else 1.0
        score += weight * ((value - mean) / safe_scale)

    margin = score - threshold
    feature_confidence = _optional_float(getattr(feature_set, "feature_confidence", None))
    feature_floor = _sanity_float(rec, "feature_confidence_floor", 0.5)
    target_quality = (
        _atlas_float(atlas, "target_typicality", "quality_score")
        if atlas is not None
        else None
    )
    quality_ceiling = _sanity_float(rec, "target_quality_ceiling", None)
    passed = margin >= 0.0
    reasons: list[str] = []
    if feature_confidence is None or (
        feature_floor is not None and feature_confidence < feature_floor
    ):
        passed = False
        reasons.append("feature_confidence_below_floor")
    if (
        quality_ceiling is not None
        and target_quality is not None
        and target_quality > quality_ceiling
    ):
        passed = False
        reasons.append("target_quality_above_ceiling")
    if margin < 0.0:
        reasons.append("pairwise_margin_below_threshold")

    return {
        "alternative": rec.alternative,
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "margin": round(float(margin), 6),
        "passed": passed,
        "features": list(features),
        "feature_values": {
            name: round(float(value), 6)
            for name, value in zip(features, values, strict=True)
        },
        "detector_kind": str(detector.get("kind") or "linear_projection"),
        "calibration_independent": detector.get("calibration_independent") is True,
        "feature_confidence": feature_confidence,
        "target_quality": target_quality,
        "target_quality_ceiling": quality_ceiling,
        "fp_estimate": _fp_rate_for(rec),
        "recall_estimate": _optional_float(detector.get("test_recall")),
        "reason": rec.reason or "pairwise_detector",
        "reasons": reasons,
    }


def _detector_rank_key(item: dict[str, object]) -> tuple[bool, float, float, float]:
    margin = _optional_float(item.get("margin"))
    fp_estimate = _optional_float(item.get("fp_estimate"))
    recall_estimate = _optional_float(item.get("recall_estimate"))
    return (
        item.get("passed") is True,
        margin if margin is not None else float("-inf"),
        -(fp_estimate if fp_estimate is not None else float("inf")),
        recall_estimate if recall_estimate is not None else float("-inf"),
    )


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        parsed = _optional_float(item)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def _atlas_float(atlas: object, *names: str) -> float | None:
    for name in names:
        parsed = _optional_float(getattr(atlas, name, None))
        if parsed is not None:
            return parsed
    return None


def _sanity_float(
    rec: ContrastAuthority,
    key: str,
    default: float | None,
) -> float | None:
    parsed = _optional_float(rec.sanity.get(key))
    return default if parsed is None else parsed
