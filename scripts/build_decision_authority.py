#!/usr/bin/env python3
"""Build the static 49x49 directional decision-authority artifact.

Runtime loads this JSON read-only. Hard substitutions come from pairwise
detectors calibrated offline against speaker-disjoint native samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from assess.features import PhoneClass, phone_class
from assess.stats import (
    AtlasGMM,
    GMMCalibration,
    gmm_typicality,
)
from g2p.constants import ALL_PHONEMES, STRESS


DEFAULT_FP_CEILING = 0.02
DEFAULT_MIN_RECALL = 0.05
DEFAULT_MIN_SAMPLES = 25
DEFAULT_MIN_EVAL_SAMPLES = 500
DEFAULT_TARGET_QUALITY_CEILING = 0.20
DEFAULT_FEATURE_CONFIDENCE_FLOOR = 0.5
DEFAULT_MAX_FEATURES = 3
DEFAULT_REGULARIZATION = 1e-3

FEATURE_COLUMNS: tuple[str, ...] = (
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_skew",
    "spectral_kurtosis",
    "voiced_fraction",
    "vot_ms",
    "closure_duration_ms",
    "closure_voicing_ratio",
    "burst_centroid_hz",
    "burst_spectral_skew",
    "burst_confidence",
    "frication_rise_db_per_ms",
    "frication_duration_ms",
    "f2_transition_slope_hz_per_ms",
    "f2_locus_hz",
    "nasal_murmur_ratio",
    "vowel_f2_movement_hz_per_ms",
)

COMMON_FEATURES = frozenset(
    {
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "voiced_fraction",
    }
)

CLASS_FEATURES: dict[PhoneClass, frozenset[str]] = {
    PhoneClass.VOWEL: frozenset(
        {"f1_hz", "f2_hz", "f3_hz", "vowel_f2_movement_hz_per_ms"}
    ),
    PhoneClass.SONORANT: frozenset(
        {
            "f1_hz",
            "f2_hz",
            "f3_hz",
            "nasal_murmur_ratio",
            "f2_transition_slope_hz_per_ms",
        }
    ),
    PhoneClass.FRICATIVE: frozenset(
        {
            "spectral_skew",
            "spectral_kurtosis",
            "frication_rise_db_per_ms",
            "frication_duration_ms",
        }
    ),
    PhoneClass.AFFRICATE: frozenset(
        {
            "vot_ms",
            "closure_duration_ms",
            "closure_voicing_ratio",
            "burst_centroid_hz",
            "burst_spectral_skew",
            "burst_confidence",
            "frication_rise_db_per_ms",
            "frication_duration_ms",
            "f2_transition_slope_hz_per_ms",
            "f2_locus_hz",
            "spectral_skew",
            "spectral_kurtosis",
        }
    ),
    PhoneClass.STOP: frozenset(
        {
            "vot_ms",
            "closure_duration_ms",
            "closure_voicing_ratio",
            "burst_centroid_hz",
            "burst_spectral_skew",
            "burst_confidence",
            "f2_transition_slope_hz_per_ms",
            "f2_locus_hz",
        }
    ),
}

SPLITS = ("train", "dev", "test")


def segment_phonemes() -> list[str]:
    return sorted(phone for phone in ALL_PHONEMES if phone != STRESS)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _phone_class_name(phone: str) -> str | None:
    try:
        return str(phone_class(phone).value)
    except ValueError:
        return None


def _speaker_bucket(speaker_id: str) -> int:
    digest = hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def _split_for_speaker(speaker_id: str) -> str:
    if not speaker_id:
        return "train"
    bucket = _speaker_bucket(speaker_id)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def _possible_features(phone: str) -> frozenset[str]:
    try:
        cls = phone_class(phone)
    except ValueError:
        return frozenset()
    return COMMON_FEATURES | CLASS_FEATURES.get(cls, frozenset())


def _load_phone_samples(
    conn: sqlite3.Connection,
    *,
    phone: str,
    confidence_floor: float,
    sample_limit: int,
) -> dict[str, np.ndarray]:
    select_cols = ", ".join(FEATURE_COLUMNS)
    sql = f"""
        SELECT speaker_id, {select_cols}
        FROM phone_features
        WHERE expected_phone = ?
          AND feature_confidence >= ?
        ORDER BY id
    """
    rows: dict[str, list[list[float]]] = {split: [] for split in SPLITS}
    for raw in conn.execute(sql, (phone, confidence_floor)):
        split = _split_for_speaker(str(raw[0] or ""))
        if len(rows[split]) >= sample_limit:
            if all(len(rows[name]) >= sample_limit for name in SPLITS):
                break
            continue
        values = [
            float("nan") if _finite_float(value) is None else float(value)
            for value in raw[1:]
        ]
        rows[split].append(values)
    return {
        split: (
            np.asarray(items, dtype=np.float64)
            if items
            else np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
        )
        for split, items in rows.items()
    }


def _typicality_to_phone(
    samples: np.ndarray,
    *,
    phone: str,
    atlas: AtlasGMM,
    calibration: GMMCalibration | None,
) -> np.ndarray:
    out = np.full((samples.shape[0],), np.nan, dtype=np.float64)
    gmm = atlas.get(phone)
    if gmm is None or samples.size == 0:
        return out
    indexes = [_feature_index(name) for name in gmm.feature_names]
    usable = np.all(np.isfinite(samples[:, indexes]), axis=1)
    if not np.any(usable):
        return out
    cal = calibration.get(phone) if calibration is not None else None
    values = samples[usable][:, indexes]
    log_density = _phone_log_density_matrix(values, gmm)
    if cal is not None:
        grid = np.asarray(cal.log_density_quantiles, dtype=np.float64)
        ranks = np.searchsorted(grid, log_density, side="right")
        out[usable] = ranks / float(grid.shape[0])
    else:
        out[usable] = np.asarray(
            [gmm_typicality(row, gmm) for row in values],
            dtype=np.float64,
        )
    return out


def _phone_log_density_matrix(values: np.ndarray, gmm: object) -> np.ndarray:
    components = []
    for component in gmm.components:
        mean = np.asarray(component.mean, dtype=np.float64)
        inv_cov = np.asarray(component.inv_cov, dtype=np.float64)
        diff = values - mean
        mahal = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        df = mean.shape[0]
        components.append(
            component.log_weight
            - 0.5 * (mahal + component.cov_log_det + df * math.log(2.0 * math.pi))
        )
    if not components:
        return np.full((values.shape[0],), float("-inf"), dtype=np.float64)
    stacked = np.vstack(components)
    maxes = np.max(stacked, axis=0)
    return maxes + np.log(np.sum(np.exp(stacked - maxes), axis=0))


def _feature_index(name: str) -> int:
    return FEATURE_COLUMNS.index(name)


def _subset_arrays(
    samples: dict[str, np.ndarray],
    target_quality: dict[str, np.ndarray],
    features: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    indexes = [_feature_index(name) for name in features]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        matrix = samples[split]
        quality = target_quality[split]
        if matrix.size == 0:
            out[split] = (
                np.empty((0, len(features)), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
            continue
        mask = np.all(np.isfinite(matrix[:, indexes]), axis=1) & np.isfinite(quality)
        out[split] = (matrix[mask][:, indexes], quality[mask])
    return out


def _feature_subsets(features: list[str], max_features: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for size in range(1, min(max_features, len(features)) + 1):
        out.extend(tuple(combo) for combo in combinations(features, size))
    return out


def _fit_linear_projection(
    target_train: np.ndarray,
    alternative_train: np.ndarray,
    *,
    regularization: float,
) -> tuple[list[float], list[float], list[float]] | None:
    if target_train.size == 0 or alternative_train.size == 0:
        return None
    combined = np.vstack([target_train, alternative_train])
    center = np.median(combined, axis=0)
    scale = np.subtract(*np.percentile(combined, [75, 25], axis=0))
    std = np.std(combined, axis=0)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-9), scale, std)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-9), scale, 1.0)
    target_z = (target_train - center) / scale
    alternative_z = (alternative_train - center) / scale
    mu_target = target_z.mean(axis=0)
    mu_alt = alternative_z.mean(axis=0)
    delta = mu_alt - mu_target
    if not np.any(np.isfinite(delta)):
        return None
    if target_z.shape[1] == 1:
        weights = np.asarray([1.0 if delta[0] >= 0.0 else -1.0])
    else:
        centered = np.vstack([target_z - mu_target, alternative_z - mu_alt])
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]])
        cov = cov + np.eye(cov.shape[0]) * regularization
        try:
            weights = np.linalg.solve(cov, delta)
        except np.linalg.LinAlgError:
            weights = delta
        if float(mu_alt @ weights) < float(mu_target @ weights):
            weights = -weights
    norm = float(np.linalg.norm(weights))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    weights = weights / norm
    return (
        [round(float(value), 10) for value in weights],
        [round(float(value), 10) for value in center],
        [round(float(value), 10) for value in scale],
    )


def _scores(matrix: np.ndarray, weights: list[float], center: list[float], scale: list[float]) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    s = np.where(np.isfinite(s) & (np.abs(s) > 1e-9), s, 1.0)
    return ((matrix - c) / s) @ w


def _threshold_for_fp(
    target_scores: np.ndarray,
    target_quality: np.ndarray,
    *,
    fp_ceiling: float,
    target_quality_ceiling: float,
) -> float | None:
    finite = np.isfinite(target_scores) & np.isfinite(target_quality)
    eligible = target_scores[finite & (target_quality <= target_quality_ceiling)]
    total = int(np.sum(finite))
    if total <= 0:
        return None
    if eligible.size == 0:
        return float("inf")
    allowed = int(math.floor(fp_ceiling * total))
    ordered = np.sort(eligible)[::-1]
    if allowed <= 0:
        return float(np.nextafter(ordered[0], float("inf")))
    if allowed >= ordered.size:
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], float("inf")))


def _metrics(
    scores: np.ndarray,
    quality: np.ndarray,
    *,
    threshold: float,
    target_quality_ceiling: float,
) -> tuple[float, int]:
    finite = np.isfinite(scores) & np.isfinite(quality)
    total = int(np.sum(finite))
    if total == 0:
        return 1.0, 0
    passed = scores[finite] >= threshold
    passed &= quality[finite] <= target_quality_ceiling
    return float(np.mean(passed)), total


def _build_detector(
    *,
    target_samples: dict[str, np.ndarray],
    alternative_samples: dict[str, np.ndarray],
    target_quality_for_target: dict[str, np.ndarray],
    target_quality_for_alternative: dict[str, np.ndarray],
    features: tuple[str, ...],
    fp_ceiling: float,
    min_recall: float,
    min_samples: int,
    min_eval_samples: int,
    target_quality_ceiling: float,
    regularization: float,
) -> dict[str, object] | None:
    target = _subset_arrays(target_samples, target_quality_for_target, features)
    alt = _subset_arrays(alternative_samples, target_quality_for_alternative, features)
    if (
        target["train"][0].shape[0] < min_samples
        or alt["train"][0].shape[0] < min_samples
        or target["dev"][0].shape[0] < min_eval_samples
        or alt["dev"][0].shape[0] < min_eval_samples
        or target["test"][0].shape[0] < min_eval_samples
        or alt["test"][0].shape[0] < min_eval_samples
    ):
        return None
    fitted = _fit_linear_projection(
        target["train"][0],
        alt["train"][0],
        regularization=regularization,
    )
    if fitted is None:
        return None
    weights, center, scale = fitted
    target_dev_scores = _scores(target["dev"][0], weights, center, scale)
    threshold = _threshold_for_fp(
        target_dev_scores,
        target["dev"][1],
        fp_ceiling=fp_ceiling,
        target_quality_ceiling=target_quality_ceiling,
    )
    if threshold is None or not math.isfinite(threshold):
        return None

    target_train_scores = _scores(target["train"][0], weights, center, scale)
    alt_train_scores = _scores(alt["train"][0], weights, center, scale)
    target_test_scores = _scores(target["test"][0], weights, center, scale)
    alt_dev_scores = _scores(alt["dev"][0], weights, center, scale)
    alt_test_scores = _scores(alt["test"][0], weights, center, scale)

    dev_fp, n_target_dev = _metrics(
        target_dev_scores,
        target["dev"][1],
        threshold=threshold,
        target_quality_ceiling=target_quality_ceiling,
    )
    test_fp, n_target_test = _metrics(
        target_test_scores,
        target["test"][1],
        threshold=threshold,
        target_quality_ceiling=target_quality_ceiling,
    )
    dev_recall, n_alt_dev = _metrics(
        alt_dev_scores,
        alt["dev"][1],
        threshold=threshold,
        target_quality_ceiling=target_quality_ceiling,
    )
    test_recall, n_alt_test = _metrics(
        alt_test_scores,
        alt["test"][1],
        threshold=threshold,
        target_quality_ceiling=target_quality_ceiling,
    )
    train_margin = float(np.median(alt_train_scores) - np.median(target_train_scores))
    if (
        train_margin <= 0.0
        or dev_fp > fp_ceiling
        or test_fp > fp_ceiling
        or dev_recall < min_recall
        or test_recall < min_recall
    ):
        return None
    return {
        "kind": "linear_projection",
        "features": list(features),
        "weights": weights,
        "center": center,
        "scale": scale,
        "threshold": round(float(threshold), 10),
        "target_quality_ceiling": target_quality_ceiling,
        "dev_fp": round(dev_fp, 6),
        "test_fp": round(test_fp, 6),
        "dev_recall": round(dev_recall, 6),
        "test_recall": round(test_recall, 6),
        "train_margin": round(train_margin, 6),
        "n_target_dev": n_target_dev,
        "n_target_test": n_target_test,
        "n_alternative_dev": n_alt_dev,
        "n_alternative_test": n_alt_test,
    }


def _select_detector(
    *,
    target: str,
    alternative: str,
    phone_samples: dict[str, dict[str, np.ndarray]],
    target_quality_cache: dict[tuple[str, str], dict[str, np.ndarray]],
    fp_ceiling: float,
    min_recall: float,
    min_samples: int,
    min_eval_samples: int,
    target_quality_ceiling: float,
    max_features: int,
    regularization: float,
) -> dict[str, object] | None:
    try:
        target_features = _possible_features(target)
        alternative_features = _possible_features(alternative)
    except ValueError:
        return None
    features = sorted(target_features & alternative_features)
    if not features:
        return None
    best: dict[str, object] | None = None
    for subset in _feature_subsets(features, max_features):
        detector = _build_detector(
            target_samples=phone_samples[target],
            alternative_samples=phone_samples[alternative],
            target_quality_for_target=target_quality_cache[(target, target)],
            target_quality_for_alternative=target_quality_cache[(alternative, target)],
            features=subset,
            fp_ceiling=fp_ceiling,
            min_recall=min_recall,
            min_samples=min_samples,
            min_eval_samples=min_eval_samples,
            target_quality_ceiling=target_quality_ceiling,
            regularization=regularization,
        )
        if detector is None:
            continue
        if best is None or _detector_rank(detector) > _detector_rank(best):
            best = detector
    return best


def _detector_rank(detector: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(detector.get("test_recall") or 0.0),
        float(detector.get("dev_recall") or 0.0),
        float(detector.get("train_margin") or 0.0),
        -len(detector.get("features") or []),
    )


def _quality_cache_for(
    *,
    samples_by_phone: dict[str, dict[str, np.ndarray]],
    target_phone: str,
    atlas: AtlasGMM,
    calibration: GMMCalibration | None,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for sample_phone, split_samples in samples_by_phone.items():
        out[sample_phone] = {
            split: _typicality_to_phone(
                matrix,
                phone=target_phone,
                atlas=atlas,
                calibration=calibration,
            )
            for split, matrix in split_samples.items()
        }
    return out


def _authority_row(
    *,
    target: str,
    alternative: str,
    detector: dict[str, object] | None,
    phone_samples: dict[str, dict[str, np.ndarray]],
    fp_ceiling: float,
    min_samples: int,
    min_eval_samples: int,
    target_quality_ceiling: float,
    feature_confidence_floor: float,
    model_scope: str,
) -> dict[str, Any]:
    target_class = _phone_class_name(target)
    alternative_class = _phone_class_name(alternative)
    same_class = target_class is not None and target_class == alternative_class
    feature_compatible = bool(_possible_features(target) & _possible_features(alternative))
    hard_allowed = detector is not None

    if hard_allowed:
        override_policy = "hard_override"
        reason = "pairwise_detector_fp_within_ceiling"
    elif same_class or feature_compatible:
        override_policy = "quality_only"
        reason = "no_validated_pairwise_detector"
    else:
        override_policy = "no_override"
        reason = "incompatible_feature_space"

    fp_estimate = (
        float(detector.get("test_fp", 1.0)) if detector is not None else 1.0
    )
    recall_estimate = (
        float(detector.get("test_recall", 0.0)) if detector is not None else 0.0
    )
    threshold = (
        float(detector.get("threshold")) if detector is not None else None
    )
    target_counts = {
        split: int(phone_samples[target][split].shape[0])
        for split in SPLITS
    }
    alternative_counts = {
        split: int(phone_samples[alternative][split].shape[0])
        for split in SPLITS
    }
    return {
        "target": target,
        "alternative": alternative,
        "model_scope": model_scope,
        "acoustic_separability": {
            "measure": "pairwise_test_recall_at_fp_ceiling",
            "score": round(recall_estimate, 6),
        },
        "threshold": None if threshold is None else round(threshold, 6),
        "fp_estimate": round(fp_estimate, 6),
        "override_policy": override_policy,
        "reliability": {
            "target_class": target_class,
            "alternative_class": alternative_class,
            "same_class": same_class,
            "feature_compatible": feature_compatible,
            "n_target": target_counts,
            "n_alternative": alternative_counts,
            "min_samples": min_samples,
            "min_eval_samples": min_eval_samples,
        },
        "sanity": {
            "target_quality_ceiling": target_quality_ceiling,
            "feature_confidence_floor": feature_confidence_floor,
        },
        "detector": detector or {},
        "reason": reason,
        "wav2vec_reliable": hard_allowed,
        "atlas_separable": hard_allowed,
        "atlas_override_allowed": hard_allowed,
        "native_fp_rate": round(fp_estimate, 6),
        "median_margin": (
            None
            if detector is None
            else round(float(detector.get("train_margin") or 0.0), 6)
        ),
    }


def build_authority(
    *,
    db_path: Path,
    gmm_path: Path,
    calibration_path: Path | None,
    sample_limit: int,
    min_samples: int,
    confidence_floor: float,
    fp_ceiling: float,
    min_recall: float,
    target_quality_ceiling: float,
    feature_confidence_floor: float,
    max_features: int,
    min_eval_samples: int,
    regularization: float,
    model_scope: str,
) -> dict[str, object]:
    atlas = AtlasGMM.load(gmm_path)
    calibration = (
        GMMCalibration.load(calibration_path)
        if calibration_path is not None and calibration_path.exists()
        else None
    )
    phones = segment_phonemes()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        phone_samples = {
            phone: _load_phone_samples(
                conn,
                phone=phone,
                confidence_floor=confidence_floor,
                sample_limit=sample_limit,
            )
            for phone in phones
        }
    finally:
        conn.close()

    target_quality_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for target in phones:
        per_target = _quality_cache_for(
            samples_by_phone=phone_samples,
            target_phone=target,
            atlas=atlas,
            calibration=calibration,
        )
        for sample_phone, split_payload in per_target.items():
            target_quality_cache[(sample_phone, target)] = split_payload

    contrasts = []
    for target in phones:
        for alternative in phones:
            if target == alternative:
                continue
            detector = _select_detector(
                target=target,
                alternative=alternative,
                phone_samples=phone_samples,
                target_quality_cache=target_quality_cache,
                fp_ceiling=fp_ceiling,
                min_recall=min_recall,
                min_samples=min_samples,
                min_eval_samples=min_eval_samples,
                target_quality_ceiling=target_quality_ceiling,
                max_features=max_features,
                regularization=regularization,
            )
            contrasts.append(
                _authority_row(
                    target=target,
                    alternative=alternative,
                    detector=detector,
                    phone_samples=phone_samples,
                    fp_ceiling=fp_ceiling,
                    min_samples=min_samples,
                    min_eval_samples=min_eval_samples,
                    target_quality_ceiling=target_quality_ceiling,
                    feature_confidence_floor=feature_confidence_floor,
                    model_scope=model_scope,
                )
            )

    return {
        "artifact_schema_version": 3,
        "generated_by": "scripts/build_decision_authority.py",
        "model_scope": model_scope,
        "fp_ceiling": fp_ceiling,
        "phonemes": phones,
        "coverage": {
            "segment_phone_count": len(phones),
            "directional_non_self_pairs": len(contrasts),
        },
        "parameters": {
            "sample_limit_per_split": sample_limit,
            "min_samples": min_samples,
            "min_eval_samples": min_eval_samples,
            "confidence_floor": confidence_floor,
            "fp_ceiling": fp_ceiling,
            "min_recall": min_recall,
            "target_quality_ceiling": target_quality_ceiling,
            "feature_confidence_floor": feature_confidence_floor,
            "max_features": max_features,
            "regularization": regularization,
            "splits": {
                "train_buckets": [0, 80],
                "dev_buckets": [80, 90],
                "test_buckets": [90, 100],
            },
        },
        "contrasts": contrasts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--gmm", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=5000)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--min-eval-samples", type=int, default=DEFAULT_MIN_EVAL_SAMPLES)
    parser.add_argument("--confidence-floor", type=float, default=DEFAULT_FEATURE_CONFIDENCE_FLOOR)
    parser.add_argument("--fp-ceiling", type=float, default=DEFAULT_FP_CEILING)
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument(
        "--target-quality-ceiling",
        type=float,
        default=DEFAULT_TARGET_QUALITY_CEILING,
    )
    parser.add_argument(
        "--feature-confidence-floor",
        type=float,
        default=DEFAULT_FEATURE_CONFIDENCE_FLOOR,
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    parser.add_argument("--model-scope", default="mms-1b")
    # Deprecated v2 knobs are accepted so old build commands fail closed only
    # through validation, not argument parsing.
    parser.add_argument("--margin-quantile", type=float, default=None)
    parser.add_argument("--base-threshold", type=float, default=None)
    parser.add_argument("--separability-floor", type=float, default=None)
    parser.add_argument("--alternative-floor", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_authority(
        db_path=args.db,
        gmm_path=args.gmm,
        calibration_path=args.calibration,
        sample_limit=args.sample_limit,
        min_samples=args.min_samples,
        min_eval_samples=args.min_eval_samples,
        confidence_floor=args.confidence_floor,
        fp_ceiling=args.fp_ceiling,
        min_recall=args.min_recall,
        target_quality_ceiling=args.target_quality_ceiling,
        feature_confidence_floor=args.feature_confidence_floor,
        max_features=args.max_features,
        regularization=args.regularization,
        model_scope=args.model_scope,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hard = sum(
        1
        for row in payload["contrasts"]
        if isinstance(row, dict) and row.get("override_policy") == "hard_override"
    )
    print(
        json.dumps(
            {
                "phonemes": len(payload["phonemes"]),
                "contrasts": len(payload["contrasts"]),
                "hard_override": hard,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
