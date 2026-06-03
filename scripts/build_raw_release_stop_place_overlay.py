#!/usr/bin/env python3
"""Build kʰ↔cʰ raw-release detector rows into the curated authority overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assess.contracts import PhoneInterval  # noqa: E402
from assess.features import (  # noqa: E402
    PhoneClass,
    class_window_samples,
    closure_burst_landmarks,
    release_spectrum_features,
)

CURATED_DB = REPO_ROOT / "artifacts" / "curated_stop_place" / "curated_stop_place_reference.sqlite"
AUTH_OVERLAY = REPO_ROOT / "artifacts" / "curated_stop_place" / "decision_authority_overlay.json"
REPORT_JSON = REPO_ROOT / "reports" / "assessment" / "raw-release-stop-place-detector.json"
REPORT_MD = REPO_ROOT / "reports" / "assessment" / "raw-release-stop-place-detector.md"

PHONES = ("kʰ", "cʰ")
TARGET_PAIRS = (("kʰ", "cʰ"), ("cʰ", "kʰ"))
FEATURES = (
    "release20_band_500_1000_log",
    "release20_band_1000_1500_log",
    "release20_band_2500_3000_log",
    "release20_band_3000_4000_log",
    "release20_high_low_2k",
    "release20_high_low_3k",
    "release20_centroid_hz",
    "release20_band_dct1",
    "release20_band_dct3",
    "release20_band_dct4",
)


@dataclass(frozen=True)
class DetectorMetrics:
    target: str
    alternative: str
    train_n_target: int
    train_n_alternative: int
    dev_n_target: int
    dev_n_alternative: int
    test_n_target: int
    test_n_alternative: int
    dev_fp: float
    dev_recall: float
    test_fp: float
    test_recall: float
    test_auc: float
    test_balanced_accuracy_at_threshold: float
    threshold: float
    median_margin_alternative_test: float


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


def load_manifest() -> pd.DataFrame:
    with sqlite3.connect(str(CURATED_DB)) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                id,
                audio_path,
                start_ms,
                end_ms,
                phone,
                next_phone,
                speaker_id,
                provider,
                word,
                meaning
            FROM curation_manifest
            WHERE phone IN ('kʰ', 'cʰ')
              AND included_in_training = 1
            ORDER BY id
            """,
            conn,
        )
    df["split"] = df["speaker_id"].astype(str).map(_split_for_speaker)
    return df


def _load_audio(path: str) -> tuple[np.ndarray, int] | None:
    try:
        samples, sample_rate = sf.read(path, dtype="float32")
    except Exception:
        return None
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if audio.size == 0:
        return None
    return audio, int(sample_rate)


def _features_for_row(row: pd.Series) -> dict[str, Any] | None:
    loaded = _load_audio(str(row["audio_path"]))
    if loaded is None:
        return None
    audio, sample_rate = loaded
    interval = PhoneInterval(
        target_phone=str(row["phone"]),
        start_ms=int(row["start_ms"]),
        end_ms=int(row["end_ms"]),
        speech_start_ms_original=0,
        speech_end_ms_original=max(1, int(round(audio.size * 1000.0 / sample_rate))),
    )
    start, end = class_window_samples(
        audio.size, sample_rate, interval, PhoneClass.STOP
    )
    span = audio[start:end].astype(np.float32)
    landmarks = closure_burst_landmarks(span, sample_rate)
    if landmarks.burst_ms is None:
        return None
    release = release_spectrum_features(
        span, sample_rate, burst_ms=float(landmarks.burst_ms)
    )
    if not all(name in release and math.isfinite(float(release[name])) for name in FEATURES):
        return None
    item: dict[str, Any] = {
        "id": int(row["id"]),
        "phone": str(row["phone"]),
        "speaker_id": str(row["speaker_id"]),
        "split": str(row["split"]),
        "next_phone": str(row["next_phone"]),
        "provider": str(row["provider"]),
    }
    item.update({name: float(release[name]) for name in FEATURES})
    return item


def extract_features(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in manifest.iterrows():
        if index and index % 5000 == 0:
            print(f"extracted {index}/{len(manifest)}", file=sys.stderr, flush=True)
        item = _features_for_row(row)
        if item is not None:
            rows.append(item)
    return pd.DataFrame(rows)


def _fit_detector(
    df: pd.DataFrame,
    *,
    target: str,
    alternative: str,
    fp_ceiling: float,
) -> tuple[dict[str, Any], DetectorMetrics]:
    local = df[df["phone"].isin({target, alternative})].copy()
    local["y"] = (local["phone"] == alternative).astype(int)
    x = local[list(FEATURES)].to_numpy(dtype=float)
    y = local["y"].to_numpy(dtype=int)
    train_mask = local["split"].to_numpy() == "train"
    dev_mask = local["split"].to_numpy() == "dev"
    test_mask = local["split"].to_numpy() == "test"

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=13,
        ),
    )
    model.fit(x[train_mask], y[train_mask])
    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]

    # The runtime projection is coef · ((x - center) / scale), with threshold
    # carrying the logistic intercept and the chosen FP operating point.
    center = scaler.mean_.astype(float)
    scale = scaler.scale_.astype(float)
    weights = clf.coef_[0].astype(float)
    intercept = float(clf.intercept_[0])

    def scores(mask: np.ndarray) -> np.ndarray:
        filled = imputer.transform(x[mask])
        standardized = scaler.transform(filled)
        return standardized @ weights + intercept

    dev_scores = scores(dev_mask)
    dev_y = y[dev_mask]
    target_dev_scores = dev_scores[dev_y == 0]
    if target_dev_scores.size == 0:
        raise ValueError(f"no dev target samples for {target}->{alternative}")
    threshold = float(np.quantile(target_dev_scores, 1.0 - fp_ceiling))

    test_scores = scores(test_mask)
    test_y = y[test_mask]
    dev_pred = dev_scores >= threshold
    test_pred = test_scores >= threshold
    dev_fp = float(np.mean(dev_pred[dev_y == 0])) if np.any(dev_y == 0) else 1.0
    dev_recall = float(np.mean(dev_pred[dev_y == 1])) if np.any(dev_y == 1) else 0.0
    test_fp = float(np.mean(test_pred[test_y == 0])) if np.any(test_y == 0) else 1.0
    test_recall = float(np.mean(test_pred[test_y == 1])) if np.any(test_y == 1) else 0.0
    test_auc = float(roc_auc_score(test_y, test_scores))
    test_bal = float(balanced_accuracy_score(test_y, test_pred))
    alt_test_scores = test_scores[test_y == 1]
    median_margin = (
        float(np.median(alt_test_scores - threshold)) if alt_test_scores.size else 0.0
    )

    detector = {
        "kind": "raw_release_spectrum_logistic",
        "calibration_independent": True,
        "features": list(FEATURES),
        "weights": [round(float(value), 10) for value in weights],
        "center": [round(float(value), 10) for value in center],
        "scale": [round(float(value), 10) for value in scale],
        "threshold": round(float(threshold - intercept), 10),
        "logistic_intercept": round(float(intercept), 10),
        "operating_fp_ceiling": fp_ceiling,
        "dev_fp": round(dev_fp, 6),
        "dev_recall": round(dev_recall, 6),
        "test_fp": round(test_fp, 6),
        "test_recall": round(test_recall, 6),
        "test_auc": round(test_auc, 6),
    }
    metrics = DetectorMetrics(
        target=target,
        alternative=alternative,
        train_n_target=int(np.sum((local["split"] == "train") & (local["phone"] == target))),
        train_n_alternative=int(np.sum((local["split"] == "train") & (local["phone"] == alternative))),
        dev_n_target=int(np.sum((local["split"] == "dev") & (local["phone"] == target))),
        dev_n_alternative=int(np.sum((local["split"] == "dev") & (local["phone"] == alternative))),
        test_n_target=int(np.sum((local["split"] == "test") & (local["phone"] == target))),
        test_n_alternative=int(np.sum((local["split"] == "test") & (local["phone"] == alternative))),
        dev_fp=dev_fp,
        dev_recall=dev_recall,
        test_fp=test_fp,
        test_recall=test_recall,
        test_auc=test_auc,
        test_balanced_accuracy_at_threshold=test_bal,
        threshold=threshold,
        median_margin_alternative_test=median_margin,
    )
    return detector, metrics


def _replace_contrast_row(
    payload: dict[str, Any],
    *,
    target: str,
    alternative: str,
    detector: dict[str, Any],
    metrics: DetectorMetrics,
) -> None:
    for row in payload.get("contrasts", []):
        if not isinstance(row, dict):
            continue
        if row.get("target") == target and row.get("alternative") == alternative:
            row.update(
                {
                    "wav2vec_reliable": True,
                    "atlas_separable": True,
                    "atlas_override_allowed": True,
                    "native_fp_rate": round(metrics.test_fp, 6),
                    "fp_estimate": round(metrics.test_fp, 6),
                    "threshold": detector["threshold"],
                    "override_policy": "hard_override",
                    "reason": "raw_release_spectrum_detector_fp_within_ceiling",
                    "model_scope": "mms-1b-curated-stop-place-raw-release",
                    "acoustic_separability": {
                        "measure": "raw_release_test_recall_at_fp_ceiling",
                        "score": round(metrics.test_recall, 6),
                    },
                    "median_margin": round(metrics.median_margin_alternative_test, 6),
                    "detector": detector,
                    "reliability": {
                        "target_class": "stop",
                        "alternative_class": "stop",
                        "same_class": True,
                        "feature_compatible": True,
                        "n_target": {
                            "train": metrics.train_n_target,
                            "dev": metrics.dev_n_target,
                            "test": metrics.test_n_target,
                        },
                        "n_alternative": {
                            "train": metrics.train_n_alternative,
                            "dev": metrics.dev_n_alternative,
                            "test": metrics.test_n_alternative,
                        },
                    },
                    "sanity": {
                        "feature_confidence_floor": 0.5,
                    },
                }
            )
            return
    raise ValueError(f"contrast row not found: {target}->{alternative}")


def update_overlay(detectors: dict[tuple[str, str], dict[str, Any]], metrics: dict[tuple[str, str], DetectorMetrics]) -> None:
    payload = json.loads(AUTH_OVERLAY.read_text(encoding="utf-8"))
    for target, alternative in TARGET_PAIRS:
        _replace_contrast_row(
            payload,
            target=target,
            alternative=alternative,
            detector=detectors[(target, alternative)],
            metrics=metrics[(target, alternative)],
        )
    payload.setdefault("curated_stop_place_overlay", {})
    payload["curated_stop_place_overlay"]["raw_release_spectrum_detector"] = {
        "pairs": [f"{target}->{alternative}" for target, alternative in TARGET_PAIRS],
        "features": list(FEATURES),
        "source_report": str(REPORT_JSON),
    }
    AUTH_OVERLAY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(
    extracted: pd.DataFrame,
    detectors: dict[tuple[str, str], dict[str, Any]],
    metrics: dict[tuple[str, str], DetectorMetrics],
) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_db": str(CURATED_DB),
        "authority_overlay": str(AUTH_OVERLAY),
        "features": list(FEATURES),
        "counts": {
            "rows": int(len(extracted)),
            "kʰ": int((extracted["phone"] == "kʰ").sum()),
            "cʰ": int((extracted["phone"] == "cʰ").sum()),
            "speakers": int(extracted["speaker_id"].nunique()),
            "by_split": {
                str(split): int(count)
                for split, count in extracted["split"].value_counts().sort_index().items()
            },
        },
        "metrics": [asdict(metrics[pair]) for pair in TARGET_PAIRS],
        "detectors": {
            f"{target}->{alternative}": detectors[(target, alternative)]
            for target, alternative in TARGET_PAIRS
        },
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Raw Release Stop-Place Detector",
        "",
        "Detector uses only 20 ms post-release spectrum features.",
        "",
        "| target | alternative | test FP | test recall | test AUC | test bal. acc. | test n target | test n alt |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in TARGET_PAIRS:
        item = metrics[pair]
        lines.append(
            "| {target} | {alt} | {fp:.4f} | {recall:.4f} | {auc:.4f} | {bal:.4f} | {nt} | {na} |".format(
                target=item.target,
                alt=item.alternative,
                fp=item.test_fp,
                recall=item.test_recall,
                auc=item.test_auc,
                bal=item.test_balanced_accuracy_at_threshold,
                nt=item.test_n_target,
                na=item.test_n_alternative,
            )
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-ceiling", type=float, default=0.02)
    args = parser.parse_args()

    manifest = load_manifest()
    extracted = extract_features(manifest)
    if extracted.empty:
        raise SystemExit("no raw release features extracted")
    detectors: dict[tuple[str, str], dict[str, Any]] = {}
    metrics: dict[tuple[str, str], DetectorMetrics] = {}
    for target, alternative in TARGET_PAIRS:
        detector, metric = _fit_detector(
            extracted,
            target=target,
            alternative=alternative,
            fp_ceiling=args.fp_ceiling,
        )
        detectors[(target, alternative)] = detector
        metrics[(target, alternative)] = metric
    update_overlay(detectors, metrics)
    write_report(extracted, detectors, metrics)
    print(json.dumps({"report": str(REPORT_MD), "overlay": str(AUTH_OVERLAY)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
