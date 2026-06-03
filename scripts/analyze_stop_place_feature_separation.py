#!/usr/bin/env python3
"""Analyze acoustic separation of Turkish dorsal stop place in curated data."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_DB = REPO_ROOT / "artifacts" / "curated_stop_place" / "curated_stop_place_reference.sqlite"
EXTERNAL_REFERENCE_DB = Path("/home/onur/Code/telaffuz-yz-thesis-db/coach_reference_features.sqlite")

REPORT_JSON = REPO_ROOT / "reports" / "assessment" / "stop-place-feature-separation.json"
REPORT_MD = REPO_ROOT / "reports" / "assessment" / "stop-place-feature-separation.md"

PHONES = ("k", "kʰ", "c", "cʰ")
PALATAL = {"c", "cʰ"}
VELAR = {"k", "kʰ"}
BACK_A_CONTEXT = {"a", "aː"}

FEATURE_COLUMNS = [
    "duration_ms",
    "active_duration_ms",
    "feature_confidence",
    "anchor_confidence",
    "formant_success",
    "duration_reliability",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_skew",
    "spectral_kurtosis",
    "voiced_fraction",
    "rms",
    "vot_ms",
    "closure_ms",
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
    "next_f1_hz",
    "next_f2_hz",
    "next_f3_hz",
    "next_duration_ms",
    "next_spectral_centroid_hz",
    "f2_locus_minus_next_f2_hz",
    "f2_locus_over_next_f2",
]

BASE_FEATURE_COLUMNS = FEATURE_COLUMNS[:28]

MODEL_FEATURE_SETS: dict[str, list[str]] = {
    "f2_locus_only": ["f2_locus_hz"],
    "f2_static": ["f1_hz", "f2_hz", "f3_hz", "f2_locus_hz"],
    "next_vowel_formants": ["next_f1_hz", "next_f2_hz", "next_f3_hz"],
    "locus_plus_next_vowel": [
        "f2_locus_hz",
        "next_f2_hz",
        "f2_locus_minus_next_f2_hz",
        "f2_locus_over_next_f2",
    ],
    "f2_transition": [
        "f2_locus_hz",
        "f2_transition_slope_hz_per_ms",
        "vowel_f2_movement_hz_per_ms",
    ],
    "burst_spectrum": [
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_skew",
        "spectral_kurtosis",
        "burst_centroid_hz",
        "burst_spectral_skew",
        "burst_confidence",
    ],
    "timing_voicing": [
        "duration_ms",
        "active_duration_ms",
        "voiced_fraction",
        "vot_ms",
        "closure_duration_ms",
        "closure_voicing_ratio",
    ],
    "current_detector": [
        "f2_locus_hz",
        "f2_transition_slope_hz_per_ms",
        "spectral_centroid_hz",
    ],
    "combined_core": [
        "f2_locus_hz",
        "f2_transition_slope_hz_per_ms",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_skew",
        "burst_centroid_hz",
        "burst_spectral_skew",
        "burst_confidence",
        "vot_ms",
        "closure_duration_ms",
        "closure_voicing_ratio",
    ],
    "all_acoustic": list(FEATURE_COLUMNS),
}


@dataclass
class FeatureMetric:
    feature: str
    n: int
    palatal_n: int
    velar_n: int
    coverage: float
    palatal_median: float | None
    velar_median: float | None
    palatal_iqr: list[float | None]
    velar_iqr: list[float | None]
    auc: float | None
    best_balanced_accuracy: float | None
    best_threshold: float | None
    direction: str | None
    cohen_d: float | None


@dataclass
class ModelMetric:
    feature_set: str
    n: int
    palatal_n: int
    velar_n: int
    auc_mean: float | None
    auc_std: float | None
    balanced_accuracy_mean: float | None
    balanced_accuracy_std: float | None
    folds: int
    skipped_reason: str | None = None


def load_data() -> pd.DataFrame:
    phones_sql = ",".join("?" for _ in PHONES)
    columns = ", ".join(f"pf.{name}" for name in BASE_FEATURE_COLUMNS)
    query = f"""
        SELECT
            pf.id,
            pf.expected_phone,
            pf.provider,
            pf.speaker_id,
            pf.segment_alignment_id,
            pf.phone_index,
            cm.prev_phone,
            cm.next_phone,
            cm.word,
            cm.meaning,
            cm.included_in_training,
            nxt.f1_hz AS next_f1_hz,
            nxt.f2_hz AS next_f2_hz,
            nxt.f3_hz AS next_f3_hz,
            nxt.duration_ms AS next_duration_ms,
            nxt.spectral_centroid_hz AS next_spectral_centroid_hz,
            {columns}
        FROM phone_features pf
        JOIN curation_manifest cm ON cm.source_ref_id = pf.id
        LEFT JOIN ext.phone_features nxt
          ON nxt.segment_alignment_id = pf.segment_alignment_id
         AND nxt.phone_index = pf.phone_index + 1
        WHERE pf.expected_phone IN ({phones_sql})
          AND cm.included_in_training = 1
    """
    with sqlite3.connect(str(CURATED_DB)) as conn:
        conn.execute("ATTACH DATABASE ? AS ext", (str(EXTERNAL_REFERENCE_DB),))
        df = pd.read_sql_query(query, conn, params=PHONES)
    df["place"] = np.where(df["expected_phone"].isin(PALATAL), "palatal", "velar")
    df["y"] = np.where(df["expected_phone"].isin(PALATAL), 1, 0)
    df["f2_locus_minus_next_f2_hz"] = (
        pd.to_numeric(df["f2_locus_hz"], errors="coerce")
        - pd.to_numeric(df["next_f2_hz"], errors="coerce")
    )
    next_f2 = pd.to_numeric(df["next_f2_hz"], errors="coerce")
    df["f2_locus_over_next_f2"] = pd.to_numeric(
        df["f2_locus_hz"], errors="coerce"
    ) / next_f2.where(next_f2 != 0)
    return df


def finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantiles(values: pd.Series) -> list[float | None]:
    finite = finite_series(values)
    if finite.empty:
        return [None, None]
    return [_safe_float(finite.quantile(0.25)), _safe_float(finite.quantile(0.75))]


def _best_threshold(values: np.ndarray, y: np.ndarray, direction: int) -> tuple[float, float]:
    scores = values * direction
    order = np.argsort(scores)
    sorted_scores = scores[order]
    candidates = np.unique(sorted_scores)
    if candidates.size > 400:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 400))
    best_acc = -1.0
    best_threshold = float(candidates[0])
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        acc = balanced_accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)
    return best_acc, best_threshold * direction


def feature_metrics(df: pd.DataFrame) -> list[FeatureMetric]:
    out: list[FeatureMetric] = []
    total = len(df)
    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(df[feature], errors="coerce")
        mask = np.isfinite(values.to_numpy())
        local = df.loc[mask, ["y", "place", feature]]
        pal = finite_series(local.loc[local["place"] == "palatal", feature])
        vel = finite_series(local.loc[local["place"] == "velar", feature])
        auc = None
        best_acc = None
        threshold = None
        direction = None
        cohen_d = None
        if len(pal) >= 10 and len(vel) >= 10:
            x = local[feature].to_numpy(dtype=float)
            y = local["y"].to_numpy(dtype=int)
            raw_auc = float(roc_auc_score(y, x))
            if raw_auc >= 0.5:
                auc = raw_auc
                sign = 1
                direction = "palatal_higher"
            else:
                auc = 1.0 - raw_auc
                sign = -1
                direction = "palatal_lower"
            best_acc, threshold = _best_threshold(x, y, sign)
            pooled = math.sqrt(
                (
                    (len(pal) - 1) * float(pal.var(ddof=1))
                    + (len(vel) - 1) * float(vel.var(ddof=1))
                )
                / max(1, (len(pal) + len(vel) - 2))
            )
            if pooled > 0.0 and math.isfinite(pooled):
                cohen_d = (float(pal.mean()) - float(vel.mean())) / pooled
        out.append(
            FeatureMetric(
                feature=feature,
                n=int(mask.sum()),
                palatal_n=int(len(pal)),
                velar_n=int(len(vel)),
                coverage=float(mask.mean()) if total else 0.0,
                palatal_median=_safe_float(pal.median()) if len(pal) else None,
                velar_median=_safe_float(vel.median()) if len(vel) else None,
                palatal_iqr=_quantiles(pal),
                velar_iqr=_quantiles(vel),
                auc=auc,
                best_balanced_accuracy=best_acc,
                best_threshold=threshold,
                direction=direction,
                cohen_d=_safe_float(cohen_d),
            )
        )
    return sorted(
        out,
        key=lambda item: (
            item.auc is not None,
            item.auc or -1.0,
            item.coverage,
        ),
        reverse=True,
    )


def _usable_model_features(df: pd.DataFrame, features: Iterable[str]) -> list[str]:
    usable: list[str] = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        if np.isfinite(values.to_numpy()).sum() >= 20:
            usable.append(feature)
    return usable


def model_metric(df: pd.DataFrame, name: str, features: list[str]) -> ModelMetric:
    usable = _usable_model_features(df, features)
    y = df["y"].to_numpy(dtype=int)
    palatal_n = int(y.sum())
    velar_n = int(len(y) - y.sum())
    if len(usable) == 0:
        return ModelMetric(name, len(df), palatal_n, velar_n, None, None, None, None, 0, "no_usable_features")
    if palatal_n < 30 or velar_n < 30:
        return ModelMetric(name, len(df), palatal_n, velar_n, None, None, None, None, 0, "too_few_class_samples")

    x = df[usable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    groups = df["speaker_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size >= 5 and min(palatal_n, velar_n) >= 50:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(x, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        splits = list(splitter.split(x, y))

    aucs: list[float] = []
    balanced: list[float] = []
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=0,
            ),
        )
        clf.fit(x[train_idx], y[train_idx])
        prob = clf.predict_proba(x[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(float(roc_auc_score(y[test_idx], prob)))
        balanced.append(float(balanced_accuracy_score(y[test_idx], pred)))
    if not aucs:
        return ModelMetric(name, len(df), palatal_n, velar_n, None, None, None, None, 0, "invalid_cv_splits")
    return ModelMetric(
        feature_set=name,
        n=len(df),
        palatal_n=palatal_n,
        velar_n=velar_n,
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        balanced_accuracy_mean=float(np.mean(balanced)),
        balanced_accuracy_std=float(np.std(balanced)),
        folds=len(aucs),
    )


def subset_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "global_all_kc": df.copy(),
        "aspirated_only": df[df["expected_phone"].isin({"kʰ", "cʰ"})].copy(),
        "plain_only": df[df["expected_phone"].isin({"k", "c"})].copy(),
        "back_a_context_all": df[df["next_phone"].isin(BACK_A_CONTEXT)].copy(),
        "back_a_context_aspirated": df[
            df["next_phone"].isin(BACK_A_CONTEXT)
            & df["expected_phone"].isin({"kʰ", "cʰ"})
        ].copy(),
        "back_a_context_plain": df[
            df["next_phone"].isin(BACK_A_CONTEXT)
            & df["expected_phone"].isin({"k", "c"})
        ].copy(),
        "curated_kar_profit_snow": df[
            df["meaning"].isin({"profit", "snow"})
            | df["word"].isin({"kar", "kâr"})
        ].copy(),
    }


def format_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def top_features_table(metrics: list[FeatureMetric], limit: int = 12) -> list[str]:
    lines = [
        "| feature | AUC | bal. acc. | direction | pal median | vel median | coverage |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for item in metrics[:limit]:
        lines.append(
            "| {feature} | {auc} | {acc} | {direction} | {pal} | {vel} | {coverage} |".format(
                feature=item.feature,
                auc=format_num(item.auc),
                acc=format_num(item.best_balanced_accuracy),
                direction=item.direction or "",
                pal=format_num(item.palatal_median, 1),
                vel=format_num(item.velar_median, 1),
                coverage=format_num(item.coverage),
            )
        )
    return lines


def model_table(metrics: list[ModelMetric]) -> list[str]:
    lines = [
        "| feature set | AUC mean | bal. acc. mean | folds | note |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        lines.append(
            "| {name} | {auc} | {acc} | {folds} | {note} |".format(
                name=item.feature_set,
                auc=format_num(item.auc_mean),
                acc=format_num(item.balanced_accuracy_mean),
                folds=item.folds,
                note=item.skipped_reason or "",
            )
        )
    return lines


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stop-Place Feature Separation",
        "",
        "Classes: velar `k/kʰ` vs palatal `c/cʰ`. Positive class is palatal.",
        "Threshold metrics are univariate and sign-adjusted so AUC >= 0.5.",
        "",
    ]
    for subset_name, subset_payload in payload["subsets"].items():
        counts = subset_payload["counts"]
        lines.extend(
            [
                f"## {subset_name}",
                "",
                f"Rows: {counts['rows']}  palatal: {counts['palatal']}  velar: {counts['velar']}",
                "",
                "### Top Single Features",
                "",
                *top_features_table(
                    [FeatureMetric(**item) for item in subset_payload["feature_metrics"]]
                ),
                "",
                "### Compact Classifiers",
                "",
                *model_table([ModelMetric(**item) for item in subset_payload["model_metrics"]]),
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    df = load_data()
    subsets = subset_frames(df)
    payload: dict[str, Any] = {
        "source_db": str(CURATED_DB),
        "phones": PHONES,
        "feature_columns": FEATURE_COLUMNS,
        "subsets": {},
    }
    for name, subset in subsets.items():
        subset = subset.reset_index(drop=True)
        feature_items = [asdict(item) for item in feature_metrics(subset)]
        model_items = [
            asdict(model_metric(subset, model_name, features))
            for model_name, features in MODEL_FEATURE_SETS.items()
        ]
        payload["subsets"][name] = {
            "counts": {
                "rows": int(len(subset)),
                "palatal": int(subset["y"].sum()),
                "velar": int(len(subset) - subset["y"].sum()),
                "phones": {
                    str(phone): int(count)
                    for phone, count in subset["expected_phone"].value_counts().sort_index().items()
                },
                "next_phone_top": {
                    str(phone): int(count)
                    for phone, count in subset["next_phone"].value_counts(dropna=False).head(12).items()
                },
            },
            "feature_metrics": feature_items,
            "model_metrics": model_items,
        }
    write_reports(payload)
    print(json.dumps({"report": str(REPORT_MD), "json": str(REPORT_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
