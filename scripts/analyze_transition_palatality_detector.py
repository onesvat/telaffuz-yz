#!/usr/bin/env python3
"""Try a small transition-aware kʰ/cʰ acoustic detector.

This is the deliberately simple challenger: F2/F3 post-release transition
features plus one release high/low energy cue. It does not use target text or
semantic context.
"""

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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assess.contracts import CoachRequest, PhoneInterval  # noqa: E402
from assess.features import (  # noqa: E402
    FeatureExtractor,
    PhoneClass,
    class_window_samples,
    closure_burst_landmarks,
    release_spectrum_features,
    stop_formant_transition_features,
)
from assess.runtime import build_default_runtime_services, run_coach  # noqa: E402

CURATED_DB = REPO_ROOT / "artifacts" / "curated_stop_place" / "curated_stop_place_reference.sqlite"
REPORT_JSON = REPO_ROOT / "reports" / "assessment" / "transition-palatality-detector.json"
REPORT_MD = REPO_ROOT / "reports" / "assessment" / "transition-palatality-detector.md"
MANUAL_DIR = REPO_ROOT / "data" / "manual" / "kar"

PHONES = ("kʰ", "cʰ")
PALATAL = "cʰ"
VELAR = "kʰ"
BACK_A_CONTEXT = {"a", "aː"}
TARGET_PHONES = ("cʰ", "aː", "ɾ̞̊")
RANDOM_SEED = 19

FEATURES = (
    "transition_f2_onset_hz",
    "transition_f2_mid_hz",
    "transition_f2_delta_hz",
    "transition_f3_onset_hz",
    "release20_high_low_2k",
)


@dataclass(frozen=True)
class ThresholdMetrics:
    fp_ceiling: float
    threshold: float
    dev_fp: float
    dev_recall: float
    test_fp: float
    test_recall: float
    test_balanced_accuracy: float


@dataclass(frozen=True)
class CvResult:
    model: str
    auc_mean: float | None
    auc_std: float | None
    balanced_accuracy_mean: float | None
    balanced_accuracy_std: float | None
    folds: int
    n: int
    note: str | None = None


@dataclass(frozen=True)
class HardContextResult:
    model: str
    train_n: int
    test_n: int
    palatal_test_n: int
    velar_test_n: int
    auc: float | None
    balanced_accuracy: float | None
    palatal_recall: float | None
    velar_recall: float | None


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


def _finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def load_manifest(per_class: int | None) -> pd.DataFrame:
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
                prev_phone,
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
    df["is_hard_a_context"] = df["next_phone"].isin(BACK_A_CONTEXT)
    df["split"] = df["speaker_id"].astype(str).map(_split_for_speaker)
    if per_class is None:
        return df

    sampled: list[pd.DataFrame] = []
    for phone in PHONES:
        local = df[df["phone"] == phone]
        hard = local[local["is_hard_a_context"]]
        non_hard = local[~local["is_hard_a_context"]]
        take = min(per_class, len(non_hard))
        sampled.append(non_hard.sample(n=take, random_state=RANDOM_SEED))
        sampled.append(hard)
    return (
        pd.concat(sampled, ignore_index=True)
        .drop_duplicates("id")
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


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


def _features_from_interval(
    audio: np.ndarray,
    sample_rate: int,
    interval: PhoneInterval,
) -> dict[str, float] | None:
    start, end = class_window_samples(
        audio.size,
        sample_rate,
        interval,
        PhoneClass.STOP,
    )
    span = audio[start:end].astype(np.float32)
    landmarks = closure_burst_landmarks(span, sample_rate)
    if landmarks.burst_ms is None:
        return None
    values = release_spectrum_features(span, sample_rate, burst_ms=landmarks.burst_ms)
    values.update(
        stop_formant_transition_features(
            span,
            sample_rate,
            burst_ms=landmarks.burst_ms,
        )
    )
    if not all(_finite_float(values.get(name)) is not None for name in FEATURES):
        return None
    return {name: float(values[name]) for name in FEATURES}


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
    values = _features_from_interval(audio, sample_rate, interval)
    if values is None:
        return None
    item: dict[str, Any] = {
        "id": int(row["id"]),
        "phone": str(row["phone"]),
        "y": 1 if str(row["phone"]) == PALATAL else 0,
        "speaker_id": str(row["speaker_id"]),
        "provider": str(row["provider"]),
        "next_phone": str(row["next_phone"]),
        "word": str(row["word"]),
        "meaning": str(row["meaning"]),
        "is_hard_a_context": bool(row["is_hard_a_context"]),
        "split": str(row["split"]),
    }
    item.update(values)
    return item


def extract_features(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in manifest.iterrows():
        if index and index % 500 == 0:
            print(f"extracted {index}/{len(manifest)}", file=sys.stderr, flush=True)
        item = _features_for_row(row)
        if item is not None:
            rows.append(item)
    return pd.DataFrame(rows)


def make_model(name: str) -> Any:
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        )
    if name == "tree_depth2":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=35,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        )
    raise ValueError(name)


def _cv_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if unique_groups.size >= 5:
        return list(GroupKFold(n_splits=5).split(x, y, groups))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(x, y))


def evaluate_cv(df: pd.DataFrame, model_name: str) -> CvResult:
    local = df.reset_index(drop=True)
    x = local[list(FEATURES)].to_numpy(dtype=float)
    y = local["y"].to_numpy(dtype=int)
    groups = local["speaker_id"].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return CvResult(model_name, None, None, None, None, 0, len(local), "one_class")
    aucs: list[float] = []
    bals: list[float] = []
    for train_idx, test_idx in _cv_splits(x, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = make_model(model_name)
        model.fit(x[train_idx], y[train_idx])
        prob = model.predict_proba(x[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(float(roc_auc_score(y[test_idx], prob)))
        bals.append(float(balanced_accuracy_score(y[test_idx], pred)))
    if not aucs:
        return CvResult(model_name, None, None, None, None, 0, len(local), "invalid_splits")
    return CvResult(
        model=model_name,
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        balanced_accuracy_mean=float(np.mean(bals)),
        balanced_accuracy_std=float(np.std(bals)),
        folds=len(aucs),
        n=len(local),
    )


def evaluate_hard_context(df: pd.DataFrame, model_name: str) -> HardContextResult:
    train = df[~df["is_hard_a_context"]].reset_index(drop=True)
    test = df[df["is_hard_a_context"]].reset_index(drop=True)
    if test.empty or len(np.unique(test["y"])) < 2:
        return HardContextResult(
            model=model_name,
            train_n=len(train),
            test_n=len(test),
            palatal_test_n=int(test["y"].sum()) if "y" in test else 0,
            velar_test_n=int(len(test) - test["y"].sum()) if "y" in test else 0,
            auc=None,
            balanced_accuracy=None,
            palatal_recall=None,
            velar_recall=None,
        )
    model = make_model(model_name)
    model.fit(train[list(FEATURES)].to_numpy(dtype=float), train["y"].to_numpy(dtype=int))
    y_test = test["y"].to_numpy(dtype=int)
    prob = model.predict_proba(test[list(FEATURES)].to_numpy(dtype=float))[:, 1]
    pred = (prob >= 0.5).astype(int)
    pal = y_test == 1
    vel = y_test == 0
    return HardContextResult(
        model=model_name,
        train_n=len(train),
        test_n=len(test),
        palatal_test_n=int(pal.sum()),
        velar_test_n=int(vel.sum()),
        auc=float(roc_auc_score(y_test, prob)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        palatal_recall=float(np.mean(pred[pal] == 1)) if pal.any() else None,
        velar_recall=float(np.mean(pred[vel] == 0)) if vel.any() else None,
    )


def univariate(df: pd.DataFrame) -> list[dict[str, Any]]:
    y = df["y"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        values = df[feature].to_numpy(dtype=float)
        mask = np.isfinite(values)
        raw_auc = float(roc_auc_score(y[mask], values[mask]))
        rows.append(
            {
                "feature": feature,
                "auc": raw_auc if raw_auc >= 0.5 else 1.0 - raw_auc,
                "direction": "palatal_higher" if raw_auc >= 0.5 else "palatal_lower",
                "palatal_median": float(np.median(values[mask & (y == 1)])),
                "velar_median": float(np.median(values[mask & (y == 0)])),
            }
        )
    return sorted(rows, key=lambda item: item["auc"], reverse=True)


def fit_logistic_with_thresholds(df: pd.DataFrame) -> tuple[Any, list[ThresholdMetrics], dict[str, Any]]:
    x = df[list(FEATURES)].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    split = df["split"].to_numpy()
    train = split == "train"
    dev = split == "dev"
    test = split == "test"
    model = make_model("logistic")
    model.fit(x[train], y[train])
    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]

    def score(mask: np.ndarray) -> np.ndarray:
        return clf.decision_function(scaler.transform(imputer.transform(x[mask])))

    dev_scores = score(dev)
    test_scores = score(test)
    dev_y = y[dev]
    test_y = y[test]
    rows: list[ThresholdMetrics] = []
    for ceiling in (0.02, 0.05, 0.10):
        target_scores = dev_scores[dev_y == 0]
        threshold = float(np.quantile(target_scores, 1.0 - ceiling))
        test_pred = test_scores >= threshold
        dev_pred = dev_scores >= threshold
        rows.append(
            ThresholdMetrics(
                fp_ceiling=ceiling,
                threshold=threshold,
                dev_fp=float(np.mean(dev_pred[dev_y == 0])),
                dev_recall=float(np.mean(dev_pred[dev_y == 1])),
                test_fp=float(np.mean(test_pred[test_y == 0])),
                test_recall=float(np.mean(test_pred[test_y == 1])),
                test_balanced_accuracy=float(balanced_accuracy_score(test_y, test_pred)),
            )
        )
    detector = {
        "features": list(FEATURES),
        "weights": [float(value) for value in clf.coef_[0]],
        "center": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "intercept": float(clf.intercept_[0]),
    }
    return model, rows, detector


def score_with_model(model: Any, values: dict[str, float]) -> float:
    x = np.asarray([[values[name] for name in FEATURES]], dtype=float)
    return float(model.named_steps["logisticregression"].decision_function(
        model.named_steps["standardscaler"].transform(
            model.named_steps["simpleimputer"].transform(x)
        )
    )[0])


def _first_stop(result: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    timed = result.get("wav2vec", {}).get("timed_phones")
    if not isinstance(timed, list):
        return None
    for index, phone in enumerate(timed):
        if isinstance(phone, dict) and phone.get("ipa") in PHONES:
            return index, phone
    return None


def manual_scores(model: Any, thresholds: list[ThresholdMetrics]) -> list[dict[str, Any]]:
    files = sorted(MANUAL_DIR.glob("*.wav"))
    if not files:
        return []
    seed = CoachRequest(
        audio_path=str(files[0]),
        model_alias="mms-1b",
        target_phones=list(TARGET_PHONES),
        stats_path=str(REPO_ROOT / "artifacts" / "curated_stop_place" / "coach_gmm_overlay.json"),
        authority_path=str(REPO_ROOT / "artifacts" / "curated_stop_place" / "decision_authority_overlay.json"),
        calibration_path=str(REPO_ROOT / "artifacts" / "curated_stop_place" / "coach_gmm_calibration_overlay.json"),
    )
    services = build_default_runtime_services(seed)
    extractor = FeatureExtractor()
    out: list[dict[str, Any]] = []
    for path in files:
        request = CoachRequest(
            audio_path=str(path),
            model_alias="mms-1b",
            target_phones=list(TARGET_PHONES),
            stats_path=seed.stats_path,
            authority_path=seed.authority_path,
            calibration_path=seed.calibration_path,
        )
        result = run_coach(request, services=services).as_dict()
        first = _first_stop(result)
        if first is None:
            out.append({"file": str(path), "error": "no_kh_ch_stop"})
            continue
        index, phone = first
        timed = result["wav2vec"]["timed_phones"]
        prev_phone = str(timed[index - 1]["ipa"]) if index > 0 else None
        next_phone = str(timed[index + 1]["ipa"]) if index + 1 < len(timed) else None
        loaded = _load_audio(str(path))
        if loaded is None:
            out.append({"file": str(path), "error": "audio_load_failed"})
            continue
        audio, sample_rate = loaded
        interval = PhoneInterval(
            target_phone=str(phone["ipa"]),
            start_ms=int(phone["start_ms"]),
            end_ms=int(phone["end_ms"]),
            speech_start_ms_original=int(result["speech_start_ms_original"]),
            speech_end_ms_original=int(result["speech_end_ms_original"]),
        )
        feature_set = extractor.extract(
            audio,
            sample_rate,
            interval,
            prev_phone=prev_phone,
            next_phone=next_phone,
        )
        values = {
            name: _finite_float(getattr(feature_set, name, None))
            for name in FEATURES
        }
        if not all(value is not None for value in values.values()):
            out.append(
                {
                    "file": str(path),
                    "decoded_stop": phone.get("ipa"),
                    "analysis_phones": result.get("analysis", {}).get("phonemes"),
                    "error": "missing_transition_features",
                    "features": values,
                }
            )
            continue
        finite_values = {name: float(value) for name, value in values.items() if value is not None}
        score = score_with_model(model, finite_values)
        out.append(
            {
                "file": str(path),
                "decoded_stop": phone.get("ipa"),
                "analysis_phones": result.get("analysis", {}).get("phonemes"),
                "score": score,
                "features": finite_values,
                "threshold_results": [
                    {
                        "fp_ceiling": item.fp_ceiling,
                        "threshold": item.threshold,
                        "margin": score - item.threshold,
                        "would_flip": score >= item.threshold,
                    }
                    for item in thresholds
                ],
            }
        )
    return out


def tree_rules(df: pd.DataFrame) -> str:
    model = make_model("tree_depth2")
    x = df[list(FEATURES)].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    model.fit(x, y)
    tree = model.named_steps["decisiontreeclassifier"]
    return export_text(tree, feature_names=list(FEATURES), decimals=2)


def write_report(payload: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(value: object, digits: int = 3) -> str:
        number = _finite_float(value)
        return "" if number is None else f"{number:.{digits}f}"

    lines = [
        "# Transition Palatality Detector",
        "",
        "Small acoustic-only challenger for `kʰ` vs `cʰ`: F2/F3 transition plus one release high/low cue.",
        "",
        f"Rows extracted: {payload['counts']['rows']}  cʰ: {payload['counts']['cʰ']}  kʰ: {payload['counts']['kʰ']}",
        f"Hard a/aː rows: {payload['counts']['hard_a_rows']}  cʰ: {payload['counts']['hard_a_cʰ']}  kʰ: {payload['counts']['hard_a_kʰ']}",
        "",
        "## Features",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in FEATURES)
    lines.extend(
        [
            "",
            "## Single Feature Separation",
            "",
            "| feature | AUC | direction | cʰ median | kʰ median |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for item in payload["univariate"]:
        lines.append(
            f"| `{item['feature']}` | {fmt(item['auc'])} | {item['direction']} | {fmt(item['palatal_median'], 1)} | {fmt(item['velar_median'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Speaker-Grouped CV",
            "",
            "| model | AUC | bal. acc. | folds |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in payload["cv"]:
        lines.append(
            f"| {item['model']} | {fmt(item['auc_mean'])} | {fmt(item['balanced_accuracy_mean'])} | {item['folds']} |"
        )
    lines.extend(
        [
            "",
            "## Train Non-a/aː, Test a/aː",
            "",
            "| model | AUC | bal. acc. | cʰ recall | kʰ recall | cʰ n | kʰ n |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["hard_context"]:
        lines.append(
            "| {model} | {auc} | {bal} | {cr} | {kr} | {cn} | {kn} |".format(
                model=item["model"],
                auc=fmt(item["auc"]),
                bal=fmt(item["balanced_accuracy"]),
                cr=fmt(item["palatal_recall"]),
                kr=fmt(item["velar_recall"]),
                cn=item["palatal_test_n"],
                kn=item["velar_test_n"],
            )
        )
    lines.extend(
        [
            "",
            "## Logistic Threshold Tradeoff",
            "",
            "| FP ceiling | threshold | dev FP | dev recall | test FP | test recall | test bal. acc. |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["threshold_metrics"]:
        lines.append(
            "| {ceil} | {th} | {dfp} | {drec} | {tfp} | {trec} | {bal} |".format(
                ceil=fmt(item["fp_ceiling"], 2),
                th=fmt(item["threshold"]),
                dfp=fmt(item["dev_fp"]),
                drec=fmt(item["dev_recall"]),
                tfp=fmt(item["test_fp"]),
                trec=fmt(item["test_recall"]),
                bal=fmt(item["test_balanced_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Manual kâr Scores",
            "",
            "| file | decoded | score | margin @2% FP | margin @5% FP | margin @10% FP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["manual_scores"]:
        margins = {
            round(float(row["fp_ceiling"]), 2): row["margin"]
            for row in item.get("threshold_results", [])
        }
        lines.append(
            "| {file} | `{decoded}` | {score} | {m2} | {m5} | {m10} |".format(
                file=f"`{Path(str(item.get('file'))).name}`",
                decoded=item.get("decoded_stop", ""),
                score=fmt(item.get("score")),
                m2=fmt(margins.get(0.02)),
                m5=fmt(margins.get(0.05)),
                m10=fmt(margins.get(0.10)),
            )
        )
    lines.extend(
        [
            "",
            "## Depth-2 Tree",
            "",
            "```text",
            payload["tree_rules"].rstrip(),
            "```",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=2500,
        help="sample this many non-a/aː rows per class; use <=0 for all rows",
    )
    args = parser.parse_args()
    per_class = None if args.per_class <= 0 else args.per_class

    manifest = load_manifest(per_class)
    features = extract_features(manifest)
    if features.empty:
        raise SystemExit("no transition features extracted")
    model, thresholds, detector = fit_logistic_with_thresholds(features)
    payload = {
        "source_db": str(CURATED_DB),
        "per_class_non_hard_sample": per_class,
        "features": list(FEATURES),
        "counts": {
            "rows": int(len(features)),
            "cʰ": int((features["phone"] == PALATAL).sum()),
            "kʰ": int((features["phone"] == VELAR).sum()),
            "hard_a_rows": int(features["is_hard_a_context"].sum()),
            "hard_a_cʰ": int(((features["phone"] == PALATAL) & features["is_hard_a_context"]).sum()),
            "hard_a_kʰ": int(((features["phone"] == VELAR) & features["is_hard_a_context"]).sum()),
            "speakers": int(features["speaker_id"].nunique()),
            "by_split": {
                str(split): int(count)
                for split, count in features["split"].value_counts().sort_index().items()
            },
        },
        "univariate": univariate(features),
        "cv": [asdict(evaluate_cv(features, name)) for name in ("logistic", "tree_depth2")],
        "hard_context": [
            asdict(evaluate_hard_context(features, name))
            for name in ("logistic", "tree_depth2")
        ],
        "threshold_metrics": [asdict(item) for item in thresholds],
        "logistic_detector": detector,
        "manual_scores": manual_scores(model, thresholds),
        "tree_rules": tree_rules(features),
    }
    write_report(payload)
    print(json.dumps({"report": str(REPORT_MD), "json": str(REPORT_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
