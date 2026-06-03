#!/usr/bin/env python3
"""Extract rich waveform features for kʰ/cʰ separation experiments.

This deliberately computes features from the WAV signal instead of relying on
the existing coach feature columns. It evaluates both speaker-grouped global
CV and the hard generalization case: train away from a/aː context, test on
the scarce cʰ+a/aː rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fftpack import dct
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_DB = REPO_ROOT / "artifacts" / "curated_stop_place" / "curated_stop_place_reference.sqlite"
REPORT_JSON = REPO_ROOT / "reports" / "assessment" / "kh-ch-rich-waveform-features.json"
REPORT_MD = REPO_ROOT / "reports" / "assessment" / "kh-ch-rich-waveform-features.md"

PHONES = ("kʰ", "cʰ")
BACK_A_CONTEXT = {"a", "aː"}
RANDOM_SEED = 7

WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("pre20_0", -20.0, 0.0),
    ("stop0_20", 0.0, 20.0),
    ("stop0_40", 0.0, 40.0),
    ("post20_60", 20.0, 60.0),
    ("phoneend_0_40", 20.0, 60.0),
    ("stop0_80", 0.0, 80.0),
)

STRICT_RELEASE_PREFIXES = ("stop0_20",)
EARLY_RELEASE_PREFIXES = ("stop0_20", "stop0_40")
ALL_PREFIXES = tuple(name for name, _start, _end in WINDOWS)


@dataclass(frozen=True)
class ModelResult:
    name: str
    feature_group: str
    n: int
    auc_mean: float | None
    auc_std: float | None
    balanced_accuracy_mean: float | None
    balanced_accuracy_std: float | None
    folds: int
    note: str | None = None


@dataclass(frozen=True)
class HardContextResult:
    name: str
    feature_group: str
    train_n: int
    test_n: int
    palatal_test_n: int
    velar_test_n: int
    auc: float | None
    balanced_accuracy: float | None
    palatal_recall: float | None
    velar_recall: float | None
    palatal_probability_median: float | None
    velar_probability_median: float | None


def load_manifest(per_class: int) -> pd.DataFrame:
    query = """
        SELECT
            id,
            audio_path,
            start_ms,
            end_ms,
            phone,
            next_phone,
            prev_phone,
            word,
            meaning,
            speaker_id,
            provider
        FROM curation_manifest
        WHERE phone IN ('kʰ', 'cʰ')
          AND included_in_training = 1
        ORDER BY id
    """
    with sqlite3.connect(str(CURATED_DB)) as conn:
        df = pd.read_sql_query(query, conn)
    df["is_hard_a_context"] = df["next_phone"].isin(BACK_A_CONTEXT)
    rng = np.random.default_rng(RANDOM_SEED)
    sampled: list[pd.DataFrame] = []
    for phone in PHONES:
        local = df[df["phone"] == phone]
        hard = local[local["is_hard_a_context"]]
        non_hard = local[~local["is_hard_a_context"]]
        take = min(per_class, len(non_hard))
        sampled.append(non_hard.sample(n=take, random_state=RANDOM_SEED))
        sampled.append(hard)
    out = pd.concat(sampled, ignore_index=True).drop_duplicates("id")
    out = out.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_audio(path: str) -> tuple[np.ndarray, int] | None:
    audio_path = Path(path)
    if not audio_path.exists():
        return None
    try:
        samples, sample_rate = sf.read(str(audio_path), dtype="float32")
    except Exception:
        return None
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    if arr.size == 0:
        return None
    return arr, int(sample_rate)


def slice_ms(
    samples: np.ndarray,
    sample_rate: int,
    absolute_start_ms: float,
    absolute_end_ms: float,
) -> np.ndarray:
    start = max(0, int(round(absolute_start_ms * sample_rate / 1000.0)))
    end = min(samples.size, int(round(absolute_end_ms * sample_rate / 1000.0)))
    if end <= start:
        return np.empty(0, dtype=np.float32)
    return np.asarray(samples[start:end], dtype=np.float32)


def spectral_vector(window: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    arr = np.asarray(window, dtype=np.float64)
    out: dict[str, float | None] = {
        "samples": float(arr.size),
        "rms": None,
        "peak": None,
        "crest": None,
        "zcr": None,
        "energy_slope": None,
        "centroid": None,
        "bandwidth": None,
        "skew": None,
        "kurtosis": None,
        "entropy": None,
        "flatness": None,
        "peak_freq": None,
        "rolloff50": None,
        "rolloff85": None,
        "rolloff95": None,
        "high_low_2k": None,
        "high_low_3k": None,
        "high_low_4k": None,
        "lpc_f1": None,
        "lpc_f2": None,
        "lpc_f3": None,
    }
    if arr.size < 24:
        return out

    arr = arr - float(np.mean(arr))
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    out["rms"] = rms
    out["peak"] = peak
    out["crest"] = None if rms <= 1e-12 else peak / rms
    out["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(arr))).astype(np.float64)))

    frame = max(8, int(round(sample_rate * 0.002)))
    if arr.size >= frame * 4:
        n = arr.size // frame
        framed = arr[: n * frame].reshape(n, frame)
        energies = np.sqrt(np.mean(framed**2, axis=1)) + 1e-12
        times = np.arange(n, dtype=np.float64)
        if n >= 2:
            slope, _intercept = np.polyfit(times, 20.0 * np.log10(energies), 1)
            out["energy_slope"] = float(slope)

    n_fft = 2048
    padded = np.zeros(n_fft, dtype=np.float64)
    usable = min(arr.size, n_fft)
    padded[:usable] = arr[:usable] * np.hanning(usable)
    mag = np.abs(np.fft.rfft(padded))
    power = mag**2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    total = float(power.sum())
    if total <= 1e-20:
        return out
    p = power / total
    mean = float(np.sum(freqs * p))
    var = float(np.sum(((freqs - mean) ** 2) * p))
    std = math.sqrt(max(var, 0.0))
    out["centroid"] = mean
    out["bandwidth"] = std
    if std > 1e-9:
        out["skew"] = float(np.sum(((freqs - mean) / std) ** 3 * p))
        out["kurtosis"] = float(np.sum(((freqs - mean) / std) ** 4 * p)) - 3.0
    out["entropy"] = float(-np.sum(p * np.log2(p + 1e-20)) / math.log2(p.size))
    out["flatness"] = float(np.exp(np.mean(np.log(power + 1e-20))) / (np.mean(power) + 1e-20))
    out["peak_freq"] = float(freqs[int(np.argmax(power))])
    cdf = np.cumsum(power)
    for q in (0.50, 0.85, 0.95):
        idx = int(np.searchsorted(cdf, q * total, side="left"))
        out[f"rolloff{int(q * 100)}"] = float(freqs[min(idx, freqs.size - 1)])

    def energy_between(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(power[mask].sum())

    low_2k = energy_between(0.0, 2000.0)
    low_3k = energy_between(0.0, 3000.0)
    low_4k = energy_between(0.0, 4000.0)
    out["high_low_2k"] = energy_between(2000.0, sample_rate / 2) / (low_2k + 1e-20)
    out["high_low_3k"] = energy_between(3000.0, sample_rate / 2) / (low_3k + 1e-20)
    out["high_low_4k"] = energy_between(4000.0, sample_rate / 2) / (low_4k + 1e-20)

    band_edges = np.array(
        [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6500, sample_rate / 2],
        dtype=np.float64,
    )
    band_energies = []
    for i in range(len(band_edges) - 1):
        energy = energy_between(float(band_edges[i]), float(band_edges[i + 1]))
        frac = energy / (total + 1e-20)
        out[f"band_{int(band_edges[i])}_{int(band_edges[i + 1])}"] = float(np.log(frac + 1e-12))
        band_energies.append(frac)
    cep = dct(np.log(np.asarray(band_energies) + 1e-12), type=2, norm="ortho")
    for idx, value in enumerate(cep[:10]):
        out[f"band_dct_{idx}"] = float(value)

    if arr.size >= int(sample_rate * 0.025):
        formants = lpc_formants(arr, sample_rate)
        for idx, value in enumerate(formants[:3], start=1):
            out[f"lpc_f{idx}"] = value
    return out


def lpc_formants(samples: np.ndarray, sample_rate: int) -> list[float | None]:
    try:
        import librosa
    except Exception:
        return [None, None, None]
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size < 32:
        return [None, None, None]
    arr = np.append(arr[0], arr[1:] - 0.97 * arr[:-1])
    order = 10 if sample_rate <= 16000 else 12
    try:
        coeffs = librosa.lpc(arr, order=order)
    except Exception:
        return [None, None, None]
    roots = np.roots(coeffs)
    roots = [root for root in roots if np.imag(root) >= 0.01]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = sorted(float(angle * sample_rate / (2.0 * np.pi)) for angle in angles)
    formants = [freq for freq in freqs if 90.0 <= freq <= 5500.0]
    return [*formants[:3], None, None, None][:3]


def extract_row(row: pd.Series) -> dict[str, Any] | None:
    loaded = load_audio(str(row["audio_path"]))
    if loaded is None:
        return None
    samples, sample_rate = loaded
    item: dict[str, Any] = {
        "id": int(row["id"]),
        "phone": str(row["phone"]),
        "y": 1 if row["phone"] == "cʰ" else 0,
        "speaker_id": str(row["speaker_id"]),
        "provider": str(row["provider"]),
        "next_phone": str(row["next_phone"]),
        "word": str(row["word"]),
        "meaning": str(row["meaning"]),
        "is_hard_a_context": bool(row["is_hard_a_context"]),
    }
    start_ms = float(row["start_ms"])
    end_ms = float(row["end_ms"])
    for name, rel_start, rel_end in WINDOWS:
        # The curation stop spans are usually 20 ms; phoneend_0_40 is expressed
        # relative to start here because the span end is start + 20 ms.
        if name == "phoneend_0_40":
            abs_start = end_ms
            abs_end = end_ms + 40.0
        else:
            abs_start = start_ms + rel_start
            abs_end = start_ms + rel_end
        window = slice_ms(samples, sample_rate, abs_start, abs_end)
        for key, value in spectral_vector(window, sample_rate).items():
            item[f"{name}__{key}"] = value
    return item


def extract_features(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in manifest.iterrows():
        if index and index % 500 == 0:
            print(f"extracted {index}/{len(manifest)}", file=sys.stderr, flush=True)
        item = extract_row(row)
        if item is not None:
            rows.append(item)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    wanted = tuple(f"{prefix}__" for prefix in prefixes)
    cols = [
        col
        for col in df.columns
        if col.startswith(wanted)
        and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 20
    ]
    return cols


def _cv_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if unique_groups.size >= 5:
        splitter = GroupKFold(n_splits=5)
        return list(splitter.split(x, y, groups))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return list(splitter.split(x, y))


def make_model(name: str) -> Any:
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1200,
                random_state=RANDOM_SEED,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )
    raise ValueError(name)


def evaluate_cv(df: pd.DataFrame, model_name: str, group_name: str, cols: list[str]) -> ModelResult:
    local = df[df["phone"].isin(PHONES)].reset_index(drop=True)
    if len(cols) == 0:
        return ModelResult(model_name, group_name, len(local), None, None, None, None, 0, "no_features")
    y = local["y"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return ModelResult(model_name, group_name, len(local), None, None, None, None, 0, "one_class")
    x = local[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    groups = local["speaker_id"].astype(str).to_numpy()
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
        return ModelResult(model_name, group_name, len(local), None, None, None, None, 0, "invalid_splits")
    return ModelResult(
        name=model_name,
        feature_group=group_name,
        n=len(local),
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        balanced_accuracy_mean=float(np.mean(bals)),
        balanced_accuracy_std=float(np.std(bals)),
        folds=len(aucs),
    )


def evaluate_hard_context(
    df: pd.DataFrame,
    model_name: str,
    group_name: str,
    cols: list[str],
) -> HardContextResult:
    train = df[~df["is_hard_a_context"]].reset_index(drop=True)
    test = df[df["is_hard_a_context"]].reset_index(drop=True)
    if len(cols) == 0 or test.empty or len(np.unique(test["y"])) < 2:
        return HardContextResult(model_name, group_name, len(train), len(test), int(test["y"].sum()), int(len(test) - test["y"].sum()), None, None, None, None, None, None)
    x_train = train[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=int)
    x_test = test[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=int)
    model = make_model(model_name)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    pal_mask = y_test == 1
    vel_mask = y_test == 0
    return HardContextResult(
        name=model_name,
        feature_group=group_name,
        train_n=len(train),
        test_n=len(test),
        palatal_test_n=int(pal_mask.sum()),
        velar_test_n=int(vel_mask.sum()),
        auc=float(roc_auc_score(y_test, prob)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        palatal_recall=float(np.mean(pred[pal_mask] == 1)) if pal_mask.any() else None,
        velar_recall=float(np.mean(pred[vel_mask] == 0)) if vel_mask.any() else None,
        palatal_probability_median=float(np.median(prob[pal_mask])) if pal_mask.any() else None,
        velar_probability_median=float(np.median(prob[vel_mask])) if vel_mask.any() else None,
    )


def univariate_auc(df: pd.DataFrame, cols: list[str], limit: int = 30) -> list[dict[str, Any]]:
    y = df["y"].to_numpy(dtype=int)
    out: list[dict[str, Any]] = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        raw_auc = float(roc_auc_score(y[mask], values[mask]))
        auc = raw_auc if raw_auc >= 0.5 else 1.0 - raw_auc
        direction = "palatal_higher" if raw_auc >= 0.5 else "palatal_lower"
        pal = values[mask & (y == 1)]
        vel = values[mask & (y == 0)]
        out.append(
            {
                "feature": col,
                "auc": auc,
                "direction": direction,
                "palatal_median": _safe_float(np.median(pal)) if pal.size else None,
                "velar_median": _safe_float(np.median(vel)) if vel.size else None,
                "coverage": float(mask.mean()),
            }
        )
    return sorted(out, key=lambda item: item["auc"], reverse=True)[:limit]


def feature_importance(df: pd.DataFrame, cols: list[str], limit: int = 30) -> list[dict[str, Any]]:
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    pipeline = make_model("extra_trees")
    pipeline.fit(x, y)
    model = pipeline.named_steps["extratreesclassifier"]
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": cols[int(idx)], "importance": float(importances[int(idx)])}
        for idx in order
    ]


def write_report(payload: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(value: float | None, digits: int = 3) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    lines = [
        "# kʰ vs cʰ Rich Waveform Feature Analysis",
        "",
        "Features are computed directly from WAV windows around the curated stop span.",
        "No existing coach feature columns are used.",
        "",
        f"Rows extracted: {payload['counts']['rows']}  cʰ: {payload['counts']['cʰ']}  kʰ: {payload['counts']['kʰ']}",
        f"Hard a/aː rows in this extracted set: {payload['counts']['hard_a_rows']}  cʰ: {payload['counts']['hard_a_cʰ']}  kʰ: {payload['counts']['hard_a_kʰ']}",
        "",
        "## Speaker-Grouped Global CV",
        "",
        "| model | feature group | AUC | bal. acc. | folds |",
        "|---|---|---:|---:|---:|",
    ]
    for item in payload["global_cv"]:
        lines.append(
            f"| {item['name']} | {item['feature_group']} | {fmt(item['auc_mean'])} | {fmt(item['balanced_accuracy_mean'])} | {item['folds']} |"
        )
    lines.extend(
        [
            "",
            "## Train Non-a/aː, Test a/aː",
            "",
            "| model | feature group | AUC | bal. acc. | cʰ recall | kʰ recall | cʰ median p | kʰ median p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["hard_context"]:
        lines.append(
            "| {name} | {group} | {auc} | {bal} | {prec} | {vrec} | {pp} | {vp} |".format(
                name=item["name"],
                group=item["feature_group"],
                auc=fmt(item["auc"]),
                bal=fmt(item["balanced_accuracy"]),
                prec=fmt(item["palatal_recall"]),
                vrec=fmt(item["velar_recall"]),
                pp=fmt(item["palatal_probability_median"]),
                vp=fmt(item["velar_probability_median"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Univariate Raw Features",
            "",
            "| feature | AUC | direction | cʰ median | kʰ median | coverage |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for item in payload["top_univariate"]:
        lines.append(
            "| {feature} | {auc} | {direction} | {pal} | {vel} | {coverage} |".format(
                feature=item["feature"],
                auc=fmt(item["auc"]),
                direction=item["direction"],
                pal=fmt(item["palatal_median"], 2),
                vel=fmt(item["velar_median"], 2),
                coverage=fmt(item["coverage"]),
            )
        )
    lines.extend(
        [
            "",
            "## Extra Trees Feature Importance",
            "",
            "| feature | importance |",
            "|---|---:|",
        ]
    )
    for item in payload["feature_importance"]:
        lines.append(f"| {item['feature']} | {fmt(item['importance'], 4)} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=1800)
    args = parser.parse_args()

    manifest = load_manifest(args.per_class)
    features = extract_features(manifest)
    if features.empty:
        raise SystemExit("no features extracted")
    feature_groups = {
        "strict_release_20ms": feature_columns(features, STRICT_RELEASE_PREFIXES),
        "early_release_40ms": feature_columns(features, EARLY_RELEASE_PREFIXES),
        "all_windows": feature_columns(features, ALL_PREFIXES),
    }
    payload: dict[str, Any] = {
        "source_db": str(CURATED_DB),
        "per_class_non_hard_sample": args.per_class,
        "window_specs": WINDOWS,
        "counts": {
            "rows": int(len(features)),
            "cʰ": int((features["phone"] == "cʰ").sum()),
            "kʰ": int((features["phone"] == "kʰ").sum()),
            "hard_a_rows": int(features["is_hard_a_context"].sum()),
            "hard_a_cʰ": int(((features["phone"] == "cʰ") & features["is_hard_a_context"]).sum()),
            "hard_a_kʰ": int(((features["phone"] == "kʰ") & features["is_hard_a_context"]).sum()),
            "speakers": int(features["speaker_id"].nunique()),
        },
        "feature_group_sizes": {name: len(cols) for name, cols in feature_groups.items()},
        "global_cv": [],
        "hard_context": [],
        "top_univariate": [],
        "feature_importance": [],
    }

    for group_name, cols in feature_groups.items():
        for model_name in ("logistic", "extra_trees"):
            payload["global_cv"].append(
                asdict(evaluate_cv(features, model_name, group_name, cols))
            )
            payload["hard_context"].append(
                asdict(evaluate_hard_context(features, model_name, group_name, cols))
            )

    all_cols = feature_groups["all_windows"]
    payload["top_univariate"] = univariate_auc(features, all_cols)
    payload["feature_importance"] = feature_importance(features, all_cols)
    write_report(payload)
    print(json.dumps({"report": str(REPORT_MD), "json": str(REPORT_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
