#!/usr/bin/env python3
"""Diagnose manual profit kâr stop-place evidence under curated overlay.

The manual files in data/manual/kar are intentionally evaluated against
profit kâr (/cʰ aː ɾ̞̊/). The diagnostic remains target-independent for the
stop-place detector: it scores the first decoded stop as produced by wav2vec,
records the active pairwise detector evidence, and compares alternate
stop-release F2 windows against curated kʰ/cʰ reference distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assess.contracts import CoachRequest, PhoneInterval  # noqa: E402
from assess.authority import AuthorityTable, _evaluate_detector  # noqa: E402
from assess.features import FeatureExtractor  # noqa: E402
from assess.runtime import build_default_runtime_services, run_coach  # noqa: E402
from audio.features import (  # noqa: E402
    closure_burst_landmarks,
    formant_track,
    linear_slope,
)


TARGET_PHONES: tuple[str, ...] = ("cʰ", "aː", "ɾ̞̊")
STOP_PLACE_PHONES: frozenset[str] = frozenset({"kʰ", "cʰ", "k", "c"})
REFERENCE_PHONES: tuple[str, str] = ("kʰ", "cʰ")
DIAGNOSTIC_FEATURES: tuple[str, ...] = (
    "f2_locus_hz",
    "f2_transition_slope_hz_per_ms",
    "spectral_centroid_hz",
)


@dataclass(frozen=True)
class WindowPolicy:
    name: str
    back_ms: float
    analysis_ms: float
    transition_start_ms: float
    transition_end_ms: float
    locus: str
    description: str


WINDOW_POLICIES: tuple[WindowPolicy, ...] = (
    WindowPolicy(
        "current",
        30.0,
        150.0,
        0.0,
        60.0,
        "first",
        "Runtime stop window: 30 ms back, 150 ms forward, first finite F2.",
    ),
    WindowPolicy(
        "no_back",
        0.0,
        150.0,
        0.0,
        60.0,
        "first",
        "No pre-release margin; otherwise current F2 policy.",
    ),
    WindowPolicy(
        "short_post",
        30.0,
        90.0,
        0.0,
        40.0,
        "first",
        "Current back margin with a shorter post-release transition.",
    ),
    WindowPolicy(
        "delayed_15_75",
        30.0,
        150.0,
        15.0,
        75.0,
        "median",
        "Delayed post-burst window with robust median F2 locus.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=REPO_ROOT / "data" / "manual" / "kar",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=REPO_ROOT / "artifacts" / "curated_stop_place" / "coach_gmm_overlay.json",
    )
    parser.add_argument(
        "--authority",
        type=Path,
        default=REPO_ROOT / "artifacts" / "curated_stop_place" / "decision_authority_overlay.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "curated_stop_place"
        / "coach_gmm_calibration_overlay.json",
    )
    parser.add_argument(
        "--reference-db",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "curated_stop_place"
        / "curated_stop_place_reference.sqlite",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "assessment"
        / "manual-kar-stop-place-diagnostic.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "assessment"
        / "manual-kar-stop-place-diagnostic.md",
    )
    parser.add_argument("--model", default="mms-1b", choices=["mms-1b", "xls-r-300m"])
    return parser


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    samples, sample_rate = sf.read(str(path), dtype="float32")
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, int(sample_rate)


def _finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _round(value: object, places: int = 6) -> float | None:
    parsed = _finite_float(value)
    return None if parsed is None else round(parsed, places)


def _percentile_rank(values: np.ndarray, value: float | None) -> float | None:
    if value is None or values.size == 0:
        return None
    return float(np.searchsorted(np.sort(values), value, side="right") / values.size)


def _robust_scale(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    scale = float((q75 - q25) / 1.349)
    if math.isfinite(scale) and scale > 1e-9:
        return scale
    std = float(np.std(values))
    return std if math.isfinite(std) and std > 1e-9 else 1.0


def _load_reference_distributions(db_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for phone in REFERENCE_PHONES:
            out[phone] = {}
            for feature in DIAGNOSTIC_FEATURES:
                rows = conn.execute(
                    f"""
                    SELECT {feature}
                    FROM phone_features
                    WHERE expected_phone = ?
                      AND {feature} IS NOT NULL
                    """,
                    (phone,),
                ).fetchall()
                values = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    out[phone][feature] = {"n": 0, "values": values}
                    continue
                out[phone][feature] = {
                    "n": int(values.size),
                    "values": values,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                    "scale": _robust_scale(values),
                }
        return out
    finally:
        conn.close()


def _reference_summary(
    distributions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for phone, features in distributions.items():
        summary[phone] = {}
        for feature, stats in features.items():
            summary[phone][feature] = {
                key: _round(stats.get(key), 3) if key != "n" else stats.get(key)
                for key in ("n", "mean", "median", "p10", "p25", "p75", "p90", "scale")
                if key in stats
            }
    return summary


def _distribution_position(
    distributions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    values: Mapping[str, float | None],
) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for phone, features in distributions.items():
        out[phone] = {}
        for feature in DIAGNOSTIC_FEATURES:
            stats = features[feature]
            reference_values = stats.get("values")
            if not isinstance(reference_values, np.ndarray):
                continue
            value = values.get(feature)
            out[phone][feature] = {
                "value": _round(value, 3),
                "percentile": _round(_percentile_rank(reference_values, value), 4),
                "median": _round(stats.get("median"), 3),
                "mean": _round(stats.get("mean"), 3),
            }
    return out


def _reference_distance(
    distributions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    values: Mapping[str, float | None],
) -> dict[str, object]:
    distances: dict[str, float | None] = {}
    feature_count: dict[str, int] = {}
    for phone, features in distributions.items():
        parts: list[float] = []
        for feature in DIAGNOSTIC_FEATURES:
            value = values.get(feature)
            if value is None:
                continue
            stats = features[feature]
            median = _finite_float(stats.get("median"))
            scale = _finite_float(stats.get("scale"))
            if median is None or scale is None:
                continue
            parts.append(((value - median) / scale) ** 2)
        feature_count[phone] = len(parts)
        distances[phone] = math.sqrt(sum(parts)) if parts else None
    finite_distances = {
        phone: value for phone, value in distances.items() if value is not None
    }
    closer_phone = (
        min(finite_distances, key=finite_distances.__getitem__)
        if finite_distances
        else None
    )
    return {
        "distances": {phone: _round(value, 4) for phone, value in distances.items()},
        "finite_feature_count": feature_count,
        "closer_phone": closer_phone,
    }


def _extract_window_values(
    audio: np.ndarray,
    sample_rate: int,
    interval: PhoneInterval,
    policy: WindowPolicy,
    *,
    spectral_centroid_hz: float | None,
) -> dict[str, object]:
    raw_start = max(0, int(round(interval.start_ms / 1000.0 * sample_rate)))
    raw_end = min(len(audio), int(round(interval.end_ms / 1000.0 * sample_rate)))
    back = int(round(policy.back_ms / 1000.0 * sample_rate))
    width = int(round(policy.analysis_ms / 1000.0 * sample_rate))
    start = max(0, raw_start - back)
    end = min(len(audio), max(raw_end, raw_start + width))
    span = audio[start:end].astype(np.float64)
    landmarks = closure_burst_landmarks(span, sample_rate)

    f2_locus: float | None = None
    f2_slope: float | None = None
    track_points = 0
    finite_points = 0
    if landmarks.burst_ms is not None:
        track = formant_track(span, sample_rate)
        if track is not None:
            lower = landmarks.burst_ms + policy.transition_start_ms
            upper = landmarks.burst_ms + policy.transition_end_ms
            selected = (track.times_ms >= lower) & (track.times_ms <= upper)
            times = track.times_ms[selected]
            f2 = track.f2[selected]
            finite = np.isfinite(f2)
            track_points = int(selected.sum())
            finite_points = int(finite.sum())
            f2_slope = linear_slope(times, f2)
            if finite_points:
                values = f2[finite]
                f2_locus = (
                    float(np.median(values))
                    if policy.locus == "median"
                    else float(values[0])
                )

    return {
        "policy": policy.name,
        "description": policy.description,
        "analysis_start_ms": _round(start * 1000.0 / sample_rate, 3),
        "analysis_end_ms": _round(end * 1000.0 / sample_rate, 3),
        "burst_ms_in_window": _round(landmarks.burst_ms, 3),
        "burst_ms_absolute": _round(
            None
            if landmarks.burst_ms is None
            else start * 1000.0 / sample_rate + landmarks.burst_ms,
            3,
        ),
        "track_points": track_points,
        "finite_f2_points": finite_points,
        "feature_values": {
            "f2_locus_hz": _round(f2_locus, 6),
            "f2_transition_slope_hz_per_ms": _round(f2_slope, 6),
            "spectral_centroid_hz": _round(spectral_centroid_hz, 6),
        },
    }


def _feature_values_for_detector(feature: object) -> dict[str, float | None]:
    return {
        name: _finite_float(getattr(feature, name, None))
        for name in DIAGNOSTIC_FEATURES
    }


def _detector_for(
    table: AuthorityTable,
    feature: object,
    raw_phone: str,
) -> dict[str, object] | None:
    alternative = {"kʰ": "cʰ", "k": "c"}.get(raw_phone)
    if alternative is None:
        return None
    rec = table.get(raw_phone, alternative)
    if not rec.detector:
        return None
    return _evaluate_detector(
        feature_set=feature,
        atlas=None,
        rec=rec,
        detector=rec.detector,
    )


def _first_stop_phone(result: Mapping[str, Any]) -> tuple[int, dict[str, Any]] | None:
    wav2vec = result.get("wav2vec")
    if not isinstance(wav2vec, Mapping):
        return None
    timed = wav2vec.get("timed_phones")
    if not isinstance(timed, list):
        return None
    for index, phone in enumerate(timed):
        if not isinstance(phone, dict):
            continue
        ipa = str(phone.get("ipa") or "")
        if ipa in STOP_PLACE_PHONES:
            return index, phone
    return None


def diagnose_file(
    path: Path,
    *,
    services: object,
    args: argparse.Namespace,
    table: AuthorityTable,
    distributions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, object]:
    request = CoachRequest(
        audio_path=str(path),
        model_alias=args.model,
        target_phones=list(TARGET_PHONES),
        stats_path=str(args.stats),
        authority_path=str(args.authority),
        calibration_path=str(args.calibration),
    )
    result = run_coach(request, services=services).as_dict()
    first = _first_stop_phone(result)
    if first is None:
        return {
            "file": str(path),
            "error": "no_decoded_stop_phone",
            "coach_result": result,
        }
    stop_index, stop_phone = first
    wav2vec_timed = result["wav2vec"]["timed_phones"]  # type: ignore[index]
    prev_phone = (
        str(wav2vec_timed[stop_index - 1]["ipa"]) if stop_index > 0 else None
    )
    next_phone = (
        str(wav2vec_timed[stop_index + 1]["ipa"])
        if stop_index + 1 < len(wav2vec_timed)
        else None
    )
    audio, sample_rate = _load_audio(path)
    interval = PhoneInterval(
        target_phone=str(stop_phone["ipa"]),
        start_ms=int(stop_phone["start_ms"]),
        end_ms=int(stop_phone["end_ms"]),
        speech_start_ms_original=int(result["speech_start_ms_original"]),
        speech_end_ms_original=int(result["speech_end_ms_original"]),
    )
    extractor = FeatureExtractor()
    feature = extractor.extract(
        audio,
        sample_rate,
        interval,
        prev_phone=prev_phone,
        next_phone=next_phone,
    )
    runtime_values = _feature_values_for_detector(feature)
    detector = _detector_for(table, feature, str(stop_phone["ipa"]))

    windows: dict[str, object] = {}
    for policy in WINDOW_POLICIES:
        window = _extract_window_values(
            audio,
            sample_rate,
            interval,
            policy,
            spectral_centroid_hz=runtime_values["spectral_centroid_hz"],
        )
        values = {
            name: _finite_float(window["feature_values"][name])  # type: ignore[index]
            for name in DIAGNOSTIC_FEATURES
        }
        windows[policy.name] = {
            **window,
            "reference_position": _distribution_position(distributions, values),
            "reference_distance": _reference_distance(distributions, values),
        }

    return {
        "file": str(path),
        "target_phones": list(TARGET_PHONES),
        "coach_analysis_phones": result.get("analysis", {}).get("phonemes"),
        "coach_result_phones": result.get("result", {}).get("phones"),
        "wav2vec_first_stop": {
            "index": stop_index,
            "ipa": stop_phone.get("ipa"),
            "start_ms": stop_phone.get("start_ms"),
            "end_ms": stop_phone.get("end_ms"),
            "confidence": _round(stop_phone.get("confidence"), 6),
        },
        "runtime_feature_set": {
            "feature_confidence": _round(feature.feature_confidence, 6),
            "burst_confidence": _round(feature.burst_confidence, 6),
            "vot_ms": _round(feature.vot_ms, 6),
            "closure_duration_ms": _round(feature.closure_duration_ms, 6),
            "feature_values": {
                name: _round(value, 6)
                for name, value in runtime_values.items()
            },
            "reference_position": _distribution_position(
                distributions,
                runtime_values,
            ),
            "reference_distance": _reference_distance(distributions, runtime_values),
        },
        "detector": detector,
        "window_policies": windows,
    }


def _table_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _policy_closer(row: Mapping[str, Any], policy: str) -> str:
    windows = row.get("window_policies")
    if not isinstance(windows, Mapping):
        return ""
    item = windows.get(policy)
    if not isinstance(item, Mapping):
        return ""
    distance = item.get("reference_distance")
    if not isinstance(distance, Mapping):
        return ""
    return str(distance.get("closer_phone") or "")


def _policy_locus(row: Mapping[str, Any], policy: str) -> float | None:
    windows = row.get("window_policies")
    if not isinstance(windows, Mapping):
        return None
    item = windows.get(policy)
    if not isinstance(item, Mapping):
        return None
    values = item.get("feature_values")
    if not isinstance(values, Mapping):
        return None
    return _finite_float(values.get("f2_locus_hz"))


def _detector_value(row: Mapping[str, Any], key: str) -> object:
    detector = row.get("detector")
    if not isinstance(detector, Mapping):
        return None
    return detector.get(key)


def _detector_feature(row: Mapping[str, Any], key: str) -> float | None:
    detector = row.get("detector")
    if not isinstance(detector, Mapping):
        return None
    values = detector.get("feature_values")
    if not isinstance(values, Mapping):
        return None
    return _finite_float(values.get(key))


def _render_markdown(payload: Mapping[str, Any]) -> str:
    rows = [row for row in payload["files"] if isinstance(row, dict)]  # type: ignore[index]
    policies = [policy.name for policy in WINDOW_POLICIES]
    all_delayed_c = all(_policy_closer(row, "delayed_15_75") == "cʰ" for row in rows)
    any_policy_all_c = any(
        all(_policy_closer(row, policy) == "cʰ" for row in rows)
        for policy in policies
    )
    detector_rows = [
        row
        for row in rows
        if isinstance(row.get("detector"), Mapping)
        and _detector_value(row, "detector_kind") == "raw_release_spectrum_logistic"
    ]
    all_detectors_reject = bool(detector_rows) and all(
        _detector_value(row, "passed") is False for row in detector_rows
    )
    f2_conclusion = (
        "At least one F2-window policy moves all three files closer to curated `cʰ`."
        if any_policy_all_c
        else "No tested F2-window policy moves all three files closer to curated `cʰ`; "
        "the manual files remain acoustically closer to curated `kʰ` under these F2 cues."
    )
    detector_conclusion = (
        " The active raw-release spectrum detector also rejects `kʰ→cʰ` for all three files."
        if all_detectors_reject
        else ""
    )
    conclusion = f2_conclusion + detector_conclusion
    if all_delayed_c:
        conclusion += " The delayed 15-75 ms policy is the supported candidate fix."
    elif not any_policy_all_c:
        conclusion += " A delayed-F2 extractor change is not supported by this diagnostic."

    lines = [
        "# Manual kâr Stop-Place Diagnostic",
        "",
        "All files in `data/manual/kar/` are evaluated as profit `kâr` with target `/cʰ aː ɾ̞̊/`. "
        "The stop-place detector is still scored from the first decoded stop, not from the target text.",
        "",
        f"Conclusion: {conclusion}",
        "",
        "## Reference Summary",
        "",
        "| phone | feature | n | mean | median | p10 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ref = payload["reference_summary"]  # type: ignore[index]
    for phone in REFERENCE_PHONES:
        for feature in DIAGNOSTIC_FEATURES:
            stats = ref[phone][feature]
            lines.append(
                "| "
                + " | ".join(
                    _table_cell(value)
                    for value in (
                        f"`{phone}`",
                        f"`{feature}`",
                        stats.get("n"),
                        stats.get("mean"),
                        stats.get("median"),
                        stats.get("p10"),
                        stats.get("p90"),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## File Summary",
            "",
            "| file | decoded stop | conf | analysis phones | runtime F2 locus | runtime slope | centroid | detector | score | threshold | margin | passed | current | no-back | short-post | delayed 15-75 |",
            "|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        stop = row.get("wav2vec_first_stop") or {}
        runtime = row.get("runtime_feature_set") or {}
        values = runtime.get("feature_values") or {}
        lines.append(
            "| "
            + " | ".join(
                _table_cell(value)
                for value in (
                    f"`{Path(str(row.get('file'))).name}`",
                    f"`{stop.get('ipa')}`",
                    stop.get("confidence"),
                    "`" + " ".join(row.get("coach_analysis_phones") or []) + "`",
                    values.get("f2_locus_hz"),
                    values.get("f2_transition_slope_hz_per_ms"),
                    values.get("spectral_centroid_hz"),
                    _detector_value(row, "detector_kind"),
                    _detector_value(row, "score"),
                    _detector_value(row, "threshold"),
                    _detector_value(row, "margin"),
                    _detector_value(row, "passed"),
                    _policy_closer(row, "current"),
                    _policy_closer(row, "no_back"),
                    _policy_closer(row, "short_post"),
                    _policy_closer(row, "delayed_15_75"),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Raw Release Evidence",
            "",
            "| file | low 500-1000 | low 1000-1500 | high/low >2k | high/low >3k | centroid | score | threshold | margin |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _table_cell(value)
                for value in (
                    f"`{Path(str(row.get('file'))).name}`",
                    _detector_feature(row, "release20_band_500_1000_log"),
                    _detector_feature(row, "release20_band_1000_1500_log"),
                    _detector_feature(row, "release20_high_low_2k"),
                    _detector_feature(row, "release20_high_low_3k"),
                    _detector_feature(row, "release20_centroid_hz"),
                    _detector_value(row, "score"),
                    _detector_value(row, "threshold"),
                    _detector_value(row, "margin"),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Window F2 Loci",
            "",
            "| file | current | no-back | short-post | delayed 15-75 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _table_cell(value)
                for value in (
                    f"`{Path(str(row.get('file'))).name}`",
                    _policy_locus(row, "current"),
                    _policy_locus(row, "no_back"),
                    _policy_locus(row, "short_post"),
                    _policy_locus(row, "delayed_15_75"),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Policy Definitions",
            "",
        ]
    )
    for policy in WINDOW_POLICIES:
        lines.append(f"- `{policy.name}`: {policy.description}")
    lines.append("")
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = sorted(args.audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"no wav files found under {args.audio_dir}")

    distributions = _load_reference_distributions(args.reference_db)
    table = AuthorityTable.load(args.authority)
    seed_request = CoachRequest(
        audio_path=str(files[0]),
        model_alias=args.model,
        target_phones=list(TARGET_PHONES),
        stats_path=str(args.stats),
        authority_path=str(args.authority),
        calibration_path=str(args.calibration),
    )
    services = build_default_runtime_services(seed_request)
    file_rows = [
        diagnose_file(
            path,
            services=services,
            args=args,
            table=table,
            distributions=distributions,
        )
        for path in files
    ]
    payload = {
        "target_phones": list(TARGET_PHONES),
        "window_policies": [asdict(policy) for policy in WINDOW_POLICIES],
        "reference_summary": _reference_summary(distributions),
        "files": file_rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.json),
                "markdown": str(args.markdown),
                "files": len(file_rows),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
