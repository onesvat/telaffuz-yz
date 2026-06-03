#!/usr/bin/env python3
"""Build the experimental curated kʰ/cʰ/k/c stop-place overlay.

The workflow is intentionally overlay-only. It writes artifacts under
``artifacts/curated_stop_place/`` and never mutates production ``configs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assess.curated_stop_place import (  # noqa: E402
    CURATED_REFERENCE_FEATURE_VERSION,
    DEFAULT_REFERENCE_FEATURE_VERSION,
    STOP_PLACE_PHONES,
    build_curated_reference_db,
    curation_summary,
    load_json,
    merge_authority_overlay,
    merge_calibration_overlay,
    merge_gmm_overlay,
    mine_stop_place_candidates,
    write_curation_csv,
    write_json,
)

DEFAULT_AUDIO_DB = Path("/home/onur/Code/telaffuz-yz-thesis-db/audio.sqlite")
DEFAULT_REFERENCE_DB = Path(
    "/home/onur/Code/telaffuz-yz-thesis-db/coach_reference_features.sqlite"
)
DEFAULT_AUDIO_ROOT = Path("/home/onur/Code/telaffuz-yz-audio")
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "curated_stop_place"
DEFAULT_CURATION_CSV = DEFAULT_OUT_DIR / "curation.csv"
DEFAULT_CURATED_DB = DEFAULT_OUT_DIR / "curated_stop_place_reference.sqlite"
DEFAULT_GMM_OVERLAY = DEFAULT_OUT_DIR / "coach_gmm_overlay.json"
DEFAULT_CAL_OVERLAY = DEFAULT_OUT_DIR / "coach_gmm_calibration_overlay.json"
DEFAULT_AUTH_OVERLAY = DEFAULT_OUT_DIR / "decision_authority_overlay.json"
DEFAULT_REPORT = DEFAULT_OUT_DIR / "curation_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine", help="Mine candidate curation CSV rows.")
    _add_source_args(mine)
    mine.add_argument("--out", type=Path, default=DEFAULT_CURATION_CSV)
    mine.add_argument("--limit", type=int)
    mine.add_argument(
        "--providers", nargs="+", default=["common_voice"], help="Reference providers."
    )
    mine.add_argument("--confidence-floor", type=float, default=0.5)

    build_db = sub.add_parser("build-sqlite", help="Build curated reference SQLite.")
    _add_source_args(build_db)
    build_db.add_argument("--curation", type=Path, default=DEFAULT_CURATION_CSV)
    build_db.add_argument("--out", type=Path, default=DEFAULT_CURATED_DB)
    build_db.add_argument("--replace", action="store_true")

    overlays = sub.add_parser("build-overlays", help="Build GMM/calibration/authority overlays.")
    _add_source_args(overlays)
    overlays.add_argument("--curation", type=Path, default=DEFAULT_CURATION_CSV)
    overlays.add_argument("--curated-db", type=Path, default=DEFAULT_CURATED_DB)
    overlays.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    overlays.add_argument("--replace-db", action="store_true")
    overlays.add_argument("--base-gmm", type=Path, default=REPO_ROOT / "configs" / "coach_gmm_v1.json")
    overlays.add_argument(
        "--base-calibration",
        type=Path,
        default=REPO_ROOT / "configs" / "coach_gmm_calibration_v1.json",
    )
    overlays.add_argument(
        "--base-authority",
        type=Path,
        default=REPO_ROOT / "configs" / "decision_authority.json",
    )
    overlays.add_argument("--providers", nargs="+", default=["common_voice"])
    overlays.add_argument("--min-samples", type=int, default=100)
    overlays.add_argument("--confidence-floor", type=float, default=0.5)
    overlays.add_argument("--per-phone-cap", type=int, default=20000)
    overlays.add_argument("--train-buckets", default="0..79")
    overlays.add_argument("--test-buckets", default="80..99")
    overlays.add_argument("--d-quantile", type=float, default=0.95)
    overlays.add_argument("--seed", type=int, default=0)
    overlays.add_argument("--min-feature-coverage", type=float, default=0.6)
    overlays.add_argument("--max-k", type=int, default=3)
    overlays.add_argument("--min-samples-per-component", type=int, default=30)
    overlays.add_argument("--quantile-count", type=int, default=1024)
    overlays.add_argument("--authority-sample-limit", type=int, default=5000)
    overlays.add_argument("--authority-min-samples", type=int, default=25)
    overlays.add_argument("--authority-min-eval-samples", type=int, default=50)
    overlays.add_argument("--authority-fp-ceiling", type=float, default=0.02)
    overlays.add_argument("--authority-min-recall", type=float, default=0.05)
    overlays.add_argument("--authority-max-features", type=int, default=3)
    overlays.add_argument("--authority-regularization", type=float, default=1e-3)
    overlays.add_argument("--authority-target-quality-ceiling", type=float, default=0.20)
    overlays.add_argument(
        "--authority-merge-scope",
        choices=["involving-target", "target-target"],
        default="involving-target",
    )
    overlays.add_argument("--allow-missing-phones", action="store_true")

    all_cmd = sub.add_parser("all", help="Mine candidates, build SQLite, then overlays.")
    _add_source_args(all_cmd)
    all_cmd.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    all_cmd.add_argument("--limit", type=int)
    all_cmd.add_argument("--replace-db", action="store_true")
    all_cmd.add_argument("--allow-missing-phones", action="store_true")

    return parser


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio-db", type=Path, default=DEFAULT_AUDIO_DB)
    parser.add_argument("--reference-db", type=Path, default=DEFAULT_REFERENCE_DB)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--reference-feature-version", default=DEFAULT_REFERENCE_FEATURE_VERSION)


def cmd_mine(args: argparse.Namespace) -> int:
    rows = mine_stop_place_candidates(
        audio_db=args.audio_db,
        reference_db=args.reference_db,
        audio_root=args.audio_root,
        feature_version=args.reference_feature_version,
        providers=tuple(args.providers),
        confidence_floor=args.confidence_floor,
        limit=args.limit,
    )
    write_curation_csv(rows, args.out)
    summary = curation_summary(rows)
    print(json.dumps({"out": str(args.out), **summary}, ensure_ascii=False, indent=2))
    return 0


def cmd_build_sqlite(args: argparse.Namespace) -> int:
    report = build_curated_reference_db(
        curation_csv=args.curation,
        reference_db=args.reference_db,
        audio_db=args.audio_db,
        output_db=args.out,
        feature_version=CURATED_REFERENCE_FEATURE_VERSION,
        replace=args.replace,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_build_overlays(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_curated_reference_db(
        curation_csv=args.curation,
        reference_db=args.reference_db,
        audio_db=args.audio_db,
        output_db=args.curated_db,
        feature_version=CURATED_REFERENCE_FEATURE_VERSION,
        replace=args.replace_db,
    )
    report_path = args.out_dir / "curation_report.json"
    write_json(report_path, report.as_dict())

    from assess.stats import (  # noqa: PLC0415
        BuildConfig,
        CalibrationBuildConfig,
        build_atlas_gmm,
        build_gmm_calibration,
    )

    config = BuildConfig(
        providers=tuple(args.providers),
        min_samples=args.min_samples,
        confidence_floor=args.confidence_floor,
        per_phone_cap=args.per_phone_cap,
        train_buckets=args.train_buckets,
        test_buckets=args.test_buckets,
        d_quantile=args.d_quantile,
        seed=args.seed,
        min_feature_coverage=args.min_feature_coverage,
    )
    curated_gmm = build_atlas_gmm(
        args.curated_db,
        feature_version="curated_stop_place_gmm_v1",
        config=config,
        max_k=args.max_k,
        min_samples_per_component=args.min_samples_per_component,
    )
    curated_gmm_path = args.out_dir / "curated_stop_place_gmm.json"
    curated_gmm.save(curated_gmm_path)

    cal_config = CalibrationBuildConfig(
        providers=tuple(args.providers),
        confidence_floor=args.confidence_floor,
        per_phone_cap=args.per_phone_cap,
        quantile_count=args.quantile_count,
        seed=args.seed,
    )
    curated_cal = build_gmm_calibration(args.curated_db, curated_gmm, cal_config)
    curated_cal_path = args.out_dir / "curated_stop_place_gmm_calibration.json"
    curated_cal.save(curated_cal_path)

    strict = not args.allow_missing_phones
    gmm_overlay = merge_gmm_overlay(
        load_json(args.base_gmm),
        curated_gmm.as_dict(),
        phones=STOP_PLACE_PHONES,
        strict=strict,
    )
    gmm_overlay_path = args.out_dir / "coach_gmm_overlay.json"
    write_json(gmm_overlay_path, gmm_overlay)

    cal_overlay = merge_calibration_overlay(
        load_json(args.base_calibration),
        curated_cal.as_dict(),
        phones=STOP_PLACE_PHONES,
        strict=strict,
    )
    cal_overlay_path = args.out_dir / "coach_gmm_calibration_overlay.json"
    write_json(cal_overlay_path, cal_overlay)

    authority_overlay_path = _build_authority_overlay(args, curated_gmm_path, curated_cal_path)
    print(
        json.dumps(
            {
                "curated_db": str(args.curated_db),
                "report": str(report_path),
                "gmm_overlay": str(gmm_overlay_path),
                "calibration_overlay": str(cal_overlay_path),
                "authority_overlay": str(authority_overlay_path),
                "per_phone": report.per_phone,
                "per_phone_speakers": report.per_phone_speakers,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_authority_overlay(
    args: argparse.Namespace, curated_gmm_path: Path, curated_cal_path: Path
) -> Path:
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from build_decision_authority import build_authority  # noqa: PLC0415

    replacement = build_authority(
        db_path=args.curated_db,
        gmm_path=curated_gmm_path,
        calibration_path=curated_cal_path,
        sample_limit=args.authority_sample_limit,
        min_samples=args.authority_min_samples,
        min_eval_samples=args.authority_min_eval_samples,
        confidence_floor=args.confidence_floor,
        fp_ceiling=args.authority_fp_ceiling,
        min_recall=args.authority_min_recall,
        target_quality_ceiling=args.authority_target_quality_ceiling,
        feature_confidence_floor=args.confidence_floor,
        max_features=args.authority_max_features,
        regularization=args.authority_regularization,
        model_scope="mms-1b-curated-stop-place",
    )
    replacement_path = args.out_dir / "curated_stop_place_decision_authority.json"
    write_json(replacement_path, replacement)

    overlay = merge_authority_overlay(
        load_json(args.base_authority),
        replacement,
        phones=STOP_PLACE_PHONES,
        scope=args.authority_merge_scope,
        skip_zero_sample_rows=True,
    )
    out = args.out_dir / "decision_authority_overlay.json"
    write_json(out, overlay)
    return out


def cmd_all(args: argparse.Namespace) -> int:
    curation = args.out_dir / "curation.csv"
    mined = argparse.Namespace(
        audio_db=args.audio_db,
        reference_db=args.reference_db,
        audio_root=args.audio_root,
        reference_feature_version=args.reference_feature_version,
        providers=["common_voice"],
        confidence_floor=0.5,
        limit=args.limit,
        out=curation,
    )
    cmd_mine(mined)

    overlays = build_parser().parse_args(
        [
            "build-overlays",
            "--audio-db",
            str(args.audio_db),
            "--reference-db",
            str(args.reference_db),
            "--audio-root",
            str(args.audio_root),
            "--curation",
            str(curation),
            "--curated-db",
            str(args.out_dir / "curated_stop_place_reference.sqlite"),
            "--out-dir",
            str(args.out_dir),
            *(["--replace-db"] if args.replace_db else []),
            *(["--allow-missing-phones"] if args.allow_missing_phones else []),
        ]
    )
    return cmd_build_overlays(overlays)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "mine":
        return cmd_mine(args)
    if args.command == "build-sqlite":
        return cmd_build_sqlite(args)
    if args.command == "build-overlays":
        return cmd_build_overlays(args)
    if args.command == "all":
        return cmd_all(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
