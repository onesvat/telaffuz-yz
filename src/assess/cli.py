"""CLI for `assess coach` subcommands.

Subcommands:
  run              — run a single coach assessment
  build-stats      — build per-phone canonical statistics from compat DB
  build-gmm        — (alias for build-stats; GMM extension in Faz 4)
  validate-batch   — run coach over a validation manifest
  validate-reports — generate audit reports from JSONL observations
  calibrate-authority — calibrate contrast authority table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assess coach",
        description="Pronunciation coach runtime and validation utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run a single coach assessment.")
    run_p.add_argument("--audio", type=Path, required=True, help="Input audio file.")
    run_p.add_argument("--model", default="mms-1b", choices=["mms-1b", "xls-r-300m"])
    _target_group = run_p.add_mutually_exclusive_group(required=True)
    _target_group.add_argument("--text", help="Target text (G2P).")
    _target_group.add_argument("--phones", nargs="+", help="Target IPA phone list.")
    run_p.add_argument("--stats", type=Path, required=True, help="Atlas stats JSON.")
    run_p.add_argument("--authority", type=Path, required=True, help="Authority table JSON.")
    run_p.add_argument(
        "--calibration", type=Path, default=None, help="GMM log-density calibration JSON."
    )

    # build-stats
    bs_p = sub.add_parser("build-stats", help="Build per-phone stats from compat DB.")
    bs_p.add_argument("--compat-db", type=Path, required=True)
    bs_p.add_argument("--out", type=Path, required=True)
    bs_p.add_argument("--feature-version", default="assess_coach_stats_v1")
    bs_p.add_argument("--providers", nargs="+", default=["common_voice", "issai_tsc"])
    bs_p.add_argument("--min-samples", type=int, default=50)
    bs_p.add_argument("--confidence-floor", type=float, default=0.5)
    bs_p.add_argument("--per-phone-cap", type=int, default=20000)
    bs_p.add_argument("--train-buckets", default="0..89")
    bs_p.add_argument("--test-buckets", default="90..99")
    bs_p.add_argument("--d-quantile", type=float, default=0.95)
    bs_p.add_argument("--seed", type=int, default=0)
    bs_p.add_argument("--min-feature-coverage", type=float, default=0.6)

    # build-gmm (Faz 4: BIC-driven per-phone GMM with auto K)
    bgmm_p = sub.add_parser(
        "build-gmm",
        help="Fit per-phone GMM artifacts from a coach reference DB.",
    )
    bgmm_p.add_argument("--compat-db", type=Path, required=True)
    bgmm_p.add_argument("--out", type=Path, required=True)
    bgmm_p.add_argument("--feature-version", default="assess_coach_gmm_v1")
    bgmm_p.add_argument("--providers", nargs="+", default=["common_voice", "issai_tsc"])
    bgmm_p.add_argument("--min-samples", type=int, default=50)
    bgmm_p.add_argument("--confidence-floor", type=float, default=0.5)
    bgmm_p.add_argument("--per-phone-cap", type=int, default=20000)
    bgmm_p.add_argument("--train-buckets", default="0..89")
    bgmm_p.add_argument("--test-buckets", default="90..99")
    bgmm_p.add_argument("--d-quantile", type=float, default=0.95)
    bgmm_p.add_argument("--seed", type=int, default=0)
    bgmm_p.add_argument("--min-feature-coverage", type=float, default=0.6)
    bgmm_p.add_argument("--max-k", type=int, default=3)
    bgmm_p.add_argument("--min-samples-per-component", type=int, default=30)

    # build-length (length channel: per-phone duration stats from wav2vec spans)
    bl_p = sub.add_parser(
        "build-length",
        help="Build per-phone duration stats from a (phone, dur_ms) JSONL.",
    )
    bl_p.add_argument("--durations", type=Path, required=True)
    bl_p.add_argument("--out", type=Path, required=True)
    bl_p.add_argument("--min-samples", type=int, default=50)

    # build-calibration (per-phone log-density rank grid)
    bc_p = sub.add_parser(
        "build-calibration",
        help="Build per-phone GMM log-density calibration from native CV samples.",
    )
    bc_p.add_argument("--compat-db", type=Path, required=True)
    bc_p.add_argument("--gmm", type=Path, required=True)
    bc_p.add_argument("--out", type=Path, required=True)
    bc_p.add_argument("--providers", nargs="+", default=["common_voice"])
    bc_p.add_argument("--confidence-floor", type=float, default=0.5)
    bc_p.add_argument("--per-phone-cap", type=int, default=20000)
    bc_p.add_argument("--quantile-count", type=int, default=1024)
    bc_p.add_argument("--seed", type=int, default=0)

    # validate-batch
    vb_p = sub.add_parser("validate-batch", help="Run validation batch over a manifest.")
    vb_p.add_argument("--manifest", type=Path, required=True)
    vb_p.add_argument("--split", required=True, choices=["dev", "test"])
    vb_p.add_argument("--model", required=True, choices=["mms-1b", "xls-r-300m"])
    vb_p.add_argument("--stats", type=Path, required=True)
    vb_p.add_argument("--authority", type=Path, required=True)
    vb_p.add_argument("--output", type=Path, required=True)
    vb_p.add_argument("--audio-root", type=Path)
    vb_p.add_argument("--no-resume", action="store_true")
    vb_p.add_argument("--limit", type=int)
    vb_p.add_argument(
        "--calibration", type=Path, default=None, help="GMM log-density calibration JSON."
    )

    # validate-reports
    vr_p = sub.add_parser("validate-reports", help="Generate validation audit reports.")
    vr_p.add_argument("--observations", type=Path, required=True)
    vr_p.add_argument("--json", type=Path, required=True)
    vr_p.add_argument("--markdown", type=Path, required=True)
    vr_p.add_argument(
        "--report-type",
        choices=["hard-fp", "unscored", "model-comparison", "atlas-override"],
        default="hard-fp",
    )

    # calibrate-authority
    ca_p = sub.add_parser("calibrate-authority", help="Calibrate contrast authority table.")
    ca_p.add_argument("--observations", type=Path, required=True)
    ca_p.add_argument(
        "--from-validation",
        action="store_true",
        help="Treat --observations as a validation-batch JSONL (val_obs.jsonl) "
             "and adapt rows into calibration observations via GMM atlas fields.",
    )
    ca_p.add_argument("--output", type=Path, required=True)
    ca_p.add_argument("--diagnostics", type=Path, required=True)
    ca_p.add_argument("--override", action="append", default=[])
    ca_p.add_argument("--min-samples", type=int, default=25)
    ca_p.add_argument("--native-fp-ceiling", type=float, default=0.02)
    ca_p.add_argument("--wav2vec-cross-confusion-ceiling", type=float, default=0.03)
    ca_p.add_argument("--atlas-margin-floor", type=float, default=0.10)
    ca_p.add_argument(
        "--scope",
        choices=["all", "fixed"],
        default="all",
        help="`fixed` restricts calibration to FIXED_SAME_CLASS_CONTRASTS "
             "(Türkçe same-class minimal pairs + smoke-observed conflicts); "
             "`all` (default) calibrates every observed (target, alt) pair.",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from assess.contracts import CoachRequest  # noqa: PLC0415
    from assess.runtime import run_coach  # noqa: PLC0415

    request = CoachRequest(
        audio_path=str(args.audio),
        model_alias=args.model,
        target_text=args.text if args.text else None,
        target_phones=list(args.phones) if args.phones else None,
        stats_path=str(args.stats),
        authority_path=str(args.authority),
        calibration_path=str(args.calibration) if args.calibration else None,
    )
    result = run_coach(request)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.status in {"ok", "model_unavailable"} else 1


def _cmd_build_stats(args: argparse.Namespace) -> int:
    from assess.stats import BuildConfig, build_atlas_stats  # noqa: PLC0415

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
    stats = build_atlas_stats(
        args.compat_db,
        feature_version=args.feature_version,
        config=config,
    )
    stats.save(args.out)
    print(
        json.dumps(
            {"feature_version": stats.feature_version, "phones": len(stats.phones)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_build_gmm(args: argparse.Namespace) -> int:
    from assess.stats import BuildConfig, build_atlas_gmm  # noqa: PLC0415

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
    gmm_atlas = build_atlas_gmm(
        args.compat_db,
        feature_version=args.feature_version,
        config=config,
        max_k=args.max_k,
        min_samples_per_component=args.min_samples_per_component,
    )
    gmm_atlas.save(args.out)
    component_counts = sorted({phone.k for phone in gmm_atlas.phones.values()})
    print(
        json.dumps(
            {
                "feature_version": gmm_atlas.feature_version,
                "phones": len(gmm_atlas.phones),
                "component_counts": component_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_build_length(args: argparse.Namespace) -> int:
    from assess.length import build_duration_stats  # noqa: PLC0415

    rows = (
        json.loads(line)
        for line in args.durations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    stats = build_duration_stats(rows, min_samples=args.min_samples)
    args.out.write_text(
        json.dumps(stats.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"phones": len(stats.phones)}, ensure_ascii=False, indent=2))
    return 0


def _cmd_build_calibration(args: argparse.Namespace) -> int:
    from assess.stats import (  # noqa: PLC0415
        AtlasGMM,
        CalibrationBuildConfig,
        build_gmm_calibration,
    )

    gmm_atlas = AtlasGMM.load(args.gmm)
    config = CalibrationBuildConfig(
        providers=tuple(args.providers),
        confidence_floor=args.confidence_floor,
        per_phone_cap=args.per_phone_cap,
        quantile_count=args.quantile_count,
        seed=args.seed,
    )
    calibration = build_gmm_calibration(args.compat_db, gmm_atlas, config)
    calibration.save(args.out)
    print(
        json.dumps(
            {
                "feature_version": calibration.feature_version,
                "phones": len(calibration.phones),
                "gmm_signature": calibration.gmm_signature,
                "quantile_count": args.quantile_count,
                "per_phone_samples": {
                    phone: cal.n for phone, cal in sorted(calibration.phones.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_validate_batch(args: argparse.Namespace) -> int:
    from assess.validation import run_validation_batch  # noqa: PLC0415

    summary = run_validation_batch(
        manifest=args.manifest,
        split=args.split,
        model_alias=args.model,
        stats_path=str(args.stats),
        authority_path=str(args.authority),
        output=args.output,
        audio_root=args.audio_root,
        resume=not args.no_resume,
        limit=args.limit,
        calibration_path=str(args.calibration) if args.calibration else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_validate_reports(args: argparse.Namespace) -> int:
    from assess.validation import (  # noqa: PLC0415
        atlas_override_report,
        hard_false_positive_report,
        load_jsonl,
        model_comparison_report,
        unscored_report,
        write_json_report,
        write_markdown_report,
    )

    rows = load_jsonl(args.observations)
    report_type = args.report_type
    if report_type == "hard-fp":
        title = "Hard False-Positive Audit"
        report = hard_false_positive_report(rows)
    elif report_type == "unscored":
        title = "Unscored Phone Audit"
        report = unscored_report(rows)
    elif report_type == "model-comparison":
        title = "Model Comparison"
        report = model_comparison_report(rows)
    else:
        title = "Atlas Override Audit"
        report = atlas_override_report(rows)
    write_json_report(report, args.json)
    write_markdown_report(title, report, args.markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_override_pairs(raw: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in raw:
        if ":" not in item:
            raise ValueError("--override must use target:alternative")
        target, alternative = item.split(":", 1)
        if not target or not alternative:
            raise ValueError("--override must use target:alternative")
        pairs.add((target, alternative))
    return pairs


def _cmd_calibrate_authority(args: argparse.Namespace) -> int:
    from assess.validation import (  # noqa: PLC0415
        CalibrationConfig,
        CalibrationObservation,
        calibrate_authority,
        calibration_observations_from_validation_rows,
        load_jsonl,
        write_authority_json,
        write_diagnostics_csv,
    )

    raw_rows = load_jsonl(args.observations)
    if args.from_validation:
        observations = calibration_observations_from_validation_rows(raw_rows)
    else:
        observations = [
            CalibrationObservation(
                speaker_id=str(r["speaker_id"]),
                target=str(r["target"]),
                alternative=str(r["alternative"]),
                target_fit=float(r["target_fit"]),
                alternative_fit=float(r["alternative_fit"]),
                wav2vec_observed=(
                    str(r["wav2vec_observed"]) if r.get("wav2vec_observed") is not None else None
                ),
            )
            for r in raw_rows
        ]
    config = CalibrationConfig(
        native_fp_ceiling=args.native_fp_ceiling,
        wav2vec_cross_confusion_ceiling=args.wav2vec_cross_confusion_ceiling,
        atlas_margin_floor=args.atlas_margin_floor,
        min_samples=args.min_samples,
        override_whitelist=_parse_override_pairs(args.override),
        scope=args.scope,
    )
    result = calibrate_authority(observations, config)
    write_authority_json(result, args.output)
    write_diagnostics_csv(result, args.diagnostics)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "build-stats":
        return _cmd_build_stats(args)
    if args.command == "build-gmm":
        return _cmd_build_gmm(args)
    if args.command == "build-length":
        return _cmd_build_length(args)
    if args.command == "build-calibration":
        return _cmd_build_calibration(args)
    if args.command == "validate-batch":
        return _cmd_validate_batch(args)
    if args.command == "validate-reports":
        return _cmd_validate_reports(args)
    if args.command == "calibrate-authority":
        return _cmd_calibrate_authority(args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
