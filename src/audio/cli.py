"""Audio pipeline CLI entry point.

Subcommands:
    init        — initialize data/audio.sqlite
    whisper     — Whisper transcription with word timestamps (desktop)
    mms         — emission cache (desktop GPU)
    pilot       — stratified atlas pilot manifest + model-selection report
    quality     — segment quality assessments
    align       — CTC forced alignment (desktop GPU)
    manual      — import Praat/TextGrid manual phone labels
    export      — export atlas parquet + G2P evidence CSV
    words       — build and QA word occurrence index
    atlas       — build phoneme + word fingerprint atlas
    eval-g2p    — audio-derived G2P evidence loop
    dataset     — curated wav2vec dataset export
    features    — coach reference feature DB utilities (audit, build, rebuild)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


SUBCOMMANDS = {
    "init": "initialize data/audio.sqlite",
    "whisper": "Whisper transcription with word timestamps (desktop)",
    "mms": "emission cache (desktop GPU)",
    "pilot": "stratified atlas pilot manifest + model-selection report",
    "quality": "compute segment quality assessments",
    "align": "CTC forced alignment (desktop GPU)",
    "manual": "import Praat/TextGrid manual phone labels",
    "export": "export atlas parquet + G2P evidence CSV",
    "words": "build and QA word occurrence index",
    "atlas": "build phoneme + word fingerprint atlas",
    "eval-g2p": "audio-derived G2P evidence loop",
    "dataset": "curated wav2vec dataset export",
    "features": "coach reference feature DB utilities (audit, build, rebuild)",
}


def _print_help() -> None:
    print("usage: audio <subcommand> [args...]")
    print()
    print(__doc__)
    print("Subcommands:")
    for name, help_text in SUBCOMMANDS.items():
        print(f"  {name:10s} {help_text}")


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0
    cmd = argv[0]
    rest = argv[1:]

    if cmd not in SUBCOMMANDS:
        print(f"audio: unknown subcommand '{cmd}'", file=sys.stderr)
        _print_help()
        return 2

    if cmd == "init":
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "init_audio_db.py"), *rest]
        ).returncode

    if cmd == "whisper":
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_whisper.py"), *rest]
        ).returncode

    if cmd == "mms":
        from audio.mms_runner import main as mms_main

        return mms_main(rest)

    if cmd == "atlas":
        from audio.atlas import main as atlas_main

        return atlas_main(rest)

    if cmd == "pilot":
        from audio.atlas_pilot import main as pilot_main

        return pilot_main(rest)

    if cmd == "align":
        from audio.alignment import main as alignment_main

        return alignment_main(rest)

    if cmd == "manual":
        from audio.manual import main as manual_main

        return manual_main(rest)

    if cmd == "export":
        from audio.atlas_export import main as export_main

        return export_main(rest)

    if cmd == "features":
        return _features_main(rest)

    print(f"Command '{cmd}' not yet implemented (stub)")
    return 1


def _features_main(argv: list[str]) -> int:
    """Dispatch `audio features` subcommands."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="audio features")
    sub = parser.add_subparsers(dest="command", required=True)

    # audit-coach-db
    audit_p = sub.add_parser("audit-coach-db", help="Audit coach SQLite databases.")
    audit_p.add_argument("--audio-db", required=True, type=Path)
    audit_p.add_argument("--features-db", required=True, type=Path)
    audit_p.add_argument("--sample-rows", type=int, default=20)

    # build-compat-db
    build_p = sub.add_parser(
        "build-compat-db", help="Build coach compat feature DB from legacy sources."
    )
    build_p.add_argument("--audio-db", required=True, type=Path)
    build_p.add_argument("--features-db", required=True, type=Path)
    build_p.add_argument("--output-db", required=True, type=Path)
    build_p.add_argument("--source-feature-version", default="phone_features_v3_policy")
    build_p.add_argument("--limit", type=int)
    build_p.add_argument("--replace", action="store_true")
    build_p.add_argument("--resume", action="store_true")

    # rebuild-reference: re-extract signal-anchored features from alignments
    rebuild_p = sub.add_parser(
        "rebuild-reference",
        help="Re-extract coach reference features (Faz 1+2 pipeline, CV-only).",
    )
    rebuild_p.add_argument("--audio-db", required=True, type=Path)
    rebuild_p.add_argument("--audio-root", required=True, type=Path)
    rebuild_p.add_argument("--output-db", required=True, type=Path)
    rebuild_p.add_argument(
        "--providers",
        nargs="+",
        default=["common_voice"],
        help="Source providers to include (default: common_voice — CV-only).",
    )
    rebuild_p.add_argument("--aligner-version", default=None)
    rebuild_p.add_argument("--feature-confidence-floor", type=float, default=0.0)
    rebuild_p.add_argument("--limit", type=int)
    rebuild_p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for reference rebuild (default: 1).",
    )
    rebuild_p.add_argument("--no-replace", action="store_true")

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    if args.command == "audit-coach-db":
        from audio.coach_db import audit_databases  # noqa: PLC0415

        report = audit_databases(
            audio_db=args.audio_db,
            features_db=args.features_db,
            sample_rows=args.sample_rows,
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "build-compat-db":
        from audio.coach_db import build_compat_db  # noqa: PLC0415

        report = build_compat_db(
            audio_db=args.audio_db,
            features_db=args.features_db,
            output_db=args.output_db,
            source_feature_version=args.source_feature_version,
            limit=args.limit,
            replace=args.replace,
            resume=args.resume,
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "rebuild-reference":
        from audio.coach_db import rebuild_reference  # noqa: PLC0415

        report = rebuild_reference(
            audio_db=args.audio_db,
            audio_root=args.audio_root,
            output_db=args.output_db,
            providers=tuple(args.providers),
            aligner_version=args.aligner_version,
            feature_confidence_floor=args.feature_confidence_floor,
            limit=args.limit,
            workers=args.workers,
            replace=not args.no_replace,
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
