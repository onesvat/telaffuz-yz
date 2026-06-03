"""Command line interface for the G2P engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from g2p.pipeline import G2PResult, transcribe_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g2p", description="Turkish G2P engine")
    parser.add_argument("text", nargs="*", help="Word or text to transcribe")
    parser.add_argument("--json", action="store_true", help="Print JSON/JSONL output")
    parser.add_argument(
        "--phones",
        action="store_true",
        help="Print space-separated CTC phone tokens",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="Read one input per line from a UTF-8 file",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable built-in reference IPA overrides",
    )
    parser.add_argument(
        "--no-pedagogical",
        action="store_true",
        help="Disable pedagogical allophones (aspiration, r-devoicing)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch is not None:
        _run_batch(
            args.batch,
            as_json=args.json,
            phones=args.phones,
            use_reference=not args.no_reference,
            pedagogical_allophones=not args.no_pedagogical,
        )
        return 0

    text = " ".join(args.text).strip()
    if not text:
        build_parser().print_usage()
        return 2

    print(
        _render(
            transcribe_text(
                text,
                use_reference=not args.no_reference,
                pedagogical_allophones=not args.no_pedagogical,
            ),
            as_json=args.json,
            phones=args.phones,
        )
    )
    return 0


def _run_batch(
    path: Path,
    *,
    as_json: bool,
    phones: bool,
    use_reference: bool,
    pedagogical_allophones: bool,
) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        result = transcribe_text(
            text,
            use_reference=use_reference,
            pedagogical_allophones=pedagogical_allophones,
        )
        if as_json or phones:
            print(_render(result, as_json=as_json, phones=phones))
        else:
            print(f"{text}\t{result.ipa}")


def _render(result: G2PResult, *, as_json: bool, phones: bool) -> str:
    if as_json:
        return result.to_json()
    if phones:
        return " ".join(result.phonemes)
    return result.ipa


if __name__ == "__main__":
    raise SystemExit(main())
