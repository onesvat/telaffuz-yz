"""Benchmark and external-reference evaluation utilities."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from g2p.constants import (
    ALL_CONSONANTS,
    CONSONANTS,
    LONG_VOWELS,
    SHORT_VOWELS,
    VOWEL_ALLOPHONES,
)
from g2p.phones import (
    PhoneParse,
    levenshtein,
    normalize_phone_text,
    parse_ipa_text,
    phones_to_text,
    strip_meta,
)
from g2p.pipeline import G2PResult, format_ipa, transcribe_text, transcribe_word


_BENCHMARKS: dict[str, Path] = {
    "blind_75": Path("data/g2p/blind_75.csv"),
    "wiktionary": Path("data/g2p/wiktionary.csv"),
    "mfa": Path("data/g2p/mfa.csv"),
}


@dataclass
class MetricCounts:
    rows: int = 0
    exact: int = 0
    edits: int = 0
    target_phone_count: int = 0

    def add(self, predicted: Iterable[str], target: Iterable[str]) -> None:
        pred_clean = strip_meta(predicted)
        target_clean = strip_meta(target)
        self.rows += 1
        if pred_clean == target_clean:
            self.exact += 1
        self.edits += levenshtein(pred_clean, target_clean)
        self.target_phone_count += len(target_clean)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "rows": self.rows,
            "exact_count": self.exact,
            "exact_match": _rate(self.exact, self.rows),
            "phone_errors": self.edits,
            "target_phone_count": self.target_phone_count,
            "per": _rate(self.edits, self.target_phone_count),
        }


@dataclass
class EvalArtifacts:
    report: dict[str, Any]
    mismatches: list[dict[str, str]]
    report_path: Path | None = None
    mismatches_path: Path | None = None
    phone_errors_path: Path | None = None
    substitutions_path: Path | None = None


@dataclass
class PhoneCounts:
    reference_count: int = 0
    correct: int = 0
    substitutions: int = 0
    deletions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions

    def to_dict(self) -> dict[str, int | float]:
        return {
            "reference_count": self.reference_count,
            "correct": self.correct,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "errors": self.errors,
            "per": _rate(self.errors, self.reference_count),
        }


@dataclass(frozen=True)
class PhoneEdit:
    op: str
    target: str | None
    predicted: str | None


@dataclass
class PhoneErrorAnalysis:
    by_target_phone: dict[str, PhoneCounts] = field(default_factory=dict)
    by_phone_class: dict[str, PhoneCounts] = field(default_factory=dict)
    insertions_by_predicted_phone: Counter[str] = field(default_factory=Counter)
    substitutions: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, predicted: Iterable[str], target: Iterable[str]) -> None:
        pred_clean = strip_meta(predicted)
        target_clean = strip_meta(target)
        for edit in _align_phones(pred_clean, target_clean):
            if edit.op == "insertion":
                if edit.predicted is not None:
                    self.insertions_by_predicted_phone[edit.predicted] += 1
                continue

            if edit.target is None:
                continue
            phone_counts = self.by_target_phone.setdefault(edit.target, PhoneCounts())
            class_counts = self.by_phone_class.setdefault(
                _phone_class(edit.target), PhoneCounts()
            )
            phone_counts.reference_count += 1
            class_counts.reference_count += 1

            if edit.op == "match":
                phone_counts.correct += 1
                class_counts.correct += 1
            elif edit.op == "substitution":
                phone_counts.substitutions += 1
                class_counts.substitutions += 1
                if edit.predicted is not None:
                    self.substitutions[(edit.target, edit.predicted)] += 1
            elif edit.op == "deletion":
                phone_counts.deletions += 1
                class_counts.deletions += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_target_phone": {
                phone: counts.to_dict()
                for phone, counts in sorted(
                    self.by_target_phone.items(),
                    key=lambda item: (-item[1].errors, -item[1].reference_count, item[0]),
                )
            },
            "by_phone_class": {
                phone_class: counts.to_dict()
                for phone_class, counts in sorted(self.by_phone_class.items())
            },
            "insertions_by_predicted_phone": dict(
                sorted(
                    self.insertions_by_predicted_phone.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "top_substitutions": [
                {"target": target, "predicted": predicted, "count": count}
                for (target, predicted), count in sorted(
                    self.substitutions.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
        }


@dataclass
class _EvalState:
    overall: MetricCounts = field(default_factory=MetricCounts)
    categories: dict[str, MetricCounts] = field(default_factory=dict)
    sources: dict[str, MetricCounts] = field(default_factory=dict)
    exception_types: dict[str, MetricCounts] = field(default_factory=dict)
    phone_errors: PhoneErrorAnalysis = field(default_factory=PhoneErrorAnalysis)
    skipped: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        *,
        row: dict[str, str],
        result: G2PResult,
        target_phones: tuple[str, ...],
    ) -> None:
        self.overall.add(result.phonemes, target_phones)
        self.phone_errors.add(result.phonemes, target_phones)
        category = row.get("category", "").strip() or "uncategorized"
        self.categories.setdefault(category, MetricCounts()).add(
            result.phonemes, target_phones
        )
        self.sources.setdefault(result.source, MetricCounts()).add(
            result.phonemes, target_phones
        )
        exception_type = result.exception_type or "none"
        self.exception_types.setdefault(exception_type, MetricCounts()).add(
            result.phonemes, target_phones
        )

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


_WORD_SPLIT_RE = re.compile(r"\s+")


def evaluate_benchmark_csv(
    input_path: Path,
    *,
    name: str | None = None,
    output_dir: Path = Path("reports/g2p"),
    use_reference: bool = False,
    pedagogical_allophones: bool = True,
    write_reports: bool = True,
) -> EvalArtifacts:
    rows = _read_csv(input_path)
    eval_name = name or input_path.stem
    artifacts = evaluate_benchmark_rows(
        rows,
        name=eval_name,
        input_path=str(input_path),
        use_reference=use_reference,
        pedagogical_allophones=pedagogical_allophones,
    )
    if write_reports:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"eval_{eval_name}.json"
        mismatches_path = output_dir / f"eval_{eval_name}_mismatches.csv"
        phone_errors_path = output_dir / f"eval_{eval_name}_phone_errors.csv"
        substitutions_path = output_dir / f"eval_{eval_name}_substitutions.csv"
        report_path.write_text(
            json.dumps(artifacts.report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_mismatches(mismatches_path, artifacts.mismatches)
        _write_phone_errors(
            phone_errors_path,
            artifacts.report["phone_error_breakdown"]["by_target_phone"],
        )
        _write_substitutions(
            substitutions_path,
            artifacts.report["phone_error_breakdown"]["top_substitutions"],
        )
        artifacts.report_path = report_path
        artifacts.mismatches_path = mismatches_path
        artifacts.phone_errors_path = phone_errors_path
        artifacts.substitutions_path = substitutions_path
    return artifacts


def evaluate_benchmark_rows(
    rows: list[dict[str, str]],
    *,
    name: str,
    input_path: str = "",
    use_reference: bool = False,
    pedagogical_allophones: bool = True,
) -> EvalArtifacts:
    state = _EvalState()
    reference_state = _EvalState() if not use_reference else None
    mismatches: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        word = row.get("word", "").strip()
        if not word:
            state.skip("missing_word")
            continue

        target_parse = target_phones_from_row(row)
        if not target_parse.phones:
            state.skip("missing_target")
            continue
        if target_parse.unknown:
            state.skip("unknown_target_tokens")
            continue

        result = transcribe_for_evaluation(
            word,
            use_reference=use_reference,
            pedagogical_allophones=pedagogical_allophones,
        )
        state.add(row=row, result=result, target_phones=target_parse.phones)

        if reference_state is not None:
            ref_result = transcribe_for_evaluation(
                word,
                use_reference=True,
                pedagogical_allophones=pedagogical_allophones,
            )
            reference_state.add(
                row=row, result=ref_result, target_phones=target_parse.phones
            )

        if strip_meta(result.phonemes) != strip_meta(target_parse.phones):
            mismatches.append(_mismatch_row(index, row, result, target_parse.phones))

    report: dict[str, Any] = {
        "name": name,
        "input_path": input_path,
        "use_reference": use_reference,
        "pedagogical_allophones": pedagogical_allophones,
        "rows_total": len(rows),
        "rows_scored": state.overall.rows,
        "rows_skipped": sum(state.skipped.values()),
        "skip_reasons": dict(sorted(state.skipped.items())),
        "metrics": state.overall.to_dict(),
        "category_breakdown": _counts_dict(state.categories),
        "source_breakdown": _counts_dict(state.sources),
        "exception_type_breakdown": _counts_dict(state.exception_types),
        "phone_error_breakdown": state.phone_errors.to_dict(),
    }

    if reference_state is not None:
        report["reference_comparison"] = {
            "metrics": reference_state.overall.to_dict(),
            "source_breakdown": _counts_dict(reference_state.sources),
        }

    return EvalArtifacts(report=report, mismatches=mismatches)


def target_phones_from_row(row: dict[str, str]) -> PhoneParse:
    for field_name, parser in (
        ("target_phones", normalize_phone_text),
        ("target_ipa", parse_ipa_text),
        ("ipa_detail", parse_ipa_text),
        ("phones", normalize_phone_text),
        ("ipa", parse_ipa_text),
    ):
        value = row.get(field_name, "").strip()
        if value:
            return parser(value)
    return PhoneParse(())


def canonical_target_ipa(phones: Iterable[str]) -> str:
    return format_ipa(tuple(phones))


def _mismatch_row(
    index: int,
    row: dict[str, str],
    result: G2PResult,
    target_phones: tuple[str, ...],
) -> dict[str, str]:
    edit_distance = levenshtein(strip_meta(result.phonemes), strip_meta(target_phones))
    target_clean = strip_meta(target_phones)
    per = edit_distance / len(target_clean) if target_clean else 0.0
    return {
        "row": str(index),
        "id": row.get("id", ""),
        "word": row.get("word", ""),
        "category": row.get("category", ""),
        "target_ipa": row.get("target_ipa", "") or canonical_target_ipa(target_phones),
        "target_phones": phones_to_text(target_phones),
        "pred_ipa": result.ipa,
        "pred_phones": phones_to_text(result.phonemes),
        "source": result.source,
        "exception_type": result.exception_type or "",
        "edit_distance": str(edit_distance),
        "per": f"{per:.6f}",
        "warnings": "; ".join(result.warnings),
        "notes": row.get("notes", ""),
    }


def transcribe_for_evaluation(
    word: str,
    *,
    use_reference: bool,
    pedagogical_allophones: bool = True,
) -> G2PResult:
    parts = [part for part in _WORD_SPLIT_RE.split(word.strip()) if part]
    if len(parts) > 1:
        return transcribe_text(
            word,
            use_reference=use_reference,
            pedagogical_allophones=pedagogical_allophones,
        )
    return transcribe_word(
        word,
        use_reference=use_reference,
        pedagogical_allophones=pedagogical_allophones,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_mismatches(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = (
        "row",
        "id",
        "word",
        "category",
        "target_ipa",
        "target_phones",
        "pred_ipa",
        "pred_phones",
        "source",
        "exception_type",
        "edit_distance",
        "per",
        "warnings",
        "notes",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_phone_errors(path: Path, rows: dict[str, dict[str, int | float]]) -> None:
    fieldnames = (
        "phone",
        "phone_class",
        "reference_count",
        "correct",
        "substitutions",
        "deletions",
        "errors",
        "per",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for phone, counts in rows.items():
            writer.writerow(
                {
                    "phone": phone,
                    "phone_class": _phone_class(phone),
                    **counts,
                }
            )


def _write_substitutions(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = ("target", "predicted", "count")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _counts_dict(groups: dict[str, MetricCounts]) -> dict[str, dict[str, int | float]]:
    return {key: value.to_dict() for key, value in sorted(groups.items())}


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _align_phones(predicted: tuple[str, ...], target: tuple[str, ...]) -> list[PhoneEdit]:
    rows = len(target) + 1
    cols = len(predicted) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if target[i - 1] == predicted[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + substitution_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    edits_reversed: list[PhoneEdit] = []
    i = len(target)
    j = len(predicted)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution_cost = 0 if target[i - 1] == predicted[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + substitution_cost:
                op = "match" if substitution_cost == 0 else "substitution"
                edits_reversed.append(PhoneEdit(op, target[i - 1], predicted[j - 1]))
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            edits_reversed.append(PhoneEdit("deletion", target[i - 1], None))
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            edits_reversed.append(PhoneEdit("insertion", None, predicted[j - 1]))
            j -= 1
            continue

        raise RuntimeError("phone alignment backtrace failed")

    return list(reversed(edits_reversed))


def _phone_class(phone: str) -> str:
    if phone in SHORT_VOWELS:
        return "short_vowel"
    if phone in LONG_VOWELS:
        return "long_vowel"
    if phone in VOWEL_ALLOPHONES:
        return "vowel_allophone"
    if phone in CONSONANTS:
        return "base_consonant"
    if phone in ALL_CONSONANTS:
        return "consonant_allophone"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the G2P engine against any benchmark CSV.",
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        choices=list(_BENCHMARKS.keys()),
        help="Built-in benchmark to evaluate",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Arbitrary benchmark CSV (overrides built-in)",
    )
    parser.add_argument(
        "--name",
        help="Report name; defaults to the input file stem",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/g2p"),
        help="Directory for eval JSON and mismatch CSV",
    )
    parser.add_argument(
        "--no-reference",
        dest="use_reference",
        action="store_false",
        help="Disable built-in reference IPA overrides (default)",
    )
    parser.set_defaults(use_reference=False)
    parser.add_argument(
        "--no-pedagogical-allophones",
        dest="pedagogical_allophones",
        action="store_false",
        help="Disable pedagogical allophones (default)",
    )
    parser.set_defaults(pedagogical_allophones=True)
    parser.add_argument(
        "--print-mismatches",
        action="store_true",
        help="Print per-word mismatch details to stdout",
    )
    args = parser.parse_args()

    if args.path is not None:
        csv_path = args.path
    elif args.benchmark is not None:
        csv_path = _BENCHMARKS[args.benchmark]
    else:
        parser.print_usage()
        print("error: either BENCHMARK or --path is required")
        return

    name = args.name or csv_path.stem
    artifacts = evaluate_benchmark_csv(
        csv_path,
        name=name,
        output_dir=args.output_dir,
        use_reference=args.use_reference,
        pedagogical_allophones=args.pedagogical_allophones,
    )

    metrics = artifacts.report["metrics"]
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Rows scored: {artifacts.report['rows_scored']}")
    print(f"Rows skipped: {artifacts.report['rows_skipped']}")
    print(f"Report: {artifacts.report_path}")
    print(f"Mismatches: {artifacts.mismatches_path}")

    if args.print_mismatches and artifacts.mismatches:
        print(f"\n{len(artifacts.mismatches)} real mismatches:")
        for m in artifacts.mismatches:
            print(
                f"  {m['word']:20s}  target={m['target_phones']:40s}  "
                f"pred={m['pred_phones']:40s}  dist={m['edit_distance']}"
            )


if __name__ == "__main__":
    main()
