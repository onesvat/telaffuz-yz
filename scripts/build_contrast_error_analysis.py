#!/usr/bin/env python3
"""Build the contrast-level error analysis used by the paper.

Input:
  reports/assessment/validation-recordings-mms1b-phones.csv

Output:
  reports/assessment/contrast-error-analysis.md

Run:
  uv run python scripts/build_contrast_error_analysis.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "reports/assessment/validation-recordings-mms1b-phones.csv"
OUT_MD = ROOT / "reports/assessment/contrast-error-analysis.md"

CLASS_ORDER = ("vowel", "consonant", "missing")


def load_rows() -> list[dict[str, str]]:
    with IN_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def phone(phone_text: str) -> str:
    return phone_text or "∅"


def contrast(row: dict[str, str]) -> str:
    return f"{phone(row.get('expected', ''))}→{phone(row.get('observed', ''))}"


def contrast_class(row: dict[str, str]) -> str:
    if row.get("status") == "missing" or not row.get("observed"):
        return "missing"
    if row.get("expected_class") == "vowel":
        return "vowel"
    return "consonant"


def pct(n: int, d: int) -> str:
    return "—" if d == 0 else f"{(n / d) * 100:.2f}%"


def count_by_class(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        cls: defaultdict(int) for cls in CLASS_ORDER
    }
    for row in rows:
        cls = contrast_class(row)
        if row["kind"] in ("CTL", "W_NAT") and row["is_flagged"] == "1":
            out[cls]["correct_fp"] += 1
        if row["kind"] == "W_ERR" and row["is_intended_error_pos"] == "1":
            out[cls]["intended"] += 1
            if row["is_flagged"] == "1":
                out[cls]["caught"] += 1
    return out


def top_contrasts(
    rows: list[dict[str, str]],
    *,
    correct_speech_fp: bool = False,
    caught_intended: bool = False,
    limit: int = 8,
) -> list[tuple[str, str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if correct_speech_fp:
            keep = row["kind"] in ("CTL", "W_NAT") and row["is_flagged"] == "1"
        elif caught_intended:
            keep = (
                row["kind"] == "W_ERR"
                and row["is_intended_error_pos"] == "1"
                and row["is_flagged"] == "1"
            )
        else:
            keep = False
        if keep:
            counts[(contrast(row), contrast_class(row))] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0]))
    return [(c, cls, n) for (c, cls), n in ranked[:limit]]


def table_rows_top(items: list[tuple[str, str, int]]) -> list[str]:
    lines = ["| Contrast | Class | Count |", "|---|---|---:|"]
    lines.extend(f"| `{c}` | {cls} | {n} |" for c, cls, n in items)
    return lines


def main() -> int:
    rows = load_rows()
    split = count_by_class(rows)
    total_fp = sum(split[cls]["correct_fp"] for cls in CLASS_ORDER)
    total_intended = sum(split[cls]["intended"] for cls in CLASS_ORDER)
    total_caught = sum(split[cls]["caught"] for cls in CLASS_ORDER)

    top_fp = top_contrasts(rows, correct_speech_fp=True)
    top_caught = top_contrasts(rows, caught_intended=True)

    lines = [
        "# Contrast Error Analysis — MMS-1B Validation Phones",
        "",
        "Source: `reports/assessment/validation-recordings-mms1b-phones.csv`. "
        "Rows are the frozen two-speaker validation phones used by the paper. "
        "`∅` marks a missing target phone.",
        "",
        "## Class split",
        "",
        "| Class | Correct-speech FP | Share of FP | Intended W_ERR loci | Caught W_ERR loci | Detection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls in CLASS_ORDER:
        fp = split[cls]["correct_fp"]
        intended = split[cls]["intended"]
        caught = split[cls]["caught"]
        lines.append(
            f"| {cls} | {fp}/{total_fp} | {pct(fp, total_fp)} | "
            f"{intended}/{total_intended} | {caught}/{total_caught} | {pct(caught, intended)} |"
        )
    lines.extend(
        [
            "",
            "## Top correct-speech false-positive contrasts",
            "",
            *table_rows_top(top_fp),
            "",
            "## Top caught W_ERR contrasts",
            "",
            *table_rows_top(top_caught),
            "",
            "## Interpretation",
            "",
            "Correct-speech false positives are dominated by consonant or missing-phone "
            f"artifacts ({split['consonant']['correct_fp'] + split['missing']['correct_fp']}/"
            f"{total_fp}, {pct(split['consonant']['correct_fp'] + split['missing']['correct_fp'], total_fp)}). "
            "Caught intended-error loci are weighted toward vowel contrasts "
            f"({split['vowel']['caught']}/{total_caught}, {pct(split['vowel']['caught'], total_caught)}). "
            "This is the empirical basis for the paper's safety-first operating "
            "point: it abstains on recognizer-unreliable consonant and missing "
            "cases, which cuts false positives but necessarily gives up the "
            "consonant errors that the detection-first policy catches.",
            "",
        ]
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
