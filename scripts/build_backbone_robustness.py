"""Build the paper-facing recogniser-backbone robustness table.

Inputs:
  reports/assessment/validation-recordings-mms1b.csv
  reports/assessment/validation-recordings-xlsr.csv

Outputs:
  reports/assessment/backbone-robustness.csv
  reports/assessment/backbone-robustness.md
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "reports/assessment"
OUT_CSV = IN_DIR / "backbone-robustness.csv"
OUT_MD = IN_DIR / "backbone-robustness.md"

MODELS = [
    ("mms-1b", "MMS-1B", "default", IN_DIR / "validation-recordings-mms1b.csv"),
    ("xls-r-300m", "XLS-R-300M", "robustness", IN_DIR / "validation-recordings-xlsr.csv"),
]


def _float(value: str | None) -> float | None:
    if value in ("", "None", None):
        return None
    return float(value)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _load_summary(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {row["kind"]: row for row in rows}


def _row(model_key: str, label: str, role: str, path: Path) -> dict[str, object]:
    by_kind = _load_summary(path)
    ctl = by_kind["CTL"]
    wnat = by_kind["W_NAT"]
    werr = by_kind["W_ERR"]
    return {
        "model": model_key,
        "label": label,
        "role": role,
        "ctl_flagged": int(ctl["flagged"]),
        "ctl_total": int(ctl["target_phones"]),
        "ctl_fp_rate": _float(ctl["hard_fp_rate"]),
        "wnat_flagged": int(wnat["flagged"]),
        "wnat_total": int(wnat["target_phones"]),
        "wnat_fp_rate": _float(wnat["hard_fp_rate"]),
        "caught": int(werr["caught"]),
        "intended_positions": int(werr["intended_positions"]),
        "detection_rate": _float(werr["detection_rate"]),
        "werr_flagged": int(werr["flagged"]),
        "precision_at_err": _float(werr["precision_at_err"]),
    }


def main() -> int:
    rows = [_row(*model) for model in MODELS]
    cols = [
        "model", "label", "role",
        "ctl_flagged", "ctl_total", "ctl_fp_rate",
        "wnat_flagged", "wnat_total", "wnat_fp_rate",
        "caught", "intended_positions", "detection_rate",
        "werr_flagged", "precision_at_err",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Recogniser Backbone Robustness",
        "",
        "Same 150 two-speaker validation recordings and the same status/score protocol; "
        "only the wav2vec backbone changes. MMS-1B remains the result-of-record default, "
        "while XLS-R-300M is reported as a robustness check.",
        "",
        "| Backbone | Role | CTL FP | W_NAT FP | Detection | Precision@err |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        ctl = f"{_pct(row['ctl_fp_rate'])} ({row['ctl_flagged']}/{row['ctl_total']})"
        wnat = f"{_pct(row['wnat_fp_rate'])} ({row['wnat_flagged']}/{row['wnat_total']})"
        det = f"{_pct(row['detection_rate'])} ({row['caught']}/{row['intended_positions']})"
        prec = f"{_pct(row['precision_at_err'])} ({row['caught']}/{row['werr_flagged']})"
        lines.append(
            f"| {row['label']} | {row['role']} | {ctl} | {wnat} | {det} | {prec} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"\nWrote {OUT_CSV} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
