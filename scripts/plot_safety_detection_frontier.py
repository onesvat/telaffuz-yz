#!/usr/bin/env python3
"""Plot the paper's safety-detection frontier from the committed CSV.

Input:
  reports/assessment/safety-detection-frontier.csv

Output:
  figures/assessment/fig_safety_detection_frontier_en.{png,svg}
  figures/assessment/fig_safety_detection_frontier_tr.{png,svg}

Run:
  uv run python scripts/plot_safety_detection_frontier.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".uv-cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IN_CSV = ROOT / "reports/assessment/safety-detection-frontier.csv"
OUT_DIR = ROOT / "figures/assessment"
TEXT = {
    "en": {
        "policies": {
            "baseline": "Detection-first\n(no softening)",
            "allofonik": "Soften\nallophones",
            "vowel_trust": "Vowels\nonly",
            "vowel_trust_miss": "Safety-first\n(no missing)",
            "vowel_or_far_cons": "Vowel or\nfar cons.",
            "vowel_or_far_cons_miss": "Vowel/far cons.\n+ no missing",
        },
        "series": {
            "ctl_fp": "CTL false positive",
            "wnat_fp": "W_NAT false positive",
            "detection": "W_ERR detection",
            "precision": "W_ERR precision",
        },
        "baseline_note": "Detection-first\n65.62% detection\n5.29% / 5.54% FP",
        "safety_note": "Safety-first\n0.22% / 0.80% FP\n29.69% detection",
        "title": "Safety-detection frontier on the frozen native validation set",
        "ylabel": "Rate (%)",
        "xlabel": "Operating policy",
        "note": (
            "Note. FP = target phones flagged incorrect or missing on correct-speech prompts. "
            "All policies are deterministic post-hoc reinterpretations of the same 150 recordings."
        ),
    },
    "tr": {
        "policies": {
            "baseline": "Tespit öncelikli\n(yumuşatma yok)",
            "allofonik": "Allofonları\nyumuşat",
            "vowel_trust": "Yalnız\nünlüler",
            "vowel_trust_miss": "Güvenlik öncelikli\n(missing yok)",
            "vowel_or_far_cons": "Ünlü veya uzak\nünsüz",
            "vowel_or_far_cons_miss": "Ünlü/uzak ünsüz\n+ missing yok",
        },
        "series": {
            "ctl_fp": "CTL yanlış pozitif",
            "wnat_fp": "W_NAT yanlış pozitif",
            "detection": "W_ERR tespit",
            "precision": "W_ERR kesinlik",
        },
        "baseline_note": "Tespit öncelikli\n%65,62 tespit\n%5,29 / %5,54 FP",
        "safety_note": "Güvenlik öncelikli\n%0,22 / %0,80 FP\n%29,69 tespit",
        "title": "Dondurulmuş anadili doğrulama setinde güvenlik-tespit dengesi",
        "ylabel": "Oran (%)",
        "xlabel": "Çalışma politikası",
        "note": (
            "Not. FP = doğru-konuşma istemlerinde incorrect veya missing işaretlenen hedef fonemler. "
            "Tüm politikalar aynı 150 kaydın deterministik sonradan yorumlanmasıdır."
        ),
    },
}

SERIES = [
    ("ctl_fp", "#CC3311", "o"),
    ("wnat_fp", "#EE7733", "s"),
    ("detection", "#0077BB", "^"),
    ("precision", "#009988", "D"),
]


def load_rows() -> list[dict[str, str]]:
    with IN_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pct(row: dict[str, str], key: str) -> float:
    return float(row[key]) * 100.0


def plot(locale: str) -> None:
    rows = load_rows()
    x = list(range(len(rows)))
    text = TEXT[locale]
    labels = [text["policies"].get(row["band"], row["band"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    ax.axvspan(-0.35, 0.35, color="#E8F1FB", zorder=0)
    ax.axvspan(2.65, 3.35, color="#E7F5EF", zorder=0)

    for key, color, marker in SERIES:
        y = [pct(row, key) for row in rows]
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
            label=text["series"][key],
            zorder=3,
        )

    baseline = rows[0]
    safety = next(row for row in rows if row["band"] == "vowel_trust_miss")
    ax.annotate(
        text["baseline_note"],
        xy=(0, pct(baseline, "detection")),
        xytext=(0.55, 68),
        arrowprops={"arrowstyle": "->", "color": "#34495E", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#D0D7DE", "lw": 0.8},
        fontsize=8.5,
    )
    ax.annotate(
        text["safety_note"],
        xy=(3, pct(safety, "detection")),
        xytext=(3.55, 47),
        arrowprops={"arrowstyle": "->", "color": "#34495E", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#D0D7DE", "lw": 0.8},
        fontsize=8.5,
    )

    ax.set_title(text["title"], fontsize=13, pad=10)
    ax.set_ylabel(text["ylabel"])
    ax.set_xlabel(text["xlabel"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 82)
    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.8)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        fontsize=8.5,
    )

    fig.text(
        0.01,
        0.01,
        text["note"],
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.98))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"fig_safety_detection_frontier_{locale}.png"
    out_svg = OUT_DIR / f"fig_safety_detection_frontier_{locale}.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


def main() -> int:
    for locale in ("en", "tr"):
        plot(locale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
