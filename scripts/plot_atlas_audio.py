#!/usr/bin/env python3
"""Generate the atlas vowel figures referenced by the thesis, bilingually.

Output files (figures/audio/):
  fig_phone_feature_vowel_formants_{en,tr}.png — median F1/F2/F3 per vowel
  fig_phone_feature_vowel_space_{en,tr}.png    — F1 vs F2 scatter for short vowels

Data sources (committed, DB-free, so the script runs on either machine):
  artifacts/audio/vowel_formant_summary.csv — per-vowel median F1/F2/F3
  artifacts/audio/vowel_space_sample.csv    — short-vowel F1/F2 scatter sample

Both aggregates are derived from the reference atlas (coach_reference_features
phone_features). Both language variants are produced from the same data, so the
_en and _tr figures are guaranteed consistent.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Noto Sans"

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMANT_CSV = REPO_ROOT / "artifacts" / "audio" / "vowel_formant_summary.csv"
SPACE_CSV = REPO_ROOT / "artifacts" / "audio" / "vowel_space_sample.csv"
OUT_DIR = REPO_ROOT / "figures" / "audio"

SHORT_VOWELS = ["a", "e", "æ", "i", "ɯ", "o", "œ", "u", "y"]
LATERALS = ["l", "ɫ"]

VOWEL_COLORS = {
    "a": "#E53935", "e": "#FF7043", "æ": "#F4511E", "i": "#FFB300",
    "ɯ": "#7CB342", "o": "#00897B", "œ": "#6D4C41", "u": "#1E88E5",
    "y": "#8E24AA", "l": "#00838F", "ɫ": "#3949AB",
}
FORMANT_COLORS = {"F1": "#C62828", "F2": "#1565C0", "F3": "#00897B"}

STRINGS = {
    "en": {
        "formant_title": "Median F1/F2/F3 across Turkish vowels and laterals",
        "formant_ylabel": "Median frequency (Hz)",
        "space_title": "F1-F2 space for short vowels and laterals",
        "space_xlabel": "F2 (Hz)",
        "space_ylabel": "F1 (Hz)",
        "vowel_legend": "Phone",
    },
    "tr": {
        "formant_title": "Ünlü ve lateral fonemlerde medyan F1/F2/F3 dağılımı",
        "formant_ylabel": "Medyan frekans (Hz)",
        "space_title": "Kısa ünlüler ve lateraller için F1-F2 uzayı",
        "space_xlabel": "F2 (Hz)",
        "space_ylabel": "F1 (Hz)",
        "vowel_legend": "Fonem",
    },
}


def plot_vowel_formants(summary: pd.DataFrame, lang: str) -> None:
    s = STRINGS[lang]
    vowels = summary["phone"].tolist()
    x = np.arange(len(vowels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, summary["f1_med_hz"], width, color=FORMANT_COLORS["F1"], label="F1")
    ax.bar(x, summary["f2_med_hz"], width, color=FORMANT_COLORS["F2"], label="F2")
    ax.bar(x + width, summary["f3_med_hz"], width, color=FORMANT_COLORS["F3"], label="F3")

    ax.set_xticks(x)
    ax.set_xticklabels([f"/{v}/" for v in vowels])
    ax.set_ylabel(s["formant_ylabel"], fontsize=12)
    ax.set_title(s["formant_title"], fontsize=13)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out = OUT_DIR / f"fig_phone_feature_vowel_formants_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_vowel_space(sample: pd.DataFrame, lang: str) -> None:
    s = STRINGS[lang]
    fig, ax = plt.subplots(figsize=(8, 6))
    for v in SHORT_VOWELS + LATERALS:
        sub = sample[sample["expected_phone"] == v]
        if sub.empty:
            continue
        color = VOWEL_COLORS.get(v, "#999999")
        # Laterals share the F1/F2 plane but are not vowels; mark them with
        # squares so the /l/-/ɫ/ F2 separation reads distinctly from the cloud.
        marker = "s" if v in LATERALS else "o"
        ax.scatter(sub["f2_hz"], sub["f1_hz"], s=10, alpha=0.30, color=color,
                   marker=marker, rasterized=True)
        mu_f1, mu_f2 = sub["f1_hz"].median(), sub["f2_hz"].median()
        ax.scatter(mu_f2, mu_f1, s=140, color=color, edgecolors="black",
                   linewidths=0.9, zorder=5, marker=marker, label=f"/{v}/")
        ax.annotate(f"/{v}/", (mu_f2, mu_f1), textcoords="offset points",
                    xytext=(6, 4), fontsize=12, color=color, fontweight="bold")

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(s["space_xlabel"], fontsize=12)
    ax.set_ylabel(s["space_ylabel"], fontsize=12)
    ax.set_title(s["space_title"], fontsize=13)
    ax.legend(title=s["vowel_legend"], loc="upper right", fontsize=9, markerscale=0.8)

    fig.tight_layout()
    out = OUT_DIR / f"fig_phone_feature_vowel_space_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(FORMANT_CSV)
    sample = pd.read_csv(SPACE_CSV)
    for lang in ("en", "tr"):
        plot_vowel_formants(summary, lang)
        plot_vowel_space(sample, lang)


if __name__ == "__main__":
    main()
