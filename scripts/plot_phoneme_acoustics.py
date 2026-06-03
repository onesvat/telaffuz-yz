#!/usr/bin/env python3
"""Generate phoneme atlas figures in both English and Turkish.

Output files (figures/audio/):
  fig_phoneme_frequency_{en,tr}.png  — instance count per phone, log scale
  fig_vowel_space_{en,tr}.png        — F1 vs F2 scatter for 8 Turkish vowels
  fig_phoneme_confidence_{en,tr}.png — mean alignment confidence per phone
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

matplotlib.rcParams["font.family"] = "Noto Sans"

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "artifacts" / "audio" / "phone_summary.csv"
ACOUSTICS_CSV = REPO_ROOT / "artifacts" / "audio" / "phoneme_acoustics.csv"
OUT_DIR = REPO_ROOT / "figures" / "audio"

EVIDENCE_COLORS = {
    "model-observable": "#2196F3",
    "feature-derived": "#FF9800",
    "manual-only": "#9C27B0",
    "rule-only": "#4CAF50",
}

VOWELS = ["a", "e", "i", "ɯ", "o", "u", "y", "œ"]
VOWEL_COLORS = {
    "a":  "#E53935",
    "e":  "#FF7043",
    "i":  "#FFB300",
    "ɯ":  "#7CB342",
    "o":  "#00897B",
    "u":  "#1E88E5",
    "y":  "#8E24AA",
    "œ":  "#6D4C41",
}

STRINGS = {
    "en": {
        "freq_title": "Phone frequency in the Turkish corpus\n(49 phones, 22.4 M windows)",
        "freq_xlabel": "Instance count (log scale)",
        "evidence_legend": "Evidence type",
        "evidence_labels": {
            "model-observable": "Model Observable",
            "feature-derived":  "Feature Derived",
            "manual-only":      "Manual Only",
            "rule-only":        "Rule Only",
        },
        "vowel_title": "Turkish vowel space (F1 × F2)\nfrom aligned corpus",
        "vowel_xlabel": "F2 (Hz)",
        "vowel_ylabel": "F1 (Hz)",
        "vowel_legend": "Vowel",
        "conf_title": "Mean alignment confidence per phone\n(49 phones)",
        "conf_xlabel": "Mean alignment confidence",
        "conf_threshold": "0.85 threshold",
    },
    "tr": {
        "freq_title": "Türkçe korpustaki fonem sıklığı\n(49 fonem, 22,4 M pencere)",
        "freq_xlabel": "Örnek sayısı (logaritmik ölçek)",
        "evidence_legend": "Kanıt türü",
        "evidence_labels": {
            "model-observable": "Model Gözlemlenebilir",
            "feature-derived":  "Özellik Türetimli",
            "manual-only":      "Yalnız Manuel",
            "rule-only":        "Yalnız Kural",
        },
        "vowel_title": "Türkçe ünlü uzayı (F1 × F2)\nhizalamalı korpustan",
        "vowel_xlabel": "F2 (Hz)",
        "vowel_ylabel": "F1 (Hz)",
        "vowel_legend": "Ünlü",
        "conf_title": "Fonem başına ortalama hizalama güveni\n(49 fonem)",
        "conf_xlabel": "Ortalama hizalama güveni",
        "conf_threshold": "0,85 eşiği",
    },
}


def plot_frequency(summary: pd.DataFrame, lang: str) -> None:
    s = STRINGS[lang]
    df = summary[summary["expected_phone"] != "ˈ"].copy()
    df = df.sort_values("instance_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = [EVIDENCE_COLORS.get(et, "#999999") for et in df["evidence_type"]]
    ax.barh(df["expected_phone"], df["instance_count"], color=colors, height=0.7)

    ax.set_xscale("log")
    ax.set_xlabel(s["freq_xlabel"], fontsize=12)
    ax.set_title(s["freq_title"], fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for key, color in EVIDENCE_COLORS.items():
        ax.barh([], [], color=color, label=s["evidence_labels"][key])
    ax.legend(title=s["evidence_legend"], loc="lower right", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / f"fig_phoneme_frequency_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_vowel_space(acoustics: pd.DataFrame, lang: str) -> None:
    s = STRINGS[lang]
    df = acoustics[acoustics["expected_phone"].isin(VOWELS)].copy()
    df = df.dropna(subset=["f1_hz", "f2_hz"])

    fig, ax = plt.subplots(figsize=(8, 6))

    for vowel in VOWELS:
        sub = df[df["expected_phone"] == vowel]
        if sub.empty:
            continue
        color = VOWEL_COLORS.get(vowel, "#999999")
        ax.scatter(sub["f2_hz"], sub["f1_hz"], s=12, alpha=0.35, color=color, rasterized=True)
        mu_f1 = sub["f1_hz"].mean()
        mu_f2 = sub["f2_hz"].mean()
        ax.scatter(mu_f2, mu_f1, s=120, color=color, edgecolors="black",
                   linewidths=0.8, zorder=5, label=f"/{vowel}/")
        ax.annotate(
            f"/{vowel}/", (mu_f2, mu_f1),
            textcoords="offset points", xytext=(5, 3), fontsize=11,
            color=color, fontweight="bold",
        )

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(s["vowel_xlabel"], fontsize=12)
    ax.set_ylabel(s["vowel_ylabel"], fontsize=12)
    ax.set_title(s["vowel_title"], fontsize=13)
    ax.legend(title=s["vowel_legend"], loc="upper right", fontsize=9, markerscale=0.8)

    fig.tight_layout()
    out = OUT_DIR / f"fig_vowel_space_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_confidence(summary: pd.DataFrame, lang: str) -> None:
    s = STRINGS[lang]
    df = summary[summary["expected_phone"] != "ˈ"].copy()
    df = df.dropna(subset=["mean_confidence"])
    df = df.sort_values("mean_confidence", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = [EVIDENCE_COLORS.get(et, "#999999") for et in df["evidence_type"]]
    ax.barh(df["expected_phone"], df["mean_confidence"], color=colors, height=0.7)

    ax.set_xlim(0, 1.02)
    ax.axvline(0.85, color="gray", linestyle="--", linewidth=0.8, label=s["conf_threshold"])
    ax.set_xlabel(s["conf_xlabel"], fontsize=12)
    ax.set_title(s["conf_title"], fontsize=13)

    for key, color in EVIDENCE_COLORS.items():
        ax.barh([], [], color=color, label=s["evidence_labels"][key])
    ax.legend(title=s["evidence_legend"], loc="lower right", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / f"fig_phoneme_confidence_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV)
    acoustics = pd.read_csv(ACOUSTICS_CSV)

    for lang in ("en", "tr"):
        plot_frequency(summary, lang)
        plot_vowel_space(acoustics, lang)
        plot_confidence(summary, lang)


if __name__ == "__main__":
    main()
