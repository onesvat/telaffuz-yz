#!/usr/bin/env python3
"""Generate per-phoneme PER bar charts for XLS-R and MMS-1B in English and Turkish.

Figures (figures/wav2vec/):
  fig_per_phone_xlsr_{en,tr}.png  — XLS-R per-phoneme PER
  fig_per_phone_mms1b_{en,tr}.png — MMS-1B per-phoneme PER

Data sources (artifacts/wav2vec/):
  xlsr_test_per_phone.csv
  mms1b_test_per_phone.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Noto Sans"

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts" / "wav2vec"
OUT_DIR = REPO_ROOT / "figures" / "wav2vec"

# Phoneme evidence type → colour (same palette as phoneme atlas figures)
EVIDENCE_COLORS = {
    "model-observable": "#2196F3",
    "feature-derived":  "#FF9800",
    "manual-only":      "#9C27B0",
    "rule-only":        "#4CAF50",
}

# Hard-coded evidence type per phone (from phone_summary.csv)
EVIDENCE_TYPE: dict[str, str] = {
    "a": "model-observable", "e": "model-observable", "i": "model-observable",
    "ɯ": "model-observable", "o": "model-observable", "u": "model-observable",
    "y": "model-observable", "œ": "model-observable",
    "aː": "feature-derived", "eː": "feature-derived", "iː": "feature-derived",
    "ɯː": "feature-derived", "oː": "feature-derived", "uː": "feature-derived",
    "yː": "feature-derived", "œː": "feature-derived",
    "æ": "rule-only",
    "p": "model-observable", "b": "model-observable", "t": "model-observable",
    "d": "model-observable", "k": "model-observable", "ɡ": "model-observable",
    "t͡ʃ": "model-observable", "d͡ʒ": "model-observable", "f": "model-observable",
    "v": "model-observable", "s": "model-observable", "z": "model-observable",
    "ʃ": "model-observable", "ʒ": "model-observable", "h": "model-observable",
    "m": "model-observable", "n": "model-observable", "l": "model-observable",
    "ɾ": "model-observable", "j": "model-observable",
    "c": "model-observable", "ɟ": "model-observable", "ɲ": "model-observable",
    "ŋ": "model-observable", "ɫ": "model-observable",
    "β": "manual-only", "β̞": "manual-only",
    "pʰ": "manual-only", "tʰ": "manual-only", "kʰ": "manual-only",
    "cʰ": "manual-only", "ɾ̞̊": "manual-only",
    "ˈ": "feature-derived",
}

STRINGS = {
    "en": {
        "title_xlsr":  "XLS-R 300M — per-phoneme PER on test split",
        "title_mms1b": "MMS-1B — per-phoneme PER on test split",
        "xlabel":      "Phone Error Rate (%)",
        "ylabel":      "Phone",
        "legend":      "Evidence type",
        "evidence_labels": {
            "model-observable": "Model Observable",
            "feature-derived":  "Feature Derived",
            "manual-only":      "Manual Only",
            "rule-only":        "Rule Only",
        },
    },
    "tr": {
        "title_xlsr":  "XLS-R 300M — test split fonem hata oranı",
        "title_mms1b": "MMS-1B — test split fonem hata oranı",
        "xlabel":      "Fonem Hata Oranı (%)",
        "ylabel":      "Fonem",
        "legend":      "Kanıt türü",
        "evidence_labels": {
            "model-observable": "Model Gözlemlenebilir",
            "feature-derived":  "Özellik Türetimli",
            "manual-only":      "Yalnız Manuel",
            "rule-only":        "Yalnız Kural",
        },
    },
}


def load_per_phone(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["evidence_type"] = df["phone"].map(EVIDENCE_TYPE).fillna("model-observable")
    df["per_pct"] = df["per"] * 100
    df = df[df["ref_count"] >= 10].copy()
    df = df.sort_values("per_pct", ascending=True)
    return df


def plot_per_phone(df: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                   legend_title: str, evidence_labels: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.28 + 1.5)))

    colors = [EVIDENCE_COLORS.get(et, "#999999") for et in df["evidence_type"]]
    ax.barh(df["phone"], df["per_pct"], color=colors, height=0.7)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    for key, color in EVIDENCE_COLORS.items():
        ax.barh([], [], color=color, label=evidence_labels[key])
    ax.legend(title=legend_title, loc="lower right", fontsize=9)

    # Annotate low-count phones with n=
    for _, row in df.iterrows():
        if row["ref_count"] < 50:
            ax.text(row["per_pct"] + 0.3, row["phone"],
                    f"n={int(row['ref_count'])}", va="center", fontsize=6.5, color="#555")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    datasets = {
        "xlsr":  ARTIFACTS / "xlsr_test_per_phone.csv",
        "mms1b": ARTIFACTS / "mms1b_test_per_phone.csv",
    }

    for model_key, csv_path in datasets.items():
        if not csv_path.exists():
            print(f"SKIP (not found): {csv_path}")
            continue
        df = load_per_phone(csv_path)

        for lang in ("en", "tr"):
            s = STRINGS[lang]
            title = s[f"title_{model_key}"]
            out = OUT_DIR / f"fig_per_phone_{model_key}_{lang}.png"
            plot_per_phone(df, title=title, xlabel=s["xlabel"], ylabel=s["ylabel"],
                           legend_title=s["legend"], evidence_labels=s["evidence_labels"],
                           out_path=out)

    print("Done.")


if __name__ == "__main__":
    main()
