#!/usr/bin/env python3
"""Summarise and plot the phone feature corpus.

The script is intentionally read-only against ``phone_features.sqlite``. It
uses exact SQL aggregates for corpus counts and deterministic sampling for
distribution plots, because the full database has tens of millions of rows.

Typical desktop run:

  .venv/bin/python scripts/analyze_phone_features_corpus.py \
    --features-db /home/onur/Code/telaffuz-yz-thesis-db/phone_features.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Noto Sans"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path("/home/onur/Code/telaffuz-yz-thesis-db/phone_features.sqlite")
DEFAULT_FEATURE_VERSION = "phone_features_v3_policy"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "assessment"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "audio"

VOWELS = ["a", "e", "æ", "i", "ɯ", "o", "œ", "u", "y", "aː", "eː", "iː", "oː", "œː"]
SHORT_VOWELS = ["a", "e", "æ", "i", "ɯ", "o", "œ", "u", "y"]
FEATURE_COLUMNS = [
    "duration_ms",
    "analysis_duration_ms",
    "voiced_fraction",
    "intensity_db",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
]


def _read_sql(conn: sqlite3.Connection, sql: str, params: Iterable[object]) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=list(params))


def _safe_name(name: str) -> str:
    return (
        name.replace("ː", "long")
        .replace("͡", "")
        .replace("̞", "")
        .replace("̊", "devoiced")
        .replace("/", "_")
    )


def load_exact_summaries(
    conn: sqlite3.Connection,
    *,
    feature_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_by_version = _read_sql(
        conn,
        """
        SELECT feature_version, COUNT(*) AS n
        FROM phone_features
        GROUP BY feature_version
        ORDER BY n DESC
        """,
        [],
    )
    provider_by_version = _read_sql(
        conn,
        """
        SELECT feature_version, provider, COUNT(*) AS n
        FROM phone_features
        GROUP BY feature_version, provider
        ORDER BY feature_version, n DESC
        """,
        [],
    )
    per_phone = _read_sql(
        conn,
        """
        SELECT
          expected_phone,
          COUNT(*) AS n,
          COUNT(DISTINCT speaker_id) AS n_speakers,
          SUM(CASE WHEN provider = 'common_voice' THEN 1 ELSE 0 END) AS n_common_voice,
          SUM(CASE WHEN provider = 'issai_tsc' THEN 1 ELSE 0 END) AS n_issai_tsc,
          SUM(CASE WHEN provider = 'audiobooks' THEN 1 ELSE 0 END) AS n_audiobooks,
          SUM(formant_attempted) AS formant_attempted,
          SUM(formant_success) AS formant_success,
          AVG(duration_ms) AS mean_duration_ms,
          AVG(analysis_duration_ms) AS mean_analysis_duration_ms,
          AVG(voiced_fraction) AS mean_voiced_fraction,
          AVG(intensity_db) AS mean_intensity_db,
          AVG(f1_hz) AS mean_f1_hz,
          AVG(f2_hz) AS mean_f2_hz,
          AVG(f3_hz) AS mean_f3_hz,
          AVG(spectral_centroid_hz) AS mean_spectral_centroid_hz,
          AVG(spectral_bandwidth_hz) AS mean_spectral_bandwidth_hz
        FROM phone_features
        WHERE feature_version = ?
        GROUP BY expected_phone
        ORDER BY n DESC
        """,
        [feature_version],
    )
    per_provider_phone = _read_sql(
        conn,
        """
        SELECT
          provider,
          expected_phone,
          COUNT(*) AS n,
          SUM(formant_success) AS formant_success,
          AVG(duration_ms) AS mean_duration_ms,
          AVG(voiced_fraction) AS mean_voiced_fraction,
          AVG(f1_hz) AS mean_f1_hz,
          AVG(f2_hz) AS mean_f2_hz,
          AVG(f3_hz) AS mean_f3_hz
        FROM phone_features
        WHERE feature_version = ?
        GROUP BY provider, expected_phone
        ORDER BY provider, n DESC
        """,
        [feature_version],
    )
    per_phone["formant_success_rate"] = per_phone["formant_success"] / per_phone["formant_attempted"]
    per_provider_phone["formant_success_rate"] = (
        per_provider_phone["formant_success"] / per_provider_phone["n"]
    )
    return total_by_version, provider_by_version, per_phone, per_provider_phone


def load_sample(
    conn: sqlite3.Connection,
    *,
    feature_version: str,
    sample_mod: int,
) -> pd.DataFrame:
    cols = ", ".join(["provider", "expected_phone", *FEATURE_COLUMNS])
    sql = f"""
        SELECT {cols}
        FROM phone_features
        WHERE feature_version = ?
          AND ABS((segment_alignment_id * 31 + phone_index * 17) % ?) = 0
    """
    return _read_sql(conn, sql, [feature_version, sample_mod])


def build_quantiles(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phone, group in sample.groupby("expected_phone", sort=False):
        row: dict[str, object] = {"expected_phone": phone, "sample_n": len(group)}
        for col in FEATURE_COLUMNS:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_n"] = len(vals)
            if len(vals):
                q = vals.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
                for label, val in q.items():
                    row[f"{col}_p{int(label * 100):02d}"] = float(val)
            else:
                for label in (5, 10, 25, 50, 75, 90, 95):
                    row[f"{col}_p{label:02d}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_phone_counts(per_phone: pd.DataFrame, out: Path) -> None:
    df = per_phone.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(df["expected_phone"], df["n"], color="#2F6B8F")
    ax.set_xscale("log")
    ax.set_xlabel("Örnek sayısı (log ölçek)")
    ax.set_ylabel("Fonem")
    ax.set_title("Phone-feature corpus: fonem başına örnek sayısı")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_formant_success(per_phone: pd.DataFrame, out: Path) -> None:
    df = per_phone.sort_values("formant_success_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = np.where(df["formant_success_rate"] >= 0.75, "#2D7D46", "#B84A3A")
    ax.barh(df["expected_phone"], df["formant_success_rate"], color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Formant success oranı")
    ax.set_ylabel("Fonem")
    ax.set_title("Fonem başına formant ölçüm başarısı")
    ax.axvline(0.75, color="#555555", linestyle="--", linewidth=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_interval_quantiles(
    quantiles: pd.DataFrame,
    *,
    column: str,
    xlabel: str,
    title: str,
    out: Path,
) -> None:
    df = quantiles.sort_values(f"{column}_p50", ascending=True)
    y = np.arange(len(df))
    med = df[f"{column}_p50"].to_numpy(dtype=float)
    p10 = df[f"{column}_p10"].to_numpy(dtype=float)
    p90 = df[f"{column}_p90"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.hlines(y, p10, p90, color="#9AA7B2", linewidth=2)
    ax.scatter(med, y, color="#1B4F72", s=22, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["expected_phone"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fonem")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_vowel_space(sample: pd.DataFrame, out: Path) -> None:
    df = sample[sample["expected_phone"].isin(SHORT_VOWELS)].copy()
    df = df.dropna(subset=["f1_hz", "f2_hz"])
    if len(df) > 120_000:
        df = df.sample(120_000, random_state=0)
    colors = {
        "a": "#C0392B",
        "e": "#D35400",
        "æ": "#8E44AD",
        "i": "#D4AC0D",
        "ɯ": "#27AE60",
        "o": "#117A65",
        "œ": "#7D6608",
        "u": "#21618C",
        "y": "#6C3483",
    }
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for phone in SHORT_VOWELS:
        sub = df[df["expected_phone"] == phone]
        if sub.empty:
            continue
        ax.scatter(
            sub["f2_hz"],
            sub["f1_hz"],
            s=5,
            alpha=0.12,
            color=colors.get(phone, "#666666"),
            rasterized=True,
        )
        ax.scatter(
            sub["f2_hz"].median(),
            sub["f1_hz"].median(),
            s=105,
            color=colors.get(phone, "#666666"),
            edgecolor="black",
            linewidth=0.8,
            label=f"/{phone}/",
        )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.set_title("Kısa ünlüler için F1-F2 uzayı")
    ax.legend(title="Ünlü", ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_vowel_formants(quantiles: pd.DataFrame, out: Path) -> None:
    df = quantiles[quantiles["expected_phone"].isin(VOWELS)].copy()
    present = [p for p in VOWELS if p in set(df["expected_phone"])]
    df["expected_phone"] = pd.Categorical(df["expected_phone"], categories=present, ordered=True)
    df = df.sort_values("expected_phone")
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, col, label, color in [
        (-width, "f1_hz_p50", "F1", "#C0392B"),
        (0.0, "f2_hz_p50", "F2", "#1F618D"),
        (width, "f3_hz_p50", "F3", "#117A65"),
    ]:
        ax.bar(x + offset, df[col], width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"/{p}/" for p in df["expected_phone"]])
    ax.set_ylabel("Medyan frekans (Hz)")
    ax.set_title("Ünlülerde medyan F1/F2/F3 dağılımı")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    *,
    path: Path,
    db_path: Path,
    feature_version: str,
    sample_mod: int,
    total_by_version: pd.DataFrame,
    provider_by_version: pd.DataFrame,
    per_phone: pd.DataFrame,
    quantiles: pd.DataFrame,
) -> None:
    current_total = int(
        total_by_version.loc[
            total_by_version["feature_version"] == feature_version, "n"
        ].iloc[0]
    )
    all_total = int(total_by_version["n"].sum())
    provider_current = provider_by_version[
        provider_by_version["feature_version"] == feature_version
    ].copy()
    top = per_phone.sort_values("n", ascending=False).head(12)
    low_formant = per_phone.sort_values("formant_success_rate", ascending=True).head(12)
    vowel_rows = quantiles[quantiles["expected_phone"].isin(SHORT_VOWELS)].copy()
    vowel_rows["expected_phone"] = pd.Categorical(
        vowel_rows["expected_phone"], categories=SHORT_VOWELS, ordered=True
    )
    vowel_rows = vowel_rows.sort_values("expected_phone")

    lines = [
        "# Phone Feature Corpus Summary",
        "",
        f"- Database: `{db_path}`",
        f"- Active feature version: `{feature_version}`",
        f"- Total rows across all versions: {all_total:,}",
        f"- Rows in active feature version: {current_total:,}",
        f"- Distinct phones in active feature version: {per_phone['expected_phone'].nunique()}",
        f"- Deterministic plot sample: every 1/{sample_mod} rows by alignment id modulo",
        "",
        "## Feature versions",
        "",
        "| feature_version | rows |",
        "|---|---:|",
    ]
    for row in total_by_version.itertuples(index=False):
        lines.append(f"| `{row.feature_version}` | {int(row.n):,} |")
    lines.extend(["", "## Providers in active version", "", "| provider | rows |", "|---|---:|"])
    for row in provider_current.itertuples(index=False):
        lines.append(f"| `{row.provider}` | {int(row.n):,} |")
    lines.extend(
        [
            "",
            "## Largest phones",
            "",
            "| phone | rows | speakers | formant_success | mean duration ms | mean voiced fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(
            f"| /{row.expected_phone}/ | {int(row.n):,} | {int(row.n_speakers):,} | "
            f"{row.formant_success_rate:.3f} | {row.mean_duration_ms:.1f} | "
            f"{row.mean_voiced_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Lowest formant-success phones",
            "",
            "| phone | rows | formant_success | mean voiced fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in low_formant.itertuples(index=False):
        lines.append(
            f"| /{row.expected_phone}/ | {int(row.n):,} | "
            f"{row.formant_success_rate:.3f} | {row.mean_voiced_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Short vowel sample medians",
            "",
            "| vowel | n_sample | F1 p50 | F2 p50 | F3 p50 | duration p50 | voiced p50 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in vowel_rows.itertuples(index=False):
        lines.append(
            f"| /{row.expected_phone}/ | {int(row.sample_n):,} | "
            f"{row.f1_hz_p50:.1f} | {row.f2_hz_p50:.1f} | {row.f3_hz_p50:.1f} | "
            f"{row.duration_ms_p50:.1f} | {row.voiced_fraction_p50:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Generated figures",
            "",
            "- `figures/audio/fig_phone_feature_counts_tr.png`",
            "- `figures/audio/fig_phone_feature_formant_success_tr.png`",
            "- `figures/audio/fig_phone_feature_duration_tr.png`",
            "- `figures/audio/fig_phone_feature_voiced_fraction_tr.png`",
            "- `figures/audio/fig_phone_feature_vowel_space_tr.png`",
            "- `figures/audio/fig_phone_feature_vowel_formants_tr.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    ap.add_argument("--sample-mod", type=int, default=25)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = ap.parse_args()

    if args.sample_mod <= 0:
        ap.error("--sample-mod must be positive")
    if not args.features_db.exists():
        print(f"missing database: {args.features_db}", file=sys.stderr)
        return 1

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(args.features_db))
    try:
        total_by_version, provider_by_version, per_phone, per_provider_phone = load_exact_summaries(
            conn,
            feature_version=args.feature_version,
        )
        if args.feature_version not in set(total_by_version["feature_version"]):
            print(f"feature version not found: {args.feature_version}", file=sys.stderr)
            return 1
        sample = load_sample(conn, feature_version=args.feature_version, sample_mod=args.sample_mod)
    finally:
        conn.close()

    quantiles = build_quantiles(sample)
    per_phone_out = args.report_dir / "phone-feature-per-phone.csv"
    provider_phone_out = args.report_dir / "phone-feature-provider-phone.csv"
    quantile_out = args.report_dir / "phone-feature-sample-quantiles.csv"
    summary_md = args.report_dir / "phone-feature-corpus-summary.md"

    per_phone.to_csv(per_phone_out, index=False)
    per_provider_phone.to_csv(provider_phone_out, index=False)
    quantiles.to_csv(quantile_out, index=False)

    plot_phone_counts(per_phone, args.figure_dir / "fig_phone_feature_counts_tr.png")
    plot_formant_success(per_phone, args.figure_dir / "fig_phone_feature_formant_success_tr.png")
    plot_interval_quantiles(
        quantiles,
        column="duration_ms",
        xlabel="Duration (ms), p10-p90 ve p50",
        title="Fonem başına duration dağılımı",
        out=args.figure_dir / "fig_phone_feature_duration_tr.png",
    )
    plot_interval_quantiles(
        quantiles,
        column="voiced_fraction",
        xlabel="Voiced fraction, p10-p90 ve p50",
        title="Fonem başına voiced fraction dağılımı",
        out=args.figure_dir / "fig_phone_feature_voiced_fraction_tr.png",
    )
    plot_vowel_space(sample, args.figure_dir / "fig_phone_feature_vowel_space_tr.png")
    plot_vowel_formants(quantiles, args.figure_dir / "fig_phone_feature_vowel_formants_tr.png")

    write_markdown(
        path=summary_md,
        db_path=args.features_db,
        feature_version=args.feature_version,
        sample_mod=args.sample_mod,
        total_by_version=total_by_version,
        provider_by_version=provider_by_version,
        per_phone=per_phone,
        quantiles=quantiles,
    )

    print(f"sample rows: {len(sample):,}")
    print(f"wrote {per_phone_out}")
    print(f"wrote {provider_phone_out}")
    print(f"wrote {quantile_out}")
    print(f"wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
