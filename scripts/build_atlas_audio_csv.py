#!/usr/bin/env python3
"""Rebuild the committed atlas-audio CSV aggregates used by ``plot_atlas_audio.py``.

The two CSVs under ``artifacts/audio/`` are the DB-free inputs that the thesis
vowel figures (and the formant deltas quoted in Bulgular) are derived from. This
script regenerates them deterministically from the reference atlas so that the
figures, the CSVs and the prose all trace back to a single query.

Outputs (artifacts/audio/):
  vowel_formant_summary.csv — median F1/F2/F3 per phone (short + long vowels and
                              the lateral allophones /l/ and /ɫ/)
  vowel_space_sample.csv    — deterministic F1/F2 scatter sample for short vowels
                              and the laterals

Source: reference atlas ``coach_reference_features.sqlite`` (table
``phone_features``, ``feature_version = assess_coach_reference_v1``). The DB lives
only on the desktop machine, so run this there; the repo is NFS-shared, so the
CSVs land in the working tree directly::

    ssh desktop '/mnt/htpc/Code/telaffuz-yz-thesis/.venv/bin/python \
        scripts/build_atlas_audio_csv.py'

Plotting (``plot_atlas_audio.py``) reads only these CSVs and needs no DB.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "artifacts" / "audio"
DEFAULT_DB = Path("/home/onur/Code/telaffuz-yz-thesis-db/coach_reference_features.sqlite")
FEATURE_VERSION = "assess_coach_reference_v1"

# Short vowels, long vowels, then the lateral allophones. The laterals are
# included because Bulgular reports their F2 separation and the vowel-space
# figure visualises it; they are not vowels but share the F1/F2 feature surface.
SHORT_VOWELS = ["a", "e", "æ", "i", "ɯ", "o", "œ", "u", "y"]
LONG_VOWELS = ["aː", "eː", "iː", "oː", "œː"]
LATERALS = ["l", "ɫ"]
SUMMARY_PHONES = SHORT_VOWELS + LONG_VOWELS + LATERALS
SAMPLE_PHONES = SHORT_VOWELS + LATERALS

SAMPLE_PER_PHONE = 600
SAMPLE_SEED = 0


def load_formants(db_path: Path) -> pd.DataFrame:
    placeholders = ",".join("?" * len(SUMMARY_PHONES))
    sql = (
        "SELECT expected_phone, f1_hz, f2_hz, f3_hz FROM phone_features "
        f"WHERE feature_version = ? AND expected_phone IN ({placeholders}) "
        "AND f1_hz IS NOT NULL AND f2_hz IS NOT NULL"
    )
    with sqlite3.connect(str(db_path)) as con:
        return pd.read_sql(sql, con, params=[FEATURE_VERSION, *SUMMARY_PHONES])


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    med = (
        df.groupby("expected_phone")[["f1_hz", "f2_hz", "f3_hz"]]
        .median()
        .round(1)
        .reindex(SUMMARY_PHONES)
        .dropna(how="all")
    )
    med = med.reset_index().rename(
        columns={
            "expected_phone": "phone",
            "f1_hz": "f1_med_hz",
            "f2_hz": "f2_med_hz",
            "f3_hz": "f3_med_hz",
        }
    )
    return med


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phone in SAMPLE_PHONES:
        sub = df.loc[df["expected_phone"] == phone, ["expected_phone", "f1_hz", "f2_hz"]]
        if sub.empty:
            continue
        n = min(SAMPLE_PER_PHONE, len(sub))
        rows.append(sub.sample(n=n, random_state=SAMPLE_SEED).round(1))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"atlas DB not found: {args.db} (run on the desktop machine)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_formants(args.db)

    summary = build_summary(df)
    summary_path = OUT_DIR / "vowel_formant_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary)} phones)")

    sample = build_sample(df)
    sample_path = OUT_DIR / "vowel_space_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Wrote {sample_path} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
