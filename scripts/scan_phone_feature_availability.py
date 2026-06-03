#!/usr/bin/env python3
"""Per-phone × per-feature availability scan over ``phone_features.sqlite``.

Single-pass read-only stream over the canonical pool. Accumulates per-phone
non-null counts, test-bucket counts (speaker-disjoint), reservoir-style first-K
sample values for percentile estimation, and MFCC null/dim metadata.

Canonical pool: ``feature_version = phone_features_v3_policy`` AND
``provider IN (common_voice, issai_tsc, audiobooks)`` — no
``formant_success`` filter, because non-formant features must be evaluated
separately. The report is canonical-side-only; no wrong-side audit data is
read or joined.

Outputs:
  reports/assessment/phone-feature-availability.csv
  reports/assessment/phone-feature-availability.md

Run on the desktop:

  ssh desktop "cd ~/Code/telaffuz-yz-thesis && \\
    .venv/bin/python scripts/scan_phone_feature_availability.py \\
      --features-db /home/onur/Code/telaffuz-yz-thesis-db/phone_features.sqlite"
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from audio.datasets import speaker_bucket  # noqa: E402

DEFAULT_FEATURE_VERSION = "phone_features_v3_policy"
DEFAULT_PROVIDERS = ("common_voice", "issai_tsc", "audiobooks")
DEFAULT_OUT_CSV = REPO_ROOT / "reports" / "assessment" / "phone-feature-availability.csv"
DEFAULT_OUT_MD = REPO_ROOT / "reports" / "assessment" / "phone-feature-availability.md"

SCALAR_FEATURES: tuple[str, ...] = (
    "rms",
    "intensity_db",
    "f0_hz",
    "f0_median_hz",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "voiced_fraction",
    "voiced_frame_count",
    "pitch_frame_count",
    "leading_unvoiced_ms",
    "trailing_unvoiced_ms",
    "duration_ms",
    "analysis_duration_ms",
    "alignment_confidence",
)

SAMPLE_CAP_PER_PAIR = 5_000  # first-K non-null values per (phone, feature)
TEST_BUCKETS = range(95, 100)
ACTIVATION_MIN_N = 100
FETCH_BATCH = 10_000


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {q: float("nan") for q in ("p5", "p25", "p50", "p75", "p95")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _new_phone_record() -> dict[str, Any]:
    return {
        "n_total": 0,
        "n_test": 0,
        "non_null": {c: 0 for c in SCALAR_FEATURES},
        "samples": {c: [] for c in SCALAR_FEATURES},
        "mfcc_non_null": 0,
    }


def _stream_canonical(
    conn: sqlite3.Connection,
    *,
    feature_version: str,
    providers: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    placeholders = ",".join("?" for _ in providers)
    column_list = ", ".join(SCALAR_FEATURES)
    sql = f"""
        SELECT expected_phone, speaker_id, {column_list}, mfcc_json
        FROM phone_features
        WHERE feature_version = ?
          AND provider IN ({placeholders})
    """
    print("  preparing query ...", flush=True)
    cur = conn.execute(sql, [feature_version, *providers])
    print("  streaming rows ...", flush=True)

    per_phone: dict[str, dict[str, Any]] = defaultdict(_new_phone_record)
    mfcc_dims: list[int] = []
    mfcc_seen = 0
    mfcc_non_null_global = 0

    n_seen = 0
    t_last = time.time()
    t_start = t_last
    feature_count = len(SCALAR_FEATURES)
    mfcc_idx = 2 + feature_count

    while True:
        batch = cur.fetchmany(FETCH_BATCH)
        if not batch:
            break
        for row in batch:
            phone = str(row[0])
            speaker_id = str(row[1])
            info = per_phone[phone]
            info["n_total"] += 1
            if speaker_bucket(speaker_id) in TEST_BUCKETS:
                info["n_test"] += 1
            for i, feat in enumerate(SCALAR_FEATURES):
                v = row[2 + i]
                if v is not None:
                    info["non_null"][feat] += 1
                    samples = info["samples"][feat]
                    if len(samples) < SAMPLE_CAP_PER_PAIR:
                        samples.append(float(v))
            mfcc_cell = row[mfcc_idx]
            mfcc_seen += 1
            if mfcc_cell:
                info["mfcc_non_null"] += 1
                mfcc_non_null_global += 1
                if len(mfcc_dims) < 50:
                    try:
                        parsed = json.loads(mfcc_cell)
                        if isinstance(parsed, list):
                            mfcc_dims.append(len(parsed))
                    except Exception:
                        pass
            n_seen += 1
        now = time.time()
        if now - t_last > 5:
            rps = n_seen / max(now - t_start, 0.001)
            print(
                f"  ...{n_seen:,} rows | {len(per_phone)} phones | {rps:,.0f} rows/s",
                flush=True,
            )
            t_last = now

    elapsed = time.time() - t_start
    print(
        f"  done: {n_seen:,} rows in {elapsed:.1f}s "
        f"({n_seen / max(elapsed, 0.001):,.0f} rows/s) "
        f"across {len(per_phone)} phones",
        flush=True,
    )
    mfcc_meta = {
        "rows_seen": mfcc_seen,
        "rows_non_null": mfcc_non_null_global,
        "null_rate": (mfcc_seen - mfcc_non_null_global) / max(mfcc_seen, 1),
        "dimensions_observed": sorted(set(mfcc_dims)),
    }
    return dict(per_phone), mfcc_meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-db", type=Path, required=True)
    ap.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    ap.add_argument("--providers", nargs="+", default=list(DEFAULT_PROVIDERS))
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"connecting to {args.features_db}", flush=True)
    conn = sqlite3.connect(str(args.features_db))
    try:
        per_phone, mfcc_meta = _stream_canonical(
            conn,
            feature_version=args.feature_version,
            providers=tuple(args.providers),
        )
    finally:
        conn.close()

    # Long-format CSV: one row per (phone, feature).
    with args.out_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "phone", "feature",
            "n_canonical_total", "n_canonical_non_null", "nan_rate_canonical",
            "n_canonical_test",
            "p5", "p25", "p50", "p75", "p95",
        ])
        for phone in sorted(per_phone.keys()):
            info = per_phone[phone]
            n_total = info["n_total"]
            n_test = info["n_test"]
            for feat in SCALAR_FEATURES:
                n_nn = info["non_null"][feat]
                nan_rate = 1.0 - (n_nn / max(n_total, 1))
                pct = _percentiles(info["samples"][feat])
                w.writerow([
                    phone, feat,
                    n_total, n_nn, f"{nan_rate:.4f}",
                    n_test,
                    f"{pct['p5']:.4f}", f"{pct['p25']:.4f}",
                    f"{pct['p50']:.4f}", f"{pct['p75']:.4f}",
                    f"{pct['p95']:.4f}",
                ])
            n_mfcc = info["mfcc_non_null"]
            w.writerow([
                phone, "mfcc_json",
                n_total, n_mfcc,
                f"{1.0 - n_mfcc / max(n_total, 1):.4f}",
                n_test, "", "", "", "", "",
            ])
    print(f"wrote {args.out_csv}", flush=True)

    # Markdown summary.
    lines: list[str] = [
        "# Phone × feature availability (canonical-side)",
        "",
        f"- DB: `{args.features_db}`",
        f"- Feature version: `{args.feature_version}`",
        f"- Providers (canonical): {', '.join(args.providers)}",
        f"- Per-(phone, feature) sample cap: {SAMPLE_CAP_PER_PAIR:,}",
        f"- Activation candidate rule: n_canonical_test ≥ {ACTIVATION_MIN_N}",
        "",
        "## MFCC corpus probe",
        "",
        f"- Rows scanned (canonical): {mfcc_meta['rows_seen']:,}",
        f"- Non-null `mfcc_json` rows: {mfcc_meta['rows_non_null']:,}",
        f"- Null rate: {mfcc_meta['null_rate']:.3f}",
        f"- Dimensions observed: {mfcc_meta['dimensions_observed']}",
        "",
        "## Per-phone activation candidacy",
        "",
        "| phone | n_canonical_total | n_canonical_test | passes_activation |",
        "|---|---:|---:|:---:|",
    ]
    for phone in sorted(per_phone.keys(), key=lambda p: -per_phone[p]["n_test"]):
        info = per_phone[phone]
        n_total = info["n_total"]
        n_test = info["n_test"]
        passes = "yes" if n_test >= ACTIVATION_MIN_N else "no"
        lines.append(
            f"| /{phone}/ | {n_total:,} | {n_test:,} | {passes} |"
        )

    lines.extend([
        "",
        "## Per-feature canonical NaN rates (1 − n_non_null / n_total)",
        "",
        "Lower is better. A feature whose NaN rate exceeds ~0.10 for a phone is "
        "downstream-dropped from that phone's density model.",
        "",
    ])
    header = ["phone"] + list(SCALAR_FEATURES) + ["mfcc_json"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for phone in sorted(per_phone.keys()):
        info = per_phone[phone]
        n_total = max(info["n_total"], 1)
        cells = [f"/{phone}/"]
        for feat in SCALAR_FEATURES:
            rate = 1.0 - (info["non_null"][feat] / n_total)
            cells.append(f"{rate:.3f}")
        cells.append(f"{1.0 - info['mfcc_non_null'] / n_total:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
