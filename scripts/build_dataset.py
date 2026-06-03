#!/usr/bin/env python3
"""Build the wav2vec phoneme-CTC training manifest from data/audio.sqlite.

Gold pool: Common Voice (native splits) + ISSAI TSC.
Output: data/wav2vec/manifest_v5.csv  (269,625 rows; not committed — too large)

Usage:
    uv run python scripts/build_dataset.py
    uv run python scripts/build_dataset.py --dry-run
    uv run python scripts/build_dataset.py --filter-provider common_voice
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from audio.datasets import speaker_bucket  # noqa: E402
from audio.db import AUDIO_DB_PATH, connect  # noqa: E402
from g2p.pipeline import transcribe_training_text  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "wav2vec" / "manifest_v5.csv"
DEFAULT_STATS = REPO_ROOT / "artifacts" / "wav2vec" / "dataset_stats.md"

GOLD_PROVIDERS = ("common_voice", "issai_tsc")

RARE_PHONEMES = frozenset({"eː", "iː", "yː", "æ", "ɟ", "ɲ", "ŋ"})

PHONEME_INVENTORY = frozenset({
    "a", "e", "i", "ɯ", "o", "œ", "u", "y",
    "aː", "eː", "iː", "ɯː", "oː", "œː", "uː", "yː",
    "æ",
    "p", "b", "t", "d", "k", "ɡ", "t͡ʃ", "d͡ʒ", "f", "v",
    "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "ɾ", "j",
    "c", "ɟ", "ɲ", "ŋ", "ɫ", "β", "β̞",
    "pʰ", "tʰ", "kʰ", "cʰ", "ɾ̞̊",
    "ˈ",
})

CTC_SPECIAL_TOKENS = frozenset({"<spc>", "<pad>", "<unk>"})

MANIFEST_COLUMNS = [
    "segment_id", "path", "provider", "transcript", "phonemes",
    "phoneme_count", "split", "source", "role", "speaker_id",
    "speaker_source", "speaker_bucket", "duration_s", "rare_phoneme_count",
    "speaker_segment_count", "source_total_hours", "in_atlas",
]


@dataclass
class ManifestRow:
    segment_id: str
    path: str
    provider: str
    transcript: str
    phonemes: str
    phoneme_count: int
    split: str
    source: str
    role: str
    speaker_id: str
    speaker_source: str
    speaker_bucket: int
    duration_s: float
    rare_phoneme_count: int
    speaker_segment_count: int = 0
    source_total_hours: float = 0.0
    in_atlas: int = 1

    def as_dict(self) -> dict[str, object]:
        return {col: getattr(self, col) for col in MANIFEST_COLUMNS}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def phonemize(transcript: str) -> tuple[list[str], list[str]]:
    try:
        result = transcribe_training_text(transcript)
    except Exception as exc:
        return [], [f"g2p_exception:{type(exc).__name__}"]
    warnings = list(result.warnings)
    if result.drop_reason is not None:
        warnings.append(f"drop_reason:{result.drop_reason}")
        return [], warnings
    return list(result.tokens), warnings


def rare_phoneme_count(phonemes: Iterable[str]) -> int:
    return sum(1 for p in phonemes if p in RARE_PHONEMES)


def inventory_ok(phonemes: Iterable[str]) -> bool:
    return all(p in PHONEME_INVENTORY or p in CTC_SPECIAL_TOKENS for p in phonemes)


def fetch_gold_segments(
    conn: sqlite3.Connection,
    *,
    provider: str,
    min_duration: float,
    max_duration: float,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    sql = """
        SELECT
            s.id              AS id,
            s.path            AS path,
            s.duration_s      AS duration_s,
            s.speaker_id      AS speaker_id,
            s.speaker_source  AS speaker_source,
            s.publisher_split AS publisher_split,
            tc.text           AS transcript
        FROM segments s
        JOIN transcript_candidates tc
            ON tc.segment_id = s.id
           AND tc.source = 'publisher_gt'
           AND tc.is_primary = 1
        WHERE s.provider = ?
          AND s.duration_s BETWEEN ? AND ?
    """
    params: list[object] = [provider, min_duration, max_duration]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def precompute_phoneme_cache(
    transcripts: Iterable[str],
    *,
    show_progress: bool = True,
) -> dict[str, tuple[list[str], list[str]]]:
    unique_sorted = sorted(set(t for t in transcripts if t))
    cache: dict[str, tuple[list[str], list[str]]] = {}
    iterator = (
        tqdm(unique_sorted, desc="G2P unique", unit="text", smoothing=0.05)
        if show_progress else unique_sorted
    )
    for text in iterator:
        cache[text] = phonemize(text)
    return cache


def build_gold_pool(
    conn: sqlite3.Connection,
    *,
    min_duration: float,
    max_duration: float,
    filter_provider: set[str] | None = None,
    limit: int | None = None,
) -> tuple[list[ManifestRow], dict[str, object]]:
    stats: dict[str, object] = {
        "queried": {}, "kept": {},
        "dropped_no_transcript": {}, "dropped_no_phonemes": {},
        "dropped_bad_inventory": {}, "dropped_no_split": {},
        "dropped_sentence_leak": {"common_voice": 0, "issai_tsc": 0},
        "cv_other_total": 0, "cv_other_kept": 0, "cv_other_dropped_quality": 0,
        "g2p_unique_total": 0, "g2p_unique_failed": 0,
        "g2p_unique_warning_total": 0, "g2p_warning_counts": {},
    }

    rows_by_provider: dict[str, list[sqlite3.Row]] = {}
    for provider in GOLD_PROVIDERS:
        if filter_provider and provider not in filter_provider:
            continue
        rows = fetch_gold_segments(
            conn, provider=provider,
            min_duration=min_duration, max_duration=max_duration, limit=limit,
        )
        rows_by_provider[provider] = rows
        stats["queried"][provider] = len(rows)
        print(f"  Fetched {provider}: {len(rows):,} rows")

    unique_transcripts: set[str] = set()
    for rows in rows_by_provider.values():
        for row in rows:
            t = (row["transcript"] or "").strip()
            if t:
                unique_transcripts.add(t)
    print(f"  Unique transcripts: {len(unique_transcripts):,}")
    cache = precompute_phoneme_cache(unique_transcripts)
    stats["g2p_unique_total"] = len(cache)
    stats["g2p_unique_failed"] = sum(1 for phones, _ in cache.values() if not phones)
    g2p_warning_counts: Counter[str] = Counter(
        w for _, warnings in cache.values() for w in warnings
    )
    stats["g2p_unique_warning_total"] = sum(g2p_warning_counts.values())
    stats["g2p_warning_counts"] = dict(g2p_warning_counts.most_common())

    dev_test_sentences: set[str] = set()
    cv_other_rows: list[sqlite3.Row] = []
    for provider, rows in rows_by_provider.items():
        for row in rows:
            if row["publisher_split"] in ("dev", "test"):
                t = (row["transcript"] or "").strip()
                if t:
                    dev_test_sentences.add(t)

    print(f"  Dev/test sentence set: {len(dev_test_sentences):,} unique sentences")

    pool: list[ManifestRow] = []
    for provider, rows in rows_by_provider.items():
        kept = no_tr = no_ph = bad_inv = no_split = sentence_leaked = 0
        for row in rows:
            split_val = row["publisher_split"]
            if split_val in (None, ""):
                if provider == "common_voice":
                    cv_other_rows.append(row)
                else:
                    no_split += 1
                continue
            if split_val not in {"train", "dev", "test"}:
                no_split += 1
                continue
            transcript = (row["transcript"] or "").strip()
            if not transcript:
                no_tr += 1
                continue
            if split_val == "train" and transcript in dev_test_sentences:
                sentence_leaked += 1
                continue
            phones, _ = cache.get(transcript, ([], []))
            if not phones:
                no_ph += 1
                continue
            if not inventory_ok(phones):
                bad_inv += 1
                continue
            speaker_id = row["speaker_id"]
            pool.append(ManifestRow(
                segment_id=str(row["id"]), path=row["path"], provider=provider,
                transcript=transcript, phonemes=" ".join(phones),
                phoneme_count=len(phones), split=split_val,
                source="gold", role=f"{split_val}+atlas",
                speaker_id=speaker_id, speaker_source=row["speaker_source"],
                speaker_bucket=speaker_bucket(speaker_id),
                duration_s=float(row["duration_s"]),
                rare_phoneme_count=rare_phoneme_count(phones), in_atlas=1,
            ))
            kept += 1
        stats["kept"][provider] = kept
        stats["dropped_no_transcript"][provider] = no_tr
        stats["dropped_no_phonemes"][provider] = no_ph
        stats["dropped_bad_inventory"][provider] = bad_inv
        stats["dropped_no_split"][provider] = no_split
        stats["dropped_sentence_leak"][provider] = sentence_leaked

    stats["cv_other_total"] = len(cv_other_rows)
    cv_other_kept = cv_other_dropped_quality = 0
    cv_other_dropped_leak = 0
    for row in cv_other_rows:
        transcript = (row["transcript"] or "").strip()
        if not transcript:
            cv_other_dropped_quality += 1
            continue
        if transcript in dev_test_sentences:
            cv_other_dropped_leak += 1
            continue
        phones, _ = cache.get(transcript, ([], []))
        if not phones or not inventory_ok(phones):
            cv_other_dropped_quality += 1
            continue
        speaker_id = row["speaker_id"]
        pool.append(ManifestRow(
            segment_id=str(row["id"]), path=row["path"], provider="common_voice",
            transcript=transcript, phonemes=" ".join(phones),
            phoneme_count=len(phones), split="train",
            source="gold", role="train+atlas",
            speaker_id=speaker_id, speaker_source=row["speaker_source"],
            speaker_bucket=speaker_bucket(speaker_id),
            duration_s=float(row["duration_s"]),
            rare_phoneme_count=rare_phoneme_count(phones), in_atlas=1,
        ))
        cv_other_kept += 1
    stats["cv_other_kept"] = cv_other_kept
    stats["dropped_sentence_leak"]["common_voice"] = (
        stats["dropped_sentence_leak"].get("common_voice", 0) + cv_other_dropped_leak
    )
    stats["cv_other_dropped_quality"] = cv_other_dropped_quality

    return pool, stats


def enrich_weights(rows: list[ManifestRow]) -> dict[str, float]:
    speaker_counts: Counter[str] = Counter()
    provider_seconds: Counter[str] = Counter()
    for row in rows:
        speaker_counts[row.speaker_id] += 1
        provider_seconds[row.provider] += row.duration_s
    provider_hours = {p: round(s / 3600.0, 4) for p, s in provider_seconds.items()}
    for row in rows:
        row.speaker_segment_count = speaker_counts[row.speaker_id]
        row.source_total_hours = provider_hours.get(row.provider, 0.0)
    return provider_hours


def sort_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    return sorted(
        rows,
        key=lambda r: (
            r.split, r.source, r.provider,
            int(r.segment_id) if r.segment_id.isdigit() else r.segment_id,
        ),
    )


def write_manifest(rows: list[ManifestRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _bucket_hours(rows: list[ManifestRow], *, key) -> dict[str, float]:
    seconds: Counter[str] = Counter()
    for row in rows:
        seconds[key(row)] += row.duration_s
    return {k: round(v / 3600.0, 4) for k, v in sorted(seconds.items())}


def _bucket_counts(rows: list[ManifestRow], *, key) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[key(row)] += 1
    return dict(sorted(counts.items()))


def write_stats(
    rows: list[ManifestRow],
    stats_path: Path,
    *,
    gold_stats: dict[str, object],
    provider_hours: dict[str, float],
    args: argparse.Namespace,
    commit: str,
) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    hours_total = round(sum(r.duration_s for r in rows) / 3600.0, 4)
    hours_by_split = _bucket_hours(rows, key=lambda r: r.split)
    counts_by_split = _bucket_counts(rows, key=lambda r: r.split)
    hours_by_provider = _bucket_hours(rows, key=lambda r: r.provider)
    counts_by_provider = _bucket_counts(rows, key=lambda r: r.provider)

    train_speakers = {r.speaker_id for r in rows if r.split == "train"}
    dev_speakers = {r.speaker_id for r in rows if r.split == "dev"}
    test_speakers = {r.speaker_id for r in rows if r.split == "test"}

    lines: list[str] = [
        "# Dataset Manifest Statistics",
        "",
        f"- Generated: {now_iso()}",
        f"- Commit: `{commit}`",
        f"- Total rows: {n_total:,}",
        f"- Total hours: {hours_total:,.2f} h",
        "",
        "## Split",
        "",
        "| Split | Rows | Hours |",
        "|---|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        lines.append(f"| {split} | {counts_by_split.get(split, 0):,} | {hours_by_split.get(split, 0.0):,.2f} |")
    lines += [
        "",
        "## Provider",
        "",
        "| Provider | Rows | Hours |",
        "|---|---:|---:|",
    ]
    for provider, n in counts_by_provider.items():
        lines.append(f"| {provider} | {n:,} | {hours_by_provider.get(provider, 0.0):,.2f} |")
    lines += [
        "",
        "## Speaker leakage",
        "",
        f"- train ∩ dev:  {len(train_speakers & dev_speakers):,}",
        f"- train ∩ test: {len(train_speakers & test_speakers):,}",
        "",
        "## Build counters",
        "",
        "```json",
        json.dumps(gold_stats, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    stats_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=AUDIO_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--filter-provider", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    filter_provider = (
        {p.strip() for p in args.filter_provider.split(",")}
        if args.filter_provider else None
    )

    print(f"DB              : {args.db}")
    print(f"Output manifest : {args.out}")
    print(f"Duration filter : [{args.min_duration}, {args.max_duration}] s")
    print(f"Dry run         : {args.dry_run}")
    print()

    conn = connect(args.db, check_schema=True)
    try:
        print("Building gold pool (Common Voice + ISSAI TSC)...")
        gold_rows, gold_stats = build_gold_pool(
            conn,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            filter_provider=filter_provider,
            limit=args.limit,
        )
        print(f"  Gold rows: {len(gold_rows):,}")
        provider_hours = enrich_weights(gold_rows)
        sorted_rows = sort_rows(gold_rows)

        if args.dry_run:
            print("[dry-run] manifest NOT written.")
        else:
            print(f"Writing manifest: {args.out}")
            write_manifest(sorted_rows, args.out)

        commit = git_commit()
        print(f"Writing stats: {args.stats}")
        write_stats(
            sorted_rows, args.stats,
            gold_stats=gold_stats, provider_hours=provider_hours,
            args=args, commit=commit,
        )
    finally:
        conn.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
