"""Phoneme atlas manifest and label-layer helpers.

The atlas treats G2P output as the canonical expected_49 hypothesis and keeps
audio evidence in separate layers:

- alignment_target: collapsed sequence visible to the reference aligner
- observed_model: MMS/POWSM/Allophant model observations
- observed_acoustic: feature-derived evidence
- manual_phone: Praat/TextGrid labels for model-invisible allophones

Corpus scope: Common Voice + ISSAI (all publisher-GT segments), audiobooks
(balanced 200 h pool). YouTube and podcasts are outside the atlas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from g2p.constants import LONG_VOWELS, STRESS
from g2p.pipeline import transcribe_training_text

from audio.db import AUDIO_DB_PATH


ATLAS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "phoneme_atlas"
DEFAULT_ATLAS_MANIFEST = ATLAS_ROOT / "atlas_audio_manifest.csv"

WHISPER_SOURCE = "whisper_large_v3_turbo"
AUDIOBOOK_TARGET_HOURS = 200.0
AUDIOBOOK_SELECTION_REASON = "audiobooks_balanced_150h_seed42"

ATLAS_MANIFEST_COLUMNS = [
    "segment_id",
    "path",
    "provider",
    "duration_s",
    "speaker_id",
    "transcript_source",
    "needs_whisper",
    "role",
    "selection_reason",
]

GOLD_ATLAS_PROVIDERS = ("common_voice", "issai_tsc")
ATLAS_PROVIDER_ORDER = {
    "common_voice": 0,
    "issai_tsc": 1,
    "audiobooks": 2,
}

MMS_ALIGNMENT_COLLAPSE = {
    STRESS: None,
    "pʰ": "p",
    "tʰ": "t",
    "kʰ": "k",
    "cʰ": "c",
    "ɾ̞̊": "ɾ",
    "β": "v",
    "β̞": "v",
    "ɲ": "n",
    "æ": "e",
}

MANUAL_ONLY_PHONES = frozenset({"pʰ", "tʰ", "kʰ", "cʰ", "ɾ̞̊", "β", "β̞"})
RULE_ONLY_PHONES = frozenset({"æ"})
FEATURE_DERIVED_PHONES = LONG_VOWELS | {STRESS}


@dataclass(frozen=True)
class AtlasManifestRow:
    segment_id: str
    path: str
    provider: str
    duration_s: float
    speaker_id: str
    transcript_source: str
    needs_whisper: bool
    role: str
    selection_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "path": self.path,
            "provider": self.provider,
            "duration_s": f"{self.duration_s:.3f}".rstrip("0").rstrip("."),
            "speaker_id": self.speaker_id,
            "transcript_source": self.transcript_source,
            "needs_whisper": "true" if self.needs_whisper else "false",
            "role": self.role,
            "selection_reason": self.selection_reason,
        }


def expected_49_from_text(text: str) -> tuple[str, ...]:
    """Return the canonical G2P hypothesis used by atlas evidence reports."""
    result = transcribe_training_text(text)
    if result.drop_reason is not None:
        raise ValueError(f"G2P training text rejected: {result.drop_reason}")
    return tuple(result.tokens)


def alignment_phone(expected_phone: str) -> str | None:
    """Collapse a canonical phone into the current MMS-observable target layer."""
    return MMS_ALIGNMENT_COLLAPSE.get(expected_phone, expected_phone)


def alignment_target_from_expected(phones: Iterable[str]) -> tuple[str, ...]:
    """Build the MMS-visible sequence from expected_49 phones."""
    collapsed: list[str] = []
    for phone in phones:
        target = alignment_phone(phone)
        if target is not None:
            collapsed.append(target)
    return tuple(collapsed)


def phone_evidence_type(phone: str) -> str:
    """Classify the strongest planned evidence source for an inventory phone."""
    if phone in MANUAL_ONLY_PHONES:
        return "manual-only"
    if phone in RULE_ONLY_PHONES:
        return "rule-only"
    if phone in FEATURE_DERIVED_PHONES:
        return "feature-derived"
    return "model-observable"


def audiobook_group_key(path: str, speaker_id: str) -> str:
    """Return the book/speaker balancing key for audiobook atlas sampling."""
    parts = [part for part in path.split("/") if part]
    try:
        idx = parts.index("audiobooks")
    except ValueError:
        return speaker_id
    if idx + 1 >= len(parts):
        return speaker_id
    return parts[idx + 1]


def _has_source_sql(source: str) -> str:
    return (
        "EXISTS ("
        "SELECT 1 FROM transcript_candidates tc "
        "WHERE tc.segment_id = s.id AND tc.source = "
        f"'{source}'"
        ")"
    )


def _segment_rows(
    conn: sqlite3.Connection,
    *,
    provider: str,
    where_sql: str = "",
    params: tuple[object, ...] = (),
) -> list[sqlite3.Row]:
    sql = f"""
        SELECT
            s.id AS segment_id,
            s.path AS path,
            s.provider AS provider,
            s.duration_s AS duration_s,
            s.speaker_id AS speaker_id,
            {_has_source_sql(WHISPER_SOURCE)} AS has_whisper,
            {_has_source_sql("publisher_gt")} AS has_publisher_gt
        FROM segments s
        WHERE s.provider = ?
        {where_sql}
        ORDER BY s.id
    """
    return conn.execute(sql, (provider, *params)).fetchall()


def _from_sql_row(
    row: sqlite3.Row,
    *,
    transcript_source: str,
    needs_whisper: bool,
    role: str,
    selection_reason: str,
) -> AtlasManifestRow:
    return AtlasManifestRow(
        segment_id=str(row["segment_id"]),
        path=row["path"],
        provider=row["provider"],
        duration_s=float(row["duration_s"]),
        speaker_id=row["speaker_id"],
        transcript_source=transcript_source,
        needs_whisper=needs_whisper,
        role=role,
        selection_reason=selection_reason,
    )


def _stable_key(seed: int, *parts: object) -> str:
    payload = "|".join(str(part) for part in (seed, *parts))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _sample_balanced_rows(
    rows: Iterable[sqlite3.Row],
    *,
    target_seconds: float,
    seed: int,
) -> list[sqlite3.Row]:
    if target_seconds <= 0:
        return []

    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[audiobook_group_key(row["path"], row["speaker_id"])].append(row)

    for group_key, group_rows in groups.items():
        group_rows.sort(
            key=lambda row: _stable_key(seed, group_key, row["segment_id"], row["path"])
        )

    cursors = {group_key: 0 for group_key in groups}
    heap: list[tuple[float, str, str]] = []
    for group_key in groups:
        heapq.heappush(heap, (0.0, _stable_key(seed, "group", group_key), group_key))

    selected: list[sqlite3.Row] = []
    total_seconds = 0.0
    while heap:
        selected_seconds, order_key, group_key = heapq.heappop(heap)
        group_rows = groups[group_key]
        idx = cursors[group_key]

        while idx < len(group_rows):
            row = group_rows[idx]
            duration = float(row["duration_s"])
            idx += 1
            if total_seconds + duration <= target_seconds:
                cursors[group_key] = idx
                selected.append(row)
                total_seconds += duration
                heapq.heappush(
                    heap,
                    (selected_seconds + duration, order_key, group_key),
                )
                break
        else:
            cursors[group_key] = idx

    return selected


def _gold_rows(conn: sqlite3.Connection) -> list[AtlasManifestRow]:
    rows: list[AtlasManifestRow] = []
    for provider in GOLD_ATLAS_PROVIDERS:
        for row in _segment_rows(
            conn,
            provider=provider,
            where_sql=f"AND {_has_source_sql('publisher_gt')}",
        ):
            rows.append(
                _from_sql_row(
                    row,
                    transcript_source="publisher_gt",
                    needs_whisper=False,
                    role="atlas_gold",
                    selection_reason=f"{provider}_all_publisher_gt",
                )
            )
    return rows


def _audiobook_rows(
    conn: sqlite3.Connection,
    *,
    target_hours: float,
    seed: int,
) -> list[AtlasManifestRow]:
    source_rows = _segment_rows(
        conn,
        provider="audiobooks",
        where_sql=f"AND {_has_source_sql(WHISPER_SOURCE)}",
    )
    selected = _sample_balanced_rows(
        source_rows,
        target_seconds=target_hours * 3600.0,
        seed=seed,
    )
    return [
        _from_sql_row(
            row,
            transcript_source=WHISPER_SOURCE,
            needs_whisper=False,
            role="atlas_silver",
            selection_reason=AUDIOBOOK_SELECTION_REASON,
        )
        for row in selected
    ]


def _sort_manifest_rows(rows: Iterable[AtlasManifestRow]) -> list[AtlasManifestRow]:
    return sorted(
        rows,
        key=lambda row: (
            ATLAS_PROVIDER_ORDER.get(row.provider, 99),
            row.path,
            int(row.segment_id) if row.segment_id.isdigit() else row.segment_id,
        ),
    )


def build_atlas_manifest_rows(
    conn: sqlite3.Connection,
    *,
    audiobook_hours: float = AUDIOBOOK_TARGET_HOURS,
    seed: int = 42,
) -> list[AtlasManifestRow]:
    """Build the deterministic phoneme-atlas audio manifest rows."""
    rows: list[AtlasManifestRow] = []
    rows.extend(_gold_rows(conn))
    rows.extend(_audiobook_rows(conn, target_hours=audiobook_hours, seed=seed))
    return _sort_manifest_rows(rows)


def write_atlas_manifest(rows: Iterable[AtlasManifestRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ATLAS_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=AUDIO_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ATLAS_MANIFEST)
    parser.add_argument("--audiobook-hours", type=float, default=AUDIOBOOK_TARGET_HOURS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        rows = build_atlas_manifest_rows(
            conn,
            audiobook_hours=args.audiobook_hours,
            seed=args.seed,
        )
    finally:
        conn.close()

    write_atlas_manifest(rows, args.out)
    total_hours = sum(row.duration_s for row in rows) / 3600.0
    print(f"Wrote {len(rows):,} rows / {total_hours:,.2f} h to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
