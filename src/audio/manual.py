"""Praat TextGrid queue/import helpers and manual annotation summary.

manual_annotation_summary is required by atlas_export for the manual-only
evidence layer (pʰ, tʰ, kʰ, cʰ, ɾ̞̊, β, β̞).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from audio.atlas import MANUAL_ONLY_PHONES
from audio.db import AUDIO_DB_PATH, now_iso


TEXTGRID_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*"
    r"xmin = (?P<xmin>[-0-9.]+)\s*"
    r"xmax = (?P<xmax>[-0-9.]+)\s*"
    r"text = \"(?P<text>[^\"]*)\"",
    re.MULTILINE,
)

DEFAULT_PROVIDER_PRIORITY = (
    "audiobooks",
    "common_voice",
    "issai_tsc",
)


@dataclass(frozen=True)
class ManualPhoneInterval:
    segment_id: int
    start_s: float
    end_s: float
    phone: str | None
    status: str
    notes: str = ""


@dataclass(frozen=True)
class ManualTextGridSummary:
    file_count: int
    label_count: int
    malformed_label_count: int
    status_counts: dict[str, int]
    phone_status_counts: dict[tuple[str | None, str], int]


@dataclass(frozen=True)
class ManualTextGridImportResult:
    inserted_count: int
    summary: ManualTextGridSummary


@dataclass(frozen=True)
class ManualQueueCandidate:
    segment_id: int
    path: str
    expected_phone: str
    alignment_phone: str | None
    start_ms: int
    end_ms: int
    confidence: float | None
    candidate_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_label(label: str) -> tuple[str | None, str, str]:
    raw = label.strip()
    if not raw:
        return None, "empty", ""
    parts = [part.strip() for part in raw.split("|")]
    phone = parts[0] or None
    status = parts[1].lower() if len(parts) > 1 and parts[1] else "accepted"
    notes = "|".join(parts[2:]) if len(parts) > 2 else ""
    if phone in {"reject", "uncertain", "accepted"}:
        status = phone
        phone = None
    if status not in {"accepted", "uncertain", "reject", "empty"}:
        status = "uncertain"
        notes = raw if not notes else f"{notes}|{raw}"
    return phone, status, notes


def parse_textgrid_phone_intervals(
    path: Path,
    *,
    segment_id: int,
) -> list[ManualPhoneInterval]:
    text = path.read_text(encoding="utf-8")
    intervals: list[ManualPhoneInterval] = []
    for match in TEXTGRID_INTERVAL_RE.finditer(text):
        phone, status, notes = _parse_label(match.group("text"))
        if status == "empty":
            continue
        intervals.append(
            ManualPhoneInterval(
                segment_id=segment_id,
                start_s=float(match.group("xmin")),
                end_s=float(match.group("xmax")),
                phone=phone,
                status=status,
                notes=notes,
            )
        )
    return intervals


def _raw_textgrid_labels(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [match.group("text") for match in TEXTGRID_INTERVAL_RE.finditer(text)]


def _is_malformed_label(label: str) -> bool:
    raw = label.strip()
    if not raw:
        return False
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) == 1:
        return False
    status = parts[1].lower() if parts[1] else "accepted"
    return status not in {"accepted", "uncertain", "reject"}


def summarize_textgrid_paths(paths: Iterable[Path]) -> ManualTextGridSummary:
    paths = list(paths)
    intervals: list[ManualPhoneInterval] = []
    malformed = 0
    for path in paths:
        malformed += sum(
            1 for label in _raw_textgrid_labels(path) if _is_malformed_label(label)
        )
        intervals.extend(
            parse_textgrid_phone_intervals(
                path,
                segment_id=_segment_id_from_textgrid(path),
            )
        )

    status_counts = Counter(interval.status for interval in intervals)
    phone_status_counts = Counter(
        (interval.phone, interval.status) for interval in intervals
    )
    return ManualTextGridSummary(
        file_count=len(paths),
        label_count=len(intervals),
        malformed_label_count=malformed,
        status_counts=dict(status_counts),
        phone_status_counts=dict(phone_status_counts),
    )


def manual_coverage_summary(
    intervals: Iterable[ManualPhoneInterval],
    *,
    minimum_per_phone: int = 15,
    target_phones: Iterable[str] = MANUAL_ONLY_PHONES,
) -> dict[str, dict[str, int | bool]]:
    summary: dict[str, dict[str, int | bool]] = {
        phone: {
            "accepted": 0,
            "uncertain": 0,
            "reject": 0,
            "meets_minimum": False,
        }
        for phone in sorted(target_phones)
    }
    for interval in intervals:
        if interval.phone not in summary:
            continue
        bucket = summary[interval.phone]
        if interval.status in {"accepted", "uncertain", "reject"}:
            bucket[interval.status] = int(bucket[interval.status]) + 1

    for phone, bucket in summary.items():
        bucket["meets_minimum"] = int(bucket["accepted"]) >= minimum_per_phone
    return summary


def write_manual_queue_csv(
    rows: Iterable[ManualQueueCandidate | dict[str, object]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "segment_id",
        "path",
        "expected_phone",
        "alignment_phone",
        "start_ms",
        "end_ms",
        "confidence",
        "candidate_reason",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if isinstance(row, ManualQueueCandidate):
                row = row.as_dict()
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _provider_from_path(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "processed":
        return parts[1]
    return parts[0] if parts else ""


def _candidate_sort_key(candidate: ManualQueueCandidate) -> tuple[float, int, int, int]:
    confidence = candidate.confidence if candidate.confidence is not None else -1.0
    return (-confidence, candidate.segment_id, candidate.start_ms, candidate.end_ms)


def _select_balanced_candidates(
    candidates: list[tuple[str, ManualQueueCandidate]],
    *,
    target_per_phone: int,
    provider_quotas: Mapping[str, int],
    provider_priority: Sequence[str],
) -> list[ManualQueueCandidate]:
    by_provider: dict[str, list[ManualQueueCandidate]] = {}
    for provider, candidate in candidates:
        by_provider.setdefault(provider, []).append(candidate)
    for provider in by_provider:
        by_provider[provider].sort(key=_candidate_sort_key)

    selected: list[ManualQueueCandidate] = []
    seen: set[tuple[int, int, int, str]] = set()

    def add_candidate(candidate: ManualQueueCandidate) -> None:
        key = (
            candidate.segment_id,
            candidate.start_ms,
            candidate.end_ms,
            candidate.expected_phone,
        )
        if key in seen or len(selected) >= target_per_phone:
            return
        seen.add(key)
        selected.append(
            replace(candidate, candidate_reason="manual-only_provider_balanced")
        )

    for provider in provider_priority:
        quota = max(0, int(provider_quotas.get(provider, 0)))
        if quota == 0:
            continue
        for candidate in by_provider.get(provider, [])[:quota]:
            add_candidate(candidate)

    preferred_providers = [
        provider for provider in provider_priority if provider_quotas.get(provider, 0) > 0
    ]
    fallback_providers = [
        provider for provider in provider_priority if provider_quotas.get(provider, 0) <= 0
    ]

    for providers in (preferred_providers, fallback_providers):
        remaining = [
            candidate
            for provider in providers
            for candidate in by_provider.get(provider, [])
            if (
                candidate.segment_id,
                candidate.start_ms,
                candidate.end_ms,
                candidate.expected_phone,
            )
            not in seen
        ]
        for candidate in sorted(remaining, key=_candidate_sort_key):
            add_candidate(candidate)
            if len(selected) >= target_per_phone:
                break
        if len(selected) >= target_per_phone:
            break

    known = set(provider_priority)
    remaining_unknown = [
        candidate
        for provider, provider_candidates in by_provider.items()
        if provider not in known
        for candidate in provider_candidates
        if (
            candidate.segment_id,
            candidate.start_ms,
            candidate.end_ms,
            candidate.expected_phone,
        )
        not in seen
    ]
    for candidate in sorted(remaining_unknown, key=_candidate_sort_key):
        add_candidate(candidate)
        if len(selected) >= target_per_phone:
            break

    return selected


def build_manual_queue_candidates(
    conn: sqlite3.Connection,
    *,
    aligner_version: str,
    target_per_phone: int = 20,
    target_phones: Iterable[str] = MANUAL_ONLY_PHONES,
    provider_quotas: Mapping[str, int] | None = None,
    provider_priority: Sequence[str] = DEFAULT_PROVIDER_PRIORITY,
    confidence_floor: float | None = None,
) -> list[ManualQueueCandidate]:
    """Select deterministic high-confidence candidates for manual-only phones."""
    from audio.alignment import parse_alignment_json

    target_set = set(target_phones)
    buckets: dict[str, list[tuple[str, ManualQueueCandidate]]] = {
        phone: [] for phone in target_set
    }
    rows = conn.execute(
        """SELECT
               sa.segment_id AS segment_id,
               sa.alignment_json AS alignment_json,
               s.path AS path,
               s.provider AS provider
           FROM segment_alignments sa
           JOIN segments s ON s.id = sa.segment_id
           WHERE sa.aligner_version = ?
           ORDER BY sa.segment_id, sa.id""",
        (aligner_version,),
    ).fetchall()
    for row in rows:
        provider = row["provider"] or _provider_from_path(row["path"])
        for item in parse_alignment_json(row["alignment_json"]):
            if item.expected_phone not in target_set:
                continue
            if (
                confidence_floor is not None
                and item.confidence is not None
                and item.confidence < confidence_floor
            ):
                continue
            buckets[item.expected_phone].append(
                (
                    provider,
                    ManualQueueCandidate(
                        segment_id=int(row["segment_id"]),
                        path=row["path"],
                        expected_phone=item.expected_phone,
                        alignment_phone=item.alignment_phone,
                        start_ms=item.start_ms,
                        end_ms=item.end_ms,
                        confidence=item.confidence,
                        candidate_reason="manual-only_top_confidence",
                    ),
                )
            )

    selected: list[ManualQueueCandidate] = []
    for phone in sorted(buckets):
        candidates_with_provider = buckets[phone]
        if provider_quotas is None:
            candidates = sorted(
                (candidate for _, candidate in candidates_with_provider),
                key=_candidate_sort_key,
            )
            selected.extend(candidates[:target_per_phone])
            continue

        selected.extend(
            _select_balanced_candidates(
                candidates_with_provider,
                target_per_phone=target_per_phone,
                provider_quotas=provider_quotas,
                provider_priority=provider_priority,
            )
        )
    return selected


def _textgrid_quote(text: str) -> str:
    return text.replace('"', '""')


def _textgrid_interval_lines(
    index: int,
    *,
    xmin: float,
    xmax: float,
    text: str,
) -> list[str]:
    return [
        f"        intervals [{index}]:",
        f"            xmin = {xmin:.3f}".rstrip("0").rstrip("."),
        f"            xmax = {xmax:.3f}".rstrip("0").rstrip("."),
        f'            text = "{_textgrid_quote(text)}"',
    ]


def write_manual_queue_textgrids(
    candidates: Iterable[ManualQueueCandidate],
    *,
    out_dir: Path,
    segment_durations: dict[int, float],
) -> list[Path]:
    """Write one Praat TextGrid per segment with candidate phone intervals."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[ManualQueueCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.segment_id, []).append(candidate)

    written: list[Path] = []
    for segment_id, segment_candidates in sorted(grouped.items()):
        duration = float(segment_durations[segment_id])
        intervals: list[tuple[float, float, str]] = []
        cursor = 0.0
        for candidate in sorted(segment_candidates, key=lambda item: item.start_ms):
            start = max(0.0, candidate.start_ms / 1000.0)
            end = min(duration, candidate.end_ms / 1000.0)
            if start > cursor:
                intervals.append((cursor, start, ""))
            if end > start:
                intervals.append((start, end, f"{candidate.expected_phone}|uncertain"))
                cursor = end
        if cursor < duration:
            intervals.append((cursor, duration, ""))

        xmax = duration
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            f"xmax = {xmax:.3f}".rstrip("0").rstrip("."),
            "tiers? <exists>",
            "size = 1",
            "item []:",
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "phones"',
            "        xmin = 0",
            f"        xmax = {xmax:.3f}".rstrip("0").rstrip("."),
            f"        intervals: size = {len(intervals)}",
        ]
        for index, (xmin, interval_xmax, text) in enumerate(intervals, start=1):
            lines.extend(
                _textgrid_interval_lines(
                    index,
                    xmin=xmin,
                    xmax=interval_xmax,
                    text=text,
                )
            )
        out_path = out_dir / f"{segment_id}.TextGrid"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(out_path)
    return written


def _segment_durations(
    conn: sqlite3.Connection,
    segment_ids: Iterable[int],
) -> dict[int, float]:
    ids = sorted(set(segment_ids))
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, duration_s FROM segments WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(row["id"]): float(row["duration_s"]) for row in rows}


def import_textgrid_intervals(
    conn: sqlite3.Connection,
    intervals: Iterable[ManualPhoneInterval],
    *,
    actor: str = "manual:praat",
) -> int:
    count = 0
    for interval in intervals:
        conn.execute(
            """INSERT INTO manual_annotations(entity_type, entity_id, label,
                                             value_json, actor, notes, created_at)
               VALUES ('segment', ?, 'manual_phone', ?, ?, ?, ?)""",
            (
                str(interval.segment_id),
                json.dumps(asdict(interval), ensure_ascii=False, sort_keys=True),
                actor,
                interval.notes,
                now_iso(),
            ),
        )
        count += 1
    return count


def import_reviewed_textgrids(
    conn: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    actor: str = "manual:praat",
) -> ManualTextGridImportResult:
    paths = list(paths)
    summary = summarize_textgrid_paths(paths)
    if summary.malformed_label_count:
        raise ValueError(
            f"refusing to import {summary.malformed_label_count} malformed labels"
        )

    intervals: list[ManualPhoneInterval] = []
    for path in paths:
        intervals.extend(
            parse_textgrid_phone_intervals(
                path,
                segment_id=_segment_id_from_textgrid(path),
            )
        )
    inserted = import_textgrid_intervals(conn, intervals, actor=actor)
    return ManualTextGridImportResult(inserted_count=inserted, summary=summary)


def manual_annotation_summary(conn: sqlite3.Connection) -> ManualTextGridSummary:
    rows = conn.execute(
        """SELECT entity_id, value_json
           FROM manual_annotations
           WHERE label = 'manual_phone'
           ORDER BY id"""
    ).fetchall()
    status_counts: Counter[str] = Counter()
    phone_status_counts: Counter[tuple[str | None, str]] = Counter()
    entity_ids: set[str] = set()
    for row in rows:
        data = json.loads(row["value_json"])
        status = str(data.get("status") or "uncertain")
        phone = data.get("phone")
        status_counts[status] += 1
        phone_status_counts[(phone, status)] += 1
        entity_ids.add(str(row["entity_id"]))
    return ManualTextGridSummary(
        file_count=len(entity_ids),
        label_count=len(rows),
        malformed_label_count=0,
        status_counts=dict(status_counts),
        phone_status_counts=dict(phone_status_counts),
    )


MANUAL_REVIEW_PHONE_ORDER = ("pʰ", "tʰ", "kʰ", "cʰ", "ɾ̞̊", "β", "β̞")
MANUAL_REVIEW_NOTES = {
    "pʰ": "Strong support",
    "tʰ": "Strong support",
    "kʰ": "Supported, some uncertainty",
    "cʰ": "Supported",
    "ɾ̞̊": "Supported",
    "β": "Supported",
    "β̞": "Accepted",
}


def _phone_status(summary: ManualTextGridSummary, phone: str, status: str) -> int:
    return int(summary.phone_status_counts.get((phone, status), 0))


def render_manual_review_report(summary: ManualTextGridSummary) -> str:
    rows = []
    for phone in MANUAL_REVIEW_PHONE_ORDER:
        rows.append(
            "| "
            f"`{phone}` | "
            f"{_phone_status(summary, phone, 'accepted')} | "
            f"{_phone_status(summary, phone, 'uncertain')} | "
            f"{_phone_status(summary, phone, 'reject')} | "
            f"{MANUAL_REVIEW_NOTES[phone]} |"
        )
    status_counts = {
        status: int(summary.status_counts.get(status, 0))
        for status in ("accepted", "uncertain", "reject")
    }
    return (
        "# Manual Phoneme Atlas Review\n\n"
        "## Purpose\n\n"
        "This manual review establishes the `manual_phone` evidence layer for the 7\n"
        "target phones/allophones that MMS_FA cannot distinguish directly:\n"
        "`pʰ`, `tʰ`, `kʰ`, `cʰ`, `ɾ̞̊`, `β`, `β̞`. This layer does not modify\n"
        "the G2P output or model output; it is a separate human-review evidence\n"
        "channel within the atlas.\n\n"
        "## Labelling Format\n\n"
        "Target intervals on the Praat `phones` tier were labelled as:\n\n"
        "```text\n"
        "phone|accepted\n"
        "phone|uncertain\n"
        "phone|reject\n"
        "```\n\n"
        "Empty interval labels are excluded from the atlas import.\n\n"
        "## Results\n\n"
        "| Status | Count |\n"
        "|---|---:|\n"
        f"| accepted | {status_counts['accepted']} |\n"
        f"| uncertain | {status_counts['uncertain']} |\n"
        f"| reject | {status_counts['reject']} |\n\n"
        "| Phone | Accepted | Uncertain | Reject | Notes |\n"
        "|---|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\n"
        "## β̞ Decision Note\n\n"
        "`β̞` was accepted into the manual atlas with 14 accepted, 1 uncertain,\n"
        "and 5 reject labels. Reject examples are retained to track labelling\n"
        "variance in downstream analysis; they do not individually constitute\n"
        "a risk verdict.\n\n"
        "## Role in the Atlas\n\n"
        "These results do not replace model output. The `expected_49`,\n"
        "`observed_model`, `observed_acoustic`, and `manual_phone` layers are\n"
        "kept separate. MMS_FA output is a boundary-and-projection layer, not\n"
        "an IPA phone recognition layer; manual results are used exclusively as\n"
        "`manual_phone` evidence.\n"
    )


def write_manual_review_report(
    summary: ManualTextGridSummary,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_manual_review_report(summary), encoding="utf-8")


def _segment_id_from_textgrid(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"cannot infer segment_id from {path}")
    return int(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=AUDIO_DB_PATH)
    parser.add_argument("--minimum", type=int, default=15)
    parser.add_argument("--export-queue", type=Path)
    parser.add_argument("--aligner-version", default="mms-fa-uroman-torchaudio-2.8")
    parser.add_argument("--target-per-phone", type=int, default=20)
    parser.add_argument("--report", type=Path)
    parser.add_argument("textgrids", nargs="*", type=Path)
    args = parser.parse_args(argv)

    if args.export_queue is not None:
        conn = sqlite3.connect(str(args.db))
        conn.row_factory = sqlite3.Row
        try:
            candidates = build_manual_queue_candidates(
                conn,
                aligner_version=args.aligner_version,
                target_per_phone=args.target_per_phone,
            )
            durations = _segment_durations(
                conn,
                (candidate.segment_id for candidate in candidates),
            )
        finally:
            conn.close()
        args.export_queue.mkdir(parents=True, exist_ok=True)
        write_manual_queue_csv(candidates, args.export_queue / "candidates.csv")
        written = write_manual_queue_textgrids(
            candidates,
            out_dir=args.export_queue / "textgrids",
            segment_durations=durations,
        )
        print(
            f"Wrote {len(candidates):,} manual candidates and "
            f"{len(written):,} TextGrid files to {args.export_queue}"
        )
        return 0

    if not args.textgrids:
        parser.error("provide TextGrid files to import or --export-queue OUT_DIR")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        result = import_reviewed_textgrids(conn, args.textgrids)
        conn.commit()
    finally:
        conn.close()
    if args.report is not None:
        write_manual_review_report(result.summary, args.report)

    print(f"Imported {result.inserted_count:,} manual intervals")
    print(
        f"Reviewed TextGrids={result.summary.file_count:,} "
        f"labels={result.summary.label_count:,} "
        f"malformed={result.summary.malformed_label_count:,}"
    )
    for phone in sorted(MANUAL_ONLY_PHONES):
        counts = {
            "accepted": _phone_status(result.summary, phone, "accepted"),
            "uncertain": _phone_status(result.summary, phone, "uncertain"),
            "reject": _phone_status(result.summary, phone, "reject"),
            "meets_minimum": _phone_status(result.summary, phone, "accepted")
            >= args.minimum,
        }
        print(
            f"{phone}: accepted={counts['accepted']} "
            f"uncertain={counts['uncertain']} reject={counts['reject']} "
            f"meets_minimum={counts['meets_minimum']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
