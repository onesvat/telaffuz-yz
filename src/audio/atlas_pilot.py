"""Pilot manifest and model-selection helpers for the phoneme atlas.

The full atlas manifest can be hundreds of hours. This module builds a small,
deterministic pilot slice that exercises each atlas provider plus an
allophone-heavy supplement before GPU-intensive MMS/Allophant runs.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from g2p.constants import ALL_PHONEMES, LONG_VOWELS

from audio.atlas import (
    ATLAS_MANIFEST_COLUMNS,
    ATLAS_PROVIDER_ORDER,
    DEFAULT_ATLAS_MANIFEST,
    MANUAL_ONLY_PHONES,
    WHISPER_SOURCE,
    AtlasManifestRow,
    _stable_key,
    expected_49_from_text,
    write_atlas_manifest,
)
from audio.db import AUDIO_DB_PATH


DEFAULT_PILOT_MANIFEST = DEFAULT_ATLAS_MANIFEST.with_name("pilot_manifest.csv")
DEFAULT_MODEL_SELECTION_REPORT = (
    Path(__file__).resolve().parent.parent.parent
    / "reports"
    / "model_selection_phoneme_atlas.md"
)

DEFAULT_PROVIDER_SECONDS: dict[str, float] = {
    "common_voice": 0.50 * 3600.0,
    "issai_tsc": 0.50 * 3600.0,
    "audiobooks": 1.00 * 3600.0,
}
DEFAULT_RARE_SECONDS = 0.25 * 3600.0

RARE_SUPPLEMENT_PHONES = MANUAL_ONLY_PHONES | LONG_VOWELS | {"ɲ", "ŋ", "ɫ", "æ"}
ALLOPHANT_RENAME = {"ø": "œ", "g": "ɡ", "ɛ": "e"}


@dataclass(frozen=True)
class PilotSpec:
    provider_seconds: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PROVIDER_SECONDS)
    )
    rare_seconds: float = DEFAULT_RARE_SECONDS
    seed: int = 42


@dataclass(frozen=True)
class AllophantMapping:
    phones: tuple[str, ...]
    unknown: tuple[str, ...]


@dataclass(frozen=True)
class ModelPilotMetrics:
    name: str
    status: str
    runtime: str = "not run"
    coverage: str = "not measured"
    agreement: str = "not measured"
    notes: str = ""


def _transcript_source_for_provider(provider: str) -> str:
    if provider in {"common_voice", "issai_tsc"}:
        return "publisher_gt"
    return WHISPER_SOURCE


def _provider_where_sql(provider: str) -> tuple[str, tuple[object, ...]]:
    return "", ()


def _candidate_rows(conn: sqlite3.Connection, provider: str) -> list[sqlite3.Row]:
    source = _transcript_source_for_provider(provider)
    where_sql, params = _provider_where_sql(provider)
    sql = f"""
        SELECT
            s.id AS segment_id,
            s.path AS path,
            s.provider AS provider,
            s.duration_s AS duration_s,
            s.speaker_id AS speaker_id,
            tc.text AS transcript_text,
            tc.source AS transcript_source
        FROM segments s
        JOIN transcript_candidates tc
          ON tc.segment_id = s.id
         AND tc.source = ?
        WHERE s.provider = ?
        {where_sql}
        ORDER BY s.id
    """
    return conn.execute(sql, (source, provider, *params)).fetchall()


def _sample_rows(
    rows: Iterable[sqlite3.Row],
    *,
    target_seconds: float,
    seed: int,
    selected_ids: set[int],
    reason: str,
) -> list[AtlasManifestRow]:
    if target_seconds <= 0:
        return []

    ordered = sorted(
        rows,
        key=lambda row: _stable_key(seed, reason, row["segment_id"], row["path"]),
    )
    selected: list[AtlasManifestRow] = []
    total = 0.0
    for row in ordered:
        segment_id = int(row["segment_id"])
        duration = float(row["duration_s"])
        if segment_id in selected_ids:
            continue
        if total + duration > target_seconds and selected:
            continue
        selected_ids.add(segment_id)
        total += duration
        selected.append(
            AtlasManifestRow(
                segment_id=str(segment_id),
                path=row["path"],
                provider=row["provider"],
                duration_s=duration,
                speaker_id=row["speaker_id"],
                transcript_source=row["transcript_source"],
                needs_whisper=False,
                role="atlas_pilot",
                selection_reason=reason,
            )
        )
        if total >= target_seconds:
            break
    return selected


def _has_rare_phone(text: str) -> bool:
    try:
        phones = expected_49_from_text(text)
    except ValueError:
        return False
    return bool(set(phones) & RARE_SUPPLEMENT_PHONES)


def build_pilot_manifest_rows(
    conn: sqlite3.Connection,
    spec: PilotSpec | None = None,
) -> list[AtlasManifestRow]:
    """Build the deterministic 3h pilot manifest rows.

    Only segments with the transcript source required by their provider are
    eligible, so every returned row has ``needs_whisper=false``.
    """
    spec = spec or PilotSpec()
    provider_rows = {
        provider: _candidate_rows(conn, provider)
        for provider in sorted(
            spec.provider_seconds,
            key=lambda item: ATLAS_PROVIDER_ORDER.get(item, 99),
        )
    }

    selected_ids: set[int] = set()
    selected: list[AtlasManifestRow] = []
    for provider, seconds in spec.provider_seconds.items():
        selected.extend(
            _sample_rows(
                provider_rows.get(provider, ()),
                target_seconds=seconds,
                seed=spec.seed,
                selected_ids=selected_ids,
                reason=f"pilot_{provider}_seed{spec.seed}",
            )
        )

    if spec.rare_seconds > 0:
        rare_candidates: list[sqlite3.Row] = []
        all_providers = set(spec.provider_seconds) | set(DEFAULT_PROVIDER_SECONDS)
        for provider in sorted(
            all_providers,
            key=lambda item: ATLAS_PROVIDER_ORDER.get(item, 99),
        ):
            rows = provider_rows.get(provider)
            if rows is None:
                rows = _candidate_rows(conn, provider)
            rare_candidates.extend(
                row for row in rows if _has_rare_phone(row["transcript_text"])
            )
        selected.extend(
            _sample_rows(
                rare_candidates,
                target_seconds=spec.rare_seconds,
                seed=spec.seed,
                selected_ids=selected_ids,
                reason=f"pilot_rare_allophone_seed{spec.seed}",
            )
        )

    return sorted(
        selected,
        key=lambda row: (
            ATLAS_PROVIDER_ORDER.get(row.provider, 99),
            row.selection_reason,
            row.path,
            int(row.segment_id),
        ),
    )


def map_allophant_inventory(phones: Iterable[str]) -> AllophantMapping:
    """Map Allophant observed phones into the local 49-phone inventory."""
    mapped: list[str] = []
    unknown: list[str] = []
    for phone in phones:
        normalized = ALLOPHANT_RENAME.get(phone, phone)
        if normalized in ALL_PHONEMES:
            mapped.append(normalized)
        elif normalized not in unknown:
            unknown.append(normalized)
    return AllophantMapping(phones=tuple(mapped), unknown=tuple(unknown))


def write_pilot_manifest(rows: Iterable[AtlasManifestRow], out_path: Path) -> None:
    write_atlas_manifest(rows, out_path)


def render_model_selection_report(
    rows: Iterable[AtlasManifestRow],
    *,
    mms: ModelPilotMetrics | None = None,
    allophant: ModelPilotMetrics | None = None,
) -> str:
    rows = list(rows)
    total_hours = sum(row.duration_s for row in rows) / 3600.0
    by_provider: dict[str, float] = {}
    for row in rows:
        by_provider[row.provider] = by_provider.get(row.provider, 0.0) + row.duration_s
    mms = mms or ModelPilotMetrics(
        name="MMS baseline",
        status="pending",
        notes="Run `uv run audio mms` and `uv run audio align` on the pilot manifest.",
    )
    allophant = allophant or ModelPilotMetrics(
        name="Allophant validator",
        status="pending",
        notes="If setup or inventory mapping fails, continue with MMS-only fallback.",
    )

    provider_lines = "\n".join(
        f"- {provider}: {seconds / 3600.0:.2f} h"
        for provider, seconds in sorted(
            by_provider.items(), key=lambda item: ATLAS_PROVIDER_ORDER.get(item[0], 99)
        )
    )
    model_lines = "\n".join(
        [
            "| Model | Status | Runtime | Coverage | Agreement | Notes |",
            "|---|---|---:|---|---|---|",
            (
                f"| {mms.name} | {mms.status} | {mms.runtime} | "
                f"{mms.coverage} | {mms.agreement} | {mms.notes} |"
            ),
            (
                f"| {allophant.name} | {allophant.status} | {allophant.runtime} | "
                f"{allophant.coverage} | {allophant.agreement} | {allophant.notes} |"
            ),
        ]
    )
    return f"""# Phoneme Atlas Model Selection Pilot

Date: {date.today().isoformat()}

## Pilot Manifest

- Rows: {len(rows):,}
- Total audio: {total_hours:.2f} h
- Seed: 42
- Policy: all rows must have `needs_whisper=false`; YouTube, podcasts, and TRT
  non-whitelist folders stay out of scope.

{provider_lines}

## Model Results

{model_lines}

## Acceptance Gate

- MMS alignment success target: >=95%.
- Allophant is a validator layer, not a blocker. If Allophant fails setup,
  inventory mapping, or runtime checks, the full atlas proceeds as MMS-only and
  records this fallback here.
- POWSM remains a second-wave candidate and is not required for the first atlas.

## Known Risk

TorchAudio `forced_align` is pinned to the 2.8 line for this project; the API is
deprecated there and removed in 2.9, so this runner must stay pinned or be
replaced before upgrading TorchAudio.
"""


def write_model_selection_report(
    rows: Iterable[AtlasManifestRow],
    out_path: Path,
    *,
    mms: ModelPilotMetrics | None = None,
    allophant: ModelPilotMetrics | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_model_selection_report(rows, mms=mms, allophant=allophant),
        encoding="utf-8",
    )


def _provider_seconds_from_csv(value: str | None) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_PROVIDER_SECONDS)
    parsed: dict[str, float] = {}
    for item in value.split(","):
        provider, raw_hours = item.split("=", 1)
        parsed[provider.strip()] = float(raw_hours) * 3600.0
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=AUDIO_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_PILOT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_MODEL_SELECTION_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider-hours",
        help="comma list like common_voice=0.5,issai_tsc=0.5",
    )
    parser.add_argument("--rare-hours", type=float, default=0.25)
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        rows = build_pilot_manifest_rows(
            conn,
            PilotSpec(
                provider_seconds=_provider_seconds_from_csv(args.provider_hours),
                rare_seconds=args.rare_hours * 3600.0,
                seed=args.seed,
            ),
        )
    finally:
        conn.close()

    write_pilot_manifest(rows, args.out)
    write_model_selection_report(rows, args.report)
    total_hours = sum(row.duration_s for row in rows) / 3600.0
    print(f"Wrote {len(rows):,} pilot rows / {total_hours:.2f} h to {args.out}")
    print(f"Wrote model-selection report to {args.report}")
    return 0


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(ATLAS_MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        return list(reader)


if __name__ == "__main__":
    raise SystemExit(main())
