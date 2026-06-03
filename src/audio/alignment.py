"""CTC forced alignment persistence.

The GPU-side emission generation is intentionally optional here. The stable
surface is the resumable DB write format: expected_49 stays in
phoneme_sequences and the observed/alignment layers stay in
segment_alignments.alignment_json.

Midpoint padding: the aligner emits one TokenSpan per romanized token where
end-start is typically a single frame (~20 ms). We post-process boundaries
so each phone occupies the continuous interval between adjacent span peaks
(first span padded to 0, last to frame_count). This prevents starvation of
downstream coach feature windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable

from g2p.constants import CTC_SPC

from audio.atlas import alignment_phone, phone_evidence_type
from audio.db import AUDIO_DB_PATH, AUDIO_ROOT, now_iso


ALIGNMENT_SCHEMA_VERSION = 1
DEFAULT_ALIGNER_VERSION = "mms-fa-uroman-torchaudio-2.8"
EXPECTED_49_SOURCE = "expected_49"
G2P_VERSION = "telaffuz-yz-g2p-1.0"
NON_ALIGNMENT_TOKENS = frozenset({CTC_SPC})

MMS_FA_TOKEN_MAP: dict[str, str] = {
    "a": "a",
    "aː": "a",
    "e": "e",
    "eː": "e",
    "i": "i",
    "iː": "i",
    "ɯ": "i",
    "ɯː": "i",
    "o": "o",
    "oː": "o",
    "œ": "o",
    "œː": "o",
    "u": "u",
    "uː": "u",
    "y": "u",
    "yː": "u",
    "æ": "e",
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "ɡ": "g",
    "ɣ": "g",
    "c": "k",
    "ɟ": "g",
    "t͡ʃ": "c",
    "d͡ʒ": "j",
    "f": "f",
    "v": "v",
    "s": "s",
    "z": "z",
    "ʃ": "s",
    "ʒ": "z",
    "h": "h",
    "m": "m",
    "n": "n",
    "ɲ": "n",
    "ŋ": "n",
    "l": "l",
    "ɫ": "l",
    "ɾ": "r",
    "j": "y",
    "pʰ": "p",
    "tʰ": "t",
    "kʰ": "k",
    "cʰ": "k",
    "ɾ̞̊": "r",
    "β": "v",
    "β̞": "v",
}


@dataclass(frozen=True)
class MMSFAProjectionItem:
    expected_phone: str
    alignment_phone: str
    model_token: str


@dataclass(frozen=True)
class AlignmentItem:
    expected_phone: str
    alignment_phone: str | None
    observed_model_phone: str | None
    start_ms: int
    end_ms: int
    confidence: float | None
    alignment_model_token: str | None = None

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def mms_fa_token_for_alignment_phone(phone: str) -> str:
    """Project a local alignment phone onto the MMS_FA uroman label set."""
    try:
        return MMS_FA_TOKEN_MAP[phone]
    except KeyError as exc:
        raise ValueError(f"no MMS_FA projection for phone {phone!r}") from exc


def project_expected_phones_for_mms_fa(
    expected_phones: Iterable[str],
) -> tuple[MMSFAProjectionItem, ...]:
    """Build a lossy MMS_FA target while preserving expected/alignment layers."""
    projected: list[MMSFAProjectionItem] = []
    for expected_phone in expected_phones:
        if expected_phone in NON_ALIGNMENT_TOKENS:
            continue
        target_phone = alignment_phone(expected_phone)
        if target_phone is None:
            continue
        projected.append(
            MMSFAProjectionItem(
                expected_phone=expected_phone,
                alignment_phone=target_phone,
                model_token=mms_fa_token_for_alignment_phone(target_phone),
            )
        )
    return tuple(projected)


def alignment_items_from_mms_fa_spans(
    projected: Iterable[MMSFAProjectionItem],
    spans: Iterable[object],
    *,
    frame_count: int,
    duration_s: float,
    signal_bounds_frames: tuple[int, int] | None = None,
) -> tuple[AlignmentItem, ...]:
    """Convert MMS_FA TokenSpan output into layered atlas alignment items.

    When ``signal_bounds_frames`` is given, the first span starts at
    ``signal_bounds_frames[0]`` and the last span ends at
    ``signal_bounds_frames[1]`` — preventing edge spans from bleeding into
    silence beyond the voiced signal. Without it, edges extend to the full
    emission span (legacy behaviour).
    """
    projected = tuple(projected)
    spans = tuple(spans)
    if len(projected) != len(spans):
        raise ValueError(
            f"MMS_FA span count mismatch: projected={len(projected)} spans={len(spans)}"
        )
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    raw_starts = [int(getattr(span, "start")) for span in spans]
    raw_ends = [int(getattr(span, "end")) for span in spans]
    n = len(spans)
    new_starts = list(raw_starts)
    new_ends = list(raw_ends)
    for i in range(1, n):
        mid = (raw_ends[i - 1] + raw_starts[i]) // 2
        new_ends[i - 1] = mid
        new_starts[i] = mid
    if n > 0:
        if signal_bounds_frames is None:
            new_starts[0] = 0
            new_ends[-1] = frame_count
        else:
            sb_start, sb_end = signal_bounds_frames
            sb_start = max(0, int(sb_start))
            sb_end = min(frame_count, int(sb_end))
            # Keep span non-empty: clamp to ensure ordering relative to the
            # adjacent midpoint boundary.
            new_starts[0] = min(sb_start, max(0, new_ends[0] - 1))
            new_ends[-1] = max(sb_end, new_starts[-1] + 1)

    duration_ms = duration_s * 1000.0
    items: list[AlignmentItem] = []
    for item, span, start_frame, end_frame in zip(
        projected, spans, new_starts, new_ends, strict=True
    ):
        items.append(
            AlignmentItem(
                expected_phone=item.expected_phone,
                alignment_phone=item.alignment_phone,
                observed_model_phone=item.model_token,
                start_ms=int(round(start_frame * duration_ms / frame_count)),
                end_ms=int(round(end_frame * duration_ms / frame_count)),
                confidence=float(getattr(span, "score")),
                alignment_model_token=item.model_token,
            )
        )
    return tuple(items)


def build_alignment_json(items: Iterable[AlignmentItem]) -> str:
    return json.dumps(
        {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "phones": [item.as_json() for item in items],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def alignment_summary(items: Iterable[AlignmentItem]) -> dict[str, object]:
    items = list(items)
    confidences = [item.confidence for item in items if item.confidence is not None]
    return {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "phone_count": len(items),
        "mean_confidence": fmean(confidences) if confidences else None,
        "start_ms": min((item.start_ms for item in items), default=None),
        "end_ms": max((item.end_ms for item in items), default=None),
    }


def _upsert_segment_sequence(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    expected_phones: tuple[str, ...],
    source: str = EXPECTED_49_SOURCE,
    g2p_version: str = G2P_VERSION,
) -> int:
    payload = json.dumps(list(expected_phones), ensure_ascii=False)
    row = conn.execute(
        """SELECT id FROM phoneme_sequences
           WHERE scope='segment'
             AND segment_id=?
             AND source=?
             AND g2p_version=?
           LIMIT 1""",
        (segment_id, source, g2p_version),
    ).fetchone()
    if row is not None:
        sequence_id = int(row["id"])
        conn.execute(
            """UPDATE phoneme_sequences
               SET phonemes_json=?, created_at=?
               WHERE id=?""",
            (payload, now_iso(), sequence_id),
        )
        return sequence_id

    cur = conn.execute(
        """INSERT INTO phoneme_sequences(scope, segment_id, word_occurrence_id,
                                         source, phonemes_json, g2p_version,
                                         created_at)
           VALUES ('segment', ?, NULL, ?, ?, ?, ?)""",
        (segment_id, source, payload, g2p_version, now_iso()),
    )
    return int(cur.lastrowid)


def upsert_segment_alignment(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    expected_phones: tuple[str, ...],
    items: Iterable[AlignmentItem],
    aligner_version: str = DEFAULT_ALIGNER_VERSION,
) -> int:
    """Insert or replace one segment alignment without duplicating rows."""
    items = tuple(items)
    sequence_id = _upsert_segment_sequence(
        conn,
        segment_id=segment_id,
        expected_phones=expected_phones,
    )
    alignment_json = build_alignment_json(items)
    summary_json = json.dumps(alignment_summary(items), ensure_ascii=False, sort_keys=True)
    existing = conn.execute(
        """SELECT id FROM segment_alignments
           WHERE segment_id=?
             AND source_sequence_id=?
             AND aligner_version=?
           LIMIT 1""",
        (segment_id, sequence_id, aligner_version),
    ).fetchone()
    if existing is not None:
        alignment_id = int(existing["id"])
        conn.execute(
            """UPDATE segment_alignments
               SET alignment_json=?, summary_json=?, created_at=?
               WHERE id=?""",
            (alignment_json, summary_json, now_iso(), alignment_id),
        )
        return alignment_id

    cur = conn.execute(
        """INSERT INTO segment_alignments(segment_id, source_sequence_id,
                                         aligner_version, alignment_json,
                                         summary_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            segment_id,
            sequence_id,
            aligner_version,
            alignment_json,
            summary_json,
            now_iso(),
        ),
    )
    return int(cur.lastrowid)


def parse_alignment_json(payload: str) -> tuple[AlignmentItem, ...]:
    data = json.loads(payload)
    return tuple(
        AlignmentItem(
            expected_phone=item["expected_phone"],
            alignment_phone=item.get("alignment_phone"),
            observed_model_phone=item.get("observed_model_phone"),
            start_ms=int(item["start_ms"]),
            end_ms=int(item["end_ms"]),
            confidence=(
                float(item["confidence"])
                if item.get("confidence") is not None
                else None
            ),
            alignment_model_token=item.get("alignment_model_token"),
        )
        for item in data.get("phones", [])
    )


def alignment_items_from_expected_fixture(
    expected_phones: Iterable[str],
    *,
    frame_ms: int = 40,
) -> tuple[AlignmentItem, ...]:
    """Build deterministic fixture alignments for tests and dry-run exports."""
    items: list[AlignmentItem] = []
    cursor = 0
    for expected_phone in expected_phones:
        target_phone = alignment_phone(expected_phone)
        if target_phone is None:
            continue
        items.append(
            AlignmentItem(
                expected_phone=expected_phone,
                alignment_phone=target_phone,
                observed_model_phone=target_phone,
                start_ms=cursor,
                end_ms=cursor + frame_ms,
                confidence=1.0,
                alignment_model_token=mms_fa_token_for_alignment_phone(target_phone),
            )
        )
        cursor += frame_ms
    return tuple(items)


def iter_alignment_phone_instances(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
    batch_size: int = 1_000,
) -> Iterable[dict[str, object]]:
    """Stream flattened atlas phone-instance rows from alignment JSON."""
    params: list[object] = []
    where = ""
    if aligner_version is not None:
        where = "WHERE sa.aligner_version = ?"
        params.append(aligner_version)
    cursor = conn.execute(
        f"""SELECT sa.segment_id, sa.alignment_json
            FROM segment_alignments sa
            {where}
            ORDER BY sa.segment_id, sa.id""",
        params,
    )

    while batch := cursor.fetchmany(batch_size):
        for row in batch:
            for item in parse_alignment_json(row["alignment_json"]):
                feature_json = json.dumps(
                    {
                        "duration_ms": item.end_ms - item.start_ms,
                        "start_ms": item.start_ms,
                        "end_ms": item.end_ms,
                        "confidence": item.confidence,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                yield {
                    "segment_id": int(row["segment_id"]),
                    "word_occurrence_id": None,
                    "expected_phone": item.expected_phone,
                    "alignment_phone": item.alignment_phone,
                    "observed_model_phone": item.observed_model_phone,
                    "manual_phone": None,
                    "evidence_type": phone_evidence_type(item.expected_phone),
                    "feature_json": feature_json,
                }


def load_alignment_phone_instances(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
) -> list[dict[str, object]]:
    """Flatten segment alignment JSON into atlas phone-instance rows."""
    return list(
        iter_alignment_phone_instances(conn, aligner_version=aligner_version)
    )


def _load_audio(path: Path, *, sample_rate: int):
    import numpy as np
    import torch
    import torchaudio.functional as F
    import soundfile as sf

    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    tensor = torch.from_numpy(audio).float()
    if source_rate != sample_rate:
        tensor = F.resample(tensor, source_rate, sample_rate)
    return tensor


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "segment_id" not in (reader.fieldnames or []):
            raise ValueError("manifest must include segment_id column")
        return list(reader)


def _transcript_text_for_segment(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    source: str,
) -> str | None:
    row = conn.execute(
        """SELECT text FROM transcript_candidates
           WHERE segment_id=? AND source=?
           ORDER BY is_primary DESC, id DESC
           LIMIT 1""",
        (segment_id, source),
    ).fetchone()
    return row["text"] if row is not None else None


def _alignment_exists(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    aligner_version: str,
) -> bool:
    row = conn.execute(
        """SELECT 1 FROM segment_alignments
           WHERE segment_id=? AND aligner_version=?
           LIMIT 1""",
        (segment_id, aligner_version),
    ).fetchone()
    return row is not None


def run_mms_fa_alignments(
    *,
    db_path: Path,
    manifest_path: Path,
    audio_root: Path = AUDIO_ROOT,
    aligner_version: str = DEFAULT_ALIGNER_VERSION,
    limit: int | None = None,
    include_existing: bool = False,
    device: str = "auto",
    commit_batch: int = 50,
) -> dict[str, int]:
    """Run MMS_FA alignment over manifest rows and persist segment_alignments."""
    import torch
    import torchaudio

    from audio.atlas import expected_49_from_text

    runtime_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    bundle = torchaudio.pipelines.MMS_FA
    sample_rate = bundle.sample_rate
    model = bundle.get_model(dl_kwargs={"progress": False}).to(runtime_device)
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    stats = {
        "aligned": 0,
        "skipped_existing": 0,
        "skipped_missing_transcript": 0,
        "skipped_empty_projection": 0,
        "failed": 0,
    }
    try:
        rows = _manifest_rows(manifest_path)
        for row in rows:
            if limit is not None and stats["aligned"] >= limit:
                break
            segment_id = int(row["segment_id"])
            if (
                not include_existing
                and _alignment_exists(
                    conn,
                    segment_id=segment_id,
                    aligner_version=aligner_version,
                )
            ):
                stats["skipped_existing"] += 1
                continue

            transcript = _transcript_text_for_segment(
                conn,
                segment_id=segment_id,
                source=row["transcript_source"],
            )
            if transcript is None:
                stats["skipped_missing_transcript"] += 1
                continue
            try:
                expected_phones = expected_49_from_text(transcript)
                projected = project_expected_phones_for_mms_fa(expected_phones)
                if not projected:
                    stats["skipped_empty_projection"] += 1
                    continue
                target_text = "".join(item.model_token for item in projected)
                target_tokens = tokenizer([target_text])
                audio_path = Path(row["path"])
                if not audio_path.is_absolute():
                    audio_path = audio_root / audio_path
                waveform = _load_audio(audio_path, sample_rate=sample_rate)
                with torch.inference_mode():
                    emission, _ = model(waveform.to(runtime_device).unsqueeze(0))
                    spans = aligner(emission[0], target_tokens)[0]
                items = alignment_items_from_mms_fa_spans(
                    projected,
                    spans,
                    frame_count=int(emission.shape[1]),
                    duration_s=float(row["duration_s"]),
                )
                upsert_segment_alignment(
                    conn,
                    segment_id=segment_id,
                    expected_phones=expected_phones,
                    items=items,
                    aligner_version=aligner_version,
                )
                stats["aligned"] += 1
                if stats["aligned"] % commit_batch == 0:
                    conn.commit()
            except Exception:
                stats["failed"] += 1
                continue
        conn.commit()
    finally:
        conn.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=AUDIO_DB_PATH)
    parser.add_argument("--aligner-version", default=DEFAULT_ALIGNER_VERSION)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument(
        "--export-phone-instances",
        type=Path,
        help="flatten existing segment_alignments into a parquet file",
    )
    args = parser.parse_args(argv)

    if args.manifest is not None:
        stats = run_mms_fa_alignments(
            db_path=args.db,
            manifest_path=args.manifest,
            audio_root=args.audio_root,
            aligner_version=args.aligner_version,
            limit=args.limit,
            include_existing=args.include_existing,
            device=args.device,
        )
        print(
            "MMS_FA alignment "
            f"aligned={stats['aligned']} "
            f"skipped_existing={stats['skipped_existing']} "
            f"skipped_missing_transcript={stats['skipped_missing_transcript']} "
            f"skipped_empty_projection={stats['skipped_empty_projection']} "
            f"failed={stats['failed']}"
        )
        if args.export_phone_instances is None:
            return 0

    if args.export_phone_instances is None:
        parser.error("provide --manifest to align or --export-phone-instances to export")

    from audio.atlas_export import PhoneInstance, write_phone_instances

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            PhoneInstance(**item)
            for item in load_alignment_phone_instances(
                conn,
                aligner_version=args.aligner_version,
            )
        ]
    finally:
        conn.close()
    write_phone_instances(rows, args.export_phone_instances)
    print(f"Wrote {len(rows):,} phone instances to {args.export_phone_instances}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
