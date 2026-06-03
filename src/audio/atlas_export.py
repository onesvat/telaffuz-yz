"""Parquet and evidence-report exports for phoneme atlas rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from g2p.constants import ALL_PHONEMES, ALL_VOWELS, CTC_SPC, STRESS
from g2p.syllabifier import syllabify

from audio.atlas import ATLAS_ROOT, MMS_ALIGNMENT_COLLAPSE, phone_evidence_type
from audio.db import AUDIO_DB_PATH, AUDIO_ROOT
from audio.manual import MANUAL_REVIEW_PHONE_ORDER, manual_annotation_summary


PHONE_INSTANCE_COLUMNS = [
    "segment_id",
    "word_occurrence_id",
    "expected_phone",
    "alignment_phone",
    "observed_model_phone",
    "manual_phone",
    "evidence_type",
    "feature_json",
]

PHONE_SUMMARY_COLUMNS = [
    "expected_phone",
    "evidence_type",
    "instance_count",
    "provider_distribution_json",
    "mean_confidence",
    "min_confidence",
    "max_confidence",
    "confidence_count",
    "model_match_count",
    "manual_phone_count",
    "manual_accepted",
    "manual_uncertain",
    "manual_reject",
    "evidence_class",
    "risk_class",
    "stress_event_count",
    "stress_evaluable_count",
    "stress_match_count",
    "stress_mismatch_count",
    "stress_low_confidence_count",
    "stress_not_applicable_count",
    "stress_match_rate",
    "example_segment_ids",
]

STRESS_SYLLABLE_INSTANCE_COLUMNS = [
    "segment_id",
    "word_index",
    "predicted_stress_syllable",
    "acoustic_peak_syllable",
    "syllable_count",
    "stressed_syllable_start_ms",
    "stressed_syllable_end_ms",
    "acoustic_peak_start_ms",
    "acoustic_peak_end_ms",
    "stressed_f0_hz",
    "stressed_intensity_db",
    "stressed_duration_ms",
    "peak_f0_hz",
    "peak_intensity_db",
    "peak_duration_ms",
    "verdict",
    "cues_json",
    "provider",
]


@dataclass(frozen=True)
class PhoneInstance:
    segment_id: int
    word_occurrence_id: int | None
    expected_phone: str
    alignment_phone: str | None
    observed_model_phone: str | None
    manual_phone: str | None
    evidence_type: str
    feature_json: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StressSyllableInstance:
    segment_id: int
    word_index: int
    predicted_stress_syllable: int | None
    acoustic_peak_syllable: int | None
    syllable_count: int
    stressed_syllable_start_ms: int | None
    stressed_syllable_end_ms: int | None
    acoustic_peak_start_ms: int | None
    acoustic_peak_end_ms: int | None
    stressed_f0_hz: float | None
    stressed_intensity_db: float | None
    stressed_duration_ms: float | None
    peak_f0_hz: float | None
    peak_intensity_db: float | None
    peak_duration_ms: float | None
    verdict: str
    cues_json: str
    provider: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _phone_instance_arrow_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised by deployment envs.
        raise RuntimeError("pyarrow is required for atlas parquet export") from exc

    return pa.schema(
        [
            pa.field("segment_id", pa.int64()),
            pa.field("word_occurrence_id", pa.int64()),
            pa.field("expected_phone", pa.string()),
            pa.field("alignment_phone", pa.string()),
            pa.field("observed_model_phone", pa.string()),
            pa.field("manual_phone", pa.string()),
            pa.field("evidence_type", pa.string()),
            pa.field("feature_json", pa.string()),
        ]
    )


def _stress_syllable_arrow_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised by deployment envs.
        raise RuntimeError("pyarrow is required for atlas parquet export") from exc

    return pa.schema(
        [
            pa.field("segment_id", pa.int64()),
            pa.field("word_index", pa.int64()),
            pa.field("predicted_stress_syllable", pa.int64()),
            pa.field("acoustic_peak_syllable", pa.int64()),
            pa.field("syllable_count", pa.int64()),
            pa.field("stressed_syllable_start_ms", pa.int64()),
            pa.field("stressed_syllable_end_ms", pa.int64()),
            pa.field("acoustic_peak_start_ms", pa.int64()),
            pa.field("acoustic_peak_end_ms", pa.int64()),
            pa.field("stressed_f0_hz", pa.float64()),
            pa.field("stressed_intensity_db", pa.float64()),
            pa.field("stressed_duration_ms", pa.float64()),
            pa.field("peak_f0_hz", pa.float64()),
            pa.field("peak_intensity_db", pa.float64()),
            pa.field("peak_duration_ms", pa.float64()),
            pa.field("verdict", pa.string()),
            pa.field("cues_json", pa.string()),
            pa.field("provider", pa.string()),
        ]
    )


def _write_phone_instance_batch(writer, batch: list[dict[str, object]], schema) -> None:
    import pyarrow as pa

    table = pa.Table.from_pylist(batch, schema=schema)
    writer.write_table(table)


def _write_stress_syllable_batch(
    writer,
    batch: list[dict[str, object]],
    schema,
) -> None:
    import pyarrow as pa

    table = pa.Table.from_pylist(batch, schema=schema)
    writer.write_table(table)


def write_phone_instances(
    rows: Iterable[PhoneInstance],
    out_path: Path,
    *,
    batch_size: int = 100_000,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by deployment envs.
        raise RuntimeError("pyarrow is required for atlas parquet export") from exc

    schema = _phone_instance_arrow_schema()
    writer = pq.ParquetWriter(out_path, schema)
    batch: list[dict[str, object]] = []
    count = 0
    try:
        for row in rows:
            batch.append(row.as_dict())
            if len(batch) >= batch_size:
                _write_phone_instance_batch(writer, batch, schema)
                count += len(batch)
                batch.clear()
        if batch:
            _write_phone_instance_batch(writer, batch, schema)
            count += len(batch)
    finally:
        writer.close()
    return count


def write_stress_syllable_instances(
    rows: Iterable[StressSyllableInstance],
    out_path: Path,
    *,
    batch_size: int = 100_000,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by deployment envs.
        raise RuntimeError("pyarrow is required for atlas parquet export") from exc

    schema = _stress_syllable_arrow_schema()
    writer = pq.ParquetWriter(out_path, schema)
    batch: list[dict[str, object]] = []
    count = 0
    try:
        for row in rows:
            batch.append(row.as_dict())
            if len(batch) >= batch_size:
                _write_stress_syllable_batch(writer, batch, schema)
                count += len(batch)
                batch.clear()
        if batch:
            _write_stress_syllable_batch(writer, batch, schema)
            count += len(batch)
    finally:
        writer.close()
    return count


def coverage_by_phone(
    rows: Iterable[PhoneInstance],
) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {
        phone: {
            "instance_count": 0,
            "model_match_count": 0,
            "manual_phone_count": 0,
        }
        for phone in sorted(ALL_PHONEMES)
    }
    for row in rows:
        bucket = coverage.setdefault(
            row.expected_phone,
            {
                "instance_count": 0,
                "model_match_count": 0,
                "manual_phone_count": 0,
            },
        )
        bucket["instance_count"] += 1
        if (
            row.alignment_phone is not None
            and row.observed_model_phone == row.alignment_phone
        ):
            bucket["model_match_count"] += 1
        if row.manual_phone == row.expected_phone:
            bucket["manual_phone_count"] += 1
    return coverage


def evidence_class(row: PhoneInstance) -> str:
    if row.evidence_type == "manual-only":
        return "supported" if row.manual_phone == row.expected_phone else "manual_review_required"
    if row.evidence_type == "feature-derived":
        return "supported" if row.feature_json else "acoustic_mismatch"
    if row.alignment_phone is not None and row.observed_model_phone != row.alignment_phone:
        return "model_mismatch"
    return "supported"


def write_g2p_audio_evidence_csv(
    rows: Iterable[PhoneInstance],
    out_path: Path,
) -> None:
    rows = list(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[PhoneInstance]] = {}
    for row in rows:
        grouped.setdefault(row.expected_phone, []).append(row)

    fieldnames = [
        "expected_phone",
        "evidence_type",
        "instance_count",
        "model_match_count",
        "manual_phone_count",
        "evidence_class",
        "example_segment_ids",
    ]
    coverage = coverage_by_phone(rows)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for phone in sorted(grouped):
            phone_rows = grouped[phone]
            classes = {evidence_class(row) for row in phone_rows}
            counts = coverage[phone]
            if (
                phone_evidence_type(phone) == "manual-only"
                and counts["manual_phone_count"] > 0
            ):
                cls = "supported"
            elif "manual_review_required" in classes:
                cls = "manual_review_required"
            elif "model_mismatch" in classes:
                cls = "model_mismatch"
            elif "acoustic_mismatch" in classes:
                cls = "acoustic_mismatch"
            else:
                cls = "supported"
            writer.writerow(
                {
                    "expected_phone": phone,
                    "evidence_type": phone_evidence_type(phone),
                    "instance_count": counts["instance_count"],
                    "model_match_count": counts["model_match_count"],
                    "manual_phone_count": counts["manual_phone_count"],
                    "evidence_class": cls,
                    "example_segment_ids": " ".join(
                        str(row.segment_id) for row in phone_rows[:10]
                    ),
                }
            )


def instances_from_alignment_db(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
) -> list[PhoneInstance]:
    return list(
        iter_instances_from_alignment_db(conn, aligner_version=aligner_version)
    )


def _load_accepted_manual_annotations(
    conn: sqlite3.Connection,
) -> dict[int, list[dict[str, object]]]:
    rows = conn.execute(
        """SELECT entity_id, value_json
           FROM manual_annotations
           WHERE entity_type = 'segment'
             AND label = 'manual_phone'
           ORDER BY id"""
    ).fetchall()
    by_segment: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        data = json.loads(row["value_json"])
        if data.get("status") != "accepted":
            continue
        phone = data.get("phone")
        if not phone:
            continue
        segment_id = int(data.get("segment_id") or row["entity_id"])
        by_segment.setdefault(segment_id, []).append(
            {
                "phone": phone,
                "start_ms": int(round(float(data["start_s"]) * 1000.0)),
                "end_ms": int(round(float(data["end_s"]) * 1000.0)),
            }
        )
    return by_segment


def _manual_overlap_ms(
    row: PhoneInstance,
    annotation: dict[str, object],
) -> int:
    feature = json.loads(row.feature_json or "{}")
    start_ms = int(feature.get("start_ms", 0))
    end_ms = int(feature.get("end_ms", 0))
    manual_start = int(annotation["start_ms"])
    manual_end = int(annotation["end_ms"])
    return max(0, min(end_ms, manual_end) - max(start_ms, manual_start))


def _overlay_manual_annotations(
    conn: sqlite3.Connection,
    rows: Iterable[PhoneInstance],
) -> list[PhoneInstance]:
    accepted_by_segment = _load_accepted_manual_annotations(conn)
    return [_overlay_manual_annotation(row, accepted_by_segment) for row in rows]


def _overlay_manual_annotation(
    row: PhoneInstance,
    accepted_by_segment: dict[int, list[dict[str, object]]],
) -> PhoneInstance:
    matches = [
        annotation
        for annotation in accepted_by_segment.get(row.segment_id, [])
        if annotation["phone"] == row.expected_phone
    ]
    if not matches:
        return row
    best = max(matches, key=lambda annotation: _manual_overlap_ms(row, annotation))
    if _manual_overlap_ms(row, best) <= 0:
        return row
    return replace(row, manual_phone=str(best["phone"]))


def iter_instances_from_alignment_db(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
) -> Iterable[PhoneInstance]:
    from audio.alignment import iter_alignment_phone_instances

    accepted_by_segment = _load_accepted_manual_annotations(conn)
    for item in iter_alignment_phone_instances(conn, aligner_version=aligner_version):
        yield _overlay_manual_annotation(PhoneInstance(**item), accepted_by_segment)


def _manifest_entries(manifest_path: Path | None) -> dict[int, dict[str, str]]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    entries: dict[int, dict[str, str]] = {}
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                segment_id = int(row.get("segment_id") or "")
            except ValueError:
                continue
            entries[segment_id] = {
                "path": row.get("path", ""),
                "provider": row.get("provider", ""),
            }
    return entries


def _audio_path_for_segment(
    *,
    segment_id: int,
    segment_path: str,
    manifest_entries: dict[int, dict[str, str]],
    audio_root: Path,
) -> Path | None:
    rel_path = manifest_entries.get(segment_id, {}).get("path") or segment_path
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path if path.exists() else None
    candidate = audio_root / rel_path
    if candidate.exists():
        return candidate
    local_candidate = ATLAS_ROOT.parent / "audio" / "local" / rel_path
    if local_candidate.exists():
        return local_candidate
    repo_candidate = ATLAS_ROOT.parent.parent / rel_path
    if repo_candidate.exists():
        return repo_candidate
    return None


def _load_audio_for_segment(path: Path | None) -> tuple[Any, int] | None:
    if path is None:
        return None
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError, ValueError):
        return None
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def _alignment_slots(
    tokens: list[str],
    alignment_items: Iterable[Any],
) -> list[Any | None]:
    items = iter(alignment_items)
    slots: list[Any | None] = []
    for token in tokens:
        if token in {STRESS, CTC_SPC}:
            slots.append(None)
            continue
        slots.append(next(items, None))
    return slots


def _word_chunks(
    tokens: list[str],
    slots: list[Any | None],
) -> list[tuple[int, list[tuple[str, Any | None]]]]:
    chunks: list[tuple[int, list[tuple[str, Any | None]]]] = []
    current: list[tuple[str, Any | None]] = []
    word_index = 0
    for token, slot in zip(tokens, slots, strict=True):
        if token == CTC_SPC:
            if current:
                chunks.append((word_index, current))
                current = []
            word_index += 1
            continue
        current.append((token, slot))
    if current:
        chunks.append((word_index, current))
    return chunks


def _syllable_index_by_plain_phone(plain: list[str]) -> list[int]:
    syllables = syllabify(plain)
    if not syllables:
        return []
    indices: list[int] = []
    for syllable_index, syllable in enumerate(syllables):
        indices.extend([syllable_index] * len(syllable))
    if len(indices) < len(plain):
        indices.extend([len(syllables) - 1] * (len(plain) - len(indices)))
    return indices


def _predicted_stress_syllable_for_chunk(
    chunk: list[tuple[str, Any | None]],
    stress_pos: int,
    chunk_to_plain: dict[int, int],
    plain_syllables: list[int],
) -> int | None:
    for idx in range(stress_pos + 1, len(chunk)):
        token = chunk[idx][0]
        if token in ALL_VOWELS:
            plain_idx = chunk_to_plain.get(idx)
            if plain_idx is not None and plain_idx < len(plain_syllables):
                return plain_syllables[plain_idx]
    return None


def _duration(item: Any | None) -> int:
    if item is None:
        return 0
    return max(0, int(item.end_ms) - int(item.start_ms))


def _extract_syllable_acoustics(
    *,
    audio_context: tuple[Any, int] | None,
    start_ms: int | None,
    end_ms: int | None,
    nucleus_phone: str | None,
) -> tuple[float | None, float | None]:
    if audio_context is None or start_ms is None or end_ms is None or end_ms <= start_ms:
        return None, None
    try:
        import numpy as np
    except ImportError:
        return None, None
    audio, sample_rate = audio_context
    start = max(0, int(round(start_ms * sample_rate / 1000.0)))
    end = min(len(audio), int(round(end_ms * sample_rate / 1000.0)))
    if end <= start:
        return None, None
    samples = np.asarray(audio[start:end], dtype=np.float32)
    if samples.size == 0:
        return None, None
    rms = float(np.sqrt(np.mean(np.square(samples))))
    intensity = 20.0 * math.log10(rms + 1e-12)
    if nucleus_phone not in ALL_VOWELS or samples.size < sample_rate // 100:
        return None, intensity

    centered = samples - float(np.mean(samples))
    if not np.any(centered):
        return None, intensity
    fft_size = 1 << int((2 * centered.size - 1).bit_length())
    spectrum = np.fft.rfft(centered, n=fft_size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: centered.size]
    if corr.size == 0 or corr[0] <= 0:
        return None, intensity
    min_lag = max(1, int(sample_rate / 500))
    max_lag = min(corr.size, int(sample_rate / 60))
    if max_lag <= min_lag:
        return None, intensity
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    if lag <= 0:
        return None, intensity
    return float(sample_rate / lag), intensity


def _syllable_metrics(
    *,
    plain: list[str],
    plain_slots: list[Any | None],
    plain_syllables: list[int],
    audio_context: tuple[Any, int] | None,
) -> list[dict[str, float | int | None]]:
    if not plain:
        return []
    syllable_count = max(plain_syllables, default=0) + 1
    metrics: list[dict[str, float | int | None]] = []
    for syllable_index in range(syllable_count):
        phone_indices = [
            idx for idx, value in enumerate(plain_syllables) if value == syllable_index
        ]
        timed_items = [
            plain_slots[idx] for idx in phone_indices if plain_slots[idx] is not None
        ]
        start_ms = (
            min(int(item.start_ms) for item in timed_items) if timed_items else None
        )
        end_ms = max(int(item.end_ms) for item in timed_items) if timed_items else None
        duration_ms = (
            float(max(0, end_ms - start_ms))
            if start_ms is not None and end_ms is not None
            else float(sum(_duration(item) for item in timed_items))
        )
        nucleus_phone = next(
            (plain[idx] for idx in phone_indices if plain[idx] in ALL_VOWELS),
            None,
        )
        f0_hz, intensity_db = _extract_syllable_acoustics(
            audio_context=audio_context,
            start_ms=start_ms,
            end_ms=end_ms,
            nucleus_phone=nucleus_phone,
        )
        metrics.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "f0_hz": f0_hz,
                "intensity_db": intensity_db,
            }
        )
    return metrics


def _minmax_norm(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def stress_peak_verdict(
    *,
    predicted_stress_syllable: int | None,
    syllables: list[dict[str, float | int | None]],
) -> tuple[str, int | None, dict[str, object]]:
    """Return stress verdict using only the syllables inside one word chunk."""
    syllable_count = len(syllables)
    if syllable_count <= 1:
        return (
            "not_applicable",
            0 if syllable_count == 1 else None,
            {"cues": ["single_syllable"], "used_dimensions": [], "scores": []},
        )
    if predicted_stress_syllable is None or not (
        0 <= predicted_stress_syllable < syllable_count
    ):
        return (
            "low_confidence",
            None,
            {
                "cues": ["no_predicted_stress_syllable"],
                "used_dimensions": [],
                "scores": [],
            },
        )

    scores = [0.0 for _ in syllables]
    used_dimensions: list[str] = []
    for key in ("f0_hz", "intensity_db", "duration_ms"):
        values: list[float | None] = []
        for syllable in syllables:
            value = syllable.get(key)
            values.append(float(value) if isinstance(value, (int, float)) else None)
        present = [value for value in values if value is not None]
        if len(present) < 2 or max(present) - min(present) < 1e-9:
            continue
        floor = min(present)
        normalized = _minmax_norm([value if value is not None else floor for value in values])
        scores = [score + normalized[idx] for idx, score in enumerate(scores)]
        used_dimensions.append(key)

    if not used_dimensions:
        return (
            "low_confidence",
            None,
            {"cues": ["flat_acoustic_profile"], "used_dimensions": [], "scores": scores},
        )

    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    peak = ranked[0]
    if len(ranked) > 1 and abs(scores[ranked[0]] - scores[ranked[1]]) < 1e-9:
        return (
            "low_confidence",
            peak,
            {
                "cues": ["tied_acoustic_peak"],
                "used_dimensions": used_dimensions,
                "scores": scores,
            },
        )

    if peak == predicted_stress_syllable:
        return (
            "ok",
            peak,
            {
                "cues": ["stress_match"],
                "used_dimensions": used_dimensions,
                "scores": scores,
            },
        )
    return (
        "mispronounced",
        peak,
        {
            "cues": ["stress_mismatch", f"acoustic_peak_syllable={peak}"],
            "used_dimensions": used_dimensions,
            "scores": scores,
        },
    )


def _metric_at(
    syllables: list[dict[str, float | int | None]],
    idx: int | None,
    key: str,
) -> float | int | None:
    if idx is None or idx < 0 or idx >= len(syllables):
        return None
    return syllables[idx].get(key)


def _stress_instances_for_segment(
    *,
    segment_id: int,
    provider: str,
    tokens: list[str],
    alignment_items: Iterable[Any],
    audio_context: tuple[Any, int] | None,
) -> list[StressSyllableInstance]:
    slots = _alignment_slots(tokens, alignment_items)
    instances: list[StressSyllableInstance] = []

    for word_index, chunk in _word_chunks(tokens, slots):
        stress_positions = [
            idx for idx, (token, _slot) in enumerate(chunk) if token == STRESS
        ]
        if not stress_positions:
            continue

        plain: list[str] = []
        plain_slots: list[Any | None] = []
        chunk_to_plain: dict[int, int] = {}
        for idx, (token, slot) in enumerate(chunk):
            if token == STRESS:
                continue
            chunk_to_plain[idx] = len(plain)
            plain.append(token)
            plain_slots.append(slot)

        plain_syllables = _syllable_index_by_plain_phone(plain)
        syllables = _syllable_metrics(
            plain=plain,
            plain_slots=plain_slots,
            plain_syllables=plain_syllables,
            audio_context=audio_context,
        )

        for stress_pos in stress_positions:
            predicted = _predicted_stress_syllable_for_chunk(
                chunk,
                stress_pos,
                chunk_to_plain,
                plain_syllables,
            )
            verdict, peak, cues = stress_peak_verdict(
                predicted_stress_syllable=predicted,
                syllables=syllables,
            )
            cues_payload = {
                **cues,
                "word_index": word_index,
                "predicted_stress_syllable": predicted,
                "acoustic_peak_syllable": peak,
            }
            instances.append(
                StressSyllableInstance(
                    segment_id=segment_id,
                    word_index=word_index,
                    predicted_stress_syllable=predicted,
                    acoustic_peak_syllable=peak,
                    syllable_count=len(syllables),
                    stressed_syllable_start_ms=_metric_at(syllables, predicted, "start_ms"),
                    stressed_syllable_end_ms=_metric_at(syllables, predicted, "end_ms"),
                    acoustic_peak_start_ms=_metric_at(syllables, peak, "start_ms"),
                    acoustic_peak_end_ms=_metric_at(syllables, peak, "end_ms"),
                    stressed_f0_hz=_metric_at(syllables, predicted, "f0_hz"),
                    stressed_intensity_db=_metric_at(
                        syllables, predicted, "intensity_db"
                    ),
                    stressed_duration_ms=_metric_at(
                        syllables, predicted, "duration_ms"
                    ),
                    peak_f0_hz=_metric_at(syllables, peak, "f0_hz"),
                    peak_intensity_db=_metric_at(syllables, peak, "intensity_db"),
                    peak_duration_ms=_metric_at(syllables, peak, "duration_ms"),
                    verdict=verdict,
                    cues_json=json.dumps(
                        cues_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    provider=provider,
                )
            )
    return instances


def iter_stress_syllable_instances_from_alignment_db(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
    manifest_path: Path | None = None,
    audio_root: Path = AUDIO_ROOT,
    batch_size: int = 1_000,
) -> Iterable[StressSyllableInstance]:
    from audio.alignment import parse_alignment_json

    manifest_entries = _manifest_entries(manifest_path)
    params: list[object] = []
    where = ""
    if aligner_version is not None:
        where = "WHERE sa.aligner_version = ?"
        params.append(aligner_version)

    cursor = conn.execute(
        f"""SELECT sa.segment_id,
                   sa.alignment_json,
                   ps.phonemes_json,
                   s.provider,
                   s.path
            FROM segment_alignments sa
            JOIN phoneme_sequences ps ON ps.id = sa.source_sequence_id
            JOIN segments s ON s.id = sa.segment_id
            {where}
            ORDER BY sa.segment_id, sa.id""",
        params,
    )

    while batch := cursor.fetchmany(batch_size):
        for row in batch:
            tokens = list(json.loads(row["phonemes_json"]))
            if STRESS not in tokens:
                continue
            segment_id = int(row["segment_id"])
            provider = (
                manifest_entries.get(segment_id, {}).get("provider")
                or str(row["provider"])
            )
            audio_path = _audio_path_for_segment(
                segment_id=segment_id,
                segment_path=str(row["path"]),
                manifest_entries=manifest_entries,
                audio_root=audio_root,
            )
            audio_context = _load_audio_for_segment(audio_path)
            yield from _stress_instances_for_segment(
                segment_id=segment_id,
                provider=provider,
                tokens=tokens,
                alignment_items=parse_alignment_json(row["alignment_json"]),
                audio_context=audio_context,
            )


def build_stress_syllable_instances(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
    manifest_path: Path | None = None,
    audio_root: Path = AUDIO_ROOT,
) -> list[StressSyllableInstance]:
    return list(
        iter_stress_syllable_instances_from_alignment_db(
            conn,
            aligner_version=aligner_version,
            manifest_path=manifest_path,
            audio_root=audio_root,
        )
    )


def _segment_provider_map(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("SELECT id, provider FROM segments").fetchall()
    return {int(row["id"]): str(row["provider"]) for row in rows}


def _row_confidence(row: PhoneInstance) -> float | None:
    feature = json.loads(row.feature_json or "{}")
    confidence = feature.get("confidence")
    return None if confidence is None else float(confidence)


def _summary_risk_class(
    *,
    phone: str,
    evidence_type: str,
    instance_count: int,
    mean_confidence: float | None,
    manual_accepted: int,
) -> str:
    if phone == STRESS:
        return "supported" if instance_count > 0 else "low_count"
    if evidence_type == "rule-only":
        return "rule_only"
    if evidence_type == "feature-derived":
        return "feature_missing"
    if instance_count == 0:
        return "low_count"
    if evidence_type == "manual-only":
        return "supported" if manual_accepted > 0 else "manual_risk"
    if mean_confidence is not None and mean_confidence < 0.80:
        return "alignment_risk"
    return "supported"


def phone_summary_rows(
    conn: sqlite3.Connection,
    rows: Iterable[PhoneInstance],
    *,
    stress_rows: Iterable[StressSyllableInstance] | None = None,
) -> list[dict[str, object]]:
    provider_by_segment = _segment_provider_map(conn)
    manual_summary = manual_annotation_summary(conn)
    buckets: dict[str, dict[str, object]] = {
        phone: {
            "instance_count": 0,
            "provider_counts": Counter(),
            "confidence_sum": 0.0,
            "confidence_count": 0,
            "min_confidence": None,
            "max_confidence": None,
            "model_match_count": 0,
            "manual_phone_count": 0,
            "stress_event_count": 0,
            "stress_evaluable_count": 0,
            "stress_match_count": 0,
            "stress_mismatch_count": 0,
            "stress_low_confidence_count": 0,
            "stress_not_applicable_count": 0,
            "example_segment_ids": [],
        }
        for phone in sorted(ALL_PHONEMES)
    }

    for row in rows:
        bucket = buckets.setdefault(
            row.expected_phone,
            {
                "instance_count": 0,
                "provider_counts": Counter(),
                "confidence_sum": 0.0,
                "confidence_count": 0,
                "min_confidence": None,
                "max_confidence": None,
                "model_match_count": 0,
                "manual_phone_count": 0,
                "stress_event_count": 0,
                "stress_evaluable_count": 0,
                "stress_match_count": 0,
                "stress_mismatch_count": 0,
                "stress_low_confidence_count": 0,
                "stress_not_applicable_count": 0,
                "example_segment_ids": [],
            },
        )
        bucket["instance_count"] = int(bucket["instance_count"]) + 1
        provider_counts = bucket["provider_counts"]
        assert isinstance(provider_counts, Counter)
        provider_counts[provider_by_segment.get(row.segment_id, "unknown")] += 1

        confidence = _row_confidence(row)
        if confidence is not None:
            bucket["confidence_sum"] = float(bucket["confidence_sum"]) + confidence
            bucket["confidence_count"] = int(bucket["confidence_count"]) + 1
            min_conf = bucket["min_confidence"]
            max_conf = bucket["max_confidence"]
            bucket["min_confidence"] = (
                confidence if min_conf is None else min(float(min_conf), confidence)
            )
            bucket["max_confidence"] = (
                confidence if max_conf is None else max(float(max_conf), confidence)
            )
        if (
            row.alignment_phone is not None
            and row.observed_model_phone == row.alignment_phone
        ):
            bucket["model_match_count"] = int(bucket["model_match_count"]) + 1
        if row.manual_phone == row.expected_phone:
            bucket["manual_phone_count"] = int(bucket["manual_phone_count"]) + 1
        examples = bucket["example_segment_ids"]
        assert isinstance(examples, list)
        if len(examples) < 10 and row.segment_id not in examples:
            examples.append(row.segment_id)

    stress_bucket = buckets[STRESS]
    stress_provider_counts = stress_bucket["provider_counts"]
    assert isinstance(stress_provider_counts, Counter)
    stress_examples = stress_bucket["example_segment_ids"]
    assert isinstance(stress_examples, list)
    for row in stress_rows or ():
        stress_bucket["stress_event_count"] = (
            int(stress_bucket["stress_event_count"]) + 1
        )
        stress_provider_counts[row.provider or "unknown"] += 1
        if row.verdict == "ok":
            stress_bucket["stress_match_count"] = (
                int(stress_bucket["stress_match_count"]) + 1
            )
            stress_bucket["stress_evaluable_count"] = (
                int(stress_bucket["stress_evaluable_count"]) + 1
            )
        elif row.verdict == "mispronounced":
            stress_bucket["stress_mismatch_count"] = (
                int(stress_bucket["stress_mismatch_count"]) + 1
            )
            stress_bucket["stress_evaluable_count"] = (
                int(stress_bucket["stress_evaluable_count"]) + 1
            )
        elif row.verdict == "low_confidence":
            stress_bucket["stress_low_confidence_count"] = (
                int(stress_bucket["stress_low_confidence_count"]) + 1
            )
        elif row.verdict == "not_applicable":
            stress_bucket["stress_not_applicable_count"] = (
                int(stress_bucket["stress_not_applicable_count"]) + 1
            )
        if len(stress_examples) < 10 and row.segment_id not in stress_examples:
            stress_examples.append(row.segment_id)

    summary_rows: list[dict[str, object]] = []
    for phone in sorted(buckets):
        bucket = buckets[phone]
        instance_count = int(bucket["instance_count"])
        if phone == STRESS:
            instance_count = int(bucket["stress_event_count"])
        confidence_count = int(bucket["confidence_count"])
        mean_confidence = (
            float(bucket["confidence_sum"]) / confidence_count
            if confidence_count
            else None
        )
        evidence_type = phone_evidence_type(phone)
        manual_accepted = manual_summary.phone_status_counts.get(
            (phone, "accepted"), 0
        )
        risk_class = _summary_risk_class(
            phone=phone,
            evidence_type=evidence_type,
            instance_count=instance_count,
            mean_confidence=mean_confidence,
            manual_accepted=manual_accepted,
        )
        provider_counts = bucket["provider_counts"]
        assert isinstance(provider_counts, Counter)
        examples = bucket["example_segment_ids"]
        assert isinstance(examples, list)
        stress_evaluable = int(bucket["stress_evaluable_count"])
        stress_match = int(bucket["stress_match_count"])
        stress_match_rate = (
            stress_match / stress_evaluable if stress_evaluable else None
        )
        summary_rows.append(
            {
                "expected_phone": phone,
                "evidence_type": evidence_type,
                "instance_count": instance_count,
                "provider_distribution_json": json.dumps(
                    dict(sorted(provider_counts.items())),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "mean_confidence": (
                    "" if mean_confidence is None else f"{mean_confidence:.6f}"
                ),
                "min_confidence": (
                    ""
                    if bucket["min_confidence"] is None
                    else f"{float(bucket['min_confidence']):.6f}"
                ),
                "max_confidence": (
                    ""
                    if bucket["max_confidence"] is None
                    else f"{float(bucket['max_confidence']):.6f}"
                ),
                "confidence_count": confidence_count,
                "model_match_count": int(bucket["model_match_count"]),
                "manual_phone_count": int(bucket["manual_phone_count"]),
                "manual_accepted": manual_accepted,
                "manual_uncertain": manual_summary.phone_status_counts.get(
                    (phone, "uncertain"), 0
                ),
                "manual_reject": manual_summary.phone_status_counts.get(
                    (phone, "reject"), 0
                ),
                "evidence_class": "supported"
                if risk_class == "supported"
                else risk_class,
                "risk_class": risk_class,
                "stress_event_count": int(bucket["stress_event_count"]),
                "stress_evaluable_count": stress_evaluable,
                "stress_match_count": stress_match,
                "stress_mismatch_count": int(bucket["stress_mismatch_count"]),
                "stress_low_confidence_count": int(
                    bucket["stress_low_confidence_count"]
                ),
                "stress_not_applicable_count": int(
                    bucket["stress_not_applicable_count"]
                ),
                "stress_match_rate": (
                    "" if stress_match_rate is None else f"{stress_match_rate:.6f}"
                ),
                "example_segment_ids": " ".join(str(item) for item in examples),
            }
        )
    return summary_rows


def build_phone_summary_rows(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
    stress_rows: Iterable[StressSyllableInstance] | None = None,
) -> list[dict[str, object]]:
    return phone_summary_rows(
        conn,
        iter_instances_from_alignment_db(conn, aligner_version=aligner_version),
        stress_rows=stress_rows,
    )


def write_phone_summary_csv(
    rows: Iterable[dict[str, object]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PHONE_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in PHONE_SUMMARY_COLUMNS}
            )


def write_g2p_audio_evidence_summary_csv(
    rows: Iterable[dict[str, object]],
    out_path: Path,
) -> None:
    fieldnames = [
        "expected_phone",
        "evidence_type",
        "instance_count",
        "model_match_count",
        "manual_phone_count",
        "evidence_class",
        "risk_class",
        "stress_event_count",
        "stress_evaluable_count",
        "stress_match_count",
        "stress_mismatch_count",
        "stress_low_confidence_count",
        "stress_not_applicable_count",
        "stress_match_rate",
        "example_segment_ids",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _manifest_coverage(
    conn: sqlite3.Connection,
    *,
    manifest_path: Path | None,
    aligner_version: str | None,
) -> dict[str, object]:
    params: list[object] = []
    where = ""
    if aligner_version is not None:
        where = "WHERE aligner_version = ?"
        params.append(aligner_version)
    aligned_ids = {
        int(row["segment_id"])
        for row in conn.execute(
            f"SELECT segment_id FROM segment_alignments {where}",
            params,
        )
    }
    if manifest_path is None:
        return {
            "manifest_segments": None,
            "aligned_manifest_segments": None,
            "missing_manifest_segments": None,
            "extra_aligned_segments": None,
            "coverage_pct": None,
        }

    with manifest_path.open(encoding="utf-8", newline="") as fh:
        manifest_ids = {
            int(row["segment_id"])
            for row in csv.DictReader(fh)
            if row.get("segment_id")
        }
    aligned_manifest = manifest_ids & aligned_ids
    missing = manifest_ids - aligned_ids
    extra = aligned_ids - manifest_ids
    coverage_pct = (
        100.0 * len(aligned_manifest) / len(manifest_ids) if manifest_ids else 0.0
    )
    return {
        "manifest_segments": len(manifest_ids),
        "aligned_manifest_segments": len(aligned_manifest),
        "missing_manifest_segments": len(missing),
        "extra_aligned_segments": len(extra),
        "coverage_pct": coverage_pct,
    }


def _overall_confidence_from_summary(rows: Iterable[dict[str, object]]) -> float | None:
    weighted_sum = 0.0
    count = 0
    for row in rows:
        mean_text = str(row.get("mean_confidence", ""))
        confidence_count = int(row.get("confidence_count") or 0)
        if not mean_text or confidence_count <= 0:
            continue
        weighted_sum += float(mean_text) * confidence_count
        count += confidence_count
    return weighted_sum / count if count else None


def render_full_atlas_report(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
    manifest_path: Path | None = None,
    summary_rows: list[dict[str, object]] | None = None,
) -> str:
    summary_rows = summary_rows or build_phone_summary_rows(
        conn,
        aligner_version=aligner_version,
    )
    coverage = _manifest_coverage(
        conn,
        manifest_path=manifest_path,
        aligner_version=aligner_version,
    )
    confidence = _overall_confidence_from_summary(summary_rows)
    confidence_text = "n/a" if confidence is None else f"{confidence:.4f}"
    phone_windows = sum(
        int(row["instance_count"])
        for row in summary_rows
        if row["expected_phone"] != STRESS
    )
    stress_row = next(
        (row for row in summary_rows if row["expected_phone"] == STRESS),
        None,
    )
    stress_events = int(stress_row.get("stress_event_count") or 0) if stress_row else 0
    stress_evaluable = (
        int(stress_row.get("stress_evaluable_count") or 0) if stress_row else 0
    )
    stress_match = int(stress_row.get("stress_match_count") or 0) if stress_row else 0
    stress_mismatch = (
        int(stress_row.get("stress_mismatch_count") or 0) if stress_row else 0
    )
    stress_low_confidence = (
        int(stress_row.get("stress_low_confidence_count") or 0) if stress_row else 0
    )
    stress_not_applicable = (
        int(stress_row.get("stress_not_applicable_count") or 0) if stress_row else 0
    )
    stress_match_rate = (
        str(stress_row.get("stress_match_rate") or "n/a") if stress_row else "n/a"
    )

    provider_lines = [
        f"| {provider} | {segment_count:,} | {seconds / 3600.0:.2f} |"
        for provider, segment_count, seconds in _alignment_provider_summary(
            conn,
            aligner_version=aligner_version,
        )
    ]
    phone_lines = [
        "| "
        f"`{row['expected_phone']}` | "
        f"{row['evidence_type']} | "
        f"{int(row['instance_count']):,} | "
        f"{row['mean_confidence'] or 'n/a'} | "
        f"{row['manual_accepted']} | "
        f"{row['manual_uncertain']} | "
        f"{row['manual_reject']} | "
        f"{row['risk_class']} |"
        for row in summary_rows
    ]
    risk_lines = [
        "| "
        f"`{row['expected_phone']}` | "
        f"{row['risk_class']} | "
        f"{int(row['instance_count']):,} | "
        f"{row['evidence_type']} |"
        for row in summary_rows
        if row["risk_class"] != "supported"
    ]
    manual_lines = []
    for phone in MANUAL_REVIEW_PHONE_ORDER:
        row = next(item for item in summary_rows if item["expected_phone"] == phone)
        manual_lines.append(
            "| "
            f"`{phone}` | "
            f"{row['manual_accepted']} | "
            f"{row['manual_uncertain']} | "
            f"{row['manual_reject']} | "
            f"{row['risk_class']} |"
        )

    manifest_lines = ""
    if coverage["manifest_segments"] is not None:
        manifest_lines = (
            f"- Manifest segments: {coverage['manifest_segments']:,}\n"
            f"- Aligned manifest segments: {coverage['aligned_manifest_segments']:,}\n"
            f"- Missing manifest segments: {coverage['missing_manifest_segments']:,}\n"
            f"- Manifest coverage: {coverage['coverage_pct']:.6f}%\n"
            f"- Extra aligned segments outside manifest: {coverage['extra_aligned_segments']:,}\n"
        )

    return (
        "# Phone Atlas Results\n\n"
        "## Coverage\n\n"
        + manifest_lines
        + f"- Aligned DB segments: {_alignment_segment_count(conn, aligner_version=aligner_version):,}\n"
        + f"- Phone windows: {phone_windows:,}\n"
        + f"- Stress events: {stress_events:,}\n"
        + f"- Mean phone confidence: {confidence_text}\n\n"
        "## Provider Distribution\n\n"
        "| Provider | Segments | Hours |\n"
        "|---|---:|---:|\n"
        + ("\n".join(provider_lines) if provider_lines else "| n/a | 0 | 0.00 |")
        + "\n\n"
        "## Stress/Syllable Acoustic Layer\n\n"
        "| Event Count | Evaluable | Match | Mismatch | Low Confidence | Single-Syllable | Match Rate |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
        + (
            f"| {stress_events:,} | {stress_evaluable:,} | {stress_match:,} | "
            f"{stress_mismatch:,} | {stress_low_confidence:,} | "
            f"{stress_not_applicable:,} | {stress_match_rate} |\n\n"
        )
        + "## 49-Phone + Allophone + Stress Coverage\n\n"
        "| Phone | Evidence Type | Count | Mean Confidence | Manual Accepted | Manual Uncertain | Manual Reject | Risk |\n"
        "|---|---|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(phone_lines)
        + "\n\n"
        "## Manual-Only Results\n\n"
        "| Phone | Accepted | Uncertain | Reject | Risk |\n"
        "|---|---:|---:|---:|---|\n"
        + "\n".join(manual_lines)
        + "\n\n"
        "## Risk Summary\n\n"
        "| Phone | Risk | Count | Evidence Type |\n"
        "|---|---|---:|---|\n"
        + ("\n".join(risk_lines) if risk_lines else "| n/a | supported | 0 | n/a |")
        + "\n\n"
        "## Notes\n\n"
        "- `β̞` is accepted in the manual atlas; reject count alone does not constitute a risk verdict.\n"
        "- `expected_49`, `observed_model`, `observed_acoustic`, and `manual_phone` are separate evidence layers.\n"
        "- MMS_FA output is a boundary-and-projection layer, not IPA phone recognition.\n"
        "- `ˈ` is not a phone window; stress evidence is reported at syllable level (F0/intensity/duration) in `stress_syllable_instances.parquet`.\n"
        "- Long vowels remain feature-derived; stress risk is `low_count` if no events, otherwise `supported` in the summary.\n"
    )


def write_full_atlas_report(
    conn: sqlite3.Connection,
    out_path: Path,
    *,
    aligner_version: str | None = None,
    manifest_path: Path | None = None,
    summary_rows: list[dict[str, object]] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_full_atlas_report(
            conn,
            aligner_version=aligner_version,
            manifest_path=manifest_path,
            summary_rows=summary_rows,
        ),
        encoding="utf-8",
    )


def _alignment_provider_summary(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None,
) -> list[tuple[str, int, float]]:
    params: list[object] = []
    where = ""
    if aligner_version is not None:
        where = "WHERE sa.aligner_version = ?"
        params.append(aligner_version)
    rows = conn.execute(
        f"""SELECT s.provider, COUNT(*) AS segment_count, SUM(s.duration_s) AS seconds
            FROM segment_alignments sa
            JOIN segments s ON s.id = sa.segment_id
            {where}
            GROUP BY s.provider
            ORDER BY s.provider""",
        params,
    ).fetchall()
    return [
        (row["provider"], int(row["segment_count"]), float(row["seconds"] or 0.0))
        for row in rows
    ]


def _alignment_segment_count(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None,
) -> int:
    params: list[object] = []
    where = ""
    if aligner_version is not None:
        where = "WHERE aligner_version = ?"
        params.append(aligner_version)
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM segment_alignments {where}",
        params,
    ).fetchone()
    return int(row["count"])


def _mean_confidence(rows: Iterable[PhoneInstance]) -> float | None:
    values: list[float] = []
    for row in rows:
        feature = json.loads(row.feature_json or "{}")
        if feature.get("confidence") is not None:
            values.append(float(feature["confidence"]))
    return fmean(values) if values else None


def render_pilot_report(
    conn: sqlite3.Connection,
    *,
    aligner_version: str | None = None,
) -> str:
    rows = instances_from_alignment_db(conn, aligner_version=aligner_version)
    grouped: dict[str, list[PhoneInstance]] = {}
    for row in rows:
        grouped.setdefault(row.expected_phone, []).append(row)
    coverage = coverage_by_phone(rows)
    manual_summary = manual_annotation_summary(conn)
    confidence = _mean_confidence(rows)
    confidence_text = "n/a" if confidence is None else f"{confidence:.4f}"

    provider_lines = [
        f"| {provider} | {segment_count} | {seconds / 3600.0:.2f} |"
        for provider, segment_count, seconds in _alignment_provider_summary(
            conn,
            aligner_version=aligner_version,
        )
    ]
    phone_lines = []
    for phone in sorted(grouped):
        phone_rows = grouped[phone]
        classes = {evidence_class(row) for row in phone_rows}
        counts = coverage[phone]
        if (
            phone_evidence_type(phone) == "manual-only"
            and counts["manual_phone_count"] > 0
        ):
            cls = "supported"
        elif "manual_review_required" in classes:
            cls = "manual_review_required"
        elif "model_mismatch" in classes:
            cls = "model_mismatch"
        elif "acoustic_mismatch" in classes:
            cls = "acoustic_mismatch"
        else:
            cls = "supported"
        phone_lines.append(
            "| "
            f"`{phone}` | "
            f"{phone_evidence_type(phone)} | "
            f"{counts['instance_count']} | "
            f"{counts['manual_phone_count']} | "
            f"{cls} |"
        )

    manual_lines = []
    for phone in MANUAL_REVIEW_PHONE_ORDER:
        manual_lines.append(
            "| "
            f"`{phone}` | "
            f"{manual_summary.phone_status_counts.get((phone, 'accepted'), 0)} | "
            f"{manual_summary.phone_status_counts.get((phone, 'uncertain'), 0)} | "
            f"{manual_summary.phone_status_counts.get((phone, 'reject'), 0)} |"
        )

    collapse_lines = [
        f"- `{expected}` -> `{target}`"
        for expected, target in sorted(MMS_ALIGNMENT_COLLAPSE.items())
        if target is not None
    ]
    collapse_lines.extend(
        f"- `{expected}` -> dropped"
        for expected, target in sorted(MMS_ALIGNMENT_COLLAPSE.items())
        if target is None
    )

    return (
        "# Pilot Atlas Evidence Report\n\n"
        "This report summarises the pilot alignment and manual-review layers\n"
        "as read from the DB/export pipeline. It is not the final atlas result.\n\n"
        "## Coverage\n\n"
        f"- Aligned segments: {_alignment_segment_count(conn, aligner_version=aligner_version):,}\n"
        f"- Phone windows: {len(rows):,}\n"
        f"- Mean alignment confidence: {confidence_text}\n\n"
        "## Provider Distribution\n\n"
        "| Provider | Segments | Hours |\n"
        "|---|---:|---:|\n"
        + ("\n".join(provider_lines) if provider_lines else "| n/a | 0 | 0.00 |")
        + "\n\n"
        "## Phone-Level Count/Confidence\n\n"
        "| Phone | Evidence Type | Count | Manual Accepted | Evidence Class |\n"
        "|---|---|---:|---:|---|\n"
        + ("\n".join(phone_lines) if phone_lines else "| n/a | n/a | 0 | 0 | n/a |")
        + "\n\n"
        "## Manual-Only Results\n\n"
        "| Phone | Accepted | Uncertain | Reject |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(manual_lines)
        + "\n\n"
        "## Collapsed Phones\n\n"
        + "\n".join(collapse_lines)
        + "\n\n"
        "## Missing Acoustic Feature Note\n\n"
        "The acoustic feature layer is absent in this pilot report. Duration and\n"
        "confidence windows are exported, but F1/F2, VOT, F0, and intensity-based\n"
        "decisions are not yet treated as final evidence.\n"
    )


def write_pilot_report(
    conn: sqlite3.Connection,
    out_path: Path,
    *,
    aligner_version: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_pilot_report(conn, aligner_version=aligner_version),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=AUDIO_DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=ATLAS_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--aligner-version")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    parser.add_argument("--phone-batch-size", type=int, default=100_000)
    parser.add_argument("--stress-batch-size", type=int, default=100_000)
    parser.add_argument(
        "--with-stress-syllables",
        action="store_true",
        help="write stress_syllable_instances.parquet and fold stress counts into summary",
    )
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    stress_count = 0
    stress_rows: list[StressSyllableInstance] = []
    try:
        phone_count = write_phone_instances(
            iter_instances_from_alignment_db(
                conn,
                aligner_version=args.aligner_version,
            ),
            args.out_dir / "phone_instances.parquet",
            batch_size=args.phone_batch_size,
        )
        if args.with_stress_syllables:
            stress_rows = build_stress_syllable_instances(
                conn,
                aligner_version=args.aligner_version,
                manifest_path=args.manifest,
                audio_root=args.audio_root,
            )
            stress_count = write_stress_syllable_instances(
                stress_rows,
                args.out_dir / "stress_syllable_instances.parquet",
                batch_size=args.stress_batch_size,
            )
        summary_rows = build_phone_summary_rows(
            conn,
            aligner_version=args.aligner_version,
            stress_rows=stress_rows,
        )
        write_phone_summary_csv(
            summary_rows,
            args.out_dir / "phone_summary.csv",
        )
        write_g2p_audio_evidence_summary_csv(
            summary_rows,
            args.reports_dir / "g2p_audio_evidence.csv",
        )
        write_full_atlas_report(
            conn,
            args.reports_dir / "phoneme_atlas.md",
            aligner_version=args.aligner_version,
            manifest_path=args.manifest,
            summary_rows=summary_rows,
        )
    finally:
        conn.close()

    print(f"Wrote {phone_count:,} phone instances")
    if args.with_stress_syllables:
        print(f"Wrote {stress_count:,} stress syllable instances")
    print(f"Wrote {len(summary_rows):,} phone summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
