"""Coach reference feature DB builder and audit helpers.

Builds and audits the SQLite feature database used by the coach runtime
for canonical statistics and GMM scoring.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

COACH_FEATURE_VERSION = "assess_coach_compat_v1"
DERIVATION_POLICY = "full_clip_midpoint_from_legacy_anchors_v1"
ENDPOINT_STATUS = "full_clip_compat"

COACH_REFERENCE_FEATURE_VERSION = "assess_coach_reference_v1"
COACH_REFERENCE_DERIVATION_POLICY = (
    "assess_coach_signal_anchored_class_features_v1"
)
_REBUILD_WORKER_STAGGER_SECONDS = 0.25

# ---------------------------------------------------------------------------
# Compat DB builder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildReport:
    output_db: str
    building_db: str
    source_feature_version: str
    inserted_rows: int
    skipped_rows: int
    last_source_feature_id: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "output_db": self.output_db,
            "building_db": self.building_db,
            "source_feature_version": self.source_feature_version,
            "inserted_rows": self.inserted_rows,
            "skipped_rows": self.skipped_rows,
            "last_source_feature_id": self.last_source_feature_id,
        }


@dataclass(frozen=True)
class NucleusWindow:
    start_ms: int
    end_ms: int
    status: str


def bound_nucleus_window(
    *,
    span_start_ms: int,
    span_end_ms: int,
    legacy_analysis_start_ms: int | None,
    legacy_analysis_end_ms: int | None,
    default_width_ms: int = 80,
) -> NucleusWindow:
    if span_end_ms <= span_start_ms:
        raise ValueError("span must be positive")

    if (
        legacy_analysis_start_ms is not None
        and legacy_analysis_end_ms is not None
        and legacy_analysis_end_ms > legacy_analysis_start_ms
    ):
        start_ms = max(span_start_ms, legacy_analysis_start_ms)
        end_ms = min(span_end_ms, legacy_analysis_end_ms)
        if end_ms > start_ms:
            return NucleusWindow(start_ms, end_ms, "legacy_intersection")

    span_duration_ms = span_end_ms - span_start_ms
    width_ms = max(1, min(default_width_ms, span_duration_ms))
    center_ms = int(round((span_start_ms + span_end_ms) / 2.0))
    start_ms = max(span_start_ms, center_ms - width_ms // 2)
    end_ms = start_ms + width_ms
    if end_ms > span_end_ms:
        end_ms = span_end_ms
        start_ms = end_ms - width_ms
    return NucleusWindow(start_ms, end_ms, "synthesized_center")


def compute_feature_confidence(
    *,
    span_duration_ms: float,
    nucleus_duration_ms: float,
    formant_success: int | None,
    provider: str,
) -> float:
    """Simple feature confidence gate for compat DB rows."""
    if span_duration_ms < 20.0:
        return 0.0
    if nucleus_duration_ms < 15.0:
        return 0.0
    # Require formant success for vowels (caller checks is_vowel)
    if formant_success is not None and formant_success == 0:
        return 0.0
    return 1.0


def build_compat_db(
    *,
    audio_db: Path,
    features_db: Path,
    output_db: Path,
    source_feature_version: str = "phone_features_v3_policy",
    limit: int | None = None,
    replace: bool = False,
    resume: bool = False,
) -> BuildReport:
    """Build a file-based coach compat feature DB from legacy SQLite sources."""
    building_db = output_db.with_suffix(".building.sqlite")

    if replace and output_db.exists():
        output_db.unlink()
    if not resume and building_db.exists():
        building_db.unlink()

    _ensure_schema(building_db)
    last_id = _last_source_feature_id(building_db) if resume else None

    src_conn = sqlite3.connect(f"file:{audio_db}?mode=ro", uri=True)
    feat_conn = sqlite3.connect(f"file:{features_db}?mode=ro", uri=True)
    src_conn.row_factory = sqlite3.Row
    feat_conn.row_factory = sqlite3.Row
    out_conn = sqlite3.connect(str(building_db))

    inserted = 0
    skipped = 0
    try:
        for row in _iter_feature_rows(feat_conn, source_feature_version, last_id, limit):
            alignment = _load_alignment(src_conn, int(row["segment_alignment_id"]))
            if alignment is None:
                skipped += 1
                continue
            inserted_row, ok = _build_compat_row(row, alignment, source_feature_version)
            if not ok:
                skipped += 1
                continue
            _insert_row(out_conn, inserted_row)
            inserted += 1
        out_conn.commit()
    finally:
        src_conn.close()
        feat_conn.close()
        out_conn.close()

    shutil.move(str(building_db), str(output_db))
    return BuildReport(
        output_db=str(output_db),
        building_db=str(building_db),
        source_feature_version=source_feature_version,
        inserted_rows=inserted,
        skipped_rows=skipped,
        last_source_feature_id=None,
    )


def _ensure_schema(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS phone_features (
            id INTEGER PRIMARY KEY,
            feature_version TEXT NOT NULL,
            expected_phone TEXT NOT NULL,
            provider TEXT,
            speaker_id TEXT,
            start_ms INTEGER,
            end_ms INTEGER,
            duration_ms REAL,
            active_duration_ms REAL,
            speech_start_ms_original INTEGER,
            speech_end_ms_original INTEGER,
            voiced_start_ms REAL,
            voiced_end_ms REAL,
            span_window_json TEXT,
            nucleus_start_ms INTEGER,
            nucleus_end_ms INTEGER,
            feature_confidence REAL,
            anchor_confidence REAL,
            formant_success INTEGER,
            duration_reliability REAL,
            quality_json TEXT,
            derivation_policy TEXT,
            f1_hz REAL,
            f2_hz REAL,
            f3_hz REAL,
            spectral_centroid_hz REAL,
            spectral_bandwidth_hz REAL,
            spectral_skew REAL,
            spectral_kurtosis REAL,
            voiced_fraction REAL,
            rms REAL,
            vot_ms REAL,
            closure_ms REAL,
            closure_duration_ms REAL,
            closure_voicing_ratio REAL,
            burst_centroid_hz REAL,
            burst_spectral_skew REAL,
            burst_confidence REAL,
            frication_rise_db_per_ms REAL,
            frication_duration_ms REAL,
            f2_transition_slope_hz_per_ms REAL,
            f2_locus_hz REAL,
            nasal_murmur_ratio REAL,
            vowel_f2_movement_hz_per_ms REAL,
            segment_alignment_id INTEGER,
            phone_index INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _last_source_feature_id(db: Path) -> int | None:
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT MAX(segment_alignment_id) FROM phone_features"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


def _iter_feature_rows(
    conn: sqlite3.Connection,
    feature_version: str,
    last_id: int | None,
    limit: int | None,
):
    clause = "WHERE feature_version = ?"
    params: list[object] = [feature_version]
    if last_id is not None:
        clause += " AND segment_alignment_id > ?"
        params.append(last_id)
    sql = f"SELECT * FROM phone_features {clause} ORDER BY segment_alignment_id, phone_index"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    yield from conn.execute(sql, params)


def _load_alignment(conn: sqlite3.Connection, alignment_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT alignment_json, provider, speaker_id FROM segment_alignments WHERE id = ?",
        (alignment_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        data = json.loads(row[0])
        data["provider"] = row[1]
        data["speaker_id"] = row[2]
        return data
    except (json.JSONDecodeError, TypeError):
        return None


def _build_compat_row(
    row: sqlite3.Row,
    alignment: dict[str, Any],
    source_feature_version: str,
) -> tuple[dict[str, Any], bool]:
    phones = alignment.get("phones") or []
    phone_index = int(row["phone_index"]) if row["phone_index"] is not None else -1
    if phone_index < 0 or phone_index >= len(phones):
        return {}, False

    phone_item = phones[phone_index]
    span_start_ms = int(phone_item.get("start_ms", 0))
    span_end_ms = int(phone_item.get("end_ms", 0))
    if span_end_ms <= span_start_ms:
        return {}, False

    duration_ms = span_end_ms - span_start_ms
    nucleus = bound_nucleus_window(
        span_start_ms=span_start_ms,
        span_end_ms=span_end_ms,
        legacy_analysis_start_ms=None,
        legacy_analysis_end_ms=None,
    )

    out: dict[str, Any] = {
        "feature_version": COACH_FEATURE_VERSION,
        "expected_phone": str(row["expected_phone"]),
        "provider": alignment.get("provider"),
        "speaker_id": alignment.get("speaker_id"),
        "start_ms": span_start_ms,
        "end_ms": span_end_ms,
        "duration_ms": float(duration_ms),
        "speech_start_ms_original": span_start_ms,
        "speech_end_ms_original": span_end_ms,
        "span_window_json": json.dumps({"start_ms": span_start_ms, "end_ms": span_end_ms}),
        "nucleus_start_ms": nucleus.start_ms,
        "nucleus_end_ms": nucleus.end_ms,
        "feature_confidence": float(row["feature_confidence"]) if row["feature_confidence"] is not None else 0.0,
        "formant_success": int(row["formant_success"]) if row["formant_success"] is not None else None,
        "quality_json": json.dumps({"derivation_policy": DERIVATION_POLICY}),
        "f1_hz": _safe_float(row, "f1_hz"),
        "f2_hz": _safe_float(row, "f2_hz"),
        "f3_hz": _safe_float(row, "f3_hz"),
        "spectral_centroid_hz": _safe_float(row, "spectral_centroid_hz"),
        "spectral_bandwidth_hz": _safe_float(row, "spectral_bandwidth_hz"),
        "voiced_fraction": _safe_float(row, "voiced_fraction"),
        "rms": _safe_float(row, "rms"),
        "segment_alignment_id": int(row["segment_alignment_id"]) if row["segment_alignment_id"] is not None else None,
        "phone_index": phone_index,
    }
    return out, True


def _safe_float(row: sqlite3.Row, col: str) -> float | None:
    try:
        v = row[col]
        return float(v) if v is not None else None
    except (TypeError, ValueError, IndexError):
        return None


def _insert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT INTO phone_features ({','.join(cols)}) VALUES ({placeholders})"
    conn.execute(sql, [row[c] for c in cols])


# ---------------------------------------------------------------------------
# Reference re-extraction (Faz 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceRebuildReport:
    output_db: str
    feature_version: str
    derivation_policy: str
    providers: tuple[str, ...]
    workers: int
    inserted_rows: int
    skipped_rows: int
    audio_root: str

    def as_dict(self) -> dict[str, object]:
        return {
            "output_db": self.output_db,
            "feature_version": self.feature_version,
            "derivation_policy": self.derivation_policy,
            "providers": list(self.providers),
            "workers": self.workers,
            "inserted_rows": self.inserted_rows,
            "skipped_rows": self.skipped_rows,
            "audio_root": self.audio_root,
        }


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    audio, sr = sf.read(str(path), dtype="float32")
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    return arr, int(sr)


_REBUILD_SELECT = """
    SELECT sa.id AS alignment_id,
           sa.alignment_json,
           sa.aligner_version,
           s.path AS audio_path,
           s.provider,
           s.speaker_id,
           s.duration_s
    FROM segment_alignments sa
    JOIN segments s ON s.id = sa.segment_id
"""


def _rebuild_where(
    providers: tuple[str, ...], aligner_version: str | None
) -> tuple[str, list[object]]:
    placeholders = ",".join("?" for _ in providers)
    clause = f"WHERE s.provider IN ({placeholders})"
    params: list[object] = list(providers)
    if aligner_version is not None:
        clause += " AND sa.aligner_version = ?"
        params.append(aligner_version)
    return clause, params


def _process_rows(
    rows: Iterable[sqlite3.Row],
    out_conn: sqlite3.Connection,
    audio_root: Path,
    feature_confidence_floor: float,
) -> tuple[int, int]:
    """Extract features for each alignment row and insert them into ``out_conn``."""
    from assess.contracts import PhoneInterval  # noqa: PLC0415
    from assess.features import FeatureExtractor  # noqa: PLC0415

    extractor = FeatureExtractor()
    inserted = 0
    skipped = 0
    for row in rows:
        try:
            alignment_data = json.loads(row["alignment_json"])
        except (TypeError, json.JSONDecodeError):
            skipped += 1
            continue
        phones = alignment_data.get("phones") or []
        if not phones:
            skipped += 1
            continue

        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_path
        if not audio_path.exists():
            skipped += 1
            continue
        try:
            audio, sample_rate = _load_audio_mono(audio_path)
        except Exception:
            skipped += 1
            continue
        duration_ms_clip = int(row["duration_s"] * 1000)

        for phone_index, phone_item in enumerate(phones):
            expected_phone = phone_item.get("expected_phone")
            if not expected_phone:
                continue
            start_ms = int(phone_item.get("start_ms", 0))
            end_ms = int(phone_item.get("end_ms", 0))
            if end_ms <= start_ms:
                continue

            interval = PhoneInterval(
                target_phone=str(expected_phone),
                start_ms=start_ms,
                end_ms=end_ms,
                speech_start_ms_original=0,
                speech_end_ms_original=max(end_ms, duration_ms_clip),
            )
            prev_phone = (
                str(phones[phone_index - 1].get("expected_phone") or "")
                if phone_index > 0
                else None
            ) or None
            next_phone = (
                str(phones[phone_index + 1].get("expected_phone") or "")
                if phone_index + 1 < len(phones)
                else None
            ) or None

            fs = extractor.extract(
                audio,
                sample_rate,
                interval,
                prev_phone=prev_phone,
                next_phone=next_phone,
            )
            if fs.feature_confidence < feature_confidence_floor:
                skipped += 1
                continue

            anchor_confidence = phone_item.get("confidence")
            anchor_confidence_value = (
                float(anchor_confidence) if anchor_confidence is not None else None
            )

            out_row = {
                "feature_version": COACH_REFERENCE_FEATURE_VERSION,
                "expected_phone": str(expected_phone),
                "provider": row["provider"],
                "speaker_id": row["speaker_id"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": float(end_ms - start_ms),
                "active_duration_ms": float(fs.active_duration_ms),
                "speech_start_ms_original": interval.speech_start_ms_original,
                "speech_end_ms_original": interval.speech_end_ms_original,
                "voiced_start_ms": fs.voiced_start_ms,
                "voiced_end_ms": fs.voiced_end_ms,
                "span_window_json": json.dumps(
                    {"start_ms": start_ms, "end_ms": end_ms}
                ),
                "nucleus_start_ms": None,
                "nucleus_end_ms": None,
                "feature_confidence": float(fs.feature_confidence),
                "anchor_confidence": anchor_confidence_value,
                "formant_success": 1 if fs.f1_hz is not None else 0,
                "duration_reliability": float(fs.duration_reliability),
                "quality_json": json.dumps(
                    {
                        "low_confidence_reasons": list(fs.low_confidence_reasons),
                        "duration_notes": list(fs.duration_notes),
                    }
                ),
                "derivation_policy": COACH_REFERENCE_DERIVATION_POLICY,
                "f1_hz": fs.f1_hz,
                "f2_hz": fs.f2_hz,
                "f3_hz": fs.f3_hz,
                "spectral_centroid_hz": fs.spectral_centroid_hz,
                "spectral_bandwidth_hz": fs.spectral_bandwidth_hz,
                "spectral_skew": fs.spectral_skew,
                "spectral_kurtosis": fs.spectral_kurtosis,
                "voiced_fraction": fs.voiced_fraction,
                "rms": fs.rms,
                "vot_ms": fs.vot_ms,
                "closure_ms": fs.closure_ms,
                "closure_duration_ms": fs.closure_duration_ms,
                "closure_voicing_ratio": fs.closure_voicing_ratio,
                "burst_centroid_hz": fs.burst_centroid_hz,
                "burst_spectral_skew": fs.burst_spectral_skew,
                "burst_confidence": fs.burst_confidence,
                "frication_rise_db_per_ms": fs.frication_rise_db_per_ms,
                "frication_duration_ms": fs.frication_duration_ms,
                "f2_transition_slope_hz_per_ms": fs.f2_transition_slope_hz_per_ms,
                "f2_locus_hz": fs.f2_locus_hz,
                "nasal_murmur_ratio": fs.nasal_murmur_ratio,
                "vowel_f2_movement_hz_per_ms": fs.vowel_f2_movement_hz_per_ms,
                "segment_alignment_id": int(row["alignment_id"]),
                "phone_index": phone_index,
            }
            _insert_row(out_conn, out_row)
            inserted += 1
    return inserted, skipped


def _prewarm_rebuild_imports() -> None:
    """Load the scipy-backed audio stack once before worker processes fan out."""
    from assess.features import FeatureExtractor  # noqa: F401, PLC0415

    import librosa  # noqa: PLC0415

    sr = 16_000
    t = np.arange(int(sr * 0.2), dtype=np.float32) / sr
    tone = (0.05 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)
    librosa.feature.spectral_centroid(y=tone, sr=sr)
    librosa.feature.spectral_bandwidth(y=tone, sr=sr)
    librosa.feature.mfcc(y=tone, sr=sr, n_mfcc=13)
    librosa.pyin(tone, fmin=75.0, fmax=600.0, sr=sr, frame_length=512)


def _rebuild_shard(
    payload: tuple[
        str, str, tuple[str, ...], str | None, float, str, list[int], int
    ],
) -> tuple[str, int, int]:
    """Worker entry point: extract a subset of alignment ids into a shard DB."""
    (
        audio_db,
        audio_root,
        providers,
        aligner_version,
        floor,
        shard_path,
        ids,
        shard_index,
    ) = payload
    if shard_index > 0:
        time.sleep(min(shard_index * _REBUILD_WORKER_STAGGER_SECONDS, 5.0))
    shard = Path(shard_path)
    if shard.exists():
        shard.unlink()
    _ensure_schema(shard)
    if not ids:
        return shard_path, 0, 0

    uri = f"file:{quote(audio_db, safe='/')}?mode=ro&immutable=1"
    src = sqlite3.connect(uri, uri=True)
    src.row_factory = sqlite3.Row
    out = sqlite3.connect(shard_path)
    where, params = _rebuild_where(providers, aligner_version)
    id_csv = ",".join(str(int(i)) for i in ids)
    sql = f"{_REBUILD_SELECT} {where} AND sa.id IN ({id_csv}) ORDER BY sa.id"
    try:
        inserted, skipped = _process_rows(
            src.execute(sql, params), out, Path(audio_root), floor
        )
        out.commit()
    finally:
        src.close()
        out.close()
    return shard_path, inserted, skipped


def _merge_shards(building_db: Path, shard_paths: list[Path]) -> None:
    """Append every shard's rows into ``building_db`` (id is reassigned)."""
    out = sqlite3.connect(str(building_db))
    try:
        cols = [
            r[1]
            for r in out.execute("PRAGMA table_info(phone_features)")
            if r[1] != "id"
        ]
        col_csv = ",".join(cols)
        for idx, shard in enumerate(shard_paths):
            schema_name = f"shard_{idx}"
            out.execute(f"ATTACH ? AS {schema_name}", (str(shard),))
            out.execute(
                f"INSERT INTO phone_features ({col_csv}) "
                f"SELECT {col_csv} FROM {schema_name}.phone_features"
            )
            out.commit()
            out.execute(f"DETACH {schema_name}")
        out.commit()
    finally:
        out.close()


def _rebuild_parallel(
    *,
    audio_db: Path,
    audio_root: Path,
    building_db: Path,
    providers: tuple[str, ...],
    aligner_version: str | None,
    feature_confidence_floor: float,
    limit: int | None,
    workers: int,
) -> tuple[int, int]:
    """Shard alignment rows across processes, then merge shards into building_db."""
    import concurrent.futures  # noqa: PLC0415
    import multiprocessing  # noqa: PLC0415

    uri = f"file:{quote(str(audio_db), safe='/')}?mode=ro&immutable=1"
    where, params = _rebuild_where(providers, aligner_version)
    id_sql = (
        "SELECT sa.id FROM segment_alignments sa "
        f"JOIN segments s ON s.id = sa.segment_id {where} ORDER BY sa.id"
    )
    if limit is not None:
        id_sql += f" LIMIT {int(limit)}"
    src = sqlite3.connect(uri, uri=True)
    try:
        ids = [int(r[0]) for r in src.execute(id_sql, params)]
    finally:
        src.close()

    _prewarm_rebuild_imports()

    shard_paths = [building_db.with_suffix(f".shard{i}.sqlite") for i in range(workers)]
    payloads = [
        (
            str(audio_db),
            str(audio_root),
            tuple(providers),
            aligner_version,
            feature_confidence_floor,
            str(shard_paths[i]),
            ids[i::workers],
            i,
        )
        for i in range(workers)
    ]

    inserted = 0
    skipped = 0
    executor_kwargs = {}
    if sys.platform.startswith("linux"):
        executor_kwargs["mp_context"] = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, **executor_kwargs
    ) as pool:
        futures = [pool.submit(_rebuild_shard, p) for p in payloads]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            _shard_path, ins, skip = future.result()
            inserted += ins
            skipped += skip
            print(
                f"[rebuild] shard {done}/{workers} done (+{ins} rows)",
                file=sys.stderr,
                flush=True,
            )

    _merge_shards(building_db, shard_paths)
    for shard in shard_paths:
        if shard.exists():
            shard.unlink()
    return inserted, skipped


def rebuild_reference(
    *,
    audio_db: Path,
    audio_root: Path,
    output_db: Path,
    providers: tuple[str, ...] = ("common_voice",),
    aligner_version: str | None = None,
    feature_confidence_floor: float = 0.0,
    limit: int | None = None,
    workers: int = 1,
    replace: bool = True,
) -> ReferenceRebuildReport:
    """Re-extract coach reference features from CV-only audio.

    Reads reference alignments from ``audio_db``, loads audio from
    ``audio_root`` (relative paths in ``segments.path`` are resolved against
    it), and runs voiced-edge windowing plus class-specific
    feature extraction. The output rows are tagged with the
    :data:`COACH_REFERENCE_FEATURE_VERSION` and
    :data:`COACH_REFERENCE_DERIVATION_POLICY` strings.
    """
    if not audio_db.exists():
        raise FileNotFoundError(f"audio_db does not exist: {audio_db}")
    if not audio_root.exists():
        raise FileNotFoundError(f"audio_root does not exist: {audio_root}")
    if not providers:
        raise ValueError("at least one provider is required")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    if replace and output_db.exists():
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)

    building_db = output_db.with_suffix(".building.sqlite")
    if building_db.exists():
        building_db.unlink()
    _ensure_schema(building_db)

    if workers > 1:
        inserted, skipped = _rebuild_parallel(
            audio_db=audio_db,
            audio_root=audio_root,
            building_db=building_db,
            providers=providers,
            aligner_version=aligner_version,
            feature_confidence_floor=feature_confidence_floor,
            limit=limit,
            workers=workers,
        )
    else:
        uri_path = quote(str(audio_db), safe="/")
        src_uri = f"file:{uri_path}?mode=ro&immutable=1"
        src_conn = sqlite3.connect(src_uri, uri=True)
        src_conn.row_factory = sqlite3.Row
        out_conn = sqlite3.connect(str(building_db))
        where, params = _rebuild_where(providers, aligner_version)
        sql = f"{_REBUILD_SELECT} {where} ORDER BY sa.id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        try:
            inserted, skipped = _process_rows(
                src_conn.execute(sql, params),
                out_conn,
                audio_root,
                feature_confidence_floor,
            )
            out_conn.commit()
        finally:
            src_conn.close()
            out_conn.close()

    shutil.move(str(building_db), str(output_db))
    return ReferenceRebuildReport(
        output_db=str(output_db),
        feature_version=COACH_REFERENCE_FEATURE_VERSION,
        derivation_policy=COACH_REFERENCE_DERIVATION_POLICY,
        providers=tuple(providers),
        workers=workers,
        inserted_rows=inserted,
        skipped_rows=skipped,
        audio_root=str(audio_root),
    )


# ---------------------------------------------------------------------------
# DB audit
# ---------------------------------------------------------------------------

REQUIRED_COACH_FEATURE_COLUMNS: tuple[str, ...] = (
    "feature_version",
    "expected_phone",
    "start_ms",
    "end_ms",
    "duration_ms",
    "speech_start_ms_original",
    "speech_end_ms_original",
    "span_window_json",
    "nucleus_start_ms",
    "nucleus_end_ms",
    "feature_confidence",
    "quality_json",
)

_AUDIO_ALIGNMENT_TABLE = "segment_alignments"
_FEATURES_TABLE = "phone_features"
_REQUIRED_ALIGNMENT_PHONE_KEYS = ("expected_phone", "start_ms", "end_ms")


@dataclass(frozen=True)
class DbSummary:
    path: str
    exists: bool
    tables: list[str]
    columns: list[str]
    row_count: int
    missing_coach_columns: tuple[str, ...]
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "tables": list(self.tables),
            "columns": list(self.columns),
            "row_count": self.row_count,
            "missing_coach_columns": list(self.missing_coach_columns),
            "error": self.error,
        }


@dataclass(frozen=True)
class AuditReport:
    audio_db: DbSummary
    features_db: DbSummary
    sample_alignment_keys: list[object]
    recommendation: str
    reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "audio_db": self.audio_db.as_dict(),
            "features_db": self.features_db.as_dict(),
            "sample_alignment_keys": [
                list(item) if isinstance(item, tuple) else dict(item)  # type: ignore[arg-type]
                for item in self.sample_alignment_keys
            ],
            "recommendation": self.recommendation,
            "reasons": list(self.reasons),
        }


def audit_databases(
    *,
    audio_db: Path,
    features_db: Path,
    sample_rows: int = 20,
) -> AuditReport:
    """Audit coach SQLite databases without mutating them."""
    audio_summary = _summarize_db(audio_db, focus_table=_AUDIO_ALIGNMENT_TABLE)
    features_summary = _summarize_db(
        features_db,
        focus_table=_FEATURES_TABLE,
        required_columns=REQUIRED_COACH_FEATURE_COLUMNS,
    )
    sample_keys = _sample_alignment_keys(audio_db, sample_rows=sample_rows)
    reasons = _audit_reasons(audio_summary, features_summary)

    return AuditReport(
        audio_db=audio_summary,
        features_db=features_summary,
        sample_alignment_keys=sample_keys,
        recommendation="rebuild" if reasons else "reuse",
        reasons=reasons,
    )


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri_path = quote(str(path), safe="/")
    uri = f"file:{uri_path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _summarize_db(
    path: Path,
    *,
    focus_table: str,
    required_columns: tuple[str, ...] = (),
) -> DbSummary:
    if not path.exists():
        return DbSummary(
            path=str(path), exists=False, tables=[], columns=[], row_count=0,
            missing_coach_columns=required_columns,
        )
    try:
        with closing(_connect_read_only(path)) as conn:
            tables = _table_names(conn)
            columns = _column_names(conn, focus_table) if focus_table in tables else []
            row_count = _row_count(conn, focus_table) if focus_table in tables else 0
    except sqlite3.DatabaseError:
        return DbSummary(
            path=str(path), exists=True, tables=[], columns=[], row_count=0,
            missing_coach_columns=required_columns, error="database_error",
        )
    return DbSummary(
        path=str(path),
        exists=True,
        tables=tables,
        columns=columns,
        row_count=row_count,
        missing_coach_columns=tuple(c for c in required_columns if c not in columns),
    )


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [str(row[1]) for row in rows]


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    return int(row[0])


def _sample_alignment_keys(path: Path, *, sample_rows: int) -> list[object]:
    if not path.exists():
        return [{"error": "missing_db", "path": str(path)}]
    try:
        with closing(_connect_read_only(path)) as conn:
            tables = _table_names(conn)
            if _AUDIO_ALIGNMENT_TABLE not in tables:
                return [{"error": "missing_table"}]
            columns = _column_names(conn, _AUDIO_ALIGNMENT_TABLE)
            if "alignment_json" not in columns:
                return [{"error": "missing_alignment_json_column"}]
            order_col = "id" if "id" in columns else "rowid"
            rows = conn.execute(
                f'SELECT alignment_json FROM "{_AUDIO_ALIGNMENT_TABLE}" '
                f'ORDER BY "{order_col}" LIMIT ?',
                (max(sample_rows, 0),),
            ).fetchall()
    except sqlite3.DatabaseError:
        return [{"error": "database_error", "path": str(path)}]

    keys: list[object] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            keys.append({"error": "invalid_json"})
            continue
        if not isinstance(payload, dict):
            keys.append({"error": "json_not_object"})
            continue
        key = tuple(sorted(str(k) for k in payload))
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _audit_reasons(audio_summary: DbSummary, features_summary: DbSummary) -> list[str]:
    reasons: list[str] = []
    if not audio_summary.exists:
        reasons.append("audio_db_missing")
    elif audio_summary.error:
        reasons.append("audio_db_malformed")
    if not features_summary.exists:
        reasons.append("features_db_missing")
    elif features_summary.error:
        reasons.append("features_db_malformed")
    elif features_summary.missing_coach_columns:
        reasons.append("features_db_missing_coach_columns")
    elif features_summary.row_count == 0:
        reasons.append("features_db_empty")
    return reasons
