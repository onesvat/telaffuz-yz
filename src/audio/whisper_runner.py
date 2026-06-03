"""Multi-process Whisper-large-v3-turbo batch transcription runner (GPU).

Model: openai/whisper-large-v3-turbo (faster-whisper / CTranslate2 backend).
Configuration: language='tr', temperature=0, beam_size=5,
word_timestamps=True, condition_on_previous_text=False.

Writes one transcript_candidates row per segment plus word_occurrences from
word timestamps. For publisher-GT providers (common_voice, issai_tsc) the
Whisper transcript is written as is_primary=0 (timestamp/validator role).
For all other providers Whisper is is_primary=1 (silver transcript).
"""

from __future__ import annotations

import hashlib
import csv
import json
import multiprocessing as mp
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from audio.db import (
    AUDIO_DB_PATH,
    AUDIO_ROOT,
    REPO_ROOT,
    audit,
    connect,
    now_iso,
    transaction,
)

WHISPER_TRANSCRIPT_SOURCE = "whisper_large_v3_turbo"
WHISPER_WORD_SOURCE = "whisper_large_v3_turbo"

GT_PROVIDERS = frozenset({"common_voice", "issai_tsc"})

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_LANGUAGE = "tr"
DEFAULT_BEAM_SIZE = 5
DEFAULT_COMPUTE_TYPE = "int8_float16"
DEFAULT_DEVICE = "cuda"
DEFAULT_WORKERS = 4
DEFAULT_COMMIT_BATCH = 100
DEFAULT_QUEUE_MULTIPLIER = 4

_PUNCT_RE = re.compile(r"[^\w\s'’]", flags=re.UNICODE)
_SHARD_RE = re.compile(r"^(\d+(?:,\d+)*)/(\d+)$")

SENTINEL: Any = "__WHISPER_SENTINEL__"


@dataclass(frozen=True)
class WhisperConfig:
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    beam_size: int = DEFAULT_BEAM_SIZE
    temperature: float = 0.0
    compute_type: str = DEFAULT_COMPUTE_TYPE
    device: str = DEFAULT_DEVICE
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    vad_filter: bool = False
    engine: str = "faster_whisper"

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "compute_type": self.compute_type,
            "condition_on_previous_text": self.condition_on_previous_text,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_canonical_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ManifestFilter:
    segment_ids: frozenset[int]
    providers: tuple[str, ...]


def load_manifest_filter(
    manifest_path: Path,
    *,
    only_needs_whisper: bool = False,
) -> ManifestFilter:
    """Load segment ids from an atlas manifest CSV.

    When ``only_needs_whisper`` is true, only rows whose ``needs_whisper`` value
    is true are returned. This lets the runner complete the selected atlas
    subset without accidentally processing a whole provider.
    """
    segment_ids: set[int] = set()
    providers: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "segment_id" not in (reader.fieldnames or []):
            raise ValueError("manifest must include segment_id column")
        if only_needs_whisper and "needs_whisper" not in (reader.fieldnames or []):
            raise ValueError("--only-needs-whisper requires needs_whisper column")
        for row in reader:
            needs_whisper = (row.get("needs_whisper") or "").strip().lower()
            if only_needs_whisper and needs_whisper not in {"1", "true", "yes"}:
                continue
            segment_ids.add(int(row["segment_id"]))
            provider = (row.get("provider") or "").strip()
            if provider:
                providers.add(provider)
    return ManifestFilter(
        segment_ids=frozenset(segment_ids),
        providers=tuple(sorted(providers)),
    )


def code_version() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return ""


def normalize_word(surface: str) -> str:
    """Lowercase and strip punctuation; Turkish locale: İ→i, I→ı."""
    s = unicodedata.normalize("NFC", surface.strip())
    s = s.replace("İ", "i").replace("I", "ı")
    s = s.lower()
    s = _PUNCT_RE.sub("", s)
    return s.strip()


def parse_shard_spec(spec: str) -> tuple[set[int], int]:
    """Parse 1-based shard specs like '6/10' or '6,7/10'."""
    match = _SHARD_RE.match(spec.strip())
    if not match:
        raise ValueError("shard must look like N/TOTAL or A,B/TOTAL")
    shard_count = int(match.group(2))
    if shard_count < 1:
        raise ValueError("shard TOTAL must be >= 1")
    shard_numbers = {int(part) for part in match.group(1).split(",")}
    invalid = sorted(n for n in shard_numbers if n < 1 or n > shard_count)
    if invalid:
        raise ValueError(
            f"shard numbers must be between 1 and {shard_count}: {invalid}"
        )
    return {n - 1 for n in shard_numbers}, shard_count


def segment_in_shard(segment_id: int, shard_indexes: set[int], shard_count: int) -> bool:
    """Stable 1-based sharding over segment ids."""
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if segment_id < 1:
        raise ValueError("segment_id must be >= 1")
    return (segment_id - 1) % shard_count in shard_indexes


def ensure_asr_config(conn: sqlite3.Connection, cfg: WhisperConfig) -> str:
    h = cfg.config_hash()
    config_json = json.dumps(cfg.to_canonical_dict(), sort_keys=True, ensure_ascii=False)
    conn.execute(
        """INSERT OR IGNORE INTO asr_configs
             (config_hash, engine, model_version, config_json, code_version, created_at)
             VALUES (?, ?, ?, ?, ?, ?)""",
        (h, cfg.engine, cfg.model, config_json, code_version(), now_iso()),
    )
    conn.commit()
    return h


def iter_pending_segments(
    conn: sqlite3.Connection,
    *,
    config_hash: str,
    providers: Iterable[str] | None = None,
    segment_ids: Iterable[int] | None = None,
    limit: int | None = None,
    shard: tuple[set[int], int] | None = None,
) -> Iterator[tuple[int, str, str]]:
    segment_filter: frozenset[int] | None = (
        frozenset(int(segment_id) for segment_id in segment_ids)
        if segment_ids is not None
        else None
    )
    if segment_filter is not None:
        if not segment_filter:
            return
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS whisper_manifest_filter "
            "(segment_id INTEGER PRIMARY KEY)"
        )
        conn.execute("DELETE FROM whisper_manifest_filter")
        conn.executemany(
            "INSERT OR IGNORE INTO whisper_manifest_filter(segment_id) VALUES (?)",
            ((segment_id,) for segment_id in sorted(segment_filter)),
        )

    params: list[Any] = [WHISPER_TRANSCRIPT_SOURCE, config_hash]
    sql = (
        "SELECT s.id AS id, s.path AS path, s.provider AS provider "
        "FROM segments s "
    )
    if segment_filter is not None:
        sql += "JOIN whisper_manifest_filter mf ON mf.segment_id = s.id "
    sql += (
        "LEFT JOIN transcript_candidates tc "
        "  ON tc.segment_id = s.id "
        " AND tc.source = ? "
        " AND tc.run_hash = ? "
        "WHERE tc.id IS NULL"
    )
    prov_list = list(providers) if providers else None
    if prov_list:
        placeholders = ",".join("?" * len(prov_list))
        sql += f" AND s.provider IN ({placeholders})"
        params.extend(prov_list)
    sql += " ORDER BY s.id"
    if limit is not None and shard is None:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql, params):
        segment_id = int(row["id"])
        if shard is not None:
            shard_indexes, shard_count = shard
            if not segment_in_shard(segment_id, shard_indexes, shard_count):
                continue
        yield segment_id, str(AUDIO_ROOT / row["path"]), row["provider"]
        if limit is not None and shard is not None:
            limit -= 1
            if limit <= 0:
                break


def _segment_confidence(words: list[dict[str, Any]]) -> float | None:
    probs = [w["probability"] for w in words if w.get("probability") is not None]
    if not probs:
        return None
    return float(sum(probs) / len(probs))


def worker_loop(
    worker_id: int,
    cfg_dict: dict[str, Any],
    input_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    """Worker process: loads WhisperModel once, drains input_q."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except Exception as exc:
        result_q.put({"ok": False, "fatal": True, "worker_id": worker_id,
                      "error_class": type(exc).__name__, "error_msg": str(exc)})
        return

    try:
        model = WhisperModel(
            cfg_dict["model"],
            device=cfg_dict["device"],
            compute_type=cfg_dict["compute_type"],
        )
    except Exception as exc:
        result_q.put({"ok": False, "fatal": True, "worker_id": worker_id,
                      "error_class": type(exc).__name__, "error_msg": str(exc)})
        return

    while True:
        job = input_q.get()
        if job == SENTINEL:
            break
        seg_id, abs_path, provider = job
        t0 = time.monotonic()
        try:
            segments_iter, info = model.transcribe(
                abs_path,
                language=cfg_dict["language"],
                beam_size=cfg_dict["beam_size"],
                temperature=cfg_dict["temperature"],
                word_timestamps=cfg_dict["word_timestamps"],
                condition_on_previous_text=cfg_dict["condition_on_previous_text"],
                vad_filter=cfg_dict["vad_filter"],
            )
            text_chunks: list[str] = []
            words_out: list[dict[str, Any]] = []
            for seg in segments_iter:
                text_chunks.append(seg.text)
                if seg.words:
                    for w in seg.words:
                        words_out.append({
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": w.word,
                            "probability": float(w.probability),
                        })
            text = " ".join(t.strip() for t in text_chunks).strip()
            result_q.put({
                "ok": True,
                "worker_id": worker_id,
                "segment_id": seg_id,
                "provider": provider,
                "text": text,
                "language": info.language,
                "duration": float(info.duration),
                "words": words_out,
                "elapsed": time.monotonic() - t0,
            })
        except Exception as exc:
            result_q.put({
                "ok": False,
                "fatal": False,
                "worker_id": worker_id,
                "segment_id": seg_id,
                "provider": provider,
                "error_class": type(exc).__name__,
                "error_msg": str(exc)[:500],
                "elapsed": time.monotonic() - t0,
            })


@dataclass
class RunStats:
    processed_ok: int = 0
    processed_err: int = 0
    empty: int = 0
    non_tr: int = 0
    fatal: int = 0
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    started_at: str = field(default_factory=now_iso)


def _insert_result(
    conn: sqlite3.Connection,
    cfg: WhisperConfig,
    config_hash: str,
    r: dict[str, Any],
) -> None:
    provider = r["provider"]
    is_primary = 0 if provider in GT_PROVIDERS else 1
    confidence = _segment_confidence(r["words"])
    created = now_iso()
    cur = conn.execute(
        """INSERT OR IGNORE INTO transcript_candidates
             (segment_id, source, model_version, run_hash, text, language,
              confidence, is_primary, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            r["segment_id"],
            WHISPER_TRANSCRIPT_SOURCE,
            cfg.model,
            config_hash,
            r["text"],
            r.get("language") or cfg.language,
            confidence,
            is_primary,
            created,
        ),
    )
    if cur.rowcount == 0:
        return
    tc_id = cur.lastrowid

    if r["words"]:
        rows = []
        for idx, w in enumerate(r["words"]):
            surface = w["word"]
            norm = normalize_word(surface)
            start_ms = int(round(w["start"] * 1000))
            end_ms = int(round(w["end"] * 1000))
            if end_ms <= start_ms:
                end_ms = start_ms + 1
            rows.append((
                r["segment_id"], tc_id, idx, surface, norm,
                start_ms, end_ms, float(w["probability"]),
                WHISPER_WORD_SOURCE, None, created,
            ))
        conn.executemany(
            """INSERT INTO word_occurrences
                 (segment_id, transcript_candidate_id, word_index,
                  word_surface, word_norm, start_ms, end_ms, confidence,
                  source, phone_alignment_json, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    if r["text"] == "":
        audit(conn, actor="whisper_runner", action="whisper_empty",
              entity="segments", entity_id=str(r["segment_id"]))
    if r.get("language") and r["language"] != cfg.language:
        audit(conn, actor="whisper_runner", action="whisper_non_tr",
              entity="segments", entity_id=str(r["segment_id"]),
              payload_json=json.dumps({"detected": r["language"]}))


def writer_loop(
    db_path: Path,
    cfg: WhisperConfig,
    config_hash: str,
    result_q: mp.Queue,
    commit_batch: int,
    stats: RunStats,
    progress_cb: Callable[[RunStats], None] | None,
    expected_sentinels: int,
) -> None:
    """Single-writer thread: drains result_q, batched DB insert + audit."""
    conn = connect(db_path)
    batch_ok: list[dict[str, Any]] = []
    batch_err: list[dict[str, Any]] = []
    sentinels_seen = 0

    def flush_ok() -> None:
        if not batch_ok:
            return
        with transaction(conn):
            for r in batch_ok:
                _insert_result(conn, cfg, config_hash, r)
        batch_ok.clear()

    def flush_err() -> None:
        if not batch_err:
            return
        with transaction(conn):
            for r in batch_err:
                audit(
                    conn, actor="whisper_runner", action="whisper_error",
                    entity="segments", entity_id=str(r.get("segment_id")),
                    payload_json=json.dumps({
                        "error_class": r.get("error_class"),
                        "error_msg": r.get("error_msg"),
                        "worker_id": r.get("worker_id"),
                    }, ensure_ascii=False),
                )
        batch_err.clear()

    try:
        while sentinels_seen < expected_sentinels:
            try:
                r = result_q.get(timeout=1.0)
            except queue.Empty:
                if batch_ok:
                    flush_ok()
                if batch_err:
                    flush_err()
                continue
            if r == SENTINEL:
                sentinels_seen += 1
                continue
            if r.get("fatal"):
                stats.fatal += 1
                print(f"[whisper-writer] FATAL worker={r.get('worker_id')} "
                      f"{r.get('error_class')}: {r.get('error_msg')}",
                      file=sys.stderr)
                with transaction(conn):
                    audit(conn, actor="whisper_runner", action="whisper_fatal",
                          entity="asr_configs", entity_id=config_hash,
                          payload_json=json.dumps({
                              "worker_id": r.get("worker_id"),
                              "error_class": r.get("error_class"),
                              "error_msg": r.get("error_msg"),
                          }, ensure_ascii=False))
                continue
            if r.get("ok"):
                batch_ok.append(r)
                stats.processed_ok += 1
                stats.audio_seconds += float(r.get("duration") or 0.0)
                if not r["text"]:
                    stats.empty += 1
                if r.get("language") and r["language"] != cfg.language:
                    stats.non_tr += 1
            else:
                batch_err.append(r)
                stats.processed_err += 1
            if progress_cb is not None:
                progress_cb(stats)
            if len(batch_ok) >= commit_batch:
                flush_ok()
            if len(batch_err) >= max(10, commit_batch // 10):
                flush_err()
    finally:
        flush_ok()
        flush_err()
        conn.close()


class WhisperRunner:
    def __init__(
        self,
        cfg: WhisperConfig,
        *,
        db_path: Path = AUDIO_DB_PATH,
        workers: int = DEFAULT_WORKERS,
        providers: Iterable[str] | None = None,
        segment_ids: Iterable[int] | None = None,
        limit: int | None = None,
        shard: tuple[set[int], int] | None = None,
        commit_batch: int = DEFAULT_COMMIT_BATCH,
        progress_every: int = 100,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.cfg = cfg
        self.db_path = db_path
        self.workers = workers
        self.providers = list(providers) if providers else None
        self.segment_ids = frozenset(int(segment_id) for segment_id in segment_ids) if segment_ids is not None else None
        self.limit = limit
        self.shard = shard
        self.commit_batch = commit_batch
        self.progress_every = progress_every

    def run(self) -> RunStats:
        ctx = mp.get_context("spawn")
        qmax = max(self.workers * DEFAULT_QUEUE_MULTIPLIER, 8)
        input_q: mp.Queue = ctx.Queue(maxsize=qmax)
        result_q: mp.Queue = ctx.Queue(maxsize=qmax)

        cfg_dict = self.cfg.to_canonical_dict()
        cfg_dict["device"] = self.cfg.device
        cfg_dict["model"] = self.cfg.model

        conn = connect(self.db_path)
        config_hash = ensure_asr_config(conn, self.cfg)
        audit(
            conn, actor="whisper_runner", action="whisper_run_start",
            entity="asr_configs", entity_id=config_hash,
            payload_json=json.dumps({
                "workers": self.workers, "providers": self.providers,
                "manifest_segment_count": (
                    len(self.segment_ids) if self.segment_ids is not None else None
                ),
                "limit": self.limit, "shard": self._shard_for_log(),
                "code_version": code_version(),
            }, ensure_ascii=False),
        )
        conn.commit()
        pending = list(iter_pending_segments(
            conn, config_hash=config_hash,
            providers=self.providers,
            segment_ids=self.segment_ids,
            limit=self.limit,
            shard=self.shard,
        ))
        conn.close()

        stats = RunStats()

        if not pending:
            print(f"[whisper] no pending segments for config_hash={config_hash}", file=sys.stderr)
            return stats

        print(
            f"[whisper] pending={len(pending)} workers={self.workers} "
            f"config_hash={config_hash} model={self.cfg.model} "
            f"shard={self._shard_for_log() or 'all'}",
            file=sys.stderr,
        )

        from multiprocessing.process import BaseProcess

        workers: list[BaseProcess] = []
        for wid in range(self.workers):
            p = ctx.Process(
                target=worker_loop,
                args=(wid, cfg_dict, input_q, result_q),
                daemon=True,
                name=f"whisper-worker-{wid}",
            )
            p.start()
            workers.append(p)

        start_time = time.monotonic()
        last_log = [start_time, 0]

        def progress_cb(s: RunStats) -> None:
            now = time.monotonic()
            done = s.processed_ok + s.processed_err
            if done - last_log[1] >= self.progress_every or now - last_log[0] >= 30:
                elapsed = now - start_time
                rtf = (elapsed / s.audio_seconds) if s.audio_seconds > 0 else 0.0
                print(
                    f"[whisper] done={done}/{len(pending)} "
                    f"ok={s.processed_ok} err={s.processed_err} "
                    f"empty={s.empty} non_tr={s.non_tr} fatal={s.fatal} "
                    f"audio_h={s.audio_seconds / 3600:.2f} "
                    f"elapsed_s={elapsed:.1f} RTF={rtf:.3f}",
                    file=sys.stderr,
                )
                last_log[0] = now
                last_log[1] = done

        writer = threading.Thread(
            target=writer_loop,
            args=(self.db_path, self.cfg, config_hash, result_q,
                  self.commit_batch, stats, progress_cb, self.workers),
            name="whisper-writer",
            daemon=False,
        )
        writer.start()

        try:
            for job in pending:
                input_q.put(job)
            for _ in workers:
                input_q.put(SENTINEL)
            for proc in workers:
                proc.join()
            for _ in workers:
                result_q.put(SENTINEL)
            writer.join()
        except KeyboardInterrupt:
            print("[whisper] SIGINT — draining...", file=sys.stderr)
            for _ in workers:
                try:
                    input_q.put_nowait(SENTINEL)
                except queue.Full:
                    pass
            for proc in workers:
                proc.join(timeout=30)
            for _ in workers:
                try:
                    result_q.put_nowait(SENTINEL)
                except queue.Full:
                    pass
            writer.join(timeout=120)
            raise

        stats.wall_seconds = time.monotonic() - start_time

        conn = connect(self.db_path)
        audit(
            conn, actor="whisper_runner", action="whisper_run_end",
            entity="asr_configs", entity_id=config_hash,
            payload_json=json.dumps({
                "processed_ok": stats.processed_ok,
                "processed_err": stats.processed_err,
                "empty": stats.empty,
                "non_tr": stats.non_tr,
                "fatal": stats.fatal,
                "audio_seconds": stats.audio_seconds,
                "wall_seconds": stats.wall_seconds,
            }, ensure_ascii=False),
        )
        conn.commit()
        conn.close()

        rtf = stats.wall_seconds / stats.audio_seconds if stats.audio_seconds else 0.0
        print(
            f"[whisper] done ok={stats.processed_ok} err={stats.processed_err} "
            f"empty={stats.empty} non_tr={stats.non_tr} fatal={stats.fatal} "
            f"audio_h={stats.audio_seconds / 3600:.2f} "
            f"wall_h={stats.wall_seconds / 3600:.2f} RTF={rtf:.3f}",
            file=sys.stderr,
        )
        return stats

    def _shard_for_log(self) -> str | None:
        if self.shard is None:
            return None
        shard_indexes, shard_count = self.shard
        shard_numbers = ",".join(str(i + 1) for i in sorted(shard_indexes))
        return f"{shard_numbers}/{shard_count}"
