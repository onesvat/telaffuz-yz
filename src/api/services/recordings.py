from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordingSlot:
    recording_id: str
    created_at: str
    wav_path: Path
    result_path: Path


def new_recording_slot(recordings_dir: Path) -> RecordingSlot:
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    recording_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}"
    day_dir = recordings_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    return RecordingSlot(
        recording_id=recording_id,
        created_at=now.isoformat(),
        wav_path=day_dir / f"{recording_id}.wav",
        result_path=day_dir / f"{recording_id}.json",
    )


def archive_existing_wav(
    recordings_dir: Path,
    source_wav: Path,
    *,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> str:
    slot = new_recording_slot(recordings_dir)
    if source_wav.exists():
        shutil.copyfile(source_wav, slot.wav_path)
    else:
        slot.wav_path.write_bytes(b"")
    write_recording_result(recordings_dir, slot, metadata=metadata, result=result)
    return slot.recording_id


def write_recording_result(
    recordings_dir: Path,
    slot: RecordingSlot,
    *,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> None:
    payload = {
        "recording_id": slot.recording_id,
        "created_at": slot.created_at,
        "audio_path": _relative(recordings_dir, slot.wav_path),
        "result_path": _relative(recordings_dir, slot.result_path),
        "metadata": metadata,
        "result": result,
    }
    slot.result_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    index_path = recordings_dir / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_index_row(payload), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def list_recordings(
    recordings_dir: Path,
    *,
    limit: int = 100,
    word: str | None = None,
    source: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    index_path = recordings_dir / "index.jsonl"
    if not index_path.exists():
        return []
    word_query = _normalize_query(word)
    rows: list[dict[str, Any]] = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    matches = [row for row in rows if _recording_matches(row, word_query=word_query, source=source, status=status)]
    return list(reversed(matches[-limit:]))


def recording_paths(recordings_dir: Path, recording_id: str) -> tuple[Path, Path]:
    if "/" in recording_id or "\\" in recording_id or recording_id in {"", ".", ".."}:
        raise ValueError("invalid recording_id")
    matches = sorted(recordings_dir.glob(f"????-??-??/{recording_id}.wav"))
    if not matches:
        raise FileNotFoundError("recording not found")
    wav_path = matches[-1]
    return wav_path, wav_path.with_suffix(".json")


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _recording_matches(
    row: dict[str, Any],
    *,
    word_query: str,
    source: str | None,
    status: str | None,
) -> bool:
    if word_query:
        haystack = " ".join(
            str(row.get(key) or "")
            for key in ("word", "prompt", "prompt_id")
        ).casefold()
        if word_query not in haystack:
            return False
    if source and source != "all" and str(row.get("source") or "—") != source:
        return False
    status_value = row.get("quality_status") or row.get("status") or "—"
    if status and status != "all" and str(status_value) != status:
        return False
    return True


def _normalize_query(value: str | None) -> str:
    return (value or "").strip().casefold()


def _index_row(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    quality = result.get("recording_quality") if isinstance(result.get("recording_quality"), dict) else {}
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    layer_summary = diagnostics.get("layer_summary") if isinstance(diagnostics.get("layer_summary"), dict) else {}
    score = _score(result)
    return {
        "recording_id": payload["recording_id"],
        "created_at": payload["created_at"],
        "audio_path": payload["audio_path"],
        "result_path": payload["result_path"],
        "source": metadata.get("source"),
        "collection": metadata.get("collection"),
        "participant_name": metadata.get("participant_name"),
        "client_id": metadata.get("client_id"),
        "word": metadata.get("word") or metadata.get("prompt"),
        "prompt": metadata.get("prompt"),
        "prompt_id": metadata.get("prompt_id"),
        "session_id": metadata.get("session_id"),
        "target_kind": metadata.get("target_kind"),
        "attempt": metadata.get("attempt"),
        "model_id": result.get("model_id") or metadata.get("model_id"),
        "status": result.get("status"),
        "quality_status": quality.get("status"),
        "quality_reason": quality.get("reason"),
        "duration_ms": quality.get("duration_ms"),
        "score": score,
        "layer_summary": layer_summary,
    }


def _score(result: dict[str, Any]) -> float | None:
    body = result.get("result")
    if not isinstance(body, dict):
        return None
    score = body.get("score")
    if not isinstance(score, dict):
        return None
    item = score.get("item_score")
    max_item = score.get("max_item_score") or 1
    if isinstance(item, (int, float)) and isinstance(max_item, (int, float)) and max_item:
        return round(float(item) / float(max_item) * 100, 1)
    return None
