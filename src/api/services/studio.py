"""Validation recording capture: prompt manifest + per-session WAV attempts.

Records bind to ``data/validation/prompts.tsv`` rows and land under
``data/validation/recordings/<session>/<prompt_id>-a<NN>.wav`` (mono 16 kHz
PCM16), the exact layout the studio evaluation runner reads back.
"""

from __future__ import annotations

import csv
import re
import wave
from collections.abc import Iterator
from pathlib import Path

from fastapi import UploadFile

from api.services.audio import RecordingQuality, normalize_upload
from api.services.recordings import archive_existing_wav

_RECORDING_STEM_RE = re.compile(r"^(?P<pid>.+)-a(?P<n>\d{2,})$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_component(value: str, label: str) -> str:
    value = (value or "").strip()
    if not value or value in {".", ".."} or not _SAFE_COMPONENT_RE.match(value):
        raise ValueError(f"invalid {label}")
    return value


def load_studio_prompts(prompts_tsv: Path) -> list[dict[str, str]]:
    if not prompts_tsv.exists():
        raise FileNotFoundError("prompts.tsv not found")
    with prompts_tsv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t")]
    return [row for row in rows if (row.get("prompt_id") or "").strip()]


def _session_dir(recordings_dir: Path, session_id: str) -> Path:
    return recordings_dir / _safe_component(session_id, "session_id")


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
        return int(frames * 1000 / rate) if rate else 0
    except (wave.Error, OSError):
        return 0


def _iter_recordings(session_dir: Path) -> Iterator[tuple[str, int, Path]]:
    if not session_dir.is_dir():
        return
    for path in sorted(session_dir.iterdir()):
        if path.suffix.lower() != ".wav":
            continue
        match = _RECORDING_STEM_RE.match(path.stem)
        if match:
            yield match.group("pid"), int(match.group("n")), path


def list_attempts(recordings_dir: Path, session_id: str, prompt_id: str) -> list[int]:
    target = _safe_component(prompt_id, "prompt_id")
    return sorted(
        attempt
        for pid, attempt, _ in _iter_recordings(_session_dir(recordings_dir, session_id))
        if pid == target
    )


def next_attempt_number(recordings_dir: Path, session_id: str, prompt_id: str) -> int:
    attempts = list_attempts(recordings_dir, session_id, prompt_id)
    return attempts[-1] + 1 if attempts else 1


def session_recordings(recordings_dir: Path, session_id: str) -> dict[str, object]:
    attempts: dict[str, list[dict[str, object]]] = {}
    for pid, attempt, path in _iter_recordings(_session_dir(recordings_dir, session_id)):
        attempts.setdefault(pid, []).append(
            {
                "attempt": attempt,
                "filename": path.name,
                "duration_ms": _wav_duration_ms(path),
            }
        )
    for items in attempts.values():
        items.sort(key=lambda item: int(item["attempt"]))
    counts = {pid: len(items) for pid, items in attempts.items()}
    return {
        "session_id": _safe_component(session_id, "session_id"),
        "recorded_prompts": sorted(attempts),
        "attempt_counts": counts,
        "attempts": attempts,
    }


def recording_path(
    recordings_dir: Path,
    session_id: str,
    prompt_id: str,
    attempt: int | None = None,
) -> Path | None:
    session_dir = _session_dir(recordings_dir, session_id)
    safe_prompt_id = _safe_component(prompt_id, "prompt_id")
    if attempt is None:
        attempts = list_attempts(recordings_dir, session_id, safe_prompt_id)
        if not attempts:
            return None
        attempt = attempts[-1]
    candidate = session_dir / f"{safe_prompt_id}-a{attempt:02d}.wav"
    return candidate if candidate.is_file() else None


async def save_recording(
    recordings_dir: Path,
    session_id: str,
    prompt_id: str,
    upload: UploadFile,
    *,
    duration_ms: int,
    archive_recordings_dir: Path | None = None,
    collection_name: str = "studio",
) -> tuple[Path | None, RecordingQuality, str | None]:
    """Transcode the upload to a fresh ``{pid}-a{NN}.wav`` attempt.

    On a failed quality gate no permanent file is written and existing
    attempts are untouched; returns ``(None, quality)``.
    """
    safe_prompt_id = _safe_component(prompt_id, "prompt_id")
    session_dir = _session_dir(recordings_dir, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = session_dir / f"{safe_prompt_id}.incoming.wav"
    quality = await normalize_upload(
        upload=upload, target_path=tmp_path, duration_ms=duration_ms
    )
    recording_id = None
    if archive_recordings_dir is not None:
        result = {
            "status": "valid" if quality.status == "valid" else "invalid",
            "recording_quality": quality.as_dict(),
            "model_id": None,
            "result": None,
            "score": None,
            "debug": {"reason": "studio_capture"},
            "error_code": None,
        }
        safe_collection = _safe_component(collection_name, "collection_name")
        recording_id = archive_existing_wav(
            archive_recordings_dir,
            tmp_path,
            metadata={
                "source": safe_collection,
                "collection": safe_collection,
                "session_id": _safe_component(session_id, "session_id"),
                "prompt": safe_prompt_id,
                "prompt_id": safe_prompt_id,
            },
            result=result,
        )
    if quality.status != "valid":
        tmp_path.unlink(missing_ok=True)
        return None, quality, recording_id

    attempt = next_attempt_number(recordings_dir, session_id, safe_prompt_id)
    final_path = session_dir / f"{safe_prompt_id}-a{attempt:02d}.wav"
    tmp_path.replace(final_path)
    return final_path, quality, recording_id


def delete_recording(
    recordings_dir: Path,
    session_id: str,
    prompt_id: str,
    attempt: int,
) -> list[int]:
    if attempt < 1:
        raise ValueError("invalid attempt")
    path = recording_path(recordings_dir, session_id, prompt_id, attempt)
    if path is None:
        raise FileNotFoundError("recording not found")
    path.unlink()
    return list_attempts(recordings_dir, session_id, prompt_id)
