from __future__ import annotations

import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from array import array
from pathlib import Path

from fastapi import UploadFile


@dataclass(frozen=True)
class RecordingQuality:
    status: str
    reason: str | None
    duration_ms: int
    sample_rate: int | None
    channels: int | None
    rms: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "rms": self.rms,
        }


async def normalize_upload(
    *,
    upload: UploadFile,
    target_path: Path,
    duration_ms: int,
) -> RecordingQuality:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = target_path.with_suffix(".raw-upload")
    raw_path.write_bytes(await upload.read())
    try:
        if _looks_like_wav(raw_path):
            _normalize_wav_or_copy(raw_path, target_path)
        else:
            _run_ffmpeg(raw_path, target_path)
        trim_wav_silence(target_path)
        quality = inspect_wav(target_path, fallback_duration_ms=duration_ms)
    except Exception as exc:
        if not target_path.exists():
            target_path.write_bytes(b"")
        return RecordingQuality(
            status="invalid",
            reason=f"audio_decode_failed:{type(exc).__name__}",
            duration_ms=duration_ms,
            sample_rate=None,
            channels=None,
            rms=None,
        )
    finally:
        raw_path.unlink(missing_ok=True)

    return _apply_quality_gate(quality, duration_ms)


def _apply_quality_gate(
    quality: RecordingQuality, duration_ms: int
) -> RecordingQuality:
    if duration_ms < 250 or quality.duration_ms < 200:
        return RecordingQuality(
            status="invalid",
            reason="too_short",
            duration_ms=quality.duration_ms,
            sample_rate=quality.sample_rate,
            channels=quality.channels,
            rms=quality.rms,
        )
    if quality.rms is not None and quality.rms < 10:
        return RecordingQuality(
            status="invalid",
            reason="silent",
            duration_ms=quality.duration_ms,
            sample_rate=quality.sample_rate,
            channels=quality.channels,
            rms=quality.rms,
        )
    return quality


def inspect_wav(path: Path, *, fallback_duration_ms: int) -> RecordingQuality:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        data = wav.readframes(frames)
    duration_ms = int(frames * 1000 / sample_rate) if sample_rate else fallback_duration_ms
    rms = _rms_16bit(data) if sample_width == 2 else None
    return RecordingQuality(
        status="valid",
        reason=None,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        rms=rms,
    )


def trim_wav_silence(
    path: Path,
    *,
    threshold: int = 320,
    padding_ms: int = 120,
    min_keep_ms: int = 80,
) -> None:
    """Trim leading/trailing silence from a mono 16-bit WAV in place."""
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            params = wav.getparams()
            data = wav.readframes(frames)
    except (wave.Error, OSError):
        return
    if channels != 1 or sample_width != 2 or not data or not sample_rate:
        return

    samples = array("h")
    samples.frombytes(data)
    if samples.itemsize != 2:
        return

    first = None
    last = None
    for idx, sample in enumerate(samples):
        if abs(sample) >= threshold:
            first = idx
            break
    if first is None:
        return
    for idx in range(len(samples) - 1, -1, -1):
        if abs(samples[idx]) >= threshold:
            last = idx
            break
    if last is None:
        return

    pad = int(sample_rate * padding_ms / 1000)
    start = max(0, first - pad)
    end = min(len(samples), last + pad + 1)
    min_keep = int(sample_rate * min_keep_ms / 1000)
    if end - start < min_keep:
        center = (first + last) // 2
        start = max(0, center - min_keep // 2)
        end = min(len(samples), start + min_keep)

    if start == 0 and end == len(samples):
        return
    trimmed = samples[start:end]
    with wave.open(str(path), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(trimmed.tobytes())


def _looks_like_wav(path: Path) -> bool:
    head = path.read_bytes()[:12]
    return head.startswith(b"RIFF") and b"WAVE" in head


def _normalize_wav_or_copy(raw_path: Path, target_path: Path) -> None:
    with wave.open(str(raw_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
    if channels == 1 and sample_rate == 16000 and sample_width == 2:
        shutil.copyfile(raw_path, target_path)
        return
    _run_ffmpeg(raw_path, target_path)


def _run_ffmpeg(raw_path: Path, target_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(target_path),
        ],
        check=True,
    )


def _rms_16bit(data: bytes) -> float:
    if not data:
        return 0.0
    total = 0
    count = 0
    for i in range(0, len(data) - 1, 2):
        sample = int.from_bytes(data[i : i + 2], byteorder="little", signed=True)
        total += sample * sample
        count += 1
    return math.sqrt(total / count) if count else 0.0
