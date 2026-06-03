"""MMS-1b-fl102 emission cache runner (GPU, desktop).

Model: facebook/mms-1b-fl102 (CC-BY-NC, academic scope).
Writes one .npz emission cache per segment; alignment consumes those caches
later. Planning is deterministic and resumable; long GPU runs can be sharded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from audio.db import AUDIO_ROOT
from audio.whisper_runner import parse_shard_spec, segment_in_shard


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EMISSION_DIR = REPO_ROOT / "data" / "audio" / "mms_emissions"
DEFAULT_MODEL_ID = "facebook/mms-1b-fl102"
DEFAULT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class MMSEmissionJob:
    segment_id: int
    audio_path: Path
    provider: str
    duration_s: float
    emission_path: Path


@dataclass(frozen=True)
class MMSConfig:
    model_id: str = DEFAULT_MODEL_ID
    sample_rate: int = DEFAULT_SAMPLE_RATE
    device: str = "auto"


def _needs_whisper_false(row: dict[str, str]) -> bool:
    value = (row.get("needs_whisper") or "false").strip().lower()
    return value in {"", "0", "false", "no"}


def _normalize_shard(
    shard: tuple[int, int] | tuple[set[int], int] | None,
) -> tuple[set[int], int] | None:
    if shard is None:
        return None
    indexes, count = shard
    if isinstance(indexes, set):
        return indexes, count
    if indexes < 1 or indexes > count:
        raise ValueError("shard index must be 1-based and within shard count")
    return {indexes - 1}, count


def plan_mms_emission_jobs(
    manifest_path: Path,
    *,
    emission_dir: Path = DEFAULT_EMISSION_DIR,
    shard: tuple[int, int] | tuple[set[int], int] | None = None,
    limit: int | None = None,
    include_cached: bool = False,
) -> list[MMSEmissionJob]:
    """Plan uncached MMS emission jobs from an atlas manifest CSV."""
    emission_dir = Path(emission_dir)
    normalized_shard = _normalize_shard(shard)
    jobs: list[MMSEmissionJob] = []
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "segment_id" not in (reader.fieldnames or []):
            raise ValueError("manifest must include segment_id column")
        for row in reader:
            if not _needs_whisper_false(row):
                continue
            segment_id = int(row["segment_id"])
            if normalized_shard and not segment_in_shard(segment_id, *normalized_shard):
                continue
            emission_path = emission_dir / f"{segment_id}.npz"
            if emission_path.exists() and not include_cached:
                continue
            jobs.append(
                MMSEmissionJob(
                    segment_id=segment_id,
                    audio_path=Path(row["path"]),
                    provider=row.get("provider", ""),
                    duration_s=float(row.get("duration_s") or 0.0),
                    emission_path=emission_path,
                )
            )
            if limit is not None and len(jobs) >= limit:
                break
    return jobs


def _load_audio(path: Path, *, target_sample_rate: int):
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate == target_sample_rate:
        return audio

    import torch
    import torchaudio.functional as F

    tensor = torch.from_numpy(audio).float()
    resampled = F.resample(tensor, sample_rate, target_sample_rate)
    return resampled.numpy()


def run_mms_emission_jobs(
    jobs: Iterable[MMSEmissionJob],
    *,
    audio_root: Path = AUDIO_ROOT,
    config: MMSConfig = MMSConfig(),
) -> int:
    """Run MMS inference and write compressed emission caches."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    processor = AutoProcessor.from_pretrained(config.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(config.model_id).to(device)
    model.eval()

    count = 0
    for job in jobs:
        audio_path = job.audio_path
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_path
        audio = _load_audio(audio_path, target_sample_rate=config.sample_rate)
        inputs = processor(
            audio,
            sampling_rate=config.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[0].detach().cpu().numpy()
        job.emission_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            job.emission_path,
            logits=logits,
            segment_id=job.segment_id,
            model_id=config.model_id,
            sample_rate=config.sample_rate,
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    parser.add_argument("--emission-dir", type=Path, default=DEFAULT_EMISSION_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", help="1-based shard spec, e.g. 1/4")
    parser.add_argument("--include-cached", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    shard = parse_shard_spec(args.shard) if args.shard else None
    jobs = plan_mms_emission_jobs(
        args.manifest,
        emission_dir=args.emission_dir,
        shard=shard,
        limit=args.limit,
        include_cached=args.include_cached,
    )
    if args.dry_run:
        total_hours = sum(job.duration_s for job in jobs) / 3600.0
        print(f"Planned {len(jobs):,} MMS emission jobs / {total_hours:.2f} h")
        return 0

    completed = run_mms_emission_jobs(
        jobs,
        audio_root=args.audio_root,
        config=MMSConfig(model_id=args.model_id, device=args.device),
    )
    print(f"Wrote {completed:,} MMS emission caches to {args.emission_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
