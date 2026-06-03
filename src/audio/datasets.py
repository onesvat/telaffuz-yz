"""Speaker-aware deterministic train/val/test split for the wav2vec dataset.

Dataset spec: curated-400h-v1
  Sources: CV-clean + ISSAI-clean + audiobook-core (~400 h)
  Split:   SHA1(speaker_id) % 100 → 0–89 train, 90–94 val, 95–99 test
  Output:  data/wav2vec/<spec_name>/{train,val,test}.parquet

The SHA1-bucket assignment is deterministic and speaker-leakage-free: all
segments from the same speaker always land in the same split regardless of
insertion order.
"""

from __future__ import annotations

import hashlib


def speaker_bucket(speaker_id: str) -> int:
    """SHA1(speaker_id) % 100 — deterministic, leakage-free bucket assignment."""
    digest = hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def split_for_bucket(bucket: int) -> str:
    """Map bucket 0..99 to split (90/5/5 ratio)."""
    if 0 <= bucket <= 89:
        return "train"
    if 90 <= bucket <= 94:
        return "val"
    if 95 <= bucket <= 99:
        return "test"
    raise ValueError(f"bucket out of range: {bucket}")


def sha1_short(value: str, *, length: int = 12) -> str:
    """Stable short hash for human-readable source-derived speaker IDs."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]


def _source_component(
    *,
    provider: str,
    path: str,
    source_folder: str,
    fallback: str,
) -> str:
    raw = (source_folder or path or fallback).strip("/")
    parts = [part for part in raw.split("/") if part]
    if len(parts) >= 2 and parts[0] == provider:
        return parts[1]
    if parts:
        return parts[0]
    return fallback


def _hashed_speaker(provider_prefix: str, value: str) -> str:
    return f"{provider_prefix}_{sha1_short(value)}"


def assign_legacy_speaker_id(
    *,
    provider: str,
    segment_id: str,
    path: str,
    source_folder: str,
    cv_client_id: str | None = None,
) -> tuple[str, str]:
    """Assign a deterministic speaker_id and source label for legacy imported segments.

    Common Voice rows without a validated.tsv match are still importable but
    marked with missing_cv_client_id so reports can quantify the risk.
    """
    if provider == "common_voice":
        if cv_client_id:
            return cv_client_id, "validated.tsv.client_id"
        return f"common_voice_missing_{sha1_short(segment_id)}", "missing_cv_client_id"

    if provider == "issai_tsc":
        split = source_folder.strip("/") if source_folder else ""
        if split not in {"train", "dev", "test"}:
            parts = segment_id.split("_")
            split = parts[1] if len(parts) >= 3 else "unknown"
        return f"issai_{split}", "issai_split_fallback"

    if provider == "audiobooks":
        book = _source_component(
            provider=provider,
            path=path,
            source_folder=source_folder,
            fallback=segment_id,
        )
        return _hashed_speaker("audiobook", book), "book_folder"

    if provider == "podcasts":
        podcast = _source_component(
            provider=provider,
            path=path,
            source_folder=source_folder,
            fallback=segment_id,
        )
        return _hashed_speaker("podcast", podcast), "podcast_name"

    if provider == "youtube":
        channel = _source_component(
            provider=provider,
            path=path,
            source_folder=source_folder,
            fallback=segment_id,
        )
        return _hashed_speaker("youtube", channel), "yt_channel"

    return _hashed_speaker("legacy_unknown", segment_id), "unknown_provider"
