"""Audio pipeline package.

Layout:
    db.py               — SQLite connection helpers and schema version check
    schema.py           — DDL and init_db for the v3 schema
    cli.py              — entry point: `uv run audio <subcommand>`
    whisper_runner.py   — Whisper-large-v3-turbo batch transcription
    mms_runner.py       — MMS-1b-fl102 emission cache runner (GPU)
    alignment.py        — MMS-CTC forced alignment and DB persistence
    atlas.py            — phoneme atlas manifest builder
    atlas_pilot.py      — deterministic pilot manifest and model-selection report
    atlas_export.py     — Parquet/CSV export and atlas evidence reports
    manual.py           — Praat TextGrid import and manual annotation summary
    features.py         — acoustic feature extraction for phone windows
    datasets.py         — speaker-aware deterministic train/val/test split

Schema v3: single `segments` table with INTEGER AUTOINCREMENT primary key.
All modules write to data/audio.sqlite (external artifact, not in Git).
GPU-bound modules (whisper_runner, mms_runner, alignment) run on a separate
desktop machine with CUDA; see REPRODUCIBILITY.md for hardware requirements.
"""

from audio.db import AUDIO_DB_PATH, AUDIO_ROOT, connect, schema_version

__all__ = ["AUDIO_DB_PATH", "AUDIO_ROOT", "connect", "schema_version"]
