from __future__ import annotations

import io
import json
import math
import wave
import asyncio
from pathlib import Path

from api.config import AppConfig
from api.services.recordings import list_recordings, recording_paths
from api.services.studio import load_studio_prompts, save_recording


def _wav_bytes(*, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for duration_ms, amplitude in [(300, 0), (450, 4000), (300, 0)]:
            count = int(sample_rate * duration_ms / 1000)
            for i in range(count):
                sample = int(amplitude * math.sin(2 * math.pi * 220 * i / sample_rate)) if amplitude else 0
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))
    return buf.getvalue()


class _Upload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _seed_prompts(path: Path, prompt_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["prompt_id\ttarget_text"] + [f"{pid}\t{pid}-metin" for pid in prompt_ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_demo_studio_prompts_isolated_from_validation(tmp_path: Path) -> None:
    cfg = AppConfig(data_root=tmp_path)
    _seed_prompts(cfg.prompts_tsv, ["VAL_1", "VAL_2"])
    _seed_prompts(cfg.demo_prompts_tsv, ["DEMO_1"])
    demo = load_studio_prompts(cfg.demo_prompts_tsv)
    val = load_studio_prompts(cfg.prompts_tsv)
    assert [item["prompt_id"] for item in demo] == ["DEMO_1"]
    assert [item["prompt_id"] for item in val] == ["VAL_1", "VAL_2"]


def test_demo_capture_paths_isolated_from_validation(tmp_path: Path) -> None:
    cfg = AppConfig(data_root=tmp_path)
    assert cfg.demo_recordings_dir != cfg.recordings_dir
    assert cfg.demo_prompts_tsv != cfg.prompts_tsv
    assert cfg.demo_capture_dir.name == "demo_capture"


def test_demo_studio_archive_metadata_preserves_collection(tmp_path: Path) -> None:
    cfg = AppConfig(data_root=tmp_path)
    path, quality, recording_id = asyncio.run(
        save_recording(
            cfg.demo_recordings_dir,
            "session-a",
            "demo01_ok",
            _Upload(_wav_bytes()),  # type: ignore[arg-type]
            duration_ms=1050,
            archive_recordings_dir=cfg.app_recordings_dir,
            collection_name="demo_studio",
        )
    )

    assert path is not None
    assert quality.status == "valid"
    assert recording_id is not None
    row = list_recordings(cfg.app_recordings_dir)[0]
    assert row["source"] == "demo_studio"
    assert row["collection"] == "demo_studio"
    assert row["prompt_id"] == "demo01_ok"
    assert row["session_id"] == "session-a"

    _, result_path = recording_paths(
        cfg.app_recordings_dir, recording_id
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["collection"] == "demo_studio"
    assert payload["metadata"]["prompt_id"] == "demo01_ok"


def test_committed_demo_prompts_are_twenty_alternating_slots() -> None:
    tsv = Path(__file__).resolve().parents[2] / "data" / "demo_capture" / "prompts.tsv"
    ids = [item["prompt_id"] for item in load_studio_prompts(tsv)]
    assert len(ids) == 20
    assert ids[0] == "demo01_ok"
    assert ids[1] == "demo01_err"
    assert ids[-1] == "demo10_err"
