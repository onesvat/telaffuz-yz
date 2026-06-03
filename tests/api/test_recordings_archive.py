from __future__ import annotations

import io
import json
import math
import wave
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.config import AppConfig
from api.routers.demo import assess
from api.services.audio import inspect_wav, normalize_upload
from api.services.recordings import (
    RecordingSlot,
    list_recordings,
    recording_paths,
    write_recording_result,
)


def _wav_bytes(*, segments: list[tuple[int, int]], sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for duration_ms, amplitude in segments:
            count = int(sample_rate * duration_ms / 1000)
            for i in range(count):
                sample = 0
                if amplitude:
                    sample = int(amplitude * math.sin(2 * math.pi * 220 * i / sample_rate))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))
    return buf.getvalue()


class _Upload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


def test_normalize_upload_trims_edge_silence(tmp_path: Path) -> None:
    data = _wav_bytes(segments=[(350, 0), (420, 4000), (350, 0)])
    target = tmp_path / "trimmed.wav"

    quality = asyncio.run(
        normalize_upload(
            upload=_Upload(data),  # type: ignore[arg-type]
            target_path=target,
            duration_ms=1120,
        )
    )
    trimmed = inspect_wav(target, fallback_duration_ms=1120)

    assert quality.status == "valid"
    assert 560 <= trimmed.duration_ms <= 700


def test_normalize_upload_marks_short_recording(tmp_path: Path) -> None:
    data = _wav_bytes(segments=[(100, 4000)])

    quality = asyncio.run(
        normalize_upload(
            upload=_Upload(data),  # type: ignore[arg-type]
            target_path=tmp_path / "short.wav",
            duration_ms=100,
        )
    )

    assert quality.status == "invalid"
    assert quality.reason == "too_short"


def test_assess_archives_and_admin_reads_recording(tmp_path: Path) -> None:
    class Runner:
        def run(self, *, model_id: str, wav_path: Path, expected_phones: list[str]):
            assert wav_path.exists()
            return {
                "status": "valid",
                "error_code": None,
                "result": {
                    "model": model_id,
                    "ipa_string": "".join(expected_phones),
                    "phones": [],
                    "gop": [],
                    "score": {
                        "item_score": 8,
                        "max_item_score": 10,
                        "interpretation_score": 78,
                        "interpretation_band": "warning",
                        "interpretation_version": "interpretation-v1",
                        "interpretation_explanation": [],
                        "diff": [],
                        "dimensions": {},
                    },
                    "stress": None,
                },
            }

    state = SimpleNamespace(config=AppConfig(data_root=tmp_path), coach_runner=Runner())
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    audio = _wav_bytes(segments=[(300, 0), (450, 4000), (300, 0)])
    body = asyncio.run(
        assess(
            request=request,  # type: ignore[arg-type]
            model_id="stub",
            word="anne",
            expected_phonemes="",
            audio=_Upload(audio),  # type: ignore[arg-type]
            duration_ms=1050,
            source="learner",
            participant_name="Ada",
            consent_audio=True,
            client_id="client-1",
            exercise_mode="",
            test_set="",
            test_index=None,
            prompt_id="W_ERR_027",
            session_id="onur",
            target_kind="canonical",
            attempt=1,
        )
    )
    recording_id = str(body["recording_id"])

    assert (tmp_path / "app" / "recordings" / "index.jsonl").exists()
    row = list_recordings(state.config.app_recordings_dir)[0]
    assert row["recording_id"] == recording_id
    assert row["source"] == "learner"
    assert row["participant_name"] == "Ada"
    assert row["prompt_id"] == "W_ERR_027"
    assert row["session_id"] == "onur"
    assert row["target_kind"] == "canonical"
    assert row["attempt"] == 1
    assert row["score"] == 80.0

    wav_path, result_path = recording_paths(state.config.app_recordings_dir, recording_id)
    assert wav_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["recording_id"] == recording_id
    assert payload["metadata"]["prompt_id"] == "W_ERR_027"


def test_coach_assess_returns_two_score_payload_and_archives(
    tmp_path: Path,
) -> None:
    class CoachRunner:
        def run(self, *, model_id: str, wav_path: Path, expected_phones: list[str]):
            assert wav_path.exists()
            assert model_id == "stub"
            assert expected_phones == ["a", "n", "e"]
            phones = [
                {
                    "index": 0,
                    "index_expected": 0,
                    "index_observed": 0,
                    "expected": "a",
                    "observed": "a",
                    "status": "correct",
                    "strict_status": "correct",
                    "score": 0.91,
                    "score_percent": 91.0,
                    "score_reason": "exact",
                    "raw_observed": "a",
                    "analysis_observed": "a",
                    "distance_features": {"exact": True},
                    "quality": 0.82,
                    "changed_by_analysis": None,
                    "timing": {"start_ms": 0, "end_ms": 120},
                }
            ]
            score = {
                "item_score": 91.0,
                "max_item_score": 100.0,
                "word_score": 0.91,
                "phoneme_core": 0.91,
                "extra_penalty": 1.0,
                "score_version": "phonetic-distance-v1",
                "phone_scores": [
                    {
                        "index": 0,
                        "index_expected": 0,
                        "expected": "a",
                        "observed": "a",
                        "score": 0.91,
                        "score_percent": 91.0,
                        "strict_status": "correct",
                        "score_reason": "exact",
                    }
                ],
            }
            return {
                "status": "ok",
                "error_code": None,
                "target": {
                    "text": None,
                    "ipa": "ane",
                    "phonemes": list(expected_phones),
                    "scoreable_phonemes": list(expected_phones),
                },
                "wav2vec": {
                    "model": model_id,
                    "ipa": "a",
                    "phonemes": ["a"],
                    "phones": [],
                    "timed_phones": [],
                },
                "analysis": {
                    "ipa": "a",
                    "phonemes": ["a"],
                    "phones": [],
                    "timed_phones": [],
                    "changes": [],
                },
                "result": {
                    "phones": phones,
                    "score": score,
                    "summary": {
                        "target_count": 1,
                        "correct_count": 1,
                        "incorrect_count": 0,
                        "missing_count": 0,
                        "extra_count": 0,
                        "accuracy": 1.0,
                        "mean_phone_score": 0.91,
                        "word_score": 0.91,
                        "quality_score": 0.82,
                        "extra_penalty": 1.0,
                        "quality_factor": 1.0,
                        "overall_score": 91.0,
                        "score_version": "phonetic-distance-v1",
                    },
                },
                "score": score,
                "debug": {"analysis_candidates": []},
            }

    state = SimpleNamespace(
        config=AppConfig(data_root=tmp_path),
        coach_runner=CoachRunner(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    audio = _wav_bytes(segments=[(300, 0), (450, 4000), (300, 0)])

    body = asyncio.run(
        assess(
            request=request,  # type: ignore[arg-type]
            model_id="stub",
            word="",
            expected_phonemes="a n e",
            audio=_Upload(audio),  # type: ignore[arg-type]
            duration_ms=1050,
            source="coach-test",
            participant_name="",
            consent_audio=False,
            client_id="",
            exercise_mode="",
            test_set="",
            test_index=None,
            prompt_id="W_ERR_027",
            session_id="onur",
            target_kind="",
            attempt=None,
        )
    )

    assert body["status"] == "ok"
    assert body["result"]["phones"][0]["score"] == 0.91
    assert body["result"]["phones"][0]["quality"] == 0.82
    assert body["score"]["item_score"] == 91.0
    assert "phones" not in body
    assert "diagnostics" not in body

    row = list_recordings(state.config.app_recordings_dir)[0]
    assert row["recording_id"] == body["recording_id"]
    assert row["source"] == "coach-test"
    assert row["target_kind"] == "coach"


def test_assess_invalid_audio_is_archived_with_metadata(tmp_path: Path) -> None:
    state = SimpleNamespace(config=AppConfig(data_root=tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    audio = _wav_bytes(segments=[(80, 4000)])
    body = asyncio.run(
        assess(
            request=request,  # type: ignore[arg-type]
            model_id="stub",
            word="",
            expected_phonemes="a n e",
            audio=_Upload(audio),  # type: ignore[arg-type]
            duration_ms=80,
            source="validation",
            participant_name="",
            consent_audio=False,
            client_id="",
            exercise_mode="",
            test_set="",
            test_index=None,
            prompt_id="W_ERR_027",
            session_id="onur",
            target_kind="intended_wrong",
            attempt=1,
        )
    )

    assert body["status"] == "invalid"
    assert body["result"] is None
    assert body["score"] is None
    assert body["debug"]["reason"] == "invalid_recording"
    assert "diagnostics" not in body

    row = list_recordings(state.config.app_recordings_dir)[0]
    assert row["status"] == "invalid"
    assert row["prompt_id"] == "W_ERR_027"
    assert row["target_kind"] == "intended_wrong"


def test_list_recordings_filters_before_limit(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    day_dir = recordings_dir / "2026-05-21"
    day_dir.mkdir(parents=True)

    def write(recording_id: str, *, word: str, source: str, status: str) -> None:
        slot = RecordingSlot(
            recording_id=recording_id,
            created_at=f"2026-05-21T12:00:0{recording_id[-1]}+00:00",
            wav_path=day_dir / f"{recording_id}.wav",
            result_path=day_dir / f"{recording_id}.json",
        )
        slot.wav_path.write_bytes(b"")
        write_recording_result(
            recordings_dir,
            slot,
            metadata={"word": word, "source": source, "participant_name": "Ada"},
            result={
                "status": status,
                "recording_quality": {"status": status, "duration_ms": 1000},
                "result": {"score": {"item_score": 8, "max_item_score": 10}},
            },
        )

    write("rec-1", word="anne", source="learner", status="valid")
    write("rec-2", word="baba", source="sandbox", status="valid")
    write("rec-3", word="annecik", source="sandbox", status="invalid")

    assert [r["recording_id"] for r in list_recordings(recordings_dir, limit=1, word="anne")] == ["rec-3"]
    assert [r["recording_id"] for r in list_recordings(recordings_dir, limit=5, word="anne")] == ["rec-3", "rec-1"]
    assert [r["recording_id"] for r in list_recordings(recordings_dir, limit=5, word="ANNE", source="sandbox")] == ["rec-3"]
    assert [r["recording_id"] for r in list_recordings(recordings_dir, limit=5, status="invalid")] == ["rec-3"]


def test_assess_rejects_unknown_target_kind(tmp_path: Path) -> None:
    state = SimpleNamespace(config=AppConfig(data_root=tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    audio = _wav_bytes(segments=[(300, 0), (450, 4000), (300, 0)])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            assess(
                request=request,  # type: ignore[arg-type]
                model_id="stub",
                word="",
                expected_phonemes="a n e",
                audio=_Upload(audio),  # type: ignore[arg-type]
                duration_ms=1050,
                source="sandbox",
                participant_name="",
                consent_audio=False,
                client_id="",
                exercise_mode="",
                test_set="",
                test_index=None,
                target_kind="wrongish",
                prompt_id="",
                session_id="",
                attempt=None,
            )
        )
    assert exc.value.status_code == 400
