from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from api.services.coach_runner import CoachRunner


def test_coach_runner_returns_target_independent_result(monkeypatch, tmp_path: Path) -> None:
    import api.services.coach_runner as coach_runner_mod
    import assess.runtime as runtime_mod

    captured: dict[str, object] = {}

    class FakeResult:
        status = "ok"
        speech_start_ms_original = 10
        speech_end_ms_original = 210

        def as_dict(self) -> dict[str, object]:
            return {
                "model_alias": "mms-1b",
                "status": "ok",
                "speech_start_ms_original": 10,
                "speech_end_ms_original": 210,
                "phones": [],
                "free_decode": [
                    {
                        "ipa": "a",
                        "start_ms": 10,
                        "end_ms": 120,
                        "confidence": 0.98,
                    }
                ],
                "target": {
                    "text": None,
                    "ipa": "a",
                    "phonemes": ["a"],
                    "scoreable_phonemes": ["a"],
                },
                "wav2vec": {
                    "model": "mms-1b",
                    "ipa": "a",
                    "phonemes": ["a"],
                    "phones": [
                        {
                            "ipa": "a",
                            "start_ms": 10,
                            "end_ms": 120,
                            "confidence": 0.98,
                        }
                    ],
                    "timed_phones": [
                        {
                            "ipa": "a",
                            "start_ms": 10,
                            "end_ms": 120,
                            "confidence": 0.98,
                        }
                    ],
                },
                "analysis": {
                    "ipa": "a",
                    "phonemes": ["a"],
                    "phones": [
                        {
                            "ipa": "a",
                            "raw_ipa": "a",
                            "start_ms": 10,
                            "end_ms": 120,
                            "confidence": 0.98,
                            "quality": 0.82,
                            "changed_by_analysis": False,
                        }
                    ],
                    "timed_phones": [
                        {
                            "ipa": "a",
                            "raw_ipa": "a",
                            "start_ms": 10,
                            "end_ms": 120,
                            "confidence": 0.98,
                            "quality": 0.82,
                            "changed_by_analysis": False,
                        }
                    ],
                    "changes": [],
                },
                "result": {
                    "phones": [
                        {
                            "index": 0,
                            "expected": "a",
                            "observed": "a",
                            "status": "correct",
                            "strict_status": "correct",
                            "score": 1.0,
                            "score_percent": 100.0,
                            "score_reason": "exact",
                            "raw_observed": "a",
                            "analysis_observed": "a",
                            "distance_features": {
                                "class": {
                                    "expected": "vowel",
                                    "observed": "vowel",
                                    "match": True,
                                },
                                "exact": True,
                            },
                            "quality": 0.82,
                            "changed_by_analysis": None,
                            "timing": {"start_ms": 10, "end_ms": 120},
                        }
                    ],
                    "score": {
                        "item_score": 100.0,
                        "max_item_score": 100.0,
                        "word_score": 1.0,
                        "phoneme_core": 1.0,
                        "extra_penalty": 1.0,
                        "score_version": "phonetic-distance-v1",
                        "phone_scores": [
                            {
                                "index": 0,
                                "index_expected": 0,
                                "expected": "a",
                                "observed": "a",
                                "score": 1.0,
                                "score_percent": 100.0,
                                "strict_status": "correct",
                                "score_reason": "exact",
                            }
                        ],
                    },
                    "summary": {
                        "target_count": 1,
                        "correct_count": 1,
                        "incorrect_count": 0,
                        "missing_count": 0,
                        "extra_count": 0,
                        "accuracy": 1.0,
                        "mean_phone_score": 1.0,
                        "word_score": 1.0,
                        "quality_score": 0.82,
                        "extra_penalty": 1.0,
                        "quality_factor": 1.0,
                        "overall_score": 100.0,
                        "score_version": "phonetic-distance-v1",
                    },
                },
                "debug": {"analysis_candidates": []},
            }

    def fake_run_coach(request: object, *, services: object | None = None) -> FakeResult:
        captured["request"] = request
        captured["services"] = services
        return FakeResult()

    monkeypatch.setattr(
        coach_runner_mod,
        "resolve_alias",
        lambda _model_id: SimpleNamespace(alias="mms-1b", available=True),
    )
    monkeypatch.setattr(
        runtime_mod, "build_default_runtime_services", lambda _request: object()
    )
    monkeypatch.setattr(runtime_mod, "run_coach", fake_run_coach)

    runner = CoachRunner(
        stats_path=tmp_path / "stats.json",
        authority_path=tmp_path / "authority.json",
    )
    result = runner.run(
        model_id="mms-1b",
        wav_path=tmp_path / "x.wav",
        expected_phones=["a"],
    )

    assert result["status"] == "ok"
    assert result["target"]["phonemes"] == ["a"]
    assert result["wav2vec"]["phonemes"] == ["a"]
    assert result["analysis"]["phonemes"] == ["a"]
    assert result["result"]["phones"][0]["status"] == "correct"
    assert result["result"]["phones"][0]["score"] == 1.0
    assert result["result"]["phones"][0]["quality"] == 0.82
    assert result["score"]["item_score"] == 100.0
    assert result["result"]["summary"]["overall_score"] == 100.0
    assert result["debug"]["analysis_candidates"] == []
    assert "aggregate" not in result
    assert "diagnostics" not in result
    assert "gop_error" not in result
    request = captured["request"]
    assert getattr(request, "model_alias") == "mms-1b"
    assert getattr(request, "target_phones") == ["a"]
    assert getattr(request, "stats_path") == str(tmp_path / "stats.json")


def test_coach_runner_non_ok_status_does_not_emit_score(
    monkeypatch, tmp_path: Path
) -> None:
    import api.services.coach_runner as coach_runner_mod
    import assess.runtime as runtime_mod

    class FakeResult:
        status = "no_speech"
        speech_start_ms_original = 0
        speech_end_ms_original = 420

        def as_dict(self) -> dict[str, object]:
            return {
                "model_alias": "mms-1b",
                "status": "no_speech",
                "speech_start_ms_original": 0,
                "speech_end_ms_original": 420,
                "phones": [],
                "free_decode": [],
            }

    monkeypatch.setattr(
        coach_runner_mod,
        "resolve_alias",
        lambda _model_id: SimpleNamespace(alias="mms-1b", available=True),
    )
    monkeypatch.setattr(
        runtime_mod, "build_default_runtime_services", lambda _request: object()
    )
    monkeypatch.setattr(
        runtime_mod,
        "run_coach",
        lambda _request, *, services=None: FakeResult(),
    )

    runner = CoachRunner(
        stats_path=tmp_path / "stats.json",
        authority_path=tmp_path / "authority.json",
    )
    result = runner.run(
        model_id="mms-1b",
        wav_path=tmp_path / "x.wav",
        expected_phones=["a"],
    )

    assert result["status"] == "no_speech"
    assert result["error_code"] == "no_speech"
    assert result["result"] is None
    assert result["score"] is None
    assert "aggregate" not in result
    assert "diagnostics" not in result
    assert "gop_error" not in result
