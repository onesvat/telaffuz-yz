"""assess.schema — versiyonlu, JSON-stabil CLI çıktı kontratı."""

from __future__ import annotations

import json

from assess.schema import (
    SCHEMA_VERSION,
    PhoneTiming,
    TranscribeResult,
)


def test_transcribe_result_as_dict_is_json_stable() -> None:
    result = TranscribeResult(
        model="mms-1b",
        ipa_string="mæɾhaba",
        phones=[
            PhoneTiming(ipa="m", start_ms=0, end_ms=40, confidence=0.91),
            PhoneTiming(ipa="æ", start_ms=40, end_ms=120, confidence=None),
        ],
    )
    payload = result.as_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] == "ok"
    assert payload["model"] == "mms-1b"
    # Nested dataclass'lar dict'e indirgenmeli (jsonlanabilir).
    assert payload["phones"][0] == {
        "ipa": "m",
        "start_ms": 0,
        "end_ms": 40,
        "confidence": 0.91,
    }
    # IPA karakterleri ascii'ye kaçırılmadan round-trip etmeli.
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "æ" in dumped
    assert json.loads(dumped)["ipa_string"] == "mæɾhaba"


def test_transcribe_result_failure_carries_error_code() -> None:
    result = TranscribeResult(
        model="xls-r-300m",
        ipa_string="",
        phones=[],
        status="model_unavailable",
        error_code="model_unavailable",
    )
    payload = result.as_dict()

    assert payload["status"] == "model_unavailable"
    assert payload["error_code"] == "model_unavailable"
    assert payload["phones"] == []
