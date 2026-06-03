"""Unified demo router.

Exposes the four endpoints the single demo frontend needs:

- ``GET  /api/v1/phonemes``       — phoneme palette metadata.
- ``GET  /api/v1/g2p/inspect``    — G2P pipeline trace + IPA + phones.
- ``GET  /api/v1/models``         — wav2vec model registry list.
- ``POST /api/v1/assess``         — two-score coach assessment for an upload.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from g2p.pipeline import G2PResult, TraceStep, transcribe_word

from api.services.audio import normalize_upload
from api.services.models import list_models
from api.services.recordings import new_recording_slot, write_recording_result
from api.services.phonemes import (
    CATEGORY_LABELS,
    SYLLABLE_BOUNDARY,
    grouped_palette,
    phonemes_as_dict,
)

router = APIRouter(prefix="/api/v1")


@router.get("/phonemes")
async def phonemes() -> dict[str, object]:
    return {
        "phonemes": phonemes_as_dict(),
        "groups": [
            {
                "key": key,
                "label": label,
                "items": [
                    {
                        "symbol": item.symbol,
                        "name_tr": item.name_tr,
                        "example": item.example,
                        "hint_tr": item.hint_tr,
                        "category": item.category,
                    }
                    for item in items
                ],
            }
            for key, label, items in grouped_palette()
        ],
        "category_labels": CATEGORY_LABELS,
        "syllable_boundary": SYLLABLE_BOUNDARY,
    }


@router.get("/g2p/inspect")
async def g2p_inspect(
    word: str = Query(..., min_length=1, max_length=128),
    pedagogical: bool = True,
    use_reference: bool = True,
) -> dict[str, Any]:
    if not word.strip():
        raise HTTPException(400, "empty word")
    return _g2p_payload(
        transcribe_word(
            word,
            use_reference=use_reference,
            pedagogical_allophones=pedagogical,
            trace=True,
        )
    )


@router.get("/models")
async def models() -> dict[str, object]:
    return {"items": list_models()}


@router.post("/assess")
async def assess(
    request: Request,
    model_id: str = Form(...),
    word: str = Form(""),
    expected_phonemes: str = Form(""),
    audio: UploadFile = File(...),
    duration_ms: int = Form(1000),
    source: str = Form("sandbox"),
    participant_name: str = Form(""),
    consent_audio: bool = Form(False),
    client_id: str = Form(""),
    exercise_mode: str = Form(""),
    test_set: str = Form(""),
    test_index: int | None = Form(None),
    prompt_id: str = Form(""),
    session_id: str = Form(""),
    target_kind: str = Form(""),
    attempt: int | None = Form(None),
) -> dict[str, object]:
    target_kind_clean = target_kind.strip() or None
    if target_kind_clean not in {None, "manual", "canonical", "intended_wrong", "coach"}:
        raise HTTPException(
            400, "target_kind must be manual, canonical, intended_wrong, or coach"
        )
    phones = _target_phones(word=word, expected_phonemes=expected_phonemes)
    if not phones:
        raise HTTPException(400, "word or expected_phonemes is required")

    slot = new_recording_slot(request.app.state.config.app_recordings_dir)
    quality = await normalize_upload(
        upload=audio,
        target_path=slot.wav_path,
        duration_ms=duration_ms,
    )
    base = {
        "recording_id": slot.recording_id,
        "recording_quality": quality.as_dict(),
        "expected_phonemes": phones,
        "model_id": model_id,
    }
    metadata = {
        "source": source or "sandbox",
        "participant_name": participant_name.strip() or None,
        "consent_audio": consent_audio,
        "client_id": client_id.strip() or None,
        "exercise_mode": exercise_mode.strip() or None,
        "test_set": test_set.strip() or None,
        "test_index": test_index,
        "word": word.strip() or None,
        "model_id": model_id,
        "prompt_id": prompt_id.strip() or None,
        "session_id": session_id.strip() or None,
        "target_kind": target_kind_clean or "coach",
        "attempt": attempt,
    }
    if quality.status != "valid":
        target = {
            "text": word.strip() or None,
            "ipa": "".join(phones),
            "phonemes": phones,
            "scoreable_phonemes": phones,
        }
        response = {
            **base,
            "status": "invalid",
            "error_code": None,
            "target": target,
            "wav2vec": {
                "model": model_id,
                "ipa": "",
                "phonemes": [],
                "phones": [],
                "timed_phones": [],
            },
            "analysis": {
                "ipa": "",
                "phonemes": [],
                "phones": [],
                "timed_phones": [],
                "changes": [],
            },
            "result": None,
            "score": None,
            "debug": {"reason": "invalid_recording"},
        }
        write_recording_result(
            request.app.state.config.app_recordings_dir,
            slot,
            metadata=metadata,
            result=response,
        )
        return response

    runner = getattr(request.app.state, "coach_runner", None)
    if runner is None:
        from api.services.coach_runner import CoachRunner

        runner = CoachRunner()
        request.app.state.coach_runner = runner

    try:
        outcome = runner.run(
            model_id=model_id,
            wav_path=slot.wav_path,
            expected_phones=phones,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    response = {**base, **outcome}
    write_recording_result(
        request.app.state.config.app_recordings_dir,
        slot,
        metadata=metadata,
        result=response,
    )
    return response


def _target_phones(*, word: str, expected_phonemes: str) -> list[str]:
    phones = [token for token in expected_phonemes.split(" ") if token]
    if phones or not word.strip():
        return phones
    from audio.atlas import expected_49_from_text

    return list(expected_49_from_text(word.strip()))


def _g2p_payload(result: G2PResult) -> dict[str, Any]:
    return {
        "word": result.word,
        "normalized": result.normalized,
        "ipa": result.ipa,
        "phonemes": list(result.phonemes),
        "syllables": [list(syllable) for syllable in result.syllables],
        "source": result.source,
        "exception_type": result.exception_type,
        "warnings": list(result.warnings),
        "trace": [_step_payload(step) for step in result.trace],
    }


def _step_payload(step: TraceStep) -> dict[str, Any]:
    return {
        "name": step.name,
        "input": list(step.input),
        "output": list(step.output),
        "fired": step.fired,
        "note": step.note,
    }
