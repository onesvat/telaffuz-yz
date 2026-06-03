"""API inference seam delegating to the assessment package.

The API layer holds a thin contract (``InferenceOutput`` + ``InferenceService``
ABC) and forwards to ``assess.inference.Wav2VecInferenceService``. When the
underlying model artifact is missing the assessment package returns a graceful
``model_unavailable`` result which is mapped here to ``status="failed"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceOutput:
    status: str
    predicted_phonemes: list[str]
    confidence: float | None
    error_code: str | None = None
    speech_duration_ms: int | None = None


class InferenceService:
    def analyze(
        self,
        *,
        model_id: str,
        wav_path: Path,
        expected_phonemes: list[str],
    ) -> InferenceOutput:
        raise NotImplementedError


class AssessInferenceService(InferenceService):
    def __init__(self) -> None:
        from assess.inference import Wav2VecInferenceService
        from assess.registry import ModelRegistry

        self._service = Wav2VecInferenceService(ModelRegistry.from_config())

    def analyze(
        self,
        *,
        model_id: str,
        wav_path: Path,
        expected_phonemes: list[str],
    ) -> InferenceOutput:
        result = self._service.analyze(model_alias=model_id, wav_path=wav_path)
        if result.status == "ok":
            speech_ms: int | None = None
            if result.phones:
                speech_ms = max(
                    1, int(result.phones[-1].end_ms - result.phones[0].start_ms)
                )
            return InferenceOutput(
                status="ok",
                predicted_phonemes=result.predicted_phonemes,
                confidence=result.mean_confidence,
                speech_duration_ms=speech_ms,
            )
        return InferenceOutput(
            status="failed",
            predicted_phonemes=[],
            confidence=None,
            error_code=result.error_code or "model_unavailable",
        )
