"""API runner for the two-score pronunciation coach endpoint."""

from __future__ import annotations

from pathlib import Path

from api.config import REPO_ROOT
from api.services.models import resolve_alias

DEFAULT_COACH_STATS_PATH = REPO_ROOT / "configs" / "coach_gmm_v1.json"
DEFAULT_COACH_AUTHORITY_PATH = REPO_ROOT / "configs" / "decision_authority.json"
DEFAULT_COACH_CALIBRATION_PATH = REPO_ROOT / "configs" / "coach_gmm_calibration_v1.json"


class CoachRunner:
    def __init__(
        self,
        *,
        stats_path: Path = DEFAULT_COACH_STATS_PATH,
        authority_path: Path = DEFAULT_COACH_AUTHORITY_PATH,
        calibration_path: Path = DEFAULT_COACH_CALIBRATION_PATH,
    ) -> None:
        self._stats_path = stats_path
        self._authority_path = authority_path
        self._calibration_path = calibration_path
        # Runtime services (and the loaded wav2vec model they hold) are reused
        # across requests, keyed by model alias. Without this each assessment
        # rebuilt the service graph and reloaded the model from disk.
        self._services: dict[str, object] = {}

    def run(
        self,
        *,
        model_id: str,
        wav_path: Path,
        expected_phones: list[str],
    ) -> dict[str, object]:
        entry = resolve_alias(model_id)
        if entry is None:
            return _empty_result(
                status="failed",
                error_code="unknown_model",
                model_alias=model_id,
                expected_phones=expected_phones,
            )
        if not entry.available:
            return _empty_result(
                status="model_unavailable",
                error_code="model_unavailable",
                model_alias=entry.alias,
                expected_phones=expected_phones,
            )

        from assess.contracts import CoachRequest  # noqa: PLC0415
        from assess.runtime import build_default_runtime_services, run_coach  # noqa: PLC0415

        request = CoachRequest(
            audio_path=str(wav_path),
            model_alias=entry.alias,
            target_phones=list(expected_phones),
            stats_path=str(self._stats_path),
            authority_path=str(self._authority_path),
            calibration_path=(
                str(self._calibration_path)
                if self._calibration_path.exists()
                else None
            ),
        )
        services = self._services.get(entry.alias)
        if services is None:
            services = build_default_runtime_services(request)
            self._services[entry.alias] = services
        result = run_coach(request, services=services)
        coach_payload = result.as_dict()
        if result.status != "ok":
            return _runtime_error_result(
                status=result.status,
                expected_phones=expected_phones,
                coach_payload=coach_payload,
            )
        error_code = None if result.status == "ok" else result.status
        target_payload = _as_dict(coach_payload.get("target")) or {
            "phonemes": list(expected_phones),
            "scoreable_phonemes": list(expected_phones),
        }
        wav2vec_payload = _as_dict(coach_payload.get("wav2vec"))
        analysis_payload = _as_dict(coach_payload.get("analysis"))
        result_payload = _as_dict(coach_payload.get("result"))
        debug_payload = _as_dict(coach_payload.get("debug"))
        return {
            "status": result.status,
            "error_code": error_code,
            "target": target_payload,
            "wav2vec": wav2vec_payload,
            "analysis": analysis_payload,
            "result": result_payload,
            "score": result_payload.get("score"),
            "debug": debug_payload,
        }


def _empty_result(
    *,
    status: str,
    error_code: str,
    model_alias: str,
    expected_phones: list[str],
) -> dict[str, object]:
    target = {
        "text": None,
        "ipa": "".join(expected_phones),
        "phonemes": list(expected_phones),
        "scoreable_phonemes": list(expected_phones),
    }
    debug = {"reason": error_code}
    return {
        "status": status,
        "error_code": error_code,
        "target": target,
        "wav2vec": {"model": model_alias, "ipa": "", "phonemes": [], "phones": [], "timed_phones": []},
        "analysis": {"ipa": "", "phonemes": [], "phones": [], "timed_phones": [], "changes": []},
        "result": None,
        "score": None,
        "debug": debug,
    }


def _runtime_error_result(
    *,
    status: str,
    expected_phones: list[str],
    coach_payload: dict[str, object],
) -> dict[str, object]:
    target_payload = _as_dict(coach_payload.get("target")) or {
        "phonemes": list(expected_phones),
        "scoreable_phonemes": list(expected_phones),
    }
    wav2vec_payload = _as_dict(coach_payload.get("wav2vec"))
    analysis_payload = _as_dict(coach_payload.get("analysis"))
    debug_payload = _as_dict(coach_payload.get("debug"))
    return {
        "status": status,
        "error_code": status,
        "target": target_payload,
        "wav2vec": wav2vec_payload,
        "analysis": analysis_payload,
        "result": None,
        "score": None,
        "debug": debug_payload,
    }


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
