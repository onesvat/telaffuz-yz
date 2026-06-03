"""assess.inference — saf CTC decode mantığı + graceful model_unavailable.

Ağır model yükleme (transformers) glue'dur, unit test edilmez; testlenebilir
çekirdek `decode_ctc` (sentetik logits) + registry-graceful yol.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from assess.inference import (
    InferenceResult,
    Wav2VecInferenceService,
    decode_ctc,
)
from assess.registry import ModelRegistry


def test_decode_ctc_collapses_repeats_and_drops_blank() -> None:
    id_to_token = {0: "<blank>", 1: "a", 2: "b"}
    # 4 frame: a, a, <blank>, b  → collapse → a, blank(drop), b
    logits = np.array(
        [
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    phones = decode_ctc(
        logits, id_to_token, drop_tokens={"<blank>"}, total_ms=400
    )

    assert [p.ipa for p in phones] == ["a", "b"]
    assert (phones[0].start_ms, phones[0].end_ms) == (0, 200)
    assert (phones[1].start_ms, phones[1].end_ms) == (300, 400)


def test_decode_ctc_confidence_is_softmax_mean_over_run() -> None:
    id_to_token = {0: "<blank>", 1: "a"}
    logits = np.array([[0.0, 2.0]])  # tek frame, token "a"
    phones = decode_ctc(
        logits, id_to_token, drop_tokens={"<blank>"}, total_ms=100
    )

    expected = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0))
    assert len(phones) == 1
    assert phones[0].ipa == "a"
    assert phones[0].confidence == pytest_approx(expected)
    assert (phones[0].start_ms, phones[0].end_ms) == (0, 100)


def test_decode_ctc_drops_spc_and_pad() -> None:
    id_to_token = {0: "<blank>", 1: "<pad>", 2: "<spc>", 3: "m"}
    logits = np.array(
        [
            [0.0, 0.0, 0.0, 9.0],  # m
            [0.0, 0.0, 9.0, 0.0],  # <spc>
            [0.0, 9.0, 0.0, 0.0],  # <pad>
        ]
    )
    phones = decode_ctc(
        logits,
        id_to_token,
        drop_tokens={"<blank>", "<pad>", "<spc>"},
        total_ms=300,
    )
    assert [p.ipa for p in phones] == ["m"]


def test_inference_result_helpers() -> None:
    from assess.schema import PhoneTiming

    result = InferenceResult(
        model="mms-1b",
        phones=[
            PhoneTiming(ipa="m", start_ms=0, end_ms=40, confidence=0.9),
            PhoneTiming(ipa="æ", start_ms=40, end_ms=120, confidence=0.7),
        ],
    )
    assert result.predicted_phonemes == ["m", "æ"]
    assert result.ipa_string == "mæ"
    assert result.mean_confidence == pytest_approx(0.8)
    assert result.status == "ok"
    assert result.frame_posteriors == ()


def test_inference_result_mean_confidence_none_when_empty() -> None:
    result = InferenceResult(model="mms-1b", phones=[])
    assert result.mean_confidence is None
    assert result.predicted_phonemes == []


def test_service_returns_model_unavailable_when_checkpoint_missing(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "models.toml"
    cfg.write_text(
        'default = "mms-1b"\n\n'
        "[models.mms-1b]\n"
        'display_name = "x"\n'
        'checkpoint = "runs/gone"\n'
        'processor = "runs/gone"\n'
        "frozen = true\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_config(cfg, base_dir=tmp_path)
    service = Wav2VecInferenceService(registry)

    result = service.analyze(model_alias="mms-1b", wav_path=tmp_path / "x.wav")

    assert result.status == "model_unavailable"
    assert result.error_code == "model_unavailable"
    assert result.phones == []


def test_service_run_returns_none_when_checkpoint_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "models.toml"
    cfg.write_text(
        'default = "mms-1b"\n\n'
        "[models.mms-1b]\n"
        'display_name = "x"\n'
        'checkpoint = "runs/gone"\n'
        'processor = "runs/gone"\n'
        "frozen = true\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_config(cfg, base_dir=tmp_path)
    service = Wav2VecInferenceService(registry)

    result = service.run("mms-1b", np.zeros(160, dtype=np.float32), 16_000)

    assert result is None


# küçük yerel approx (pytest.approx ister; importu sade tutuyoruz)
def pytest_approx(value: float, rel: float = 1e-6):
    import pytest

    return pytest.approx(value, rel=rel)
