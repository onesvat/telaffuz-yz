"""Pure wav2vec phoneme-CTC inference.

The testable core decode_ctc (numpy only, no model required) is separated
from the heavy model loader Wav2VecInferenceService (lazy torch/transformers
imports; graceful model_unavailable when checkpoint is absent).

InferenceService ABC provides the seam shared by CLI and evaluation scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from g2p.constants import CTC_SPECIAL_TOKENS

from assess.posterior import PosteriorFrame, frame_posteriors_from_logits
from assess.registry import ModelRegistry, UnknownModelError
from assess.schema import PhoneTiming

SAMPLE_RATE = 16_000
# blank/pad/unk/spc are dropped; the legacy "<sil>" marker is not in the
# deployed CTC vocabulary but is dropped defensively if an older processor emits it.
DEFAULT_DROP_TOKENS: frozenset[str] = frozenset(CTC_SPECIAL_TOKENS) | {"<sil>"}


def decode_ctc(
    logits: np.ndarray,
    id_to_token: Mapping[int, str],
    *,
    drop_tokens: set[str] | frozenset[str],
    total_ms: int,
) -> list[PhoneTiming]:
    """CTC argmax → collapse repeated runs → drop special tokens →
    return per-run (start_ms, end_ms, mean softmax confidence)."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"logits must be 2-D (T,V), got shape {arr.shape}")
    n_frames = arr.shape[0]
    if n_frames == 0:
        return []
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    ids = probs.argmax(axis=1)

    phones: list[PhoneTiming] = []
    i = 0
    while i < n_frames:
        cur = int(ids[i])
        j = i
        while j < n_frames and int(ids[j]) == cur:
            j += 1
        token = id_to_token.get(cur)
        if token is not None and token not in drop_tokens:
            phones.append(
                PhoneTiming(
                    ipa=token,
                    start_ms=round(i * total_ms / n_frames),
                    end_ms=round(j * total_ms / n_frames),
                    confidence=float(probs[i:j, cur].mean()),
                )
            )
        i = j
    return phones


@dataclass(frozen=True)
class InferenceResult:
    model: str
    phones: list[PhoneTiming] = field(default_factory=list)
    status: str = "ok"
    error_code: str | None = None
    frame_posteriors: tuple[PosteriorFrame, ...] = field(default_factory=tuple)

    @property
    def predicted_phonemes(self) -> list[str]:
        return [p.ipa for p in self.phones]

    @property
    def ipa_string(self) -> str:
        return "".join(p.ipa for p in self.phones)

    @property
    def mean_confidence(self) -> float | None:
        vals = [p.confidence for p in self.phones if p.confidence is not None]
        return float(sum(vals) / len(vals)) if vals else None


class InferenceService(ABC):
    """Shared inference seam for CLI and evaluation scripts."""

    @abstractmethod
    def analyze(self, *, model_alias: str | None, wav_path: Path) -> InferenceResult:
        ...


class Wav2VecInferenceService(InferenceService):
    """Real Wav2Vec2ForCTC loader with lazy imports and graceful unavailability."""

    def __init__(self, registry: ModelRegistry, *, device: str = "auto") -> None:
        self._registry = registry
        self._device = device
        self._cache: dict[str, tuple[Any, Any, str]] = {}
        # (alias, wav_path, mtime_ns) → InferenceResult. The studio runner
        # re-assesses the same wav against canonical vs intended-wrong
        # targets and benefits from the cache hit; mtime guards the demo's
        # `latest.wav`, which is rewritten on every upload under the same path.
        self._inference_cache: dict[tuple[str, str, int], InferenceResult] = {}

    def analyze(
        self, *, model_alias: str | None, wav_path: Path
    ) -> InferenceResult:
        try:
            entry = self._registry.get(model_alias)
        except UnknownModelError:
            return InferenceResult(
                model=model_alias or "?",
                status="model_unavailable",
                error_code="unknown_model",
            )
        if not entry.available:
            return InferenceResult(
                model=entry.alias,
                status="model_unavailable",
                error_code="model_unavailable",
            )

        try:
            mtime_ns = wav_path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        cache_key = (entry.alias, str(wav_path), mtime_ns)
        cached = self._inference_cache.get(cache_key)
        if cached is not None:
            return cached

        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32")
        result = self._infer_array(entry.alias, audio, int(sr))
        self._inference_cache[cache_key] = result
        return result

    def run(
        self,
        model_alias: str | None,
        audio: np.ndarray,
        sample_rate: int,
    ) -> InferenceResult | None:
        """Run wav2vec inference over in-memory audio.

        This is the runtime/API path used by the coach assessment package so the same
        wav2vec call can feed alignment and posterior evidence without writing
        a temporary WAV. ``None`` mirrors the legacy coach adapter contract for
        unavailable models.
        """
        try:
            entry = self._registry.get(model_alias)
        except UnknownModelError:
            return None
        if not entry.available:
            return None
        return self._infer_array(entry.alias, audio, sample_rate)

    def _infer_array(
        self,
        alias: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> InferenceResult:
        model, processor, device = self._load(alias)
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        total_ms = round(len(audio) / SAMPLE_RATE * 1000)
        inputs = processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
        )
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        vocab = processor.tokenizer.get_vocab()
        id_to_token = {int(v): k for k, v in vocab.items()}
        arr = logits[0].cpu().numpy()
        phones = decode_ctc(
            arr,
            id_to_token,
            drop_tokens=set(DEFAULT_DROP_TOKENS),
            total_ms=total_ms,
        )
        posteriors = frame_posteriors_from_logits(
            arr,
            id_to_token,
            drop_tokens=set(DEFAULT_DROP_TOKENS),
            total_ms=total_ms,
        )
        result = InferenceResult(
            model=alias,
            phones=phones,
            frame_posteriors=posteriors,
        )
        return result

    def _load(self, alias: str) -> tuple[Any, Any, str]:
        if alias in self._cache:
            return self._cache[alias]
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        entry = self._registry.get(alias)
        device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self._device == "auto"
            else self._device
        )
        processor = Wav2Vec2Processor.from_pretrained(str(entry.processor))
        model = Wav2Vec2ForCTC.from_pretrained(str(entry.checkpoint))
        model = model.to(device)  # type: ignore[arg-type]
        model.eval()
        self._cache[alias] = (model, processor, device)
        return self._cache[alias]
