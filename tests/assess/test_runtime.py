"""Pure runtime helper tests for assess.runtime."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


def _interval(target_phone: str = "a"):
    from assess.contracts import PhoneInterval

    return PhoneInterval(
        target_phone=target_phone,
        start_ms=0,
        end_ms=30,
        speech_start_ms_original=0,
        speech_end_ms_original=30,
    )


def test_evidence_for_span_uses_peak_frame_not_span_mean() -> None:
    from assess.runtime import evidence_for_span
    from assess.posterior import PosteriorFrame

    frames = [
        PosteriorFrame(
            frame_index=0,
            start_ms=0,
            end_ms=10,
            token_logprobs={"a": -5.0, "e": -0.1},
        ),
        PosteriorFrame(
            frame_index=1,
            start_ms=10,
            end_ms=20,
            token_logprobs={"a": -0.2, "e": -0.3},
        ),
        PosteriorFrame(
            frame_index=2,
            start_ms=20,
            end_ms=30,
            token_logprobs={"a": -5.0, "e": -0.1},
        ),
    ]

    evidence = evidence_for_span(
        _interval("a"),
        frames,
        normalized_vocab={"a", "e"},
        min_frames=1,
    )

    assert evidence.observed_phone == "a"
    assert evidence.observed_matches_target is True
    assert evidence.peak_frame_index == 1
    assert evidence.target_logprob == pytest.approx(-0.2)
    assert evidence.best_logprob == pytest.approx(-0.2)
    assert evidence.gop_raw == pytest.approx(0.0)
    assert evidence.identity_score is not None
    assert evidence.identity_score > 0.7


def test_evidence_for_spans_normalizes_length_colon_tokens() -> None:
    from assess.runtime import evidence_for_spans
    from assess.posterior import PosteriorFrame

    frames = [
        PosteriorFrame(
            frame_index=0,
            start_ms=0,
            end_ms=30,
            token_logprobs={"a:": -0.1, "a": -2.0},
        )
    ]

    evidence = evidence_for_spans(
        [_interval("aː")],
        frames,
        vocab={"a:", "a"},
        min_frames=1,
    )[0]

    assert evidence.observed_phone == "aː"
    assert evidence.observed_matches_target is True
    assert evidence.peak_frame_index == 0


def test_timing_from_wav2vec_posteriors_builds_phone_intervals() -> None:
    from assess.runtime import timing_from_wav2vec_posteriors
    from assess.posterior import PosteriorFrame
    from audio.timing import SpeechRegion

    frames = [
        PosteriorFrame(0, 0, 10, {"a": -0.1, "b": -4.0}),
        PosteriorFrame(1, 10, 20, {"a": -0.2, "b": -3.0}),
        PosteriorFrame(2, 20, 30, {"a": -3.0, "b": -0.2}),
        PosteriorFrame(3, 30, 40, {"a": -4.0, "b": -0.1}),
    ]

    timing = timing_from_wav2vec_posteriors(
        ["a", "b"],
        frames,
        SpeechRegion(0, 40),
    )

    assert timing.status == "ok"
    assert [span.target_phone for span in timing.intervals] == ["a", "b"]
    assert [(span.start_ms, span.end_ms) for span in timing.intervals] == [
        (0, 20),
        (20, 40),
    ]


def test_timing_from_wav2vec_posteriors_respects_speech_region() -> None:
    from assess.runtime import timing_from_wav2vec_posteriors
    from assess.posterior import PosteriorFrame
    from audio.timing import SpeechRegion

    frames = [
        PosteriorFrame(0, 0, 10, {"a": -0.1, "b": -4.0}),
        PosteriorFrame(1, 10, 20, {"a": -0.1, "b": -4.0}),
        PosteriorFrame(2, 20, 30, {"a": -4.0, "b": -0.1}),
        PosteriorFrame(3, 30, 40, {"a": -4.0, "b": -0.1}),
    ]

    timing = timing_from_wav2vec_posteriors(
        ["a", "b"],
        frames,
        SpeechRegion(10, 30),
    )

    assert timing.status == "ok"
    assert [(span.start_ms, span.end_ms) for span in timing.intervals] == [
        (10, 20),
        (20, 30),
    ]


def test_timing_from_wav2vec_posteriors_normalizes_length_tokens() -> None:
    from assess.runtime import timing_from_wav2vec_posteriors
    from assess.posterior import PosteriorFrame
    from audio.timing import SpeechRegion

    timing = timing_from_wav2vec_posteriors(
        ["aː"],
        [PosteriorFrame(0, 0, 20, {"a:": -0.1, "a": -3.0})],
        SpeechRegion(0, 20),
    )

    assert timing.status == "ok"
    assert timing.intervals[0].target_phone == "aː"
    assert (timing.intervals[0].start_ms, timing.intervals[0].end_ms) == (0, 20)


def test_timing_from_wav2vec_posteriors_returns_alignment_failed() -> None:
    from assess.runtime import timing_from_wav2vec_posteriors
    from assess.posterior import PosteriorFrame
    from audio.timing import SpeechRegion

    timing = timing_from_wav2vec_posteriors(
        ["a", "b"],
        [
            PosteriorFrame(0, 0, 10, {"a": -0.1}),
            PosteriorFrame(1, 10, 20, {"a": -0.1}),
        ],
        SpeechRegion(0, 20),
    )

    assert timing.status == "alignment_failed"
    assert timing.intervals == []


def test_default_runtime_services_use_wav2vec_for_alignment_once(monkeypatch) -> None:
    import assess.authority as authority_mod
    import assess.runtime as runtime_mod
    import assess.stats as stats_mod
    import assess.inference as inference_mod
    import assess.registry as registry_mod
    from assess.contracts import CoachRequest
    from assess.posterior import PosteriorFrame

    frame_posteriors = (
        PosteriorFrame(0, 0, 250, {"a": -0.1, "b": -4.0}),
        PosteriorFrame(1, 250, 500, {"a": -0.2, "b": -3.0}),
        PosteriorFrame(2, 500, 750, {"a": -3.0, "b": -0.2}),
        PosteriorFrame(3, 750, 1000, {"a": -4.0, "b": -0.1}),
    )

    class FakeWav2VecInferenceService:
        calls = 0

        def __init__(self, _registry: object) -> None:
            pass

        def run(
            self,
            _model_alias: str,
            _audio: np.ndarray,
            _sample_rate: int,
        ) -> object:
            type(self).calls += 1
            return SimpleNamespace(
                status="ok",
                phones=[],
                frame_posteriors=frame_posteriors,
            )

    monkeypatch.setattr(stats_mod.AtlasStats, "load", staticmethod(lambda _p: object()))
    monkeypatch.setattr(
        authority_mod.AuthorityTable,
        "load",
        staticmethod(lambda _p: object()),
    )
    monkeypatch.setattr(
        registry_mod.ModelRegistry,
        "from_config",
        classmethod(lambda _cls, *_args, **_kwargs: object()),
    )
    monkeypatch.setattr(
        inference_mod,
        "Wav2VecInferenceService",
        FakeWav2VecInferenceService,
    )
    monkeypatch.setattr(runtime_mod, "_is_gmm_artifact", lambda _p: False)

    request = CoachRequest(
        audio_path="x.wav",
        model_alias="mms-1b",
        target_phones=["a", "b"],
        stats_path="stats.json",
        authority_path="authority.json",
    )
    services = runtime_mod.build_default_runtime_services(request)
    audio = np.zeros(16_000, dtype=np.float32)
    audio[4_000:12_000] = 0.2

    timing = services.align(audio, 16_000, ["a", "b"])
    wav_run = services.run_wav2vec(audio, 16_000, timing.intervals, "mms-1b")

    assert timing.status == "ok"
    assert len(timing.intervals) == 2
    assert len(wav_run.spans) == 2
    assert FakeWav2VecInferenceService.calls == 1

    second_timing = services.align(audio.copy(), 16_000, ["a", "b"])

    assert second_timing.status == "ok"
    assert FakeWav2VecInferenceService.calls == 2


def test_default_runtime_services_uses_wav2vec_decode_when_vad_falls_back(
    monkeypatch,
) -> None:
    import assess.authority as authority_mod
    import assess.inference as inference_mod
    import assess.registry as registry_mod
    import assess.runtime as runtime_mod
    import assess.stats as stats_mod
    import audio.timing as timing_mod
    from assess.contracts import CoachRequest
    from assess.posterior import PosteriorFrame

    frame_posteriors = (
        PosteriorFrame(0, 0, 250, {"a": -0.1, "b": -4.0}),
        PosteriorFrame(1, 250, 500, {"a": -0.2, "b": -3.0}),
        PosteriorFrame(2, 500, 750, {"a": -3.0, "b": -0.2}),
        PosteriorFrame(3, 750, 1000, {"a": -4.0, "b": -0.1}),
    )

    class FakeWav2VecInferenceService:
        def __init__(self, _registry: object) -> None:
            pass

        def run(
            self,
            _model_alias: str,
            _audio: np.ndarray,
            _sample_rate: int,
        ) -> object:
            return SimpleNamespace(
                status="ok",
                phones=[SimpleNamespace(ipa="a"), SimpleNamespace(ipa="b")],
                frame_posteriors=frame_posteriors,
            )

    monkeypatch.setattr(stats_mod.AtlasStats, "load", staticmethod(lambda _p: object()))
    monkeypatch.setattr(
        authority_mod.AuthorityTable,
        "load",
        staticmethod(lambda _p: object()),
    )
    monkeypatch.setattr(
        registry_mod.ModelRegistry,
        "from_config",
        classmethod(lambda _cls, *_args, **_kwargs: object()),
    )
    monkeypatch.setattr(
        inference_mod,
        "Wav2VecInferenceService",
        FakeWav2VecInferenceService,
    )
    monkeypatch.setattr(runtime_mod, "_is_gmm_artifact", lambda _p: False)
    monkeypatch.setattr(
        timing_mod,
        "detect_speech_region",
        lambda _audio, _sample_rate: timing_mod.SpeechRegion(
            0, 1000, "no_speech_fallback"
        ),
    )

    request = CoachRequest(
        audio_path="x.wav",
        model_alias="mms-1b",
        target_phones=["a", "b"],
        stats_path="stats.json",
        authority_path="authority.json",
    )
    services = runtime_mod.build_default_runtime_services(request)

    timing = services.align(np.ones(16_000, dtype=np.float32), 16_000, ["a", "b"])

    assert timing.status == "ok"
    assert [span.target_phone for span in timing.intervals] == ["a", "b"]


def test_reconcile_speech_region_extends_over_truncated_quiet_syllable() -> None:
    from assess.runtime import _reconcile_speech_region
    from audio.timing import SpeechRegion

    # RMS gate dropped the quietly produced second syllable (ends at 170 ms)
    # while wav2vec free-decoded its content out at 264-325 ms.
    region = SpeechRegion(0, 170, "ok")
    free_decode = [
        SimpleNamespace(ipa="pʰ", start_ms=20, end_ms=41),
        SimpleNamespace(ipa="a", start_ms=81, end_ms=102),
        SimpleNamespace(ipa="pʰ", start_ms=264, end_ms=284),
        SimpleNamespace(ipa="a", start_ms=304, end_ms=325),
    ]
    out = _reconcile_speech_region(
        region, free_decode, sample_count=16_000 * 406 // 1000, sample_rate=16_000
    )
    assert out.status == "ok"
    assert out.start_ms_original == 0
    assert out.end_ms_original == 325


def test_reconcile_speech_region_leaves_consistent_region_untouched() -> None:
    from assess.runtime import _reconcile_speech_region
    from audio.timing import SpeechRegion

    region = SpeechRegion(0, 400, "ok")
    free_decode = [SimpleNamespace(ipa="a", start_ms=20, end_ms=380)]
    out = _reconcile_speech_region(
        region, free_decode, sample_count=16_000 * 406 // 1000, sample_rate=16_000
    )
    assert (out.start_ms_original, out.end_ms_original, out.status) == (0, 400, "ok")


def test_reconcile_speech_region_repairs_fallback_without_timestamps() -> None:
    from assess.runtime import _reconcile_speech_region
    from audio.timing import SpeechRegion

    region = SpeechRegion(0, 1000, "no_speech_fallback")
    free_decode = [SimpleNamespace(ipa="a"), SimpleNamespace(ipa="b")]
    out = _reconcile_speech_region(
        region, free_decode, sample_count=16_000, sample_rate=16_000
    )
    assert (out.start_ms_original, out.end_ms_original, out.status) == (0, 1000, "ok")


def test_load_acoustic_scorer_uses_gmm_artifact(tmp_path) -> None:
    from assess.features import FeatureSet
    from assess.runtime import _load_acoustic_scorer
    from assess.stats import AtlasGMM, GMMComponent, PhoneGMM

    atlas = AtlasGMM(
        feature_version="assess_coach_gmm_v1",
        phones={
            "a": PhoneGMM(
                phone="a",
                feature_names=("f1_hz", "f2_hz", "f3_hz"),
                components=(
                    GMMComponent(
                        mean=[800.0, 1200.0, 2400.0],
                        inv_cov=[
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        cov_log_det=0.0,
                        log_weight=0.0,
                    ),
                ),
                d_threshold=10.0,
                duration_median=80.0,
                duration_scale=10.0,
                n=200,
            )
        },
    )
    path = tmp_path / "gmm.json"
    atlas.save(path)
    feature_set = FeatureSet(
        duration_ms=80.0,
        rms=0.2,
        spectral_centroid_hz=2000.0,
        spectral_bandwidth_hz=1100.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.7,
        f0_mean_hz=180.0,
        f1_hz=800.0,
        f2_hz=1200.0,
        f3_hz=2400.0,
        formant_confidence=1.0,
        feature_confidence=1.0,
        low_confidence_reasons=[],
        active_duration_ms=80.0,
    )

    evidence = _load_acoustic_scorer(path)(feature_set, "a")

    assert evidence.as_dict()["quality_score"] == pytest.approx(1.0)
    assert evidence.as_dict()["component_count"] == 1


@dataclass(frozen=True)
class _AtlasEvidence:
    target_fit: float
    confidence: float = 1.0
    best_match: str | None = None
    best_match_fit: float | None = None
    margin: float | None = None
    atlas_verdict: str = "fit_ok"

    def as_dict(self) -> dict[str, object]:
        return {
            "target_fit": self.target_fit,
            "confidence": self.confidence,
            "best_match": self.best_match,
            "best_match_fit": self.best_match_fit,
            "margin": self.margin,
            "atlas_verdict": self.atlas_verdict,
        }


@dataclass(frozen=True)
class _WavRun:
    status: str
    spans: list[object]
    free_decode: list[object]
    frame_posteriors: tuple[object, ...] = ()


def _pairwise_row(
    alternative: str,
    *,
    passed: bool = True,
    margin: float = 0.4,
    fp_estimate: float = 0.01,
    recall_estimate: float = 0.5,
    target_quality: float | None = None,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "alternative": alternative,
        "score": 1.0 + margin,
        "threshold": 1.0,
        "margin": margin,
        "passed": passed,
        "features": ["f2_locus_hz"],
        "feature_values": {"f2_locus_hz": 2400.0},
        "fp_estimate": fp_estimate,
        "recall_estimate": recall_estimate,
        "target_quality": target_quality,
        "reason": "pairwise_detector_fp_within_ceiling",
        "reasons": [] if reasons is None else list(reasons),
    }


def test_run_coach_promotes_identity_and_quality_scores() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(160, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult(
            status="ok",
            speech_start_ms_original=0,
            speech_end_ms_original=30,
            intervals=[],
        ),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=30, confidence=0.91)
            ],
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.wav2vec is not None
    assert result.analysis is not None
    assert result.result is not None
    assert result.wav2vec["phonemes"] == ["a"]
    assert result.analysis["phonemes"] == ["a"]
    assert result.analysis["phones"][0]["quality"] == pytest.approx(0.82)
    assert result.result["phones"][0]["status"] == "correct"
    assert result.result["phones"][0]["quality"] == pytest.approx(0.82)
    assert result.result["phones"][0]["score"] == pytest.approx(1.0)
    assert result.result["summary"]["overall_score"] == pytest.approx(100.0)
    assert result.result["score"]["item_score"] == pytest.approx(100.0)


def test_run_coach_quality_ignores_length_channel() -> None:
    """A well-articulated but naturally long/short phone keeps its atlas quality.

    Regression: an earlier ``min(atlas, length_score)`` combination let the soft
    duration typicality gate the headline quality, collapsing a native vowel's
    quality to ~0.1 whenever its duration drifted from the corpus median. Length
    must stay an independent, non-gating channel.
    """
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    # /a/ reference sits tight around 120 ms; the 30 ms decode span lands many
    # robust scales short, so the length cue collapses toward zero.
    length_stats = DurationStats(
        phones={"a": PhoneDuration(phone="a", median_ms=120.0, scale_ms=10.0, n=500)}
    )
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(160, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 30, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=30, confidence=0.91)
            ],
        ),
        length_stats=length_stats,
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.result is not None
    analysis_phone = result.analysis["phones"][0]
    result_phone = result.result["phones"][0]
    # The length channel collapsed and is surfaced as its own evidence...
    assert analysis_phone["length_score"] < 0.1
    assert analysis_phone["length_note"] == "short"
    assert result_phone["length"]["score"] < 0.1
    assert result_phone["length"]["note"] == "short"
    # ...but quality stays pinned to the atlas typicality, not the length cue.
    assert analysis_phone["quality"] == pytest.approx(0.82)
    assert result_phone["quality"] == pytest.approx(0.82)
    assert result_phone["status"] == "correct"


def test_run_coach_exposes_raw_posterior_candidates() -> None:
    from assess.contracts import CoachRequest
    from assess.posterior import PosteriorFrame
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(160, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 30, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=30, confidence=0.91)
            ],
            frame_posteriors=(
                PosteriorFrame(0, 0, 15, {"a": -0.1, "e": -2.0}),
                PosteriorFrame(1, 15, 30, {"a": -0.2, "e": -1.7}),
            ),
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.wav2vec is not None
    assert result.debug is not None
    candidates = result.debug["posterior_candidates"][0]["candidates"]
    assert candidates[0]["phone"] == "a"
    assert candidates[0]["probability"] > candidates[1]["probability"]
    assert result.wav2vec["posterior_candidates"][0]["raw_ipa"] == "a"


def test_run_coach_uses_raw_duration_to_correct_short_vowel_long() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(
        active_duration_ms=340.0,
        duration_reliability=1.0,
    )

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(6800, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult(
            status="ok",
            speech_start_ms_original=0,
            speech_end_ms_original=340,
            intervals=[],
        ),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=340, confidence=0.9)
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["aː"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.result is not None
    assert result.analysis["phonemes"] == ["aː"]
    assert result.analysis["changes"][0]["from"] == "a"
    assert result.analysis["changes"][0]["to"] == "aː"
    assert result.result["phones"][0]["status"] == "correct"
    assert result.result["phones"][0]["score"] == pytest.approx(1.0)
    assert result.result["phones"][0]["score_reason"] == "analysis_corrected"
    assert result.result["phones"][0]["quality"] is not None


def test_run_coach_uses_neighbor_quantity_span_for_long_vowel() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=20.0, duration_reliability=0.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(4000, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 250, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=20, confidence=0.9),
                SimpleNamespace(ipa="a", start_ms=60, end_ms=80, confidence=0.9),
                SimpleNamespace(ipa="ɾ̞̊", start_ms=203, end_ms=223, confidence=0.9),
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["aː"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.analysis["phonemes"] == ["kʰ", "aː", "ɾ̞̊"]
    assert result.analysis["changes"][0]["from"] == "a"
    assert result.analysis["changes"][0]["to"] == "aː"
    vowel = result.analysis["phones"][1]
    assert vowel["duration_ms"] == pytest.approx(183.0)
    assert vowel["quantity_span"]["source"] == "ctc_neighbor_span"


def test_neighbor_quantity_span_does_not_lengthen_borderline_short_vowel() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=20.0, duration_reliability=0.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(3200, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 180, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=20, confidence=0.9),
                SimpleNamespace(ipa="a", start_ms=60, end_ms=80, confidence=0.9),
                SimpleNamespace(ipa="ɾ̞̊", start_ms=141, end_ms=161, confidence=0.9),
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.analysis["phonemes"] == ["kʰ", "a", "ɾ̞̊"]
    assert result.analysis["changes"] == []
    assert result.analysis["phones"][1]["duration_ms"] == pytest.approx(121.0)


def test_neighbor_quantity_span_applies_to_other_long_vowels() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=20.0, duration_reliability=0.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(4000, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 250, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="tʰ", start_ms=0, end_ms=20, confidence=0.9),
                SimpleNamespace(ipa="e", start_ms=60, end_ms=80, confidence=0.9),
                SimpleNamespace(ipa="n", start_ms=220, end_ms=240, confidence=0.9),
            ],
        ),
        length_stats=DurationStats(
            phones={
                "e": PhoneDuration("e", median_ms=70.0, scale_ms=10.0, n=1000),
                "eː": PhoneDuration("eː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["eː"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.analysis["phonemes"] == ["tʰ", "eː", "n"]
    assert result.analysis["changes"][0]["from"] == "e"
    assert result.analysis["changes"][0]["to"] == "eː"


def test_duration_change_can_make_target_short_vowel_incorrect() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=340.0, duration_reliability=1.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(6800, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 340, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=340, confidence=0.9)
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.result is not None
    assert result.analysis["phonemes"] == ["aː"]
    row = result.result["phones"][0]
    assert row["status"] == "incorrect"
    assert row["quality"] is None
    assert row["changed_by_analysis"] == {
        "from": "a",
        "to": "aː",
        "source": "duration",
        "change_id": "analysis-0",
    }


def test_duration_change_can_shorten_raw_long_vowel() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=70.0, duration_reliability=1.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 70, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="aː", start_ms=0, end_ms=70, confidence=0.9)
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.result is not None
    assert result.analysis["phonemes"] == ["a"]
    assert result.analysis["changes"][0]["from"] == "aː"
    assert result.analysis["changes"][0]["to"] == "a"
    assert result.result["phones"][0]["status"] == "correct"


def test_short_raw_vowel_does_not_change_for_long_target() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=70.0, duration_reliability=1.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 70, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=70, confidence=0.9)
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["aː"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.result is not None
    assert result.analysis["phonemes"] == ["a"]
    assert result.analysis["changes"] == []
    row = result.result["phones"][0]
    assert row["status"] == "incorrect"
    assert 0.0 < row["score"] < 1.0
    assert row["quality"] is None
    assert row["reason"] == "long_vowel_heard_short"


def test_allophonic_stop_mismatch_is_not_masked() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["cʰ"]),
        services=services,
    )

    assert result.result is not None
    assert result.result["phones"][0]["status"] == "incorrect"
    assert result.result["phones"][0]["expected"] == "cʰ"
    assert result.result["phones"][0]["observed"] == "kʰ"
    assert result.result["phones"][0]["score"] > 0.70
    assert result.result["phones"][0]["strict_status"] == "incorrect"


def test_pairwise_contrast_corrects_raw_phone_target_independently() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    seen_current_phones: list[str] = []

    def assess_contrasts(
        _feature: object,
        current_phone: str,
        _atlas: object,
    ) -> list[dict[str, object]]:
        seen_current_phones.append(current_phone)
        return [_pairwise_row("cʰ")] if current_phone == "kʰ" else []

    def services() -> CoachRuntimeServices:
        return CoachRuntimeServices(
            load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
            align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
            extract_feature=lambda *_args, **_kwargs: object(),
            assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
            run_wav2vec=lambda *_args, **_kwargs: _WavRun(
                status="ok",
                spans=[],
                free_decode=[
                    SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=80, confidence=0.9)
                ],
            ),
            assess_contrasts=assess_contrasts,
        )

    correct_result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["cʰ"]),
        services=services(),
    )
    incorrect_result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["kʰ"]),
        services=services(),
    )

    assert seen_current_phones == ["kʰ", "kʰ"]
    assert correct_result.analysis is not None
    assert correct_result.result is not None
    assert correct_result.analysis["phonemes"] == ["cʰ"]
    assert correct_result.analysis["changes"][0]["source"] == "pairwise_contrast"
    assert correct_result.result["phones"][0]["status"] == "correct"

    assert incorrect_result.analysis is not None
    assert incorrect_result.result is not None
    assert incorrect_result.analysis["phonemes"] == ["cʰ"]
    assert incorrect_result.result["phones"][0]["expected"] == "kʰ"
    assert incorrect_result.result["phones"][0]["observed"] == "cʰ"
    assert incorrect_result.result["phones"][0]["status"] == "incorrect"


def test_rejected_pairwise_contrast_keeps_raw_phone_in_debug() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
        assess_contrasts=lambda *_args, **_kwargs: [
            _pairwise_row(
                "cʰ",
                passed=False,
                margin=-0.2,
                reasons=["pairwise_margin_below_threshold"],
            )
        ],
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["kʰ"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["kʰ"]
    assert result.analysis["changes"] == []
    candidate = result.debug["analysis_candidates"][0]
    assert candidate["source"] == "pairwise_contrast"
    assert candidate["passed"] is False
    assert candidate["applied"] is False
    assert candidate["evidence"]["margin"] == -0.2
    assert "pairwise_margin_below_threshold" in candidate["reasons"]


def test_pairwise_contrast_requires_runtime_quality_lift() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    def atlas_for(_feature: object, phone: str) -> _AtlasEvidence:
        return _AtlasEvidence(target_fit=0.25 if phone == "cʰ" else 0.20)

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=atlas_for,
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
        assess_contrasts=lambda *_args, **_kwargs: [
            _pairwise_row("cʰ", margin=0.4, recall_estimate=0.5, target_quality=0.20)
        ],
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["kʰ"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["kʰ"]
    candidate = result.debug["analysis_candidates"][0]
    assert candidate["passed"] is False
    assert candidate["evidence"]["alternative_quality"] == pytest.approx(0.25)
    assert candidate["evidence"]["quality_lift"] == pytest.approx(0.05)
    assert "pairwise_quality_lift_below_runtime_floor" in candidate["reasons"]


def test_pairwise_contrast_requires_stronger_unscored_evidence() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=None),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="k", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
        assess_contrasts=lambda *_args, **_kwargs: [
            _pairwise_row("d", margin=0.30, recall_estimate=0.19)
        ],
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["k"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["k"]
    candidate = result.debug["analysis_candidates"][0]
    assert candidate["passed"] is False
    assert "pairwise_unscored_evidence_below_runtime_floor" in candidate["reasons"]


def test_multiple_pairwise_contrasts_choose_best_ranked_candidate() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    rows = [
        _pairwise_row("cʰ", margin=0.4, fp_estimate=0.01, recall_estimate=0.9),
        _pairwise_row("d", margin=0.5, fp_estimate=0.02, recall_estimate=0.9),
        _pairwise_row("pʰ", margin=0.5, fp_estimate=0.01, recall_estimate=0.2),
        _pairwise_row("tʰ", margin=0.5, fp_estimate=0.01, recall_estimate=0.4),
    ]
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(target_fit=0.82),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
        assess_contrasts=lambda *_args, **_kwargs: [dict(row) for row in rows],
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["tʰ"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["tʰ"]
    assert result.analysis["changes"][0]["to"] == "tʰ"
    assert result.result["phones"][0]["status"] == "correct"
    applied = [
        candidate
        for candidate in result.debug["analysis_candidates"]
        if candidate["source"] == "pairwise_contrast" and candidate["applied"] is True
    ]
    assert [candidate["to"] for candidate in applied] == ["tʰ"]


def test_duration_candidate_precedes_pairwise_and_nearest() -> None:
    from assess.contracts import CoachRequest
    from assess.length import DurationStats, PhoneDuration
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    feature = SimpleNamespace(active_duration_ms=340.0, duration_reliability=1.0)
    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(6800, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 340, []),
        extract_feature=lambda *_args, **_kwargs: feature,
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(
            target_fit=0.01,
            best_match="o",
            best_match_fit=0.9,
            margin=-0.5,
            atlas_verdict="likely_substitution",
        ),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=340, confidence=0.9)
            ],
        ),
        length_stats=DurationStats(
            phones={
                "a": PhoneDuration("a", median_ms=70.0, scale_ms=10.0, n=1000),
                "aː": PhoneDuration("aː", median_ms=160.0, scale_ms=10.0, n=500),
            }
        ),
        assess_contrasts=lambda *_args, **_kwargs: [_pairwise_row("e")],
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["aː"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["aː"]
    assert result.analysis["changes"][0]["source"] == "duration"
    assert any(
        candidate["source"] == "pairwise_contrast" and candidate["applied"] is False
        for candidate in result.debug["analysis_candidates"]
    )
    assert any(
        candidate["source"] == "gmm_nearest" and candidate["applied"] is False
        for candidate in result.debug["analysis_candidates"]
    )


def test_pairwise_contrast_precedes_nearest_fallback() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(
            target_fit=0.01,
            best_match="d",
            best_match_fit=0.9,
            margin=-0.5,
            atlas_verdict="likely_substitution",
        ),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="kʰ", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
        assess_contrasts=lambda *_args, **_kwargs: [_pairwise_row("cʰ")],
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["cʰ"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["cʰ"]
    assert result.analysis["changes"][0]["source"] == "pairwise_contrast"
    assert any(
        candidate["source"] == "gmm_nearest"
        and candidate["passed"] is True
        and candidate["applied"] is False
        for candidate in result.debug["analysis_candidates"]
    )


def test_result_alignment_orders_missing_and_extra() -> None:
    from assess.runtime import _result_payload

    result = _result_payload(
        target_phones=["a", "b", "c"],
        analysis_phones=[
            {"ipa": "a", "quality": 0.9},
            {"ipa": "c", "quality": 0.8},
            {"ipa": "y", "quality": 0.1},
        ],
    )

    assert [row["status"] for row in result["phones"]] == [
        "correct",
        "missing",
        "correct",
        "extra",
    ]
    assert result["phones"][1]["expected"] == "b"
    assert result["phones"][3]["observed"] == "y"
    assert result["summary"]["extra_count"] == 1
    assert result["phones"][1]["score"] == pytest.approx(0.0)
    assert result["phones"][3]["score_reason"] == "extra_penalty"
    assert result["summary"]["mean_phone_score"] == pytest.approx(2 / 3)
    assert result["summary"]["extra_penalty"] == pytest.approx(0.75)
    assert result["summary"]["overall_score"] == pytest.approx(50.0)


def test_result_extra_phone_penalty_reduces_word_score() -> None:
    from assess.runtime import _result_payload

    result = _result_payload(
        target_phones=["a"],
        analysis_phones=[
            {"ipa": "a", "raw_ipa": "a", "quality": 0.9},
            {"ipa": "b", "raw_ipa": "b", "quality": 0.9},
        ],
    )

    assert [row["strict_status"] for row in result["phones"]] == ["correct", "extra"]
    assert result["phones"][0]["score"] == pytest.approx(1.0)
    assert result["summary"]["mean_phone_score"] == pytest.approx(1.0)
    assert result["summary"]["extra_penalty"] == pytest.approx(0.5)
    assert result["summary"]["overall_score"] == pytest.approx(50.0)
    assert result["score"]["item_score"] == pytest.approx(50.0)


def test_rejected_nearest_candidate_keeps_raw_phone_in_debug() -> None:
    from assess.contracts import CoachRequest
    from assess.runtime import CoachRuntimeServices, TimingResult, run_coach

    services = CoachRuntimeServices(
        load_audio=lambda _path: (np.zeros(1600, dtype=np.float32), 16_000),
        align=lambda _audio, _sr, _phones: TimingResult("ok", 0, 80, []),
        extract_feature=lambda *_args, **_kwargs: object(),
        assess_atlas=lambda _feature, _phone: _AtlasEvidence(
            target_fit=0.7,
            best_match="e",
            best_match_fit=0.8,
            margin=-0.1,
            atlas_verdict="atypical",
        ),
        run_wav2vec=lambda *_args, **_kwargs: _WavRun(
            status="ok",
            spans=[],
            free_decode=[
                SimpleNamespace(ipa="a", start_ms=0, end_ms=80, confidence=0.9)
            ],
        ),
    )

    result = run_coach(
        CoachRequest(audio_path="x.wav", model_alias="mms-1b", target_phones=["a"]),
        services=services,
    )

    assert result.analysis is not None
    assert result.debug is not None
    assert result.analysis["phonemes"] == ["a"]
    assert result.analysis["changes"] == []
    candidate = result.debug["analysis_candidates"][0]
    assert candidate["source"] == "gmm_nearest"
    assert candidate["passed"] is False
    assert candidate["applied"] is False
    assert "target_quality_above_floor" in candidate["reasons"]
