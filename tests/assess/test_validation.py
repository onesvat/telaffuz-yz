"""Smoke tests for assess.validation — pure logic, no I/O."""

from __future__ import annotations

from types import SimpleNamespace


def test_speaker_bucket_is_deterministic() -> None:
    from assess.validation import speaker_bucket

    b1 = speaker_bucket("speaker_001")
    b2 = speaker_bucket("speaker_001")
    assert b1 == b2
    assert 0 <= b1 <= 99


def test_split_for_speaker_dev_or_test() -> None:
    from assess.validation import split_for_speaker

    splits = {split_for_speaker(f"speaker_{i:04d}") for i in range(200)}
    assert splits == {"dev", "test"}


def test_run_validation_batch_writes_observations_with_stub_services(tmp_path) -> None:
    """The batch runner integrates run_coach and writes per-phone rows."""
    import csv as _csv
    from dataclasses import dataclass

    from assess.contracts import PhoneInterval, Wav2VecEvidence
    from assess.runtime import CoachRuntimeServices, TimingResult
    from assess.validation import load_jsonl, run_validation_batch
    from assess.validation import speaker_bucket

    @dataclass(frozen=True)
    class _Atlas:
        confidence: float = 1.0
        atlas_verdict: str = "fit_ok"
        duration_verdict: str = "expected"
        best_match: str | None = None

        def as_dict(self) -> dict[str, object]:
            return {"confidence": self.confidence, "atlas_verdict": self.atlas_verdict}

    @dataclass(frozen=True)
    class _WavRun:
        status: str
        spans: list[object]
        free_decode: list[object]

    def _interval(phone: str) -> PhoneInterval:
        return PhoneInterval(phone, 0, 30, 0, 30)

    def _wav(phone: str) -> Wav2VecEvidence:
        return Wav2VecEvidence(
            target_phone=phone, observed_phone=phone, target_logprob=-0.2,
            best_logprob=-0.2, gop_raw=0.0, observed_matches_target=True,
            frame_count=1, confidence=1.0, reasons=[], peak_frame_index=0,
            identity_score=0.9,
        )

    services = CoachRuntimeServices(
        load_audio=lambda _p: (__import__("numpy").zeros(160, dtype="float32"), 16_000),
        align=lambda _a, _sr, phones: TimingResult(
            status="ok", speech_start_ms_original=0, speech_end_ms_original=30,
            intervals=[_interval(p) for p in phones],
        ),
        extract_feature=lambda *_a, **_k: object(),
        assess_atlas=lambda _f, _p: _Atlas(),
        run_wav2vec=lambda _a, _sr, intervals, _m: _WavRun(
            status="ok",
            spans=[_wav(i.target_phone) for i in intervals],
            free_decode=[
                SimpleNamespace(ipa=phone, start_ms=i * 30, end_ms=i * 30 + 30, confidence=0.9)
                for i, phone in enumerate(["a", "m", "a"])
            ],
        ),
    )

    # Pick a speaker that lands in the test split.
    spk = next(f"spk{i}" for i in range(1000) if speaker_bucket(f"spk{i}") >= 90)
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["speaker_id", "path", "text"])
        w.writeheader()
        w.writerow({"speaker_id": spk, "path": "x.wav", "text": "ama"})

    out = tmp_path / "obs.jsonl"
    counts = run_validation_batch(
        manifest=manifest, split="test", model_alias="mms-1b",
        stats_path="unused", authority_path="unused", output=out,
        resume=False, services=services,
    )
    assert counts["processed"] == 1
    assert counts["errors"] == 0
    rows = load_jsonl(out)
    assert rows and all(r["status"] == "correct" for r in rows)
    assert rows and all(r["is_flagged"] == 0 for r in rows)


def test_hard_false_positive_report_empty() -> None:
    from assess.validation import hard_false_positive_report

    report = hard_false_positive_report([])
    assert report["overall"]["total"] == 0
    assert report["overall"]["hard_fp"] == 0
    assert report["overall"]["hard_fp_rate"] is None


def test_hard_false_positive_report_counts_flagged_status() -> None:
    from assess.validation import hard_false_positive_report

    rows = [
        {"target_phone": "a", "status": "correct"},
        {"target_phone": "a", "status": "incorrect"},
        {"target_phone": "e", "status": "missing"},
        {"target_phone": "e", "status": "extra"},
        {"target_phone": "e", "status": "unscored"},
    ]
    report = hard_false_positive_report(rows)
    assert report["overall"]["total"] == 5
    assert report["overall"]["hard_fp"] == 2
    assert report["by_phone"]["a"]["hard_fp"] == 1
    assert report["by_phone"]["e"]["hard_fp"] == 1


def test_unscored_report() -> None:
    from assess.validation import unscored_report

    rows = [
        {"status": "correct", "reasons": []},
        {"status": "unscored", "reasons": ["span_too_short"]},
        {"status": "unscored", "reasons": ["no_frames"]},
    ]
    report = unscored_report(rows)
    assert report["overall"]["unscored"] == 2
    assert report["reasons"]["span_too_short"] == 1


def test_calibration_authority_table_written_with_new_namespace(tmp_path) -> None:
    from assess.validation import (
        CalibrationConfig,
        CalibrationObservation,
        calibrate_authority,
        write_authority_json,
    )
    import json

    observations = [
        CalibrationObservation(
            speaker_id=f"spk_{i}",
            target="a",
            alternative="e",
            target_fit=0.8 + i * 0.01,
            alternative_fit=0.3 + i * 0.005,
            wav2vec_observed=None,
        )
        for i in range(30)
    ]
    config = CalibrationConfig(min_samples=10)
    result = calibrate_authority(observations, config)
    assert result.generated_by == "assess.validation.calibration"

    path = tmp_path / "authority.json"
    write_authority_json(result, path)
    data = json.loads(path.read_text())
    assert "generated_by" in data
    assert data["generated_by"] == "assess.validation.calibration"


def test_calibration_observations_from_validation_rows() -> None:
    from assess.validation import calibration_observations_from_validation_rows

    rows = [
        {
            "speaker_id": "spk1",
            "atlas": {
                "target_phone": "u",
                "best_match": "i",
                "target_typicality": 0.42,
                "best_match_typicality": 0.71,
            },
            "wav2vec": {"observed_phone": "u"},
        },
        {
            "speaker_id": "spk2",
            "atlas": {
                "target_phone": "a",
                "best_match": None,
                "target_typicality": 0.9,
                "best_match_typicality": None,
            },
            "wav2vec": {"observed_phone": "a"},
        },
        {
            "speaker_id": "spk3",
            "atlas": {
                "target_phone": "o",
                "best_match": "u",
                "target_typicality": 0.5,
                "best_match_typicality": 0.6,
            },
            "wav2vec": {"observed_phone": None},
        },
        {
            "speaker_id": "spk4",
            "atlas": {
                "target_phone": "e",
                "best_match": "e",
                "target_typicality": 0.8,
                "best_match_typicality": 0.8,
            },
            "wav2vec": {"observed_phone": "e"},
        },
    ]

    observations = calibration_observations_from_validation_rows(rows)
    # spk2 dropped (best_match None); spk4 dropped (target == alternative)
    assert len(observations) == 2
    by_target = {o.target: o for o in observations}
    assert by_target["u"].alternative == "i"
    assert by_target["u"].target_fit == 0.42
    assert by_target["u"].alternative_fit == 0.71
    assert by_target["u"].wav2vec_observed == "u"
    assert by_target["o"].wav2vec_observed is None


def test_calibration_observations_skip_malformed_rows() -> None:
    from assess.validation import calibration_observations_from_validation_rows

    rows = [
        {"speaker_id": "spk1"},  # no atlas
        {"speaker_id": "spk2", "atlas": "not-a-dict"},
        {
            "speaker_id": "spk3",
            "atlas": {"target_phone": "a", "best_match": "e",
                      "target_typicality": None, "best_match_typicality": 0.3},
        },
    ]
    assert calibration_observations_from_validation_rows(rows) == []


def test_fixed_same_class_contrasts_symmetric_and_includes_observed() -> None:
    from assess.validation import FIXED_SAME_CLASS_CONTRASTS

    # Plan'da gözlenen 7 çakışma sabit listede olmalı.
    observed = [("u", "i"), ("a", "o"), ("a", "u"), ("o", "u"), ("cʰ", "ɟ")]
    for pair in observed:
        assert pair in FIXED_SAME_CLASS_CONTRASTS, f"missing {pair}"

    # Liste her iki yönde olmalı (kalibrasyon yön-duyarlı).
    for target, alternative in FIXED_SAME_CLASS_CONTRASTS:
        assert (alternative, target) in FIXED_SAME_CLASS_CONTRASTS


def test_calibrate_authority_scope_fixed_filters_observations() -> None:
    from assess.validation import (
        CalibrationConfig,
        CalibrationObservation,
        calibrate_authority,
    )

    def _obs(target: str, alt: str, idx: int) -> CalibrationObservation:
        return CalibrationObservation(
            speaker_id=f"s{idx}", target=target, alternative=alt,
            target_fit=0.6, alternative_fit=0.5, wav2vec_observed=target,
        )

    in_scope = [_obs("u", "i", i) for i in range(30)]
    out_of_scope = [_obs("a", "j", i) for i in range(30)]
    observations = in_scope + out_of_scope

    fixed = calibrate_authority(observations, CalibrationConfig(scope="fixed", min_samples=10))
    fixed_pairs = {(c.target, c.alternative) for c in fixed.contrasts}
    assert ("u", "i") in fixed_pairs
    assert ("a", "j") not in fixed_pairs

    all_ = calibrate_authority(observations, CalibrationConfig(scope="all", min_samples=10))
    all_pairs = {(c.target, c.alternative) for c in all_.contrasts}
    assert ("u", "i") in all_pairs
    assert ("a", "j") in all_pairs


def test_load_jsonl_roundtrip(tmp_path) -> None:
    import json
    from assess.validation import load_jsonl

    rows = [{"a": 1}, {"b": 2}]
    p = tmp_path / "obs.jsonl"
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    loaded = load_jsonl(p)
    assert loaded == rows
