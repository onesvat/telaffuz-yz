"""Smoke tests for assess.stats — pure math, no GPU."""

from __future__ import annotations

import numpy as np
import pytest


def test_robust_center_scatter_returns_correct_shapes() -> None:
    from assess.stats import robust_center_scatter

    rng = np.random.default_rng(0)
    samples = rng.normal(loc=[2.0, 3.0], scale=1.0, size=(50, 2))
    center, cov, inv_cov = robust_center_scatter(samples)
    assert center.shape == (2,)
    assert cov.shape == (2, 2)
    assert inv_cov.shape == (2, 2)


def test_fit_score_at_center_is_one() -> None:
    from assess.stats import fit_score, robust_center_scatter

    rng = np.random.default_rng(0)
    samples = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(50, 2))
    center, _cov, inv_cov = robust_center_scatter(samples)
    score = fit_score(center, center, inv_cov, d_threshold=1.0)
    assert score == pytest.approx(1.0)


def test_fit_score_decreases_with_distance() -> None:
    from assess.stats import fit_score, robust_center_scatter

    rng = np.random.default_rng(0)
    samples = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(100, 2))
    center, _cov, inv_cov = robust_center_scatter(samples)
    near = np.array([0.1, 0.1])
    far = np.array([5.0, 5.0])
    score_near = fit_score(near, center, inv_cov, d_threshold=2.0)
    score_far = fit_score(far, center, inv_cov, d_threshold=2.0)
    assert score_near > score_far


def test_atlas_stats_save_load_roundtrip(tmp_path) -> None:
    import json
    from assess.stats import AtlasStats, PhoneStat

    stat = PhoneStat(
        phone="a",
        feature_names=("f1_hz", "f2_hz"),
        center=[800.0, 1200.0],
        inv_cov=[[0.01, 0.0], [0.0, 0.01]],
        d_threshold=3.0,
        duration_median=80.0,
        duration_scale=20.0,
        n=100,
    )
    stats = AtlasStats(feature_version="assess_coach_test_v1", phones={"a": stat})
    path = tmp_path / "stats.json"
    stats.save(path)
    loaded = AtlasStats.load(path)
    assert loaded.feature_version == "assess_coach_test_v1"
    assert loaded.has("a")
    loaded_stat = loaded.get("a")
    assert loaded_stat is not None
    assert loaded_stat.center == pytest.approx([800.0, 1200.0])


def test_assess_atlas_low_confidence_for_empty_feature_set() -> None:
    from assess.features import FeatureSet
    from assess.stats import AtlasStats, assess_atlas

    feature_set = FeatureSet(
        duration_ms=10.0,  # too short → low confidence
        rms=0.0,
        spectral_centroid_hz=0.0,
        spectral_bandwidth_hz=0.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.0,
        f0_mean_hz=None,
        f1_hz=None,
        f2_hz=None,
        f3_hz=None,
        formant_confidence=0.0,
        feature_confidence=0.0,
        low_confidence_reasons=["span_too_short"],
    )
    stats = AtlasStats(feature_version="v1", phones={})
    evidence = assess_atlas(feature_set, "a", stats)
    assert evidence.confidence == 0.0
    assert evidence.atlas_verdict == "low_confidence"


def test_phone_stat_roundtrip_dict() -> None:
    from assess.stats import PhoneStat

    stat = PhoneStat(
        phone="e",
        feature_names=("f1_hz", "f2_hz", "f3_hz"),
        center=[500.0, 1800.0, 2600.0],
        inv_cov=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        d_threshold=2.5,
        duration_median=70.0,
        duration_scale=15.0,
        n=200,
    )
    d = stat.as_dict()
    loaded = PhoneStat.from_dict(d)
    assert loaded.phone == "e"
    assert loaded.n == 200
    assert loaded.center == pytest.approx([500.0, 1800.0, 2600.0])


# ---------------------------------------------------------------------------
# Faz 4: per-phone GMM scoring
# ---------------------------------------------------------------------------

def _phone_gmm(
    *,
    phone: str,
    feature_names: tuple[str, ...],
    mean: list[float],
    inv_cov_diag: float = 1.0,
    duration_median: float = 70.0,
    duration_scale: float = 10.0,
    d_threshold: float = 1.0,
    n: int = 200,
):
    from assess.stats import GMMComponent, PhoneGMM

    eye = [
        [inv_cov_diag if i == j else 0.0 for j in range(len(mean))]
        for i in range(len(mean))
    ]
    return PhoneGMM(
        phone=phone,
        feature_names=feature_names,
        components=(
            GMMComponent(
                mean=list(mean),
                inv_cov=eye,
                cov_log_det=0.0,
                log_weight=0.0,
            ),
        ),
        d_threshold=d_threshold,
        duration_median=duration_median,
        duration_scale=duration_scale,
        n=n,
    )


def test_atlas_gmm_save_load_roundtrip(tmp_path) -> None:
    from assess.stats import AtlasGMM

    gmm = _phone_gmm(
        phone="a",
        feature_names=("f1_hz", "f2_hz", "f3_hz"),
        mean=[800.0, 1200.0, 2400.0],
    )
    atlas = AtlasGMM(feature_version="assess_coach_gmm_v1", phones={"a": gmm})
    path = tmp_path / "gmm.json"
    atlas.save(path)
    loaded = AtlasGMM.load(path)
    assert loaded.feature_version == "assess_coach_gmm_v1"
    assert loaded.has("a")
    a = loaded.get("a")
    assert a is not None
    assert a.components[0].mean == pytest.approx([800.0, 1200.0, 2400.0])
    assert a.k == 1


def test_gmm_typicality_at_centre_is_one() -> None:
    from assess.stats import gmm_typicality

    gmm = _phone_gmm(
        phone="a", feature_names=("f1_hz", "f2_hz"), mean=[800.0, 1200.0]
    )
    score = gmm_typicality(np.array([800.0, 1200.0]), gmm)
    assert score == pytest.approx(1.0)


def test_assess_atlas_gmm_returns_evidence_for_vowel() -> None:
    from assess.features import FeatureSet
    from assess.stats import AtlasGMM, assess_atlas_gmm

    fs = FeatureSet(
        duration_ms=80.0,
        rms=0.2,
        spectral_centroid_hz=2000.0,
        spectral_bandwidth_hz=1100.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.7,
        f0_mean_hz=180.0,
        f1_hz=820.0,
        f2_hz=1180.0,
        f3_hz=2400.0,
        formant_confidence=1.0,
        feature_confidence=1.0,
        low_confidence_reasons=[],
        active_duration_ms=80.0,
    )
    target = _phone_gmm(
        phone="a", feature_names=("f1_hz", "f2_hz", "f3_hz"), mean=[800.0, 1200.0, 2400.0]
    )
    alt = _phone_gmm(
        phone="e", feature_names=("f1_hz", "f2_hz", "f3_hz"), mean=[500.0, 1800.0, 2600.0]
    )
    atlas = AtlasGMM(
        feature_version="assess_coach_gmm_v1",
        phones={"a": target, "e": alt},
    )

    ev = assess_atlas_gmm(fs, "a", atlas)
    assert ev.target_typicality is not None
    assert 0.0 < ev.target_typicality <= 1.0
    assert ev.best_match == "e"
    assert ev.margin is not None
    assert ev.confidence == 1.0
    assert ev.component_count == 1
    assert ev.as_dict()["quality_score"] == pytest.approx(ev.target_typicality)
    # Two-score wiring: a verdict in the shared vocabulary is always produced.
    assert ev.atlas_verdict in {"fit_ok", "likely_substitution", "atypical"}
    assert ev.as_dict()["atlas_verdict"] == ev.atlas_verdict


def test_degenerate_duration_is_not_scored() -> None:
    from assess.features import FeatureSet
    from assess.stats import _duration_from_gmm, _duration_scale

    # A reference floored at one value (e.g. ~20 ms span) collapses IQR.
    median, scale = _duration_scale([20.0] * 50)
    assert scale == 0.0
    gmm = _phone_gmm(
        phone="a", feature_names=("f1_hz",), mean=[800.0],
        duration_median=median, duration_scale=scale,
    )
    fs = FeatureSet(
        duration_ms=300.0,  # real runtime duration, far from the 20 ms floor
        rms=0.2, spectral_centroid_hz=2000.0, spectral_bandwidth_hz=1100.0,
        mfcc=[0.0] * 13, voiced_fraction=0.7, f0_mean_hz=180.0,
        f1_hz=820.0, f2_hz=1180.0, f3_hz=2400.0,
        formant_confidence=1.0, feature_confidence=1.0, low_confidence_reasons=[],
        active_duration_ms=300.0,
    )
    z, verdict = _duration_from_gmm(fs, gmm, tau=1.5)
    assert z is None
    assert verdict == "expected"


def test_gmm_atlas_verdict_mapping() -> None:
    from assess.stats import _INSIDE_FIT_FLOOR, _gmm_atlas_verdict

    assert _gmm_atlas_verdict(_INSIDE_FIT_FLOOR + 0.1, None, delta=0.15) == "fit_ok"
    assert _gmm_atlas_verdict(0.2, -0.5, delta=0.15) == "likely_substitution"
    assert _gmm_atlas_verdict(0.2, -0.05, delta=0.15) == "atypical"
    assert _gmm_atlas_verdict(0.2, None, delta=0.15) == "atypical"


def _stop_reference_db(path, *, n: int, vot_present: int) -> None:
    """Tiny coach reference DB for phone /b/ with a sparse VOT column."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE phone_features (
            id INTEGER PRIMARY KEY,
            expected_phone TEXT,
            provider TEXT,
            speaker_id TEXT,
            formant_success INTEGER,
            feature_confidence REAL,
            duration_ms REAL,
            vot_ms REAL,
            burst_centroid_hz REAL,
            burst_spectral_skew REAL,
            f2_transition_slope_hz_per_ms REAL,
            f2_locus_hz REAL,
            voiced_fraction REAL,
            closure_voicing_ratio REAL
        )
        """
    )
    rows = []
    for i in range(n):
        vot = 18.0 + (i % 5) if i < vot_present else None
        rows.append(
            (
                "b", "common_voice", f"spk{i:04d}", 0, 1.0, 95.0 + (i % 7),
                vot, 1500.0 + i, 0.3, -2.0 + (i % 3), 1400.0 + i, 0.9, 0.85,
            )
        )
    conn.executemany(
        """
        INSERT INTO phone_features (
            expected_phone, provider, speaker_id, formant_success,
            feature_confidence, duration_ms, vot_ms, burst_centroid_hz,
            burst_spectral_skew, f2_transition_slope_hz_per_ms, f2_locus_hz,
            voiced_fraction, closure_voicing_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_build_gmm_drops_sparse_feature_keeps_rows(tmp_path) -> None:
    from assess.stats import BuildConfig, build_atlas_gmm

    db_path = tmp_path / "ref.sqlite"
    _stop_reference_db(db_path, n=80, vot_present=16)  # VOT coverage 20%
    config = BuildConfig(
        providers=("common_voice",),
        min_samples=10,
        confidence_floor=0.5,
        per_phone_cap=0,
        train_buckets="0..99",
        test_buckets="",
        d_quantile=0.95,
        seed=0,
        min_feature_coverage=0.6,
    )
    atlas = build_atlas_gmm(db_path, feature_version="cov_test_v1", config=config)
    gmm = atlas.get("b")
    assert gmm is not None
    # Sparse VOT (20% coverage) dropped; dense cues retained.
    assert "vot_ms" not in gmm.feature_names
    assert "burst_centroid_hz" in gmm.feature_names
    # Rows are retained rather than collapsing to the 16 VOT-bearing rows.
    assert gmm.n >= 70


def test_assess_atlas_gmm_scores_consonant_without_abstain() -> None:
    from assess.features import FeatureSet
    from assess.stats import AtlasGMM, assess_atlas_gmm

    fs = FeatureSet(
        duration_ms=90.0,
        rms=0.15,
        spectral_centroid_hz=6500.0,
        spectral_bandwidth_hz=1500.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.0,
        f0_mean_hz=None,
        f1_hz=None,
        f2_hz=None,
        f3_hz=None,
        formant_confidence=0.0,
        feature_confidence=1.0,
        low_confidence_reasons=[],
        active_duration_ms=90.0,
        spectral_skew=1.2,
        spectral_kurtosis=3.0,
    )
    target = _phone_gmm(
        phone="s",
        feature_names=(
            "spectral_centroid_hz",
            "spectral_bandwidth_hz",
            "spectral_skew",
            "spectral_kurtosis",
            "voiced_fraction",
        ),
        mean=[6800.0, 1400.0, 1.1, 2.5, 0.05],
    )
    atlas = AtlasGMM(
        feature_version="assess_coach_gmm_v1",
        phones={"s": target},
    )

    ev = assess_atlas_gmm(fs, "s", atlas)
    # Consonants now get a typicality score, no abstain.
    assert ev.target_typicality is not None
    assert ev.as_dict()["quality_score"] == pytest.approx(ev.target_typicality)
    assert ev.component_count == 1
    assert "consonant_abstain" not in ev.reasons


# ---------------------------------------------------------------------------
# GMM log-density calibration
# ---------------------------------------------------------------------------

def _vowel_reference_db(path, *, phone: str, n: int, mean: tuple[float, float]) -> None:
    """Tiny coach reference DB carrying F1/F2 rows for a single vowel."""
    import sqlite3

    rng = np.random.default_rng(0)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE phone_features (
            id INTEGER PRIMARY KEY,
            expected_phone TEXT,
            provider TEXT,
            speaker_id TEXT,
            formant_success INTEGER,
            feature_confidence REAL,
            duration_ms REAL,
            f1_hz REAL,
            f2_hz REAL,
            segment_alignment_id INTEGER,
            phone_index INTEGER
        )
        """
    )
    rows = []
    for i in range(n):
        f1 = float(mean[0] + rng.normal(0.0, 30.0))
        f2 = float(mean[1] + rng.normal(0.0, 80.0))
        rows.append(
            (phone, "common_voice", f"spk{i:04d}", 1, 1.0, 80.0, f1, f2, i, 0)
        )
    conn.executemany(
        """
        INSERT INTO phone_features (
            expected_phone, provider, speaker_id, formant_success,
            feature_confidence, duration_ms, f1_hz, f2_hz,
            segment_alignment_id, phone_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_calibration_roundtrip(tmp_path) -> None:
    from assess.stats import (
        GMMCalibration,
        PhoneLogDensityCalibration,
    )

    cal = GMMCalibration(
        feature_version="assess_coach_gmm_v1",
        gmm_signature="abc123",
        phones={
            "a": PhoneLogDensityCalibration(
                phone="a",
                log_density_quantiles=(-10.0, -8.0, -5.0, -2.0, 0.5),
                n=500,
            )
        },
    )
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = GMMCalibration.load(path)
    assert loaded.feature_version == "assess_coach_gmm_v1"
    assert loaded.gmm_signature == "abc123"
    a = loaded.get("a")
    assert a is not None
    assert a.log_density_quantiles == (-10.0, -8.0, -5.0, -2.0, 0.5)
    assert a.n == 500


def test_calibrated_typicality_is_native_rank() -> None:
    from assess.stats import (
        PhoneLogDensityCalibration,
        calibrated_typicality,
    )

    cal = PhoneLogDensityCalibration(
        phone="a",
        log_density_quantiles=tuple(float(v) for v in range(-9, 1)),  # -9..0
        n=10,
    )
    assert calibrated_typicality(-100.0, cal) == pytest.approx(0.0)
    assert calibrated_typicality(100.0, cal) == pytest.approx(1.0)
    # Mid value falls in the middle of the grid.
    mid = calibrated_typicality(-4.5, cal)
    assert 0.4 <= mid <= 0.6


def test_calibration_rejects_descending_grid() -> None:
    from assess.stats import PhoneLogDensityCalibration

    with pytest.raises(ValueError):
        PhoneLogDensityCalibration(
            phone="a", log_density_quantiles=(0.0, -1.0), n=1
        )


def test_build_gmm_calibration_produces_uniform_native_typicality(tmp_path) -> None:
    from assess.stats import (
        BuildConfig,
        CalibrationBuildConfig,
        assess_atlas_gmm,
        build_atlas_gmm,
        build_gmm_calibration,
    )

    db_path = tmp_path / "ref.sqlite"
    _vowel_reference_db(db_path, phone="a", n=400, mean=(800.0, 1200.0))

    build_config = BuildConfig(
        providers=("common_voice",),
        min_samples=50,
        confidence_floor=0.5,
        per_phone_cap=0,
        train_buckets="0..99",
        test_buckets="",
        d_quantile=0.95,
        seed=0,
        min_feature_coverage=0.6,
    )
    atlas = build_atlas_gmm(db_path, feature_version="cal_test_v1", config=build_config)
    assert atlas.get("a") is not None

    cal_config = CalibrationBuildConfig(
        providers=("common_voice",),
        confidence_floor=0.5,
        per_phone_cap=0,
        quantile_count=128,
        seed=0,
    )
    calibration = build_gmm_calibration(db_path, atlas, cal_config)
    a_cal = calibration.get("a")
    assert a_cal is not None
    assert a_cal.n >= 300
    # The grid is ascending — searchsorted invariant.
    assert list(a_cal.log_density_quantiles) == sorted(a_cal.log_density_quantiles)
    # A sample drawn at the GMM mean should land near the top of the native rank.
    fs = _make_vowel_feature_set(f1=800.0, f2=1200.0)
    ev = assess_atlas_gmm(fs, "a", atlas, calibration=calibration)
    assert ev.target_typicality is not None
    assert ev.target_typicality > 0.5


def _make_vowel_feature_set(*, f1: float, f2: float):
    from assess.features import FeatureSet

    return FeatureSet(
        duration_ms=80.0,
        rms=0.2,
        spectral_centroid_hz=2000.0,
        spectral_bandwidth_hz=1100.0,
        mfcc=[0.0] * 13,
        voiced_fraction=0.7,
        f0_mean_hz=180.0,
        f1_hz=f1,
        f2_hz=f2,
        f3_hz=2400.0,
        formant_confidence=1.0,
        feature_confidence=1.0,
        low_confidence_reasons=[],
        active_duration_ms=80.0,
    )


def test_assess_atlas_gmm_uses_calibrated_thresholds() -> None:
    from assess.stats import (
        AtlasGMM,
        GMMCalibration,
        PhoneLogDensityCalibration,
        assess_atlas_gmm,
    )

    fs = _make_vowel_feature_set(f1=800.0, f2=1200.0)
    target = _phone_gmm(
        phone="a", feature_names=("f1_hz", "f2_hz"), mean=[800.0, 1200.0]
    )
    alt = _phone_gmm(
        phone="e", feature_names=("f1_hz", "f2_hz"), mean=[500.0, 1800.0]
    )
    atlas = AtlasGMM(
        feature_version="assess_coach_gmm_v1",
        phones={"a": target, "e": alt},
    )
    # Calibration where a sample at the GMM mean lands at the top of the grid.
    calibration = GMMCalibration(
        feature_version="assess_coach_gmm_v1",
        gmm_signature=None,
        phones={
            "a": PhoneLogDensityCalibration(
                phone="a",
                log_density_quantiles=(-50.0, -20.0, -10.0, -5.0),
                n=4,
            ),
            "e": PhoneLogDensityCalibration(
                phone="e",
                log_density_quantiles=(-200.0, -100.0, -80.0, -60.0),
                n=4,
            ),
        },
    )
    ev = assess_atlas_gmm(fs, "a", atlas, calibration=calibration)
    assert ev.target_typicality is not None
    # Sample at the centre clears the calibrated fit floor.
    assert ev.atlas_verdict == "fit_ok"
