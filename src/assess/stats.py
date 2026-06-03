"""Coach acoustic statistics: robust distance stats, GMM scoring, atlas evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np
try:  # pragma: no cover - exercised when scipy is installed.
    from scipy.stats import chi2
except ModuleNotFoundError:  # pragma: no cover - small fallback for lean test envs.
    class _Chi2Fallback:
        @staticmethod
        def ppf(q: float, df: int) -> float:
            z = NormalDist().inv_cdf(float(q))
            base = 1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))
            return float(df * max(0.0, base) ** 3)

    chi2 = _Chi2Fallback()

from assess.features import (
    FeatureSet,
    feature_names_for,
    feature_vector_from_names,
    is_vowel,
    phone_class,
    short_identity,
)

# ---------------------------------------------------------------------------
# Robust distance helpers
# ---------------------------------------------------------------------------

_TRIM_QUANTILE = 0.999


def mahalanobis_sq(x: np.ndarray, center: np.ndarray, inv_cov: np.ndarray) -> float:
    diff = np.asarray(x, dtype=np.float64) - center
    return float(diff @ inv_cov @ diff)


def _mahalanobis_sq_rows(
    samples: np.ndarray, center: np.ndarray, inv_cov: np.ndarray
) -> np.ndarray:
    diff = samples - center
    return np.einsum("ij,jk,ik->i", diff, inv_cov, diff)


def _ridged_scatter(samples: np.ndarray, center: np.ndarray, ridge: float) -> np.ndarray:
    diff = samples - center
    n = samples.shape[0]
    cov = (diff.T @ diff) / max(1, n - 1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    df = cov.shape[0]
    cov = cov + ridge * np.trace(cov) / df * np.eye(df)
    return cov


def _sample_matrix_2d(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("samples must be a non-empty 2D array")
    if not np.isfinite(arr).all():
        raise ValueError("samples must contain only finite values")
    return arr


def robust_center_scatter(
    samples: np.ndarray, *, ridge: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median center + median-centered scatter with chi2 trim.

    Returns (center, cov, inv_cov), all float64.
    """
    samples = _sample_matrix_2d(samples)
    df = samples.shape[1]
    center0 = np.median(samples, axis=0)
    cov0 = _ridged_scatter(samples, center0, ridge)
    inv0 = np.linalg.pinv(cov0)
    d2 = _mahalanobis_sq_rows(samples, center0, inv0)
    keep = d2 <= float(chi2.ppf(_TRIM_QUANTILE, df))
    kept = samples[keep] if int(keep.sum()) >= df + 1 else samples
    center = np.median(kept, axis=0)
    cov = _ridged_scatter(kept, center, ridge)
    inv = np.linalg.pinv(cov)
    return center, cov, inv


def distance_threshold(
    samples: np.ndarray,
    center: np.ndarray,
    inv_cov: np.ndarray,
    *,
    quantile: float = 0.95,
) -> float:
    """sqrt of the requested robust-distance quantile over samples."""
    d2 = _mahalanobis_sq_rows(_sample_matrix_2d(samples), center, inv_cov)
    return float(math.sqrt(float(np.quantile(d2, quantile))))


def fit_score(
    x: np.ndarray,
    center: np.ndarray,
    inv_cov: np.ndarray,
    *,
    d_threshold: float,
) -> float:
    """exp(-0.5 (d / d_threshold)^2): 1.0 at center, decaying outward."""
    x_arr = np.asarray(x, dtype=np.float64)
    center_arr = np.asarray(center, dtype=np.float64)
    d = math.sqrt(max(0.0, mahalanobis_sq(x_arr, center_arr, inv_cov)))
    if d_threshold <= 0:
        return 1.0 if np.array_equal(x_arr, center_arr) else 0.0
    return float(math.exp(-0.5 * (d / d_threshold) ** 2))


# ---------------------------------------------------------------------------
# Per-phone canonical statistics artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhoneStat:
    phone: str
    feature_names: tuple[str, ...]
    center: list[float]
    inv_cov: list[list[float]]
    d_threshold: float
    duration_median: float
    duration_scale: float
    n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "phone": self.phone,
            "feature_names": list(self.feature_names),
            "center": list(self.center),
            "inv_cov": [list(row) for row in self.inv_cov],
            "d_threshold": self.d_threshold,
            "duration_median": self.duration_median,
            "duration_scale": self.duration_scale,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PhoneStat:
        return cls(
            phone=str(data["phone"]),
            feature_names=tuple(str(x) for x in data["feature_names"]),
            center=[float(x) for x in data["center"]],
            inv_cov=[[float(v) for v in row] for row in data["inv_cov"]],
            d_threshold=float(data["d_threshold"]),
            duration_median=float(data["duration_median"]),
            duration_scale=float(data["duration_scale"]),
            n=int(data["n"]),
        )


@dataclass(frozen=True)
class AtlasStats:
    feature_version: str
    phones: dict[str, PhoneStat]

    def has(self, phone: str) -> bool:
        return phone in self.phones

    def get(self, phone: str) -> PhoneStat | None:
        return self.phones.get(phone)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "phones": {k: v.as_dict() for k, v in self.phones.items()},
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> AtlasStats:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Atlas stats artifact must be a JSON object")
        if "feature_version" not in data:
            raise ValueError("Atlas stats artifact missing feature_version")
        if "phones" not in data:
            raise ValueError("Atlas stats artifact missing phones")
        phones_payload = data["phones"]
        if not isinstance(phones_payload, dict):
            raise ValueError("Atlas stats phones must be a JSON object")
        phones = {str(k): PhoneStat.from_dict(v) for k, v in phones_payload.items()}
        return cls(feature_version=str(data["feature_version"]), phones=phones)


# ---------------------------------------------------------------------------
# Per-phone GMM artifact (Faz 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GMMComponent:
    """One Gaussian component of a per-phone density."""

    mean: list[float]
    inv_cov: list[list[float]]
    cov_log_det: float
    log_weight: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": list(self.mean),
            "inv_cov": [list(row) for row in self.inv_cov],
            "cov_log_det": float(self.cov_log_det),
            "log_weight": float(self.log_weight),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GMMComponent:
        return cls(
            mean=[float(x) for x in data["mean"]],
            inv_cov=[[float(v) for v in row] for row in data["inv_cov"]],
            cov_log_det=float(data["cov_log_det"]),
            log_weight=float(data["log_weight"]),
        )


@dataclass(frozen=True)
class PhoneGMM:
    """K-component GMM for a single phone; K=1 is a single robust component."""

    phone: str
    feature_names: tuple[str, ...]
    components: tuple[GMMComponent, ...]
    d_threshold: float
    duration_median: float
    duration_scale: float
    n: int

    @property
    def k(self) -> int:
        return len(self.components)

    def as_dict(self) -> dict[str, object]:
        return {
            "phone": self.phone,
            "feature_names": list(self.feature_names),
            "components": [c.as_dict() for c in self.components],
            "d_threshold": self.d_threshold,
            "duration_median": self.duration_median,
            "duration_scale": self.duration_scale,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PhoneGMM:
        return cls(
            phone=str(data["phone"]),
            feature_names=tuple(str(x) for x in data["feature_names"]),
            components=tuple(
                GMMComponent.from_dict(c) for c in data["components"]
            ),
            d_threshold=float(data["d_threshold"]),
            duration_median=float(data["duration_median"]),
            duration_scale=float(data["duration_scale"]),
            n=int(data["n"]),
        )


@dataclass(frozen=True)
class AtlasGMM:
    feature_version: str
    phones: dict[str, PhoneGMM]

    def has(self, phone: str) -> bool:
        return phone in self.phones

    def get(self, phone: str) -> PhoneGMM | None:
        return self.phones.get(phone)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "phones": {k: v.as_dict() for k, v in self.phones.items()},
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> AtlasGMM:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Atlas GMM artifact must be a JSON object")
        if "feature_version" not in data:
            raise ValueError("Atlas GMM artifact missing feature_version")
        if "phones" not in data:
            raise ValueError("Atlas GMM artifact missing phones")
        phones_payload = data["phones"]
        if not isinstance(phones_payload, dict):
            raise ValueError("Atlas GMM phones must be a JSON object")
        phones = {
            str(k): PhoneGMM.from_dict(v) for k, v in phones_payload.items()
        }
        return cls(feature_version=str(data["feature_version"]), phones=phones)


# ---------------------------------------------------------------------------
# GMM log-density calibration artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhoneLogDensityCalibration:
    """Ascending log-density quantile grid drawn from native reference samples.

    The grid maps a runtime log-density into the native cumulative rank for
    that phone, so per-phone typicality scores share a single uniform scale.
    """

    phone: str
    log_density_quantiles: tuple[float, ...]
    n: int

    def __post_init__(self) -> None:
        grid = list(self.log_density_quantiles)
        if not grid:
            raise ValueError("log_density_quantiles must not be empty")
        for prev, curr in zip(grid, grid[1:]):
            if curr < prev:
                raise ValueError("log_density_quantiles must be ascending")
        if self.n < 1:
            raise ValueError("n must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "phone": self.phone,
            "log_density_quantiles": list(self.log_density_quantiles),
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PhoneLogDensityCalibration:
        return cls(
            phone=str(data["phone"]),
            log_density_quantiles=tuple(float(x) for x in data["log_density_quantiles"]),
            n=int(data["n"]),
        )


@dataclass(frozen=True)
class GMMCalibration:
    feature_version: str
    gmm_signature: str | None
    phones: dict[str, PhoneLogDensityCalibration]

    def has(self, phone: str) -> bool:
        return phone in self.phones

    def get(self, phone: str) -> PhoneLogDensityCalibration | None:
        return self.phones.get(phone)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "gmm_signature": self.gmm_signature,
            "phones": {k: v.as_dict() for k, v in self.phones.items()},
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> GMMCalibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("GMM calibration artifact must be a JSON object")
        if "feature_version" not in data:
            raise ValueError("GMM calibration artifact missing feature_version")
        phones_payload = data.get("phones")
        if not isinstance(phones_payload, dict):
            raise ValueError("GMM calibration phones must be a JSON object")
        phones = {
            str(k): PhoneLogDensityCalibration.from_dict(v)
            for k, v in phones_payload.items()
        }
        signature = data.get("gmm_signature")
        return cls(
            feature_version=str(data["feature_version"]),
            gmm_signature=str(signature) if signature is not None else None,
            phones=phones,
        )


def calibrated_typicality(
    log_density: float, calibration: PhoneLogDensityCalibration
) -> float:
    """Native cumulative rank of ``log_density`` against the per-phone grid."""
    grid = np.asarray(calibration.log_density_quantiles, dtype=np.float64)
    idx = int(np.searchsorted(grid, float(log_density), side="right"))
    return idx / float(grid.shape[0])


# ---------------------------------------------------------------------------
# Atlas stats builder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildConfig:
    providers: tuple[str, ...]
    min_samples: int
    confidence_floor: float
    per_phone_cap: int
    train_buckets: str
    test_buckets: str
    d_quantile: float
    seed: int
    ridge: float = 1e-3
    # Per-phone coverage gate: a class feature enters the fitted vector only if
    # it is non-null in at least this fraction of the phone's training rows.
    # Sparse-but-valuable cues (e.g. VOT in /k/ /c/ /t/) drop out where they are
    # rarely measurable, so those rows survive and the runtime never demands a
    # feature the artifact was not fit on.
    min_feature_coverage: float = 0.6


@dataclass(frozen=True)
class _Sample:
    features: tuple[float, ...]
    duration_ms: float
    sort_key: str


def _parse_buckets(spec: str) -> frozenset[int]:
    buckets: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ".." in chunk:
            lo_s, hi_s = chunk.split("..", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"bucket range is descending: {chunk}")
            for b in range(lo, hi + 1):
                buckets.add(b)
        else:
            buckets.add(int(chunk))
    for b in buckets:
        if b < 0 or b > 99:
            raise ValueError(f"speaker bucket out of range 0..99: {b}")
    return frozenset(buckets)


def _speaker_bucket(speaker_id: str) -> int:
    digest = hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def _stable_sort_key(*parts: object) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _table_columns(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[1]) for row in conn.execute("PRAGMA table_info(phone_features)")
    )


def _provider_filter_sql(providers: tuple[str, ...]) -> str:
    if not providers:
        raise ValueError("BuildConfig.providers must not be empty")
    return ",".join("?" for _ in providers)


def _distinct_phones(conn: sqlite3.Connection, providers: tuple[str, ...]) -> list[str]:
    placeholders = _provider_filter_sql(providers)
    rows = conn.execute(
        f"""
        SELECT DISTINCT expected_phone
        FROM phone_features
        WHERE provider IN ({placeholders})
        ORDER BY expected_phone
        """,
        providers,
    )
    return [str(row[0]) for row in rows if row[0]]


def _sample_matrix(samples: list[_Sample], width: int) -> np.ndarray:
    if not samples:
        return np.empty((0, width), dtype=np.float64)
    return np.asarray([s.features for s in samples], dtype=np.float64)


def _duration_scale(durations: list[float]) -> tuple[float, float]:
    """Robust (median, scale) for duration.

    A non-positive scale signals a degenerate duration distribution — e.g. an
    older reference source floors most phones at an approximately 20 ms span, so the IQR
    collapses to zero. Callers treat ``scale <= 0`` as *duration not scoreable*
    rather than inventing a unit scale that turns every real runtime duration
    into a spurious deviation.
    """
    arr = np.asarray(durations, dtype=np.float64)
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale <= 0.0:
        return median, 0.0
    return median, scale


@dataclass(frozen=True)
class _RawRow:
    values: tuple[float | None, ...]
    duration_ms: float
    sort_key: str


_EMPTY_SAMPLES: tuple[np.ndarray, np.ndarray, list[float], tuple[str, ...]] = (
    np.empty((0, 0), dtype=np.float64),
    np.empty((0, 0), dtype=np.float64),
    [],
    (),
)


def _load_phone_samples(
    conn: sqlite3.Connection,
    *,
    phone: str,
    config: BuildConfig,
    table_columns: frozenset[str],
) -> tuple[np.ndarray, np.ndarray, list[float], tuple[str, ...]]:
    """Load samples for ``phone`` with per-phone coverage-based feature selection.

    The class feature set is the candidate pool. A candidate enters the fitted
    vector only if it is non-null in at least ``config.min_feature_coverage`` of
    the phone's training rows; sparse cues drop out (keeping their rows) rather
    than forcing every row that lacks them to be discarded. The selected feature
    names are returned so callers record exactly which slice was fit, and the
    runtime scores against that same slice.
    """
    requested = feature_names_for(phone)
    names = tuple(name for name in requested if name in table_columns)
    if not names:
        return _EMPTY_SAMPLES
    placeholders = _provider_filter_sql(config.providers)
    has_alignment_order = {"segment_alignment_id", "phone_index"}.issubset(table_columns)
    select_cols = [
        "speaker_id", "formant_success", "feature_confidence", "duration_ms", *names,
    ]
    if has_alignment_order:
        select_cols.extend(["segment_alignment_id", "phone_index"])

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM phone_features
        WHERE expected_phone = ?
          AND provider IN ({placeholders})
          AND feature_confidence >= ?
    """
    params: tuple[object, ...] = (phone, *config.providers, config.confidence_floor)
    train_buckets = _parse_buckets(config.train_buckets)
    test_buckets = _parse_buckets(config.test_buckets)

    train_raw: list[_RawRow] = []
    test_raw: list[_RawRow] = []
    for row in conn.execute(sql, params):
        speaker_id = row["speaker_id"]
        if speaker_id is None or str(speaker_id) == "":
            continue
        if is_vowel(phone) and row["formant_success"] != 1:
            continue
        confidence = _finite_float(row["feature_confidence"])
        duration = _finite_float(row["duration_ms"])
        if confidence is None or duration is None:
            continue
        values = tuple(_finite_float(row[name]) for name in names)
        if has_alignment_order:
            sort_key = _stable_sort_key(
                config.seed, row["segment_alignment_id"], row["phone_index"]
            )
        else:
            sort_key = _stable_sort_key(
                config.seed, phone, speaker_id, duration, *values
            )
        raw = _RawRow(values=values, duration_ms=duration, sort_key=sort_key)
        bucket = _speaker_bucket(str(speaker_id))
        if bucket in train_buckets:
            train_raw.append(raw)
        elif bucket in test_buckets:
            test_raw.append(raw)

    if not train_raw:
        return _EMPTY_SAMPLES

    selected = _select_features_by_coverage(
        train_raw, len(names), config.min_feature_coverage
    )
    selected_names = tuple(names[i] for i in selected)

    def to_samples(raw_rows: list[_RawRow]) -> list[_Sample]:
        out: list[_Sample] = []
        for raw in raw_rows:
            picked = tuple(raw.values[i] for i in selected)
            if any(v is None for v in picked):
                continue
            out.append(
                _Sample(
                    features=tuple(float(v) for v in picked),  # type: ignore[arg-type]
                    duration_ms=raw.duration_ms,
                    sort_key=raw.sort_key,
                )
            )
        return out

    train_samples = to_samples(train_raw)
    test_samples = to_samples(test_raw)
    train_samples.sort(key=lambda s: s.sort_key)
    test_samples.sort(key=lambda s: s.sort_key)
    if config.per_phone_cap > 0:
        train_samples = train_samples[: config.per_phone_cap]

    return (
        _sample_matrix(train_samples, len(selected_names)),
        _sample_matrix(test_samples, len(selected_names)),
        [s.duration_ms for s in train_samples],
        selected_names,
    )


def _select_features_by_coverage(
    train_raw: list[_RawRow], width: int, min_coverage: float
) -> list[int]:
    """Indices of candidate features whose non-null coverage clears the gate.

    Falls back to the single densest feature when no candidate clears the gate,
    so every phone with training rows still yields a non-empty vector.
    """
    n = len(train_raw)
    coverage = [
        sum(1 for raw in train_raw if raw.values[i] is not None) / n
        for i in range(width)
    ]
    selected = [i for i in range(width) if coverage[i] >= min_coverage]
    if not selected:
        selected = [max(range(width), key=lambda i: coverage[i])]
    return selected


def build_atlas_stats(
    db_path: Path | str | sqlite3.Connection,
    feature_version: str,
    config: BuildConfig,
) -> AtlasStats:
    """Build per-phone robust canonical statistics from a compat SQLite DB."""
    if config.min_samples < 1:
        raise ValueError("BuildConfig.min_samples must be positive")
    if not 0.0 < config.d_quantile <= 1.0:
        raise ValueError("BuildConfig.d_quantile must be in (0, 1]")
    if config.per_phone_cap < 0:
        raise ValueError("BuildConfig.per_phone_cap must be non-negative")

    close_conn = not isinstance(db_path, sqlite3.Connection)
    conn = (
        db_path
        if isinstance(db_path, sqlite3.Connection)
        else sqlite3.connect(db_path)
    )
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        table_columns = _table_columns(conn)
        phones: dict[str, PhoneStat] = {}
        for phone in _distinct_phones(conn, config.providers):
            train, test, train_durations, usable_names = _load_phone_samples(
                conn, phone=phone, config=config, table_columns=table_columns
            )
            if train.shape[0] < config.min_samples or not usable_names:
                continue
            center, _cov, inv_cov = robust_center_scatter(train, ridge=config.ridge)
            threshold_samples = test if test.shape[0] >= config.min_samples else train
            d_threshold = distance_threshold(
                threshold_samples, center, inv_cov, quantile=config.d_quantile
            )
            duration_median, duration_scale = _duration_scale(train_durations)
            phones[phone] = PhoneStat(
                phone=phone,
                feature_names=usable_names,
                center=center.tolist(),
                inv_cov=inv_cov.tolist(),
                d_threshold=d_threshold,
                duration_median=duration_median,
                duration_scale=duration_scale,
                n=int(train.shape[0]),
            )
        return AtlasStats(feature_version=feature_version, phones=phones)
    finally:
        conn.row_factory = original_row_factory
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Atlas evidence scorer
# ---------------------------------------------------------------------------

_DEFAULT_DELTA = 0.15
_DEFAULT_TAU = 1.5
_INSIDE_FIT_FLOOR = math.exp(-0.5)

# Calibrated typicality lives on a per-phone native-rank scale, so the same
# inside-fit and same-class margin thresholds carry equivalent semantics across
# phones. The floor mirrors the 0.95 native quantile the raw d_threshold encodes;
# the delta is wider than the raw 0.15 because percentile differences are
# uniformly distributed under the native reference.
_CALIBRATED_INSIDE_FIT_FLOOR = 0.05
_CALIBRATED_DEFAULT_DELTA = 0.20


@dataclass(frozen=True)
class AtlasEvidence:
    target_phone: str
    target_fit: float | None
    best_match: str | None
    best_match_fit: float | None
    margin: float | None
    duration_z: float | None
    duration_verdict: str | None
    atlas_verdict: str
    confidence: float
    reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "target_phone": self.target_phone,
            "target_fit": self.target_fit,
            "best_match": self.best_match,
            "best_match_fit": self.best_match_fit,
            "margin": self.margin,
            "duration_z": self.duration_z,
            "duration_verdict": self.duration_verdict,
            "atlas_verdict": self.atlas_verdict,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


def _fit_for(vec: list[float], stat: PhoneStat) -> float:
    return fit_score(
        np.asarray(vec, dtype=np.float64),
        np.asarray(stat.center, dtype=np.float64),
        np.asarray(stat.inv_cov, dtype=np.float64),
        d_threshold=stat.d_threshold,
    )


def fit_to_phone(
    values: Mapping[str, float | None],
    phone: str,
    stats: AtlasStats,
) -> float | None:
    """Score a raw feature-value mapping against a named phone's PhoneStat."""
    stat = stats.get(phone) or stats.get(short_identity(phone))
    if stat is None:
        return None
    vec: list[float] = []
    for name in stat.feature_names:
        raw = values.get(name)
        if raw is None:
            return None
        value = float(raw)
        if not math.isfinite(value):
            return None
        vec.append(value)
    return _fit_for(vec, stat)


def _duration_from_stat(
    feature_set: FeatureSet, stat: PhoneStat, tau: float
) -> tuple[float | None, str]:
    scale = stat.duration_scale
    if not math.isfinite(scale) or scale <= 0.0:
        return None, "expected"
    z = (float(feature_set.duration_ms) - stat.duration_median) / scale
    if z < -tau:
        return z, "short"
    if z > tau:
        return z, "long"
    return z, "expected"


def assess_atlas(
    feature_set: FeatureSet,
    target_phone: str,
    stats: AtlasStats,
    *,
    delta: float = _DEFAULT_DELTA,
    tau: float = _DEFAULT_TAU,
) -> AtlasEvidence:
    """Score a FeatureSet against the atlas canonical statistics."""
    identity = target_phone if stats.get(target_phone) is not None else short_identity(target_phone)
    reasons = list(feature_set.low_confidence_reasons)

    if float(feature_set.feature_confidence) == 0.0:
        return _low_confidence_evidence(target_phone, reasons)

    stat = stats.get(identity)
    if stat is None:
        reasons.append("phone_not_in_atlas")
        return _low_confidence_evidence(target_phone, reasons)

    vec = feature_vector_from_names(feature_set, stat.feature_names)
    if vec is None:
        reasons.append("missing_features")
        return _low_confidence_evidence(target_phone, reasons)

    target_fit = _fit_for(vec, stat)
    duration_z, duration_verdict = _duration_from_stat(feature_set, stat, tau)
    confidence = float(feature_set.feature_confidence)

    if not is_vowel(identity):
        return AtlasEvidence(
            target_phone=target_phone,
            target_fit=target_fit,
            best_match=None,
            best_match_fit=None,
            margin=None,
            duration_z=duration_z,
            duration_verdict=duration_verdict,
            atlas_verdict="consonant_abstain",
            confidence=confidence,
            reasons=[*reasons, "consonant_abstain"],
        )

    best_match: str | None = None
    best_match_fit: float | None = None
    for other_phone, other_stat in stats.phones.items():
        other_identity = other_phone
        if other_identity == identity or not is_vowel(other_identity):
            continue
        other_vec = feature_vector_from_names(feature_set, other_stat.feature_names)
        if other_vec is None:
            continue
        other_fit = _fit_for(other_vec, other_stat)
        if best_match_fit is None or other_fit > best_match_fit:
            best_match = other_identity
            best_match_fit = other_fit

    margin = target_fit - best_match_fit if best_match_fit is not None else None
    if target_fit >= _INSIDE_FIT_FLOOR:
        atlas_verdict = "fit_ok"
    elif margin is not None and margin <= -delta:
        atlas_verdict = "likely_substitution"
    else:
        atlas_verdict = "atypical"

    return AtlasEvidence(
        target_phone=target_phone,
        target_fit=target_fit,
        best_match=best_match,
        best_match_fit=best_match_fit,
        margin=margin,
        duration_z=duration_z,
        duration_verdict=duration_verdict,
        atlas_verdict=atlas_verdict,
        confidence=confidence,
        reasons=reasons,
    )


def _low_confidence_evidence(target_phone: str, reasons: list[str]) -> AtlasEvidence:
    return AtlasEvidence(
        target_phone=target_phone,
        target_fit=None,
        best_match=None,
        best_match_fit=None,
        margin=None,
        duration_z=None,
        duration_verdict=None,
        atlas_verdict="low_confidence",
        confidence=0.0,
        reasons=list(reasons),
    )


# ---------------------------------------------------------------------------
# Per-phone GMM builder (Faz 4)
# ---------------------------------------------------------------------------

_GMM_DEFAULT_MAX_K = 3
_GMM_DEFAULT_MIN_SAMPLES_PER_COMPONENT = 30
_GMM_REG_COVAR = 1e-3


def _fit_phone_gmm(
    samples: np.ndarray,
    *,
    max_k: int = _GMM_DEFAULT_MAX_K,
    min_samples_per_component: int = _GMM_DEFAULT_MIN_SAMPLES_PER_COMPONENT,
    seed: int = 0,
) -> tuple[GMMComponent, ...]:
    """Fit a GMM with auto-K via BIC; falls back to K=1 robust covariance."""
    try:
        from sklearn.mixture import GaussianMixture  # noqa: PLC0415
    except ModuleNotFoundError:
        return _fit_single_component_gmm(samples)

    n, _df = samples.shape
    max_k_allowed = max(1, min(max_k, n // min_samples_per_component))
    best_components: tuple[GMMComponent, ...] | None = None
    best_bic = float("inf")
    for k in range(1, max_k_allowed + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=_GMM_REG_COVAR,
                random_state=seed,
                max_iter=200,
            )
            gmm.fit(samples)
        except Exception:
            continue
        bic = float(gmm.bic(samples))
        if bic < best_bic:
            components: list[GMMComponent] = []
            for idx in range(k):
                mean = np.asarray(gmm.means_[idx], dtype=np.float64)
                cov = np.asarray(gmm.covariances_[idx], dtype=np.float64)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    cov = cov + _GMM_REG_COVAR * np.eye(cov.shape[0])
                    sign, logdet = np.linalg.slogdet(cov)
                inv_cov = np.linalg.pinv(cov)
                weight = float(gmm.weights_[idx])
                log_weight = math.log(max(weight, 1e-12))
                components.append(
                    GMMComponent(
                        mean=mean.tolist(),
                        inv_cov=inv_cov.tolist(),
                        cov_log_det=float(logdet),
                        log_weight=log_weight,
                    )
                )
            best_components = tuple(components)
            best_bic = bic
    if best_components is None:
        best_components = _fit_single_component_gmm(samples)
    return best_components


def _fit_single_component_gmm(samples: np.ndarray) -> tuple[GMMComponent, ...]:
    center, cov, inv_cov = robust_center_scatter(samples)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = 0.0
    return (
        GMMComponent(
            mean=center.tolist(),
            inv_cov=inv_cov.tolist(),
            cov_log_det=float(logdet),
            log_weight=0.0,
        ),
    )


def build_atlas_gmm(
    db_path: Path | str | sqlite3.Connection,
    feature_version: str,
    config: BuildConfig,
    *,
    max_k: int = _GMM_DEFAULT_MAX_K,
    min_samples_per_component: int = _GMM_DEFAULT_MIN_SAMPLES_PER_COMPONENT,
) -> AtlasGMM:
    """Fit per-phone GMMs using BIC-driven K selection; K=1 is one component.

    All 49 phones get scored — no consonant abstain. The output JSON is
    compatible with :class:`AtlasGMM`.
    """
    if config.min_samples < 1:
        raise ValueError("BuildConfig.min_samples must be positive")
    if not 0.0 < config.d_quantile <= 1.0:
        raise ValueError("BuildConfig.d_quantile must be in (0, 1]")
    if max_k < 1:
        raise ValueError("max_k must be >= 1")

    close_conn = not isinstance(db_path, sqlite3.Connection)
    conn = (
        db_path
        if isinstance(db_path, sqlite3.Connection)
        else sqlite3.connect(db_path)
    )
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        table_columns = _table_columns(conn)
        phones: dict[str, PhoneGMM] = {}
        for phone in _distinct_phones(conn, config.providers):
            train, test, train_durations, usable_names = _load_phone_samples(
                conn, phone=phone, config=config, table_columns=table_columns
            )
            if train.shape[0] < config.min_samples or not usable_names:
                continue
            components = _fit_phone_gmm(
                train,
                max_k=max_k,
                min_samples_per_component=min_samples_per_component,
                seed=config.seed,
            )
            # d_threshold derived from K=1 robust covariance for backward compat
            center, _cov, inv_cov = robust_center_scatter(train, ridge=config.ridge)
            threshold_samples = test if test.shape[0] >= config.min_samples else train
            d_threshold = distance_threshold(
                threshold_samples, center, inv_cov, quantile=config.d_quantile
            )
            duration_median, duration_scale = _duration_scale(train_durations)
            phones[phone] = PhoneGMM(
                phone=phone,
                feature_names=usable_names,
                components=components,
                d_threshold=d_threshold,
                duration_median=duration_median,
                duration_scale=duration_scale,
                n=int(train.shape[0]),
            )
        return AtlasGMM(feature_version=feature_version, phones=phones)
    finally:
        conn.row_factory = original_row_factory
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# GMM log-density calibration builder
# ---------------------------------------------------------------------------

_CALIBRATION_DEFAULT_QUANTILES = 1024


@dataclass(frozen=True)
class CalibrationBuildConfig:
    providers: tuple[str, ...]
    confidence_floor: float
    per_phone_cap: int
    quantile_count: int = _CALIBRATION_DEFAULT_QUANTILES
    seed: int = 0


def _load_calibration_log_densities(
    conn: sqlite3.Connection,
    *,
    phone: str,
    feature_names: tuple[str, ...],
    config: CalibrationBuildConfig,
    table_columns: frozenset[str],
    gmm: PhoneGMM,
) -> np.ndarray:
    missing = [name for name in feature_names if name not in table_columns]
    if missing:
        return np.empty((0,), dtype=np.float64)
    placeholders = _provider_filter_sql(config.providers)
    has_alignment_order = {"segment_alignment_id", "phone_index"}.issubset(table_columns)
    select_cols = [
        "speaker_id", "formant_success", "feature_confidence", *feature_names,
    ]
    if has_alignment_order:
        select_cols.extend(["segment_alignment_id", "phone_index"])
    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM phone_features
        WHERE expected_phone = ?
          AND provider IN ({placeholders})
          AND feature_confidence >= ?
    """
    params: tuple[object, ...] = (phone, *config.providers, config.confidence_floor)
    samples: list[tuple[str, np.ndarray]] = []
    for row in conn.execute(sql, params):
        speaker_id = row["speaker_id"]
        if speaker_id is None or str(speaker_id) == "":
            continue
        if is_vowel(phone) and row["formant_success"] != 1:
            continue
        values = tuple(_finite_float(row[name]) for name in feature_names)
        if any(v is None for v in values):
            continue
        vec = np.asarray(values, dtype=np.float64)
        if has_alignment_order:
            sort_key = _stable_sort_key(
                config.seed, row["segment_alignment_id"], row["phone_index"]
            )
        else:
            sort_key = _stable_sort_key(config.seed, phone, speaker_id, *values)
        samples.append((sort_key, vec))
    if not samples:
        return np.empty((0,), dtype=np.float64)
    samples.sort(key=lambda item: item[0])
    if config.per_phone_cap > 0:
        samples = samples[: config.per_phone_cap]
    log_densities = np.fromiter(
        (_phone_log_density(vec, gmm) for _, vec in samples),
        dtype=np.float64,
        count=len(samples),
    )
    finite = log_densities[np.isfinite(log_densities)]
    return finite


def _downsample_quantiles(values: np.ndarray, quantile_count: int) -> np.ndarray:
    if values.size == 0:
        return values
    if quantile_count < 2:
        raise ValueError("quantile_count must be >= 2")
    sorted_values = np.sort(values)
    if sorted_values.size <= quantile_count:
        return sorted_values
    probs = np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    return np.quantile(sorted_values, probs, method="linear")


def _gmm_signature(gmm_atlas: AtlasGMM) -> str:
    h = hashlib.sha1()
    h.update(gmm_atlas.feature_version.encode("utf-8"))
    for phone, gmm in sorted(gmm_atlas.phones.items()):
        h.update(b"|")
        h.update(phone.encode("utf-8"))
        h.update(str(gmm.k).encode("utf-8"))
        for component in gmm.components:
            for value in component.mean:
                h.update(f"{value:.6f}".encode("utf-8"))
    return h.hexdigest()


def build_gmm_calibration(
    db_path: Path | str | sqlite3.Connection,
    gmm_atlas: AtlasGMM,
    config: CalibrationBuildConfig,
) -> GMMCalibration:
    """Build a per-phone log-density calibration from native reference samples.

    For every phone the GMM atlas covers, native rows are scored against that
    phone's GMM and the resulting log-density distribution is reduced to a
    fixed quantile grid. The grid is later searched at runtime to convert any
    log-density into a comparable per-phone native rank.
    """
    if config.quantile_count < 2:
        raise ValueError("CalibrationBuildConfig.quantile_count must be >= 2")
    if not config.providers:
        raise ValueError("CalibrationBuildConfig.providers must not be empty")
    if config.per_phone_cap < 0:
        raise ValueError("CalibrationBuildConfig.per_phone_cap must be non-negative")

    close_conn = not isinstance(db_path, sqlite3.Connection)
    conn = (
        db_path
        if isinstance(db_path, sqlite3.Connection)
        else sqlite3.connect(db_path)
    )
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        table_columns = _table_columns(conn)
        phones: dict[str, PhoneLogDensityCalibration] = {}
        for phone, gmm in sorted(gmm_atlas.phones.items()):
            log_densities = _load_calibration_log_densities(
                conn,
                phone=phone,
                feature_names=gmm.feature_names,
                config=config,
                table_columns=table_columns,
                gmm=gmm,
            )
            if log_densities.size < 2:
                continue
            grid = _downsample_quantiles(log_densities, config.quantile_count)
            phones[phone] = PhoneLogDensityCalibration(
                phone=phone,
                log_density_quantiles=tuple(float(v) for v in grid),
                n=int(log_densities.size),
            )
        return GMMCalibration(
            feature_version=gmm_atlas.feature_version,
            gmm_signature=_gmm_signature(gmm_atlas),
            phones=phones,
        )
    finally:
        conn.row_factory = original_row_factory
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# GMM scoring (Faz 4)
# ---------------------------------------------------------------------------

def _component_log_pdf(
    x: np.ndarray, component: GMMComponent
) -> float:
    mean = np.asarray(component.mean, dtype=np.float64)
    inv_cov = np.asarray(component.inv_cov, dtype=np.float64)
    diff = x - mean
    mahal = float(diff @ inv_cov @ diff)
    df = mean.shape[0]
    return float(
        -0.5 * (mahal + component.cov_log_det + df * math.log(2.0 * math.pi))
    )


def _phone_log_density(
    x: np.ndarray, gmm: PhoneGMM
) -> float:
    log_components = [
        component.log_weight + _component_log_pdf(x, component)
        for component in gmm.components
    ]
    if not log_components:
        return float("-inf")
    m = max(log_components)
    return m + math.log(sum(math.exp(lp - m) for lp in log_components))


def _min_mahalanobis(x: np.ndarray, gmm: PhoneGMM) -> float:
    mins: list[float] = []
    for component in gmm.components:
        mean = np.asarray(component.mean, dtype=np.float64)
        inv_cov = np.asarray(component.inv_cov, dtype=np.float64)
        diff = x - mean
        mins.append(float(diff @ inv_cov @ diff))
    return float(min(mins))


def gmm_typicality(x: np.ndarray, gmm: PhoneGMM) -> float:
    """Map the closest component distance to a (0, 1] typicality score."""
    d2 = _min_mahalanobis(x, gmm)
    d = math.sqrt(max(0.0, d2))
    if gmm.d_threshold <= 0:
        return 1.0
    return float(math.exp(-0.5 * (d / gmm.d_threshold) ** 2))


@dataclass(frozen=True)
class GMMEvidence:
    """Per-phone GMM evidence; all phones produce one (no consonant abstain).

    ``atlas_verdict`` is a compact quality label derived from target typicality
    and the margin to the most typical same-class alternative.
    """

    target_phone: str
    target_typicality: float | None
    best_match: str | None
    best_match_typicality: float | None
    margin: float | None
    confidence: float
    duration_z: float | None
    duration_verdict: str | None
    log_density: float | None
    component_count: int
    reasons: list[str]
    atlas_verdict: str = "low_confidence"
    contrast_candidates: list[dict[str, object]] = field(default_factory=list)
    calibrated: bool = False
    fit_floor: float | None = None
    delta: float | None = None

    @property
    def quality_score(self) -> float | None:
        return self.target_typicality

    def as_dict(self) -> dict[str, object]:
        return {
            "target_phone": self.target_phone,
            "quality_score": self.quality_score,
            "target_typicality": self.target_typicality,
            "best_match": self.best_match,
            "best_match_typicality": self.best_match_typicality,
            "margin": self.margin,
            "atlas_verdict": self.atlas_verdict,
            "confidence": self.confidence,
            "duration_z": self.duration_z,
            "duration_verdict": self.duration_verdict,
            "log_density": self.log_density,
            "component_count": self.component_count,
            "contrast_candidates": list(self.contrast_candidates),
            "reasons": list(self.reasons),
            "calibrated": self.calibrated,
            "fit_floor": self.fit_floor,
            "delta": self.delta,
        }

    def with_contrast_candidates(
        self,
        candidates: list[dict[str, object]],
    ) -> "GMMEvidence":
        return GMMEvidence(
            target_phone=self.target_phone,
            target_typicality=self.target_typicality,
            best_match=self.best_match,
            best_match_typicality=self.best_match_typicality,
            margin=self.margin,
            confidence=self.confidence,
            duration_z=self.duration_z,
            duration_verdict=self.duration_verdict,
            log_density=self.log_density,
            component_count=self.component_count,
            reasons=list(self.reasons),
            atlas_verdict=self.atlas_verdict,
            contrast_candidates=list(candidates),
            calibrated=self.calibrated,
            fit_floor=self.fit_floor,
            delta=self.delta,
        )


def _gmm_atlas_verdict(
    target_typicality: float,
    margin: float | None,
    *,
    delta: float,
    fit_floor: float = _INSIDE_FIT_FLOOR,
) -> str:
    """Map typicality + same-class margin onto the shared verdict vocabulary."""
    if target_typicality >= fit_floor:
        return "fit_ok"
    if margin is not None and margin <= -delta:
        return "likely_substitution"
    return "atypical"


def _duration_from_gmm(
    feature_set: FeatureSet, gmm: PhoneGMM, tau: float
) -> tuple[float | None, str]:
    scale = gmm.duration_scale
    if not math.isfinite(scale) or scale <= 0.0:
        return None, "expected"
    z = (float(feature_set.duration_ms) - gmm.duration_median) / scale
    if z < -tau:
        return z, "short"
    if z > tau:
        return z, "long"
    return z, "expected"


def _gmm_low_confidence(target_phone: str, reasons: list[str]) -> GMMEvidence:
    return GMMEvidence(
        target_phone=target_phone,
        target_typicality=None,
        best_match=None,
        best_match_typicality=None,
        margin=None,
        confidence=0.0,
        duration_z=None,
        duration_verdict=None,
        log_density=None,
        component_count=0,
        reasons=list(reasons),
    )


def assess_atlas_gmm(
    feature_set: FeatureSet,
    target_phone: str,
    gmm_atlas: AtlasGMM,
    *,
    tau: float = _DEFAULT_TAU,
    delta: float | None = None,
    calibration: GMMCalibration | None = None,
    fit_floor: float | None = None,
) -> GMMEvidence:
    """Score a FeatureSet against the atlas GMM. No consonant abstain.

    When a calibration artifact is supplied, per-phone typicality is the
    native cumulative rank of the GMM log-density (uniformly distributed under
    the reference population), and the quality-label thresholds switch to the
    calibrated floor/delta. The raw robust-distance fallback applies when a phone
    has no calibration row.
    """
    identity = (
        target_phone
        if gmm_atlas.get(target_phone) is not None
        else short_identity(target_phone)
    )
    reasons = list(feature_set.low_confidence_reasons)

    if float(feature_set.feature_confidence) == 0.0:
        return _gmm_low_confidence(target_phone, reasons)

    target_gmm = gmm_atlas.get(identity)
    if target_gmm is None:
        reasons.append("phone_not_in_gmm")
        return _gmm_low_confidence(target_phone, reasons)

    vec = feature_vector_from_names(feature_set, target_gmm.feature_names)
    if vec is None:
        reasons.append("missing_features")
        return _gmm_low_confidence(target_phone, reasons)
    x = np.asarray(vec, dtype=np.float64)

    log_density = _phone_log_density(x, target_gmm)
    target_cal = calibration.get(identity) if calibration is not None else None
    if target_cal is not None:
        target_typicality = calibrated_typicality(log_density, target_cal)
    else:
        target_typicality = gmm_typicality(x, target_gmm)
    duration_z, duration_verdict = _duration_from_gmm(feature_set, target_gmm, tau)
    confidence = float(feature_set.feature_confidence)

    target_class = phone_class(identity)
    best_match: str | None = None
    best_match_typicality: float | None = None
    for other_phone, other_gmm in gmm_atlas.phones.items():
        other_identity = other_phone
        if other_identity == identity:
            continue
        try:
            other_class = phone_class(other_identity)
        except ValueError:
            continue
        if other_class is not target_class:
            continue
        if other_gmm.feature_names != target_gmm.feature_names:
            continue
        other_vec = feature_vector_from_names(feature_set, other_gmm.feature_names)
        if other_vec is None:
            continue
        other_x = np.asarray(other_vec, dtype=np.float64)
        other_cal = calibration.get(other_identity) if calibration is not None else None
        if other_cal is not None:
            other_log_density = _phone_log_density(other_x, other_gmm)
            other_typ = calibrated_typicality(other_log_density, other_cal)
        else:
            other_typ = gmm_typicality(other_x, other_gmm)
        if best_match_typicality is None or other_typ > best_match_typicality:
            best_match = other_identity
            best_match_typicality = other_typ

    margin = (
        target_typicality - best_match_typicality
        if best_match_typicality is not None
        else None
    )

    using_calibration = target_cal is not None
    effective_delta = (
        delta
        if delta is not None
        else (_CALIBRATED_DEFAULT_DELTA if using_calibration else _DEFAULT_DELTA)
    )
    effective_floor = (
        fit_floor
        if fit_floor is not None
        else (
            _CALIBRATED_INSIDE_FIT_FLOOR if using_calibration else _INSIDE_FIT_FLOOR
        )
    )

    return GMMEvidence(
        target_phone=target_phone,
        target_typicality=target_typicality,
        best_match=best_match,
        best_match_typicality=best_match_typicality,
        margin=margin,
        confidence=confidence,
        duration_z=duration_z,
        duration_verdict=duration_verdict,
        log_density=log_density,
        component_count=target_gmm.k,
        reasons=reasons,
        atlas_verdict=_gmm_atlas_verdict(
            target_typicality, margin, delta=effective_delta, fit_floor=effective_floor
        ),
        calibrated=using_calibration,
        fit_floor=effective_floor,
        delta=effective_delta,
    )
