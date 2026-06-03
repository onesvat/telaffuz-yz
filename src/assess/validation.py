"""Coach validation pipeline: split, batch runner, calibration, reports."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Sabit same-class kontrast listesi (Türkçe fonoloji + smoke gözlemleri)
#
# Authority kalibrasyonu için kapsam darıtma listesi. Runtime'da gözlenen tüm
# (target, alternative) çiftlerine değil yalnız bu listeye filtrelendiğinde,
# native CV verisindeki wav2vec drift'i geniş authority'e dönüşmez.
# ---------------------------------------------------------------------------

_DESIGN_CONTRASTS: tuple[tuple[str, str], ...] = (
    # Yuvarlama (rounding flip, aynı yükseklik/arkalık)
    ("i", "y"),    # front high
    ("e", "œ"),    # front mid
    ("ɯ", "u"),    # back high
    ("a", "o"),    # back low/mid (yaklaşık)
    # Arkalık (back flip, aynı yükseklik/yuvarlama)
    ("i", "ɯ"),    # high unrounded
    ("y", "u"),    # high rounded
    ("e", "a"),    # mid/low unrounded
    ("œ", "o"),    # mid rounded
    # Smoke'ta gözlenen ünlü çakışmaları (yükseklik komşuları)
    ("u", "i"),
    ("a", "u"),
    ("o", "u"),
    # Ünlü yükseklik (curated learner contrasts)
    ("i", "e"),    # height flip front unrounded
    ("e", "æ"),    # closed-syl ae overgeneralization
    ("o", "y"),    # rounding+backness
    ("o", "i"),    # observed escape
    ("œ", "u"),    # rounding loss, height flip
    # Ünlü uzunluk (long↔short)
    ("a", "aː"),
    ("e", "eː"),
    ("i", "iː"),
    ("ɯ", "ɯː"),
    ("o", "oː"),
    ("œ", "œː"),
    ("u", "uː"),
    ("y", "yː"),
    # Damaksıllaşma (palatal/velar yer)
    ("c", "k"),
    ("cʰ", "kʰ"),
    ("ɟ", "ɡ"),
    # Palatal seslilenme + palatalization with voicing
    ("c", "ɟ"),
    ("cʰ", "ɟ"),
    ("ɟ", "kʰ"),   # palatal voiced -> velar voiceless aspirated
    ("c", "kʰ"),
    ("cʰ", "k"),
    ("cʰ", "ɡ"),
    # Ünsüz seslilenme (voicing flip, aynı yer)
    ("s", "z"),
    ("t͡ʃ", "d͡ʒ"),
    ("kʰ", "ɡ"),
    ("b", "pʰ"),
    ("d", "tʰ"),
    # Ünsüz yer/manner (curated learner contrasts)
    ("j", "ɣ"),    # soft-g articulated
    ("j", "ɾ"),    # observed escape
    ("l", "ɫ"),    # light↔dark l
    ("l", "ɲ"),    # palatal nasal substitution
    ("l", "ɾ̞̊"),  # lateral -> tap-like realization
    ("ŋ", "n"),    # velar nasal↔alveolar
    ("ŋ", "m"),    # velar nasal↔bilabial nasal
    ("ɾ̞̊", "ɾ"),  # voiceless tap↔tap
    ("ʒ", "z"),    # postalveolar↔alveolar voiced fricative
    ("ʃ", "s"),    # postalveolar↔alveolar voiceless fricative
)


def _bidirectional(pairs: tuple[tuple[str, str], ...]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for a, b in pairs:
        out.add((a, b))
        out.add((b, a))
    return out


FIXED_SAME_CLASS_CONTRASTS: frozenset[tuple[str, str]] = frozenset(
    _bidirectional(_DESIGN_CONTRASTS)
)

# ---------------------------------------------------------------------------
# Speaker-disjoint split
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    dev_buckets: range = range(0, 90)
    test_buckets: range = range(90, 100)

    def as_dict(self) -> dict[str, object]:
        return {
            "scheme": "sha1_speaker_id_mod_100",
            "dev_buckets": [self.dev_buckets.start, self.dev_buckets.stop],
            "test_buckets": [self.test_buckets.start, self.test_buckets.stop],
        }


def speaker_bucket(speaker_id: str) -> int:
    if not speaker_id:
        raise ValueError("speaker_id must not be empty")
    digest = hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def split_for_bucket(bucket: int, config: SplitConfig | None = None) -> str:
    cfg = config or SplitConfig()
    if bucket in cfg.dev_buckets:
        return "dev"
    if bucket in cfg.test_buckets:
        return "test"
    raise ValueError(f"bucket {bucket} is outside dev/test ranges")


def split_for_speaker(speaker_id: str, config: SplitConfig | None = None) -> str:
    return split_for_bucket(speaker_bucket(speaker_id), config)


# ---------------------------------------------------------------------------
# Calibration observations
# ---------------------------------------------------------------------------

DEFAULT_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("o", "u"), ("u", "o"),
    ("œ", "y"), ("y", "œ"),
    ("ɯ", "i"), ("i", "ɯ"),
    ("e", "i"), ("i", "e"),
)

_VOWEL_FEATURE_COLUMNS = ("f1_hz", "f2_hz", "f3_hz")


def build_calibration_observations(
    *,
    db_path: Path,
    stats: object,
    output: Path,
    split: str = "dev",
    contrasts: tuple[tuple[str, str], ...] = DEFAULT_CONTRASTS,
    limit: int | None = None,
) -> dict[str, object]:
    from assess.stats import fit_to_phone, short_identity  # noqa: PLC0415

    targets = {target for target, _ in contrasts}
    alternatives_by_target: dict[str, list[str]] = {}
    for target, alternative in contrasts:
        alternatives_by_target.setdefault(target, []).append(alternative)

    output.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        expected_phones = _expected_phones_for_targets(con, targets)
        scanned = 0
        emitted = 0
        per_contrast: Counter[str] = Counter()
        with output.open("w", encoding="utf-8") as out:
            for row in _iter_rows(con, expected_phones):
                scanned += 1
                speaker_id = row["speaker_id"] or ""
                if not speaker_id or split_for_speaker(speaker_id) != split:
                    continue
                target = short_identity(row["expected_phone"])
                if target not in targets:
                    continue
                values = {name: row[name] for name in _VOWEL_FEATURE_COLUMNS}
                target_fit = fit_to_phone(values, target, stats)
                if target_fit is None:
                    continue
                for alternative in alternatives_by_target[target]:
                    alternative_fit = fit_to_phone(values, alternative, stats)
                    if alternative_fit is None:
                        continue
                    out.write(
                        json.dumps(
                            {
                                "speaker_id": speaker_id,
                                "target": target,
                                "alternative": alternative,
                                "target_fit": target_fit,
                                "alternative_fit": alternative_fit,
                                "wav2vec_observed": None,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    emitted += 1
                    per_contrast[f"{target}->{alternative}"] += 1
                if limit is not None and emitted >= limit:
                    break
    finally:
        con.close()

    return {
        "split": split,
        "scanned": scanned,
        "emitted": emitted,
        "per_contrast": dict(sorted(per_contrast.items())),
    }


def _expected_phones_for_targets(con: sqlite3.Connection, targets: set[str]) -> list[str]:
    from assess.stats import short_identity  # noqa: PLC0415
    phones: list[str] = []
    for row in con.execute(
        "SELECT DISTINCT expected_phone FROM phone_features WHERE expected_phone IS NOT NULL"
    ):
        phone = row["expected_phone"]
        if short_identity(phone) in targets:
            phones.append(phone)
    return phones


def _iter_rows(con: sqlite3.Connection, expected_phones: list[str]):
    if not expected_phones:
        return
    placeholders = ",".join("?" for _ in expected_phones)
    query = (
        "SELECT speaker_id, expected_phone, f1_hz, f2_hz, f3_hz "
        "FROM phone_features "
        f"WHERE expected_phone IN ({placeholders}) "
        "AND feature_confidence > 0 "
        "AND f1_hz IS NOT NULL AND f2_hz IS NOT NULL AND f3_hz IS NOT NULL"
    )
    yield from con.execute(query, expected_phones)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationConfig:
    native_fp_ceiling: float = 0.02
    wav2vec_cross_confusion_ceiling: float = 0.03
    atlas_margin_floor: float = 0.10
    min_samples: int = 25
    override_whitelist: set[tuple[str, str]] = field(default_factory=set)
    scope: Literal["all", "fixed"] = "all"

    def as_dict(self) -> dict[str, object]:
        return {
            "native_fp_ceiling": self.native_fp_ceiling,
            "wav2vec_cross_confusion_ceiling": self.wav2vec_cross_confusion_ceiling,
            "atlas_margin_floor": self.atlas_margin_floor,
            "min_samples": self.min_samples,
            "override_whitelist": [
                {"target": t, "alternative": a}
                for t, a in sorted(self.override_whitelist)
            ],
            "scope": self.scope,
        }


@dataclass(frozen=True)
class CalibrationObservation:
    speaker_id: str
    target: str
    alternative: str
    target_fit: float
    alternative_fit: float
    wav2vec_observed: str | None

    @property
    def margin(self) -> float:
        return self.target_fit - self.alternative_fit


def calibration_observations_from_validation_rows(
    rows: list[dict[str, Any]],
) -> list[CalibrationObservation]:
    """Adapt validation-batch observation rows to calibration observations.

    Each ``val_obs.jsonl`` row carries the GMM atlas label (target/alt
    typicality, best_match) and the wav2vec observed phone. The calibration
    uses GMM typicality as the fit signal so it stays aligned with what the
    runtime acoustic analysis evaluates.
    """
    observations: list[CalibrationObservation] = []
    for row in rows:
        atlas = row.get("atlas") if isinstance(row.get("atlas"), dict) else None
        wav2vec = row.get("wav2vec") if isinstance(row.get("wav2vec"), dict) else None
        if not atlas:
            continue
        target = atlas.get("target_phone")
        alternative = atlas.get("best_match")
        target_fit = atlas.get("target_typicality")
        alternative_fit = atlas.get("best_match_typicality")
        if not target or not alternative:
            continue
        if target == alternative:
            continue
        if target_fit is None or alternative_fit is None:
            continue
        observed = wav2vec.get("observed_phone") if wav2vec else None
        observations.append(
            CalibrationObservation(
                speaker_id=str(row.get("speaker_id") or ""),
                target=str(target),
                alternative=str(alternative),
                target_fit=float(target_fit),
                alternative_fit=float(alternative_fit),
                wav2vec_observed=str(observed) if observed is not None else None,
            )
        )
    return observations


@dataclass(frozen=True)
class CalibratedContrast:
    target: str
    alternative: str
    wav2vec_reliable: bool
    atlas_separable: bool
    atlas_override_allowed: bool
    native_fp_rate: float
    n_samples: int
    median_margin: float | None
    wav2vec_cross_confusion_rate: float | None

    def authority_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "alternative": self.alternative,
            "wav2vec_reliable": self.wav2vec_reliable,
            "atlas_separable": self.atlas_separable,
            "atlas_override_allowed": self.atlas_override_allowed,
            "native_fp_rate": self.native_fp_rate,
        }

    def diagnostic_dict(self) -> dict[str, object]:
        return {
            **self.authority_dict(),
            "n_samples": self.n_samples,
            "median_margin": self.median_margin,
            "wav2vec_cross_confusion_rate": self.wav2vec_cross_confusion_rate,
        }


@dataclass(frozen=True)
class CalibrationResult:
    generated_by: str
    fp_ceiling: float
    contrasts: list[CalibratedContrast]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_by": self.generated_by,
            "fp_ceiling": self.fp_ceiling,
            "contrasts": [c.authority_dict() for c in self.contrasts],
        }


def calibrate_authority(
    observations: list[CalibrationObservation],
    config: CalibrationConfig,
    *,
    generated_by: str = "assess.validation.calibration",
) -> CalibrationResult:
    if config.scope == "fixed":
        observations = [
            o for o in observations
            if (o.target, o.alternative) in FIXED_SAME_CLASS_CONTRASTS
        ]

    by_contrast: dict[tuple[str, str], list[CalibrationObservation]] = defaultdict(list)
    for obs in observations:
        by_contrast[(obs.target, obs.alternative)].append(obs)

    fixed_force_reliable = config.scope == "fixed"
    if fixed_force_reliable:
        for pair in FIXED_SAME_CLASS_CONTRASTS:
            by_contrast.setdefault(pair, [])

    contrasts: list[CalibratedContrast] = []
    for (target, alternative), obs_list in sorted(by_contrast.items()):
        n = len(obs_list)
        if n < config.min_samples:
            # In fixed scope, ensure every curated contrast lands in the
            # authority even when native CV had too few observations to
            # calibrate it (e.g., curated-only contrasts). Default to
            # wav2vec_reliable=True (curated as learner-relevant) and
            # atlas_separable=False (no atlas evidence either way).
            if (
                config.scope == "fixed"
                and (target, alternative) in FIXED_SAME_CLASS_CONTRASTS
            ):
                contrasts.append(
                    CalibratedContrast(
                        target=target, alternative=alternative,
                        wav2vec_reliable=True, atlas_separable=False,
                        atlas_override_allowed=False,
                        native_fp_rate=0.0, n_samples=n,
                        median_margin=None,
                        wav2vec_cross_confusion_rate=None,
                    )
                )
            continue

        in_whitelist = (target, alternative) in config.override_whitelist
        margins = [o.margin for o in obs_list]
        med_margin = median(margins) if margins else None
        atlas_separable = bool(
            med_margin is not None and med_margin >= config.atlas_margin_floor
        )
        atlas_override = in_whitelist or atlas_separable

        # Count native FP: observations where target fits worse than alternative
        native_fp_count = sum(1 for o in obs_list if o.margin < 0)
        native_fp_rate = native_fp_count / n

        # wav2vec cross-confusion
        wav2vec_obs = [
            o for o in obs_list if o.wav2vec_observed is not None
        ]
        if wav2vec_obs:
            cross_confusion = sum(
                1 for o in wav2vec_obs
                if o.wav2vec_observed == alternative
            ) / len(wav2vec_obs)
        else:
            cross_confusion = None

        if (
            fixed_force_reliable
            and (target, alternative) in FIXED_SAME_CLASS_CONTRASTS
        ):
            wav2vec_reliable = True
        else:
            wav2vec_reliable = bool(
                cross_confusion is not None
                and cross_confusion <= config.wav2vec_cross_confusion_ceiling
            )
        atlas_override_allowed = atlas_override and native_fp_rate <= config.native_fp_ceiling

        contrasts.append(
            CalibratedContrast(
                target=target,
                alternative=alternative,
                wav2vec_reliable=wav2vec_reliable,
                atlas_separable=atlas_separable,
                atlas_override_allowed=atlas_override_allowed,
                native_fp_rate=native_fp_rate,
                n_samples=n,
                median_margin=med_margin,
                wav2vec_cross_confusion_rate=cross_confusion,
            )
        )

    return CalibrationResult(
        generated_by=generated_by,
        fp_ceiling=config.native_fp_ceiling,
        contrasts=contrasts,
    )


def write_authority_json(result: CalibrationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_diagnostics_csv(result: CalibrationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not result.contrasts:
        path.write_text("", encoding="utf-8")
        return
    fields = list(result.contrasts[0].diagnostic_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for contrast in result.contrasts:
            writer.writerow(contrast.diagnostic_dict())


# ---------------------------------------------------------------------------
# Validation reports
# ---------------------------------------------------------------------------

FLAGGED_TARGET_STATUSES = frozenset({"incorrect", "missing"})


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def hard_false_positive_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_phone: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hard_fp": 0})
    total = 0
    hard_fp = 0
    for row in rows:
        phone = str(row.get("target_phone") or "unknown")
        status = str(row.get("status") or "")
        total += 1
        by_phone[phone]["total"] += 1
        if status in FLAGGED_TARGET_STATUSES:
            hard_fp += 1
            by_phone[phone]["hard_fp"] += 1

    return {
        "overall": {
            "total": total,
            "hard_fp": hard_fp,
            "hard_fp_rate": _rate(hard_fp, total),
        },
        "by_phone": {
            phone: {
                "total": counts["total"],
                "hard_fp": counts["hard_fp"],
                "hard_fp_rate": _rate(counts["hard_fp"], counts["total"]),
            }
            for phone, counts in sorted(by_phone.items())
        },
    }


def unscored_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    unscored = 0
    reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "unscored":
            continue
        unscored += 1
        for reason in row.get("reasons") or []:
            reasons[str(reason)] += 1
    return {
        "overall": {
            "total": total,
            "unscored": unscored,
            "unscored_rate": _rate(unscored, total),
        },
        "reasons": dict(sorted(reasons.items())),
    }


def model_comparison_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    models = sorted({str(row.get("model_alias") or "unknown") for row in rows})
    out: list[dict[str, Any]] = []
    for model in models:
        subset = [row for row in rows if str(row.get("model_alias") or "unknown") == model]
        total = len(subset)
        hard_fp = sum(1 for row in subset if row.get("status") in FLAGGED_TARGET_STATUSES)
        low = sum(1 for row in subset if row.get("status") == "unscored")
        out.append({
            "model_alias": model,
            "total": total,
            "hard_fp_rate": _rate(hard_fp, total),
            "unscored_rate": _rate(low, total),
        })
    return {"models": out}


def atlas_override_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_contrast: dict[str, dict[str, int]] = defaultdict(lambda: {"override_count": 0})
    total = len(rows)
    override_count = 0
    for row in rows:
        if row.get("analysis_source") not in {"pairwise_contrast", "gmm_nearest", "duration"}:
            continue
        override_count += 1
        target = str(row.get("target_phone") or "unknown")
        alternative = str(row.get("observed_phone") or "unknown")
        by_contrast[f"{target}->{alternative}"]["override_count"] += 1
    return {
        "overall": {
            "total": total,
            "override_count": override_count,
            "override_rate": _rate(override_count, total),
        },
        "by_contrast": dict(sorted(by_contrast.items())),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(title: str, report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}", "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        "```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch validation runner
# ---------------------------------------------------------------------------

_PROGRESS_EVERY = 50


def run_validation_batch(
    *,
    manifest: Path,
    split: str,
    model_alias: str,
    stats_path: str,
    authority_path: str,
    output: Path,
    audio_root: Path | None = None,
    limit: int | None = None,
    resume: bool = True,
    services: object | None = None,
    calibration_path: str | None = None,
) -> dict[str, int]:
    from assess.runtime import run_coach  # noqa: PLC0415
    from assess.contracts import CoachRequest  # noqa: PLC0415

    rows = _load_manifest_rows(manifest, split=split, limit=limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = _progress_path(output)

    if resume:
        done_segments = _prepare_resume(output) | _prepare_progress(progress)
    else:
        done_segments = set()
        for path in (output, progress):
            if path.exists():
                path.unlink()

    pending = [row for row in rows if _segment_key(row) not in done_segments]
    skipped = len(rows) - len(pending)

    runtime_services = services or _build_batch_services(
        model_alias=model_alias,
        stats_path=stats_path,
        authority_path=authority_path,
        calibration_path=calibration_path,
    )

    counts = {"total": len(rows), "skipped": skipped, "processed": 0, "errors": 0}
    with output.open("a", encoding="utf-8") as fh, progress.open("a", encoding="utf-8") as pfh:
        for i, row in enumerate(pending):
            segment_key = _segment_key(row)
            audio_path = _resolve_audio_path(row, audio_root)
            try:
                request = CoachRequest(
                    audio_path=str(audio_path),
                    model_alias=model_alias,
                    target_text=str(row.get("text") or row.get("transcript") or ""),
                    stats_path=stats_path,
                    authority_path=authority_path,
                    calibration_path=calibration_path,
                )
                result = run_coach(request, services=runtime_services)
                payload = result.as_dict()
                result_block = payload.get("result")
                result_phones = (
                    result_block.get("phones")
                    if isinstance(result_block, dict)
                    else []
                )
                for phone in result_phones if isinstance(result_phones, list) else []:
                    if not isinstance(phone, dict):
                        continue
                    d = dict(phone)
                    d["model_alias"] = model_alias
                    d["segment_key"] = segment_key
                    d["speaker_id"] = str(row.get("speaker_id") or "")
                    d["target_phone"] = d.get("expected")
                    d["observed_phone"] = d.get("observed")
                    reason = d.get("reason")
                    d["reasons"] = [reason] if isinstance(reason, str) and reason else []
                    changed = d.get("changed_by_analysis")
                    source = None
                    if isinstance(changed, dict):
                        source = changed.get("source")
                    d["analysis_source"] = source or "alignment"
                    d["is_flagged"] = int(
                        str(d.get("status") or "") in FLAGGED_TARGET_STATUSES
                    )
                    fh.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:
                counts["errors"] += 1
                pfh.write(
                    json.dumps(
                        {"segment_key": segment_key, "error": str(exc)},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            else:
                pfh.write(
                    json.dumps({"segment_key": segment_key, "ok": True}, ensure_ascii=False)
                    + "\n"
                )
            counts["processed"] += 1
            if (i + 1) % _PROGRESS_EVERY == 0:
                print(
                    f"  {i + 1}/{len(pending)} processed, {counts['errors']} errors",
                    flush=True,
                )

    return counts


def _load_manifest_rows(manifest: Path, *, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            speaker_id = str(row.get("speaker_id") or "")
            if speaker_id and split_for_speaker(speaker_id) != split:
                continue
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _segment_key(row: dict[str, Any]) -> str:
    return str(row.get("segment_id") or row.get("path") or json.dumps(row, sort_keys=True))


def _progress_path(output: Path) -> Path:
    return output.with_suffix(".progress.jsonl")


def _prepare_resume(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done: set[str] = set()
    with output.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                try:
                    row = json.loads(stripped)
                    key = row.get("segment_key")
                    if key:
                        done.add(str(key))
                except json.JSONDecodeError:
                    pass
    return done


def _prepare_progress(progress: Path) -> set[str]:
    if not progress.exists():
        return set()
    done: set[str] = set()
    with progress.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                try:
                    row = json.loads(stripped)
                    key = row.get("segment_key")
                    if key:
                        done.add(str(key))
                except json.JSONDecodeError:
                    pass
    return done


def _resolve_audio_path(row: dict[str, Any], audio_root: Path | None) -> Path:
    raw = str(row.get("path") or row.get("audio_path") or "")
    p = Path(raw)
    if audio_root is not None and not p.is_absolute():
        return audio_root / p
    return p


def _build_batch_services(
    *,
    model_alias: str,
    stats_path: str,
    authority_path: str,
    calibration_path: str | None = None,
) -> object:
    from assess.runtime import build_default_runtime_services  # noqa: PLC0415
    from assess.contracts import CoachRequest  # noqa: PLC0415

    # audio_path is irrelevant to service construction (only model/stats/
    # authority are read), but the request contract requires it non-empty.
    return build_default_runtime_services(
        CoachRequest(
            audio_path="placeholder.wav",
            model_alias=model_alias,
            target_phones=["a"],
            stats_path=stats_path,
            authority_path=authority_path,
            calibration_path=calibration_path,
        )
    )
