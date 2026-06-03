#!/usr/bin/env python3
"""Check DB kar/kâr word spans as profit kâr under both coach systems."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assess.contracts import CoachRequest  # noqa: E402
from assess.runtime import build_default_runtime_services, run_coach  # noqa: E402

AUDIO_DB = Path("/home/onur/Code/telaffuz-yz-thesis-db/audio.sqlite")
CURATED_DB = REPO_ROOT / "artifacts" / "curated_stop_place" / "curated_stop_place_reference.sqlite"
AUDIO_ROOT = Path("/home/onur/Code/telaffuz-yz-audio")

TARGET_PHONES = ["cʰ", "aː", "ɾ̞̊"]
STOP_PLACE_PHONES = {"kʰ", "k", "cʰ", "c"}

SYSTEMS = {
    "default": {
        "stats": REPO_ROOT / "configs" / "coach_gmm_v1.json",
        "authority": REPO_ROOT / "configs" / "decision_authority.json",
        "calibration": REPO_ROOT / "configs" / "coach_gmm_calibration_v1.json",
    },
    "curated_overlay": {
        "stats": REPO_ROOT / "artifacts" / "curated_stop_place" / "coach_gmm_overlay.json",
        "authority": REPO_ROOT / "artifacts" / "curated_stop_place" / "decision_authority_overlay.json",
        "calibration": REPO_ROOT
        / "artifacts"
        / "curated_stop_place"
        / "coach_gmm_calibration_overlay.json",
    },
}


@dataclass
class WordSpan:
    word_occurrence_id: int
    segment_id: int
    word_surface: str
    word_norm: str
    start_ms: int
    end_ms: int
    path: str
    provider: str
    speaker_id: str
    sources: set[str] = field(default_factory=set)

    def source_path(self) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else AUDIO_ROOT / path

    def as_json(self) -> dict[str, Any]:
        item = asdict(self)
        item["sources"] = sorted(self.sources)
        item["audio_path"] = str(self.source_path())
        return item


def _dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    columns = [desc[0] for desc in cursor.description or []]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _load_literal_kar_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cursor = conn.execute(
        """
        SELECT
            wo.id AS word_occurrence_id,
            wo.segment_id,
            wo.word_surface,
            wo.word_norm,
            wo.start_ms,
            wo.end_ms,
            s.path,
            s.provider,
            s.speaker_id
        FROM word_occurrences wo
        JOIN segments s ON s.id = wo.segment_id
        WHERE lower(wo.word_norm) = 'kâr'
        ORDER BY wo.id
        """
    )
    return _dicts(cursor)


def _curated_profit_rows() -> list[dict[str, Any]]:
    with sqlite3.connect(str(CURATED_DB)) as conn:
        cursor = conn.execute(
            """
            SELECT segment_id, start_ms AS phone_start_ms, end_ms AS phone_end_ms
            FROM curation_manifest
            WHERE meaning = 'profit'
              AND phone = 'cʰ'
              AND included_in_training = 1
            ORDER BY id
            """
        )
        return _dicts(cursor)


def _load_context_profit_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for curated in _curated_profit_rows():
        cursor = conn.execute(
            """
            SELECT
                wo.id AS word_occurrence_id,
                wo.segment_id,
                wo.word_surface,
                wo.word_norm,
                wo.start_ms,
                wo.end_ms,
                s.path,
                s.provider,
                s.speaker_id,
                ABS(((wo.start_ms + wo.end_ms) / 2.0) - ?) AS center_distance_ms
            FROM word_occurrences wo
            JOIN segments s ON s.id = wo.segment_id
            WHERE wo.segment_id = ?
              AND lower(wo.word_norm) IN ('kar', 'kâr')
            ORDER BY
              CASE WHEN wo.start_ms <= ? AND wo.end_ms >= ? THEN 0 ELSE 1 END,
              center_distance_ms
            LIMIT 1
            """,
            (
                (curated["phone_start_ms"] + curated["phone_end_ms"]) / 2.0,
                curated["segment_id"],
                curated["phone_start_ms"],
                curated["phone_end_ms"],
            ),
        )
        match = cursor.fetchone()
        if match is None:
            continue
        columns = [desc[0] for desc in cursor.description or []]
        row = dict(zip(columns, match, strict=True))
        row.pop("center_distance_ms", None)
        rows.append(row)
    return rows


def load_word_spans() -> list[WordSpan]:
    spans: dict[int, WordSpan] = {}
    with sqlite3.connect(str(AUDIO_DB)) as conn:
        for row in _load_literal_kar_rows(conn):
            span = WordSpan(**row)
            span.sources.add("literal_kâr")
            spans[span.word_occurrence_id] = span
        for row in _load_context_profit_rows(conn):
            key = int(row["word_occurrence_id"])
            if key not in spans:
                spans[key] = WordSpan(**row)
            spans[key].sources.add("curated_profit_context")
    return sorted(spans.values(), key=lambda item: item.word_occurrence_id)


def crop_word(span: WordSpan, out_dir: Path) -> Path:
    source = span.source_path()
    audio, sample_rate = sf.read(str(source), dtype="float32")
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    start = max(0, int(round(span.start_ms * sample_rate / 1000.0)))
    end = min(samples.shape[0], int(round(span.end_ms * sample_rate / 1000.0)))
    if end <= start:
        raise ValueError(f"invalid crop for word occurrence {span.word_occurrence_id}")
    clip = samples[start:end]
    out = out_dir / f"kar-profit-{span.word_occurrence_id}.wav"
    sf.write(str(out), clip, sample_rate)
    return out


def _pairwise_candidate(
    payload: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any] | None:
    debug = payload.get("debug")
    if not isinstance(debug, dict):
        return None
    candidates = debug.get("analysis_candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if (
            isinstance(candidate, dict)
            and candidate.get("index") == index
            and candidate.get("source") == "pairwise_contrast"
            and candidate.get("from") in {"kʰ", "k", "cʰ", "c"}
            and candidate.get("to") in {"kʰ", "k", "cʰ", "c"}
        ):
            return candidate
    return None


def _first_relevant_stop(phones: object) -> tuple[int | None, dict[str, Any]]:
    if not isinstance(phones, list):
        return None, {}
    for index, phone in enumerate(phones):
        if not isinstance(phone, dict):
            continue
        raw = phone.get("raw_ipa")
        final = phone.get("ipa")
        if raw in STOP_PLACE_PHONES or final in STOP_PLACE_PHONES:
            return index, phone
    return None, {}


def _run_system(
    *,
    clip_path: Path,
    system_name: str,
    services: Any,
) -> dict[str, Any]:
    config = SYSTEMS[system_name]
    request = CoachRequest(
        audio_path=str(clip_path),
        model_alias="mms-1b",
        target_phones=list(TARGET_PHONES),
        stats_path=str(config["stats"]),
        authority_path=str(config["authority"]),
        calibration_path=str(config["calibration"]),
    )
    payload = run_coach(request, services=services).as_dict()
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    phones = analysis.get("phones") if isinstance(analysis, dict) else []
    first_phone = phones[0] if isinstance(phones, list) and phones else {}
    stop_index, stop_phone = _first_relevant_stop(phones)
    candidate = (
        _pairwise_candidate(payload, index=stop_index)
        if stop_index is not None
        else None
    )
    evidence = candidate.get("evidence", {}) if isinstance(candidate, dict) else {}
    feature_values = (
        evidence.get("feature_values", {}) if isinstance(evidence, dict) else {}
    )
    return {
        "status": payload.get("status"),
        "analysis_ipa": analysis.get("ipa") if isinstance(analysis, dict) else None,
        "analysis_phonemes": analysis.get("phonemes") if isinstance(analysis, dict) else [],
        "first_raw_phone": first_phone.get("raw_ipa") if isinstance(first_phone, dict) else None,
        "first_final_phone": first_phone.get("ipa") if isinstance(first_phone, dict) else None,
        "first_changed": first_phone.get("changed_by_analysis") if isinstance(first_phone, dict) else None,
        "first_confidence": first_phone.get("confidence") if isinstance(first_phone, dict) else None,
        "first_quality": first_phone.get("quality") if isinstance(first_phone, dict) else None,
        "detected_cʰ": first_phone.get("ipa") == "cʰ" if isinstance(first_phone, dict) else False,
        "relevant_stop_index": stop_index,
        "relevant_stop_raw_phone": stop_phone.get("raw_ipa") if isinstance(stop_phone, dict) else None,
        "relevant_stop_final_phone": stop_phone.get("ipa") if isinstance(stop_phone, dict) else None,
        "relevant_stop_changed": stop_phone.get("changed_by_analysis") if isinstance(stop_phone, dict) else None,
        "relevant_stop_confidence": stop_phone.get("confidence") if isinstance(stop_phone, dict) else None,
        "relevant_stop_quality": stop_phone.get("quality") if isinstance(stop_phone, dict) else None,
        "relevant_stop_detected_cʰ": stop_phone.get("ipa") == "cʰ" if isinstance(stop_phone, dict) else False,
        "pairwise_to": candidate.get("to") if isinstance(candidate, dict) else None,
        "pairwise_passed": candidate.get("passed") if isinstance(candidate, dict) else None,
        "pairwise_applied": candidate.get("applied") if isinstance(candidate, dict) else None,
        "pairwise_reasons": candidate.get("reasons", []) if isinstance(candidate, dict) else [],
        "pairwise_margin": evidence.get("margin") if isinstance(evidence, dict) else None,
        "pairwise_score": evidence.get("score") if isinstance(evidence, dict) else None,
        "pairwise_threshold": evidence.get("threshold") if isinstance(evidence, dict) else None,
        "feature_values": feature_values,
    }


def build_services() -> dict[str, Any]:
    services: dict[str, Any] = {}
    for name, config in SYSTEMS.items():
        request = CoachRequest(
            audio_path="placeholder.wav",
            model_alias="mms-1b",
            target_phones=list(TARGET_PHONES),
            stats_path=str(config["stats"]),
            authority_path=str(config["authority"]),
            calibration_path=str(config["calibration"]),
        )
        services[name] = build_default_runtime_services(request)
    return services


def write_reports(records: list[dict[str, Any]]) -> None:
    out_json = REPO_ROOT / "reports" / "assessment" / "database-kar-profit-both-systems.json"
    out_md = REPO_ROOT / "reports" / "assessment" / "database-kar-profit-both-systems.md"
    out_csv = REPO_ROOT / "reports" / "assessment" / "database-kar-profit-both-systems.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for system_name in SYSTEMS:
        results = [record["systems"][system_name] for record in records]
        summary[system_name] = {
            "clips": len(results),
            "first_final_cʰ": sum(1 for result in results if result["first_final_phone"] == "cʰ"),
            "first_raw_cʰ": sum(1 for result in results if result["first_raw_phone"] == "cʰ"),
            "first_raw_kʰ": sum(1 for result in results if result["first_raw_phone"] == "kʰ"),
            "changed_to_cʰ": sum(
                1
                for result in results
                if result["first_raw_phone"] != "cʰ" and result["first_final_phone"] == "cʰ"
            ),
            "relevant_stop_final_cʰ": sum(
                1 for result in results if result["relevant_stop_final_phone"] == "cʰ"
            ),
            "relevant_stop_raw_cʰ": sum(
                1 for result in results if result["relevant_stop_raw_phone"] == "cʰ"
            ),
            "relevant_stop_raw_kʰ": sum(
                1 for result in results if result["relevant_stop_raw_phone"] == "kʰ"
            ),
            "relevant_stop_missing": sum(
                1 for result in results if result["relevant_stop_raw_phone"] is None
            ),
            "relevant_stop_changed_to_cʰ": sum(
                1
                for result in results
                if result["relevant_stop_raw_phone"] != "cʰ"
                and result["relevant_stop_final_phone"] == "cʰ"
            ),
        }

    payload = {
        "target_context": "profit kâr",
        "target_phones": TARGET_PHONES,
        "systems": {
            name: {key: str(value) for key, value in config.items()}
            for name, config in SYSTEMS.items()
        },
        "summary": summary,
        "records": records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "word_occurrence_id",
        "segment_id",
        "word_surface",
        "word_norm",
        "provider",
        "sources",
        "default_raw",
        "default_final",
        "default_margin",
        "curated_raw",
        "curated_final",
        "curated_margin",
        "default_stop_raw",
        "default_stop_final",
        "default_stop_margin",
        "curated_stop_raw",
        "curated_stop_final",
        "curated_stop_margin",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            default = record["systems"]["default"]
            curated = record["systems"]["curated_overlay"]
            writer.writerow(
                {
                    "word_occurrence_id": record["word_occurrence_id"],
                    "segment_id": record["segment_id"],
                    "word_surface": record["word_surface"],
                    "word_norm": record["word_norm"],
                    "provider": record["provider"],
                    "sources": ",".join(record["sources"]),
                    "default_raw": default["first_raw_phone"],
                    "default_final": default["first_final_phone"],
                    "default_margin": default["pairwise_margin"],
                    "curated_raw": curated["first_raw_phone"],
                    "curated_final": curated["first_final_phone"],
                    "curated_margin": curated["pairwise_margin"],
                    "default_stop_raw": default["relevant_stop_raw_phone"],
                    "default_stop_final": default["relevant_stop_final_phone"],
                    "default_stop_margin": default["pairwise_margin"],
                    "curated_stop_raw": curated["relevant_stop_raw_phone"],
                    "curated_stop_final": curated["relevant_stop_final_phone"],
                    "curated_stop_margin": curated["pairwise_margin"],
                }
            )

    lines = [
        "# Database kar/kâr profit-context check",
        "",
        "Target forced to profit `kâr`: `cʰ aː ɾ̞̊`.",
        "",
        "## Summary",
        "",
        "| system | clips | relevant raw cʰ | relevant raw kʰ | relevant final cʰ | changed to cʰ | no k/c stop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        lines.append(
            "| {name} | {clips} | {raw_c} | {raw_k} | {final_c} | {changed} | {missing} |".format(
                name=name,
                clips=item["clips"],
                raw_c=item["relevant_stop_raw_cʰ"],
                raw_k=item["relevant_stop_raw_kʰ"],
                final_c=item["relevant_stop_final_cʰ"],
                changed=item["relevant_stop_changed_to_cʰ"],
                missing=item["relevant_stop_missing"],
            )
        )
    lines.extend(
        [
            "",
            "## Per Clip",
            "",
            "| id | source | provider | word | default decoded | default k/c stop | overlay decoded | overlay k/c stop | overlay margin |",
            "|---:|---|---|---|---|---|---|---|---:|",
        ]
    )
    for record in records:
        default = record["systems"]["default"]
        curated = record["systems"]["curated_overlay"]
        lines.append(
            "| {id} | {sources} | {provider} | `{word}` | {d_decoded} | {d_stop} | {c_decoded} | {c_stop} | {margin} |".format(
                id=record["word_occurrence_id"],
                sources=", ".join(record["sources"]),
                provider=record["provider"],
                word=record["word_surface"].strip(),
                d_decoded=" ".join(default["analysis_phonemes"]),
                d_stop=f"{default['relevant_stop_raw_phone']}→{default['relevant_stop_final_phone']}",
                c_decoded=" ".join(curated["analysis_phonemes"]),
                c_stop=f"{curated['relevant_stop_raw_phone']}→{curated['relevant_stop_final_phone']}",
                margin=curated["pairwise_margin"],
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(out_md)}, ensure_ascii=False, indent=2))


def main() -> int:
    spans = load_word_spans()
    if not spans:
        raise SystemExit("no kar/kâr DB spans found")
    services = build_services()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="kar_profit_db_") as tmp:
        tmp_dir = Path(tmp)
        for span in spans:
            clip = crop_word(span, tmp_dir)
            record = span.as_json()
            record["duration_ms"] = span.end_ms - span.start_ms
            record["systems"] = {}
            for system_name, service in services.items():
                record["systems"][system_name] = _run_system(
                    clip_path=clip,
                    system_name=system_name,
                    services=service,
                )
            records.append(record)
    write_reports(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
