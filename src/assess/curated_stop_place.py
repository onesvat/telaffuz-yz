"""Curated stop-place overlay helpers for /k/-/c/ variants.

This module deliberately builds an experimental overlay: production artifacts
remain the base, and only curated rows for kʰ/cʰ/k/c are allowed to replace
their matching model entries.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STOP_PLACE_PHONES: tuple[str, ...] = ("kʰ", "cʰ", "k", "c")
STOP_PLACE_PHONE_SET: frozenset[str] = frozenset(STOP_PLACE_PHONES)

CSV_FIELDS: tuple[str, ...] = (
    "segment_id",
    "audio_path",
    "start_ms",
    "end_ms",
    "phone",
    "word",
    "meaning",
    "prev_phone",
    "next_phone",
    "speaker_id",
    "provider",
    "split",
    "label_source",
    "status",
    "notes",
)

ALLOWED_STATUSES: frozenset[str] = frozenset(
    {"trusted_auto", "accepted", "uncertain", "reject"}
)
TRAINING_STATUSES: frozenset[str] = frozenset({"trusted_auto", "accepted"})

DEFAULT_REFERENCE_FEATURE_VERSION = "assess_coach_reference_v1"
CURATED_REFERENCE_FEATURE_VERSION = "curated_stop_place_reference_v1"

_REF_ID_RE = re.compile(r"(?:^|[;,\s])ref_feature_id=(\d+)(?:$|[;,\s])")
_PUNCT_RE = re.compile(r"^[^\wçğıöşüÇĞİÖŞÜâîûÂÎÛ]+|[^\wçğıöşüÇĞİÖŞÜâîûÂÎÛ]+$")

_PROFIT_STRONG_CUES: frozenset[str] = frozenset(
    {
        "bilanço",
        "bilanco",
        "borsa",
        "brüt",
        "brut",
        "ciro",
        "ekonomi",
        "ekonomik",
        "elde",
        "faiz",
        "finans",
        "firma",
        "gelir",
        "geliri",
        "gelirleri",
        "hisse",
        "kazanç",
        "kazanc",
        "kazancı",
        "kazanci",
        "kazançları",
        "kazançlari",
        "mali",
        "marj",
        "net",
        "oran",
        "pay",
        "piyasa",
        "satış",
        "satis",
        "sermaye",
        "şirket",
        "sirket",
        "ticaret",
        "vergi",
        "yatırım",
        "yatirim",
        "zarar",
        "zararı",
        "zarari",
    }
)
_PROFIT_AFTER_CUES: frozenset[str] = frozenset(
    {"etti", "etmek", "ediyor", "eden", "eder", "elde", "sağladı", "sagladi"}
)
_SNOW_STRONG_CUES: frozenset[str] = frozenset(
    {
        "beyaz",
        "buz",
        "dağ",
        "dag",
        "don",
        "eridi",
        "erime",
        "hava",
        "kayak",
        "kış",
        "kis",
        "soğuk",
        "soguk",
        "tipi",
        "yağan",
        "yagan",
        "yağar",
        "yagar",
        "yağacak",
        "yagacak",
        "yağdı",
        "yagdi",
        "yağıyor",
        "yagiyor",
        "yağış",
        "yagis",
        "yağışı",
        "yagisi",
    }
)
_SNOW_AFTER_CUES: frozenset[str] = frozenset(
    {"yağdı", "yagdi", "yağıyor", "yagiyor", "yağacak", "yagacak", "yağar", "yagar"}
)


class CurationError(ValueError):
    """Raised when a curation CSV or source DB cannot be used safely."""


@dataclass(frozen=True)
class KarContext:
    meaning: str
    phone: str | None
    status: str
    label_source: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CurationRow:
    segment_id: int
    audio_path: str
    start_ms: int
    end_ms: int
    phone: str
    word: str
    meaning: str
    prev_phone: str
    next_phone: str
    speaker_id: str
    provider: str
    split: str
    label_source: str
    status: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        return {
            "segment_id": str(self.segment_id),
            "audio_path": self.audio_path,
            "start_ms": str(self.start_ms),
            "end_ms": str(self.end_ms),
            "phone": self.phone,
            "word": self.word,
            "meaning": self.meaning,
            "prev_phone": self.prev_phone,
            "next_phone": self.next_phone,
            "speaker_id": self.speaker_id,
            "provider": self.provider,
            "split": self.split,
            "label_source": self.label_source,
            "status": self.status,
            "notes": self.notes,
        }

    @property
    def training_eligible(self) -> bool:
        return self.status in TRAINING_STATUSES


@dataclass(frozen=True)
class CuratedBuildReport:
    output_db: str
    manifest_rows: int
    training_rows: int
    inserted_rows: int
    skipped_rows: int
    per_phone: dict[str, int]
    per_phone_speakers: dict[str, int]
    split_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "output_db": self.output_db,
            "manifest_rows": self.manifest_rows,
            "training_rows": self.training_rows,
            "inserted_rows": self.inserted_rows,
            "skipped_rows": self.skipped_rows,
            "per_phone": dict(self.per_phone),
            "per_phone_speakers": dict(self.per_phone_speakers),
            "split_counts": dict(self.split_counts),
        }


def _lower_tr(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize_word(text: str) -> str:
    """Turkish-aware lowercasing plus light punctuation stripping."""
    text = _lower_tr(text.strip())
    text = text.replace("'", "")
    return _PUNCT_RE.sub("", text)


def split_for_speaker(speaker_id: str) -> str:
    if not speaker_id:
        return "train"
    bucket = int(sha1(speaker_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def classify_kar_context(
    words: Sequence[str],
    kar_index: int,
    *,
    surface: str | None = None,
) -> KarContext:
    """Classify exact-word ``kar`` as snow, profit, or uncertain.

    Plain-written ``kar`` is accepted only when nearby lexical cues make the
    sense defensible. Circumflex-written ``kâr`` is accepted as profit.
    """
    if kar_index < 0 or kar_index >= len(words):
        raise IndexError("kar_index out of range")

    norm_words = [normalize_word(word) for word in words]
    token = normalize_word(surface or words[kar_index])
    raw_surface = surface or words[kar_index]
    if token not in {"kar", "kâr"}:
        return KarContext("", None, "trusted_auto", "g2p_expected_phone", ())

    if "â" in raw_surface or token == "kâr":
        return KarContext(
            "profit",
            "cʰ",
            "trusted_auto",
            "kar_context_profit_circumflex",
            ("circumflex_written_kâr",),
        )

    lo = max(0, kar_index - 4)
    hi = min(len(norm_words), kar_index + 5)
    window = norm_words[lo:hi]
    prev_word = norm_words[kar_index - 1] if kar_index > 0 else ""
    next_word = norm_words[kar_index + 1] if kar_index + 1 < len(norm_words) else ""

    profit = bool(_PROFIT_STRONG_CUES & set(window))
    snow = bool(_SNOW_STRONG_CUES & set(window))
    if next_word in _PROFIT_AFTER_CUES:
        profit = True
    if next_word in _SNOW_AFTER_CUES or prev_word in {"yağan", "yagan"}:
        snow = True

    if profit and not snow:
        return KarContext(
            "profit",
            "cʰ",
            "trusted_auto",
            "kar_context_profit",
            ("plain_kar_profit_context",),
        )
    if snow and not profit:
        return KarContext(
            "snow",
            "kʰ",
            "trusted_auto",
            "kar_context_snow",
            ("plain_kar_snow_context",),
        )
    if profit and snow:
        return KarContext(
            "",
            None,
            "uncertain",
            "kar_context_conflict",
            ("plain_kar_conflicting_context",),
        )
    return KarContext(
        "",
        None,
        "uncertain",
        "kar_context_uncertain",
        ("plain_kar_no_high_confidence_context",),
    )


def parse_curation_csv(path: Path) -> list[CurationRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [field for field in CSV_FIELDS if field not in fieldnames]
        if missing:
            raise CurationError(f"curation CSV missing columns: {', '.join(missing)}")
        rows = [_row_from_mapping(raw, line_no=idx + 2) for idx, raw in enumerate(reader)]
    return rows


def write_curation_csv(rows: Iterable[CurationRow], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _row_from_mapping(raw: Mapping[str, object], *, line_no: int) -> CurationRow:
    def field(name: str) -> str:
        value = raw.get(name)
        return "" if value is None else str(value)

    phone = field("phone").strip()
    if phone not in STOP_PLACE_PHONE_SET:
        raise CurationError(
            f"line {line_no}: phone must be one of {', '.join(STOP_PLACE_PHONES)}"
        )
    status = field("status").strip()
    if status not in ALLOWED_STATUSES:
        raise CurationError(
            f"line {line_no}: status must be one of {', '.join(sorted(ALLOWED_STATUSES))}"
        )
    try:
        segment_id = int(field("segment_id"))
        start_ms = int(field("start_ms"))
        end_ms = int(field("end_ms"))
    except ValueError as exc:
        raise CurationError(f"line {line_no}: segment_id/start_ms/end_ms must be integers") from exc
    if end_ms <= start_ms:
        raise CurationError(f"line {line_no}: end_ms must be greater than start_ms")
    return CurationRow(
        segment_id=segment_id,
        audio_path=field("audio_path"),
        start_ms=start_ms,
        end_ms=end_ms,
        phone=phone,
        word=field("word"),
        meaning=field("meaning"),
        prev_phone=field("prev_phone"),
        next_phone=field("next_phone"),
        speaker_id=field("speaker_id"),
        provider=field("provider"),
        split=field("split") or split_for_speaker(field("speaker_id")),
        label_source=field("label_source"),
        status=status,
        notes=field("notes"),
    )


def curation_summary(rows: Sequence[CurationRow]) -> dict[str, object]:
    per_phone = {phone: 0 for phone in STOP_PLACE_PHONES}
    per_phone_speakers: dict[str, set[str]] = {phone: set() for phone in STOP_PLACE_PHONES}
    split_counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    status_counts: dict[str, int] = {status: 0 for status in sorted(ALLOWED_STATUSES)}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
        if not row.training_eligible:
            continue
        per_phone[row.phone] = per_phone.get(row.phone, 0) + 1
        if row.speaker_id:
            per_phone_speakers.setdefault(row.phone, set()).add(row.speaker_id)
        split = row.split or split_for_speaker(row.speaker_id)
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "rows": len(rows),
        "training_rows": sum(per_phone.values()),
        "status_counts": status_counts,
        "per_phone": per_phone,
        "per_phone_speakers": {
            phone: len(speakers) for phone, speakers in per_phone_speakers.items()
        },
        "split_counts": split_counts,
    }


def mine_stop_place_candidates(
    *,
    audio_db: Path,
    reference_db: Path,
    audio_root: Path,
    feature_version: str = DEFAULT_REFERENCE_FEATURE_VERSION,
    providers: Sequence[str] = ("common_voice",),
    confidence_floor: float = 0.5,
    limit: int | None = None,
) -> list[CurationRow]:
    """Mine kʰ/cʰ/k/c candidates from the external audio and reference DBs."""
    if not audio_db.exists():
        raise FileNotFoundError(f"audio_db not found: {audio_db}")
    if not reference_db.exists():
        raise FileNotFoundError(f"reference_db not found: {reference_db}")

    conn = sqlite3.connect(f"file:{audio_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS ref", (f"file:{reference_db}?mode=ro",))
        rows = _candidate_query(
            conn,
            feature_version=feature_version,
            providers=tuple(providers),
            confidence_floor=confidence_floor,
            limit=limit,
        )
    finally:
        conn.close()

    candidates: list[CurationRow] = []
    for raw in rows:
        candidates.append(_candidate_from_row(raw, audio_root=audio_root))
    return candidates


def _candidate_query(
    conn: sqlite3.Connection,
    *,
    feature_version: str,
    providers: tuple[str, ...],
    confidence_floor: float,
    limit: int | None,
) -> list[sqlite3.Row]:
    phone_placeholders = ",".join("?" for _ in STOP_PLACE_PHONES)
    provider_clause = ""
    params: list[object] = [feature_version, *STOP_PLACE_PHONES, confidence_floor]
    if providers:
        provider_clause = f"AND pf.provider IN ({','.join('?' for _ in providers)})"
        params.extend(providers)
    limit_clause = "" if limit is None else f"LIMIT {int(limit)}"
    sql = f"""
        WITH timed_words AS (
            SELECT
                wo.*,
                tc.text AS transcript_text,
                tc.is_primary
            FROM word_occurrences wo
            JOIN transcript_candidates tc
              ON tc.id = wo.transcript_candidate_id
        ),
        word_context AS (
            SELECT
                transcript_candidate_id,
                GROUP_CONCAT(word_norm, ' ') AS segment_word_norms
            FROM (
                SELECT transcript_candidate_id, word_norm
                FROM word_occurrences
                ORDER BY transcript_candidate_id, word_index
            )
            GROUP BY transcript_candidate_id
        )
        SELECT
            pf.id AS ref_feature_id,
            pf.segment_alignment_id,
            pf.phone_index,
            pf.expected_phone,
            pf.provider,
            pf.speaker_id,
            pf.start_ms,
            pf.end_ms,
            pf.feature_confidence,
            sa.segment_id,
            sa.alignment_json,
            s.path AS audio_rel_path,
            pw.word_surface,
            pw.word_norm,
            pw.word_index,
            pw.transcript_candidate_id,
            pw.transcript_text,
            pw.is_primary AS transcript_is_primary,
            wc.segment_word_norms,
            CASE
              WHEN pw.id IS NULL THEN 0
              ELSE MIN(pf.end_ms, pw.end_ms) - MAX(pf.start_ms, pw.start_ms)
            END AS overlap_ms
        FROM ref.phone_features pf
        JOIN segment_alignments sa ON sa.id = pf.segment_alignment_id
        JOIN segments s ON s.id = sa.segment_id
        LEFT JOIN timed_words pw
          ON pw.segment_id = sa.segment_id
         AND pf.start_ms < pw.end_ms
         AND pf.end_ms > pw.start_ms
        LEFT JOIN word_context wc
          ON wc.transcript_candidate_id = pw.transcript_candidate_id
        WHERE pf.feature_version = ?
          AND pf.expected_phone IN ({phone_placeholders})
          AND pf.feature_confidence >= ?
          {provider_clause}
        ORDER BY pf.id, pw.is_primary DESC, overlap_ms DESC
        {limit_clause}
    """
    by_ref: dict[int, sqlite3.Row] = {}
    for row in conn.execute(sql, params):
        ref_id = int(row["ref_feature_id"])
        by_ref.setdefault(ref_id, row)
    return list(by_ref.values())


def _candidate_from_row(raw: sqlite3.Row, *, audio_root: Path) -> CurationRow:
    expected_phone = str(raw["expected_phone"])
    word = str(raw["word_norm"] or raw["word_surface"] or "")
    segment_words = str(raw["segment_word_norms"] or word).split()
    word_index = _safe_int(raw["word_index"])
    if word_index is None or word_index >= len(segment_words):
        word_index = segment_words.index(normalize_word(word)) if normalize_word(word) in segment_words else 0

    phone = expected_phone
    meaning = ""
    status = "trusted_auto"
    label_source = "g2p_expected_phone"
    label_notes: list[str] = []
    if normalize_word(word) in {"kar", "kâr"} and expected_phone in {"kʰ", "cʰ"}:
        kar = classify_kar_context(
            segment_words,
            word_index,
            surface=str(raw["word_surface"] or word),
        )
        status = kar.status
        label_source = kar.label_source
        label_notes.extend(kar.notes)
        meaning = kar.meaning
        if kar.phone is not None:
            phone = kar.phone

    prev_phone, next_phone = _prev_next_from_alignment(
        str(raw["alignment_json"] or ""), _safe_int(raw["phone_index"]) or 0
    )
    speaker_id = str(raw["speaker_id"] or "")
    notes = [
        f"ref_feature_id={int(raw['ref_feature_id'])}",
        f"segment_alignment_id={int(raw['segment_alignment_id'])}",
        f"phone_index={int(raw['phone_index'])}",
        f"expected_phone={expected_phone}",
        *label_notes,
    ]
    audio_rel = str(raw["audio_rel_path"] or "")
    return CurationRow(
        segment_id=int(raw["segment_id"]),
        audio_path=str(audio_root / audio_rel) if audio_rel else "",
        start_ms=int(raw["start_ms"]),
        end_ms=int(raw["end_ms"]),
        phone=phone,
        word=word,
        meaning=meaning,
        prev_phone=prev_phone,
        next_phone=next_phone,
        speaker_id=speaker_id,
        provider=str(raw["provider"] or ""),
        split=split_for_speaker(speaker_id),
        label_source=label_source,
        status=status,
        notes=";".join(notes),
    )


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prev_next_from_alignment(alignment_json: str, phone_index: int) -> tuple[str, str]:
    try:
        payload = json.loads(alignment_json)
    except json.JSONDecodeError:
        return "", ""
    phones = payload.get("phones") if isinstance(payload, dict) else None
    if not isinstance(phones, list):
        return "", ""

    def expected_at(index: int) -> str:
        if index < 0 or index >= len(phones):
            return ""
        item = phones[index]
        return str(item.get("expected_phone") or item.get("phone") or "") if isinstance(item, dict) else ""

    return expected_at(phone_index - 1), expected_at(phone_index + 1)


def build_curated_reference_db(
    *,
    curation_csv: Path,
    reference_db: Path,
    audio_db: Path,
    output_db: Path,
    feature_version: str = CURATED_REFERENCE_FEATURE_VERSION,
    replace: bool = False,
) -> CuratedBuildReport:
    """Materialize trusted/accepted curation rows as a coach-compatible DB."""
    rows = parse_curation_csv(curation_csv)
    if output_db.exists():
        if replace:
            output_db.unlink()
        else:
            raise FileExistsError(f"output DB already exists: {output_db}")
    output_db.parent.mkdir(parents=True, exist_ok=True)

    out = sqlite3.connect(str(output_db))
    src = sqlite3.connect(f"file:{reference_db}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        src.execute("ATTACH DATABASE ? AS aud", (f"file:{audio_db}?mode=ro",))
        _create_output_schema(src, out)
        report = _insert_curated_rows(
            rows=rows,
            src=src,
            out=out,
            feature_version=feature_version,
            output_db=output_db,
        )
        out.commit()
    except Exception:
        out.close()
        if output_db.exists():
            output_db.unlink()
        raise
    finally:
        src.close()
        out.close()
    return report


def _create_output_schema(src: sqlite3.Connection, out: sqlite3.Connection) -> None:
    schema_row = src.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'phone_features'"
    ).fetchone()
    if schema_row is None or not schema_row[0]:
        raise CurationError("source reference DB has no phone_features table")
    out.execute(str(schema_row[0]))
    out.execute(
        """
        CREATE TABLE curation_manifest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            audio_path TEXT NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            phone TEXT NOT NULL,
            word TEXT NOT NULL,
            meaning TEXT NOT NULL,
            prev_phone TEXT NOT NULL,
            next_phone TEXT NOT NULL,
            speaker_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            split TEXT NOT NULL,
            label_source TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT NOT NULL,
            source_ref_id INTEGER,
            original_expected_phone TEXT,
            included_in_training INTEGER NOT NULL
        )
        """
    )


def _insert_curated_rows(
    *,
    rows: Sequence[CurationRow],
    src: sqlite3.Connection,
    out: sqlite3.Connection,
    feature_version: str,
    output_db: Path,
) -> CuratedBuildReport:
    columns = [
        str(row[1])
        for row in src.execute("PRAGMA table_info(phone_features)").fetchall()
    ]
    if not columns:
        raise CurationError("source reference DB has no phone_features columns")

    inserted = 0
    skipped = 0
    seen_training_keys: set[tuple[int | None, int, int, str]] = set()
    per_phone: dict[str, int] = {phone: 0 for phone in STOP_PLACE_PHONES}
    speakers: dict[str, set[str]] = {phone: set() for phone in STOP_PLACE_PHONES}
    split_counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}

    for row in rows:
        source_row = _lookup_source_row(src, row)
        source_ref_id = int(source_row["id"]) if source_row is not None else None
        original_expected = (
            str(source_row["expected_phone"]) if source_row is not None else ""
        )
        included = row.training_eligible
        _insert_manifest_row(
            out,
            row,
            source_ref_id=source_ref_id,
            original_expected_phone=original_expected,
            included=included,
        )
        if not included:
            skipped += 1
            continue
        if source_row is None:
            raise CurationError(
                "could not find source feature row for "
                f"segment_id={row.segment_id} start_ms={row.start_ms} end_ms={row.end_ms}"
            )
        key = (source_ref_id, row.start_ms, row.end_ms, row.phone)
        if key in seen_training_keys:
            skipped += 1
            continue
        seen_training_keys.add(key)

        payload = {name: source_row[name] for name in columns}
        payload["feature_version"] = feature_version
        payload["expected_phone"] = row.phone
        if "provider" in payload and row.provider:
            payload["provider"] = row.provider
        if "speaker_id" in payload and row.speaker_id:
            payload["speaker_id"] = row.speaker_id
        _insert_phone_feature(out, columns, payload)
        inserted += 1
        per_phone[row.phone] = per_phone.get(row.phone, 0) + 1
        if row.speaker_id:
            speakers.setdefault(row.phone, set()).add(row.speaker_id)
        split = row.split or split_for_speaker(row.speaker_id)
        split_counts[split] = split_counts.get(split, 0) + 1

    return CuratedBuildReport(
        output_db=str(output_db),
        manifest_rows=len(rows),
        training_rows=sum(1 for row in rows if row.training_eligible),
        inserted_rows=inserted,
        skipped_rows=skipped,
        per_phone=per_phone,
        per_phone_speakers={phone: len(value) for phone, value in speakers.items()},
        split_counts=split_counts,
    )


def _lookup_source_row(src: sqlite3.Connection, row: CurationRow) -> sqlite3.Row | None:
    ref_id = _source_ref_id_from_notes(row.notes)
    if ref_id is not None:
        found = src.execute(
            "SELECT * FROM phone_features WHERE id = ?", (ref_id,)
        ).fetchone()
        if found is None:
            raise CurationError(f"source ref_feature_id not found: {ref_id}")
        return found

    candidates = src.execute(
        f"""
        SELECT pf.*
        FROM phone_features pf
        JOIN aud.segment_alignments sa ON sa.id = pf.segment_alignment_id
        WHERE sa.segment_id = ?
          AND pf.start_ms = ?
          AND pf.end_ms = ?
          AND pf.expected_phone IN ({','.join('?' for _ in STOP_PLACE_PHONES)})
        ORDER BY pf.id
        """,
        (row.segment_id, row.start_ms, row.end_ms, *STOP_PLACE_PHONES),
    ).fetchall()
    if not candidates:
        return None
    if len(candidates) > 1:
        exact = [item for item in candidates if str(item["expected_phone"]) == row.phone]
        if len(exact) == 1:
            return exact[0]
        raise CurationError(
            "ambiguous source feature match for "
            f"segment_id={row.segment_id} start_ms={row.start_ms} end_ms={row.end_ms}"
        )
    return candidates[0]


def _source_ref_id_from_notes(notes: str) -> int | None:
    match = _REF_ID_RE.search(notes)
    return int(match.group(1)) if match else None


def _insert_manifest_row(
    out: sqlite3.Connection,
    row: CurationRow,
    *,
    source_ref_id: int | None,
    original_expected_phone: str,
    included: bool,
) -> None:
    out.execute(
        """
        INSERT INTO curation_manifest (
            segment_id, audio_path, start_ms, end_ms, phone, word, meaning,
            prev_phone, next_phone, speaker_id, provider, split, label_source,
            status, notes, source_ref_id, original_expected_phone,
            included_in_training
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.segment_id,
            row.audio_path,
            row.start_ms,
            row.end_ms,
            row.phone,
            row.word,
            row.meaning,
            row.prev_phone,
            row.next_phone,
            row.speaker_id,
            row.provider,
            row.split,
            row.label_source,
            row.status,
            row.notes,
            source_ref_id,
            original_expected_phone,
            1 if included else 0,
        ),
    )


def _insert_phone_feature(
    out: sqlite3.Connection, columns: Sequence[str], payload: Mapping[str, object]
) -> None:
    placeholders = ",".join("?" for _ in columns)
    column_sql = ",".join(columns)
    values = [payload.get(name) for name in columns]
    out.execute(
        f"INSERT INTO phone_features ({column_sql}) VALUES ({placeholders})",
        values,
    )


def merge_gmm_overlay(
    base: Mapping[str, object],
    replacement: Mapping[str, object],
    *,
    phones: Iterable[str] = STOP_PLACE_PHONES,
    strict: bool = True,
) -> dict[str, object]:
    merged = _json_clone(base)
    base_phones = _phones_payload(merged, artifact_name="base GMM")
    replacement_phones = _phones_payload(replacement, artifact_name="replacement GMM")
    _replace_phone_payloads(base_phones, replacement_phones, phones=phones, strict=strict)
    _mark_overlay(merged, "curated_stop_place_gmm_overlay")
    return merged


def merge_calibration_overlay(
    base: Mapping[str, object],
    replacement: Mapping[str, object],
    *,
    phones: Iterable[str] = STOP_PLACE_PHONES,
    strict: bool = True,
) -> dict[str, object]:
    merged = _json_clone(base)
    base_phones = _phones_payload(merged, artifact_name="base calibration")
    replacement_phones = _phones_payload(
        replacement, artifact_name="replacement calibration"
    )
    _replace_phone_payloads(base_phones, replacement_phones, phones=phones, strict=strict)
    merged["gmm_signature"] = None
    _mark_overlay(merged, "curated_stop_place_calibration_overlay")
    return merged


def merge_authority_overlay(
    base: Mapping[str, object],
    replacement: Mapping[str, object],
    *,
    phones: Iterable[str] = STOP_PLACE_PHONES,
    scope: str = "involving-target",
    skip_zero_sample_rows: bool = True,
) -> dict[str, object]:
    if scope not in {"involving-target", "target-target"}:
        raise ValueError("scope must be 'involving-target' or 'target-target'")
    phone_set = frozenset(phones)
    merged = _json_clone(base)
    base_rows = _contrast_rows(merged, artifact_name="base authority")
    replacement_rows = _contrast_rows(replacement, artifact_name="replacement authority")
    replacement_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in replacement_rows:
        if not _authority_row_in_scope(row, phone_set, scope):
            continue
        if skip_zero_sample_rows and not _authority_row_has_samples(row):
            continue
        replacement_by_key[(str(row.get("target")), str(row.get("alternative")))] = row

    used: set[tuple[str, str]] = set()
    out_rows: list[dict[str, object]] = []
    for row in base_rows:
        key = (str(row.get("target")), str(row.get("alternative")))
        if key in replacement_by_key:
            out_rows.append(_json_clone(replacement_by_key[key]))
            used.add(key)
        else:
            out_rows.append(row)
    for key, row in sorted(replacement_by_key.items()):
        if key not in used:
            out_rows.append(_json_clone(row))
    merged["contrasts"] = out_rows
    _mark_overlay(merged, "curated_stop_place_authority_overlay")
    merged["curated_stop_place_overlay"]["authority_merge_scope"] = scope  # type: ignore[index]
    return merged


def copy_json_with_overlay_metadata(src: Path, dst: Path, overlay_kind: str) -> None:
    payload = json.loads(Path(src).read_text(encoding="utf-8"))
    _mark_overlay(payload, overlay_kind)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CurationError(f"JSON artifact must be an object: {path}")
    return payload


def atomic_copy(src: Path, dst: Path) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _replace_phone_payloads(
    base_phones: dict[str, object],
    replacement_phones: Mapping[str, object],
    *,
    phones: Iterable[str],
    strict: bool,
) -> None:
    missing = [phone for phone in phones if phone not in replacement_phones]
    if strict and missing:
        raise CurationError(f"replacement artifact missing phones: {', '.join(missing)}")
    for phone in phones:
        if phone in replacement_phones:
            base_phones[phone] = _json_clone(replacement_phones[phone])


def _phones_payload(payload: Mapping[str, object], *, artifact_name: str) -> dict[str, object]:
    phones = payload.get("phones")
    if not isinstance(phones, dict):
        raise CurationError(f"{artifact_name} artifact has no phones object")
    return phones


def _contrast_rows(payload: Mapping[str, object], *, artifact_name: str) -> list[dict[str, object]]:
    rows = payload.get("contrasts")
    if not isinstance(rows, list):
        raise CurationError(f"{artifact_name} artifact has no contrasts list")
    out: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise CurationError(f"{artifact_name} contrast row is not an object")
        out.append(_json_clone(row))
    return out


def _authority_row_in_scope(
    row: Mapping[str, object], phones: frozenset[str], scope: str
) -> bool:
    target = str(row.get("target") or "")
    alternative = str(row.get("alternative") or "")
    if scope == "target-target":
        return target in phones and alternative in phones
    return target in phones or alternative in phones


def _authority_row_has_samples(row: Mapping[str, object]) -> bool:
    reliability = row.get("reliability")
    if not isinstance(reliability, Mapping):
        return True
    target_counts = reliability.get("n_target")
    alternative_counts = reliability.get("n_alternative")
    return _count_total(target_counts) > 0 and _count_total(alternative_counts) > 0


def _count_total(value: object) -> int:
    if isinstance(value, Mapping):
        total = 0
        for item in value.values():
            if isinstance(item, (int, float)) and math.isfinite(float(item)):
                total += int(item)
        return total
    return 1


def _mark_overlay(payload: dict[str, object], kind: str) -> None:
    payload["curated_stop_place_overlay"] = {
        "kind": kind,
        "phones": list(STOP_PLACE_PHONES),
        "rollout": "experimental_overlay",
        "production_replacement": False,
    }


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
