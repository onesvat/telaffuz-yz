"""Leksikal istisna yükleyici.

`data/g2p/exceptions_seed.csv` istisna setini yükler. Pipeline'ın
ilk leksikal adımıdır: bir sözcük listede varsa kayıtlı fonem dizisi
base seed olarak kullanılır; surface allofon ve stres kuralları yine pipeline
tarafından uygulanır.

CSV şeması:
    word,phonemes,ipa_detail,type,suffix_behavior,status,source,notes

`phonemes` alanı boşluk ile ayrılmış IPA token dizisidir (ör. ``ɟ e l i d͡ʒ
eː m``). Tied affricate'ler (``t͡ʃ``, ``d͡ʒ``) ve uzun ünlüler (``aː``)
tek token olarak korunur.

İstisna tipleri final pipeline tarafından kategorik olarak ele alınır:

- ``manual``, ``gemini-adjudicated`` — genel manuel/AI doğrulama
- ``long_vowel`` — Arapça/Farsça alıntı uzun ünlü
- ``narrowing``, ``irregular_narrowing`` — daralma istisnaları
- ``stress`` — vurgu istisnaları
- ``mutation`` — ünsüz yumuşaması istisnaları
- ``y_weakening``, ``h_contraction`` — ünlü uzaması nedenli düşmeler
- ``epenthesis`` — türeme ünlüsü
- ``yor_r_dropping`` — şimdiki zaman /ɾ/ düşmesi
- ``thin_l`` — ince /l/ (alıntı)
- ``clitic`` — klitik (ile, sonra)
- ``special`` — özel durum
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from g2p.constants import ALL_PHONEMES

DATA_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "data" / "g2p" / "exceptions_seed.csv"
)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "word",
    "phonemes",
    "ipa_detail",
    "type",
    "suffix_behavior",
    "status",
    "source",
    "notes",
)
EXCEPTION_TYPES: frozenset[str] = frozenset(
    {
        "clitic",
        "epenthesis",
        "gemini-adjudicated",
        "h_contraction",
        "irregular_narrowing",
        "long_vowel",
        "manual",
        "mutation",
        "narrowing",
        "no_e_open",
        "place_name",
        "special",
        "stress",
        "thin_l",
        "y_weakening",
        "yor_r_dropping",
    }
)
SUFFIX_BEHAVIORS: frozenset[str] = frozenset({"none", "preserve", "apply"})
STATUSES: frozenset[str] = frozenset({"active", "manual_verified", "gemini_verified"})
FORBIDDEN_SYMBOLS: frozenset[str] = frozenset(
    {"ɛ", "ɣ", "ʔ", "ˤ", "ʲ", "̪", "̞", "͡", "ː", "ʰ"}
)


@dataclass(frozen=True)
class ExceptionEntry:
    """Tek bir leksikal istisna kaydı."""

    word: str
    phonemes: tuple[str, ...]
    ipa_detail: str
    type: str
    suffix_behavior: str
    status: str
    source: str
    notes: str


@cache
def _load(path: Path = DATA_PATH) -> dict[str, ExceptionEntry]:
    """CSV'yi yükle ve case-sensitive sözlük döndür."""
    entries: dict[str, ExceptionEntry] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            phonemes_str = row.get("phonemes", "").strip()
            phonemes = tuple(phonemes_str.split(" ")) if phonemes_str else ()
            entries[word] = ExceptionEntry(
                word=word,
                phonemes=phonemes,
                ipa_detail=row.get("ipa_detail", ""),
                type=row.get("type", ""),
                suffix_behavior=row.get("suffix_behavior", ""),
                status=row.get("status", ""),
                source=row.get("source", ""),
                notes=row.get("notes", ""),
            )
    return entries


def lookup(word: str) -> ExceptionEntry | None:
    """Case-sensitive istisna araması.

    `Olğun` ile `olğun` ayrımı (özel ad kontrolü) bu fonksiyona aittir.
    Args:
        word: Yüzey form (büyük/küçük harf orijinal).

    Returns:
        Eşleşen ``ExceptionEntry`` veya ``None``.
    """
    return _load().get(word)


def lookup_case_insensitive(word: str) -> ExceptionEntry | None:
    """Önce case-sensitive dene; bulunamazsa küçük harfli formla."""
    direct = lookup(word)
    if direct is not None:
        return direct
    lower = _lower_tr(word)
    if lower != word:
        return lookup(lower)
    return None


def all_words() -> frozenset[str]:
    """Yüklenmiş istisna sözlüğündeki tüm sözcükler."""
    return frozenset(_load().keys())


def by_type(exception_type: str) -> tuple[ExceptionEntry, ...]:
    """Belirli bir tipteki tüm girdileri döndür (örn. ``"long_vowel"``)."""
    return tuple(e for e in _load().values() if e.type == exception_type)


def _lower_tr(text: str) -> str:
    """Türkçe-duyarlı küçük harf çevrimi."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def validate_csv(path: Path = DATA_PATH) -> tuple[str, ...]:
    """Exception CSV için schema, enum ve envanter bütünlüğü denetimi."""
    errors: list[str] = []
    seen_words: set[str] = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            errors.append("header does not match EXPECTED_COLUMNS")

        for line_no, row in enumerate(reader, start=2):
            word = row.get("word", "")
            if not word:
                errors.append(f"line {line_no}: empty word")
            elif word in seen_words:
                errors.append(f"line {line_no}: duplicate word {word!r}")
            seen_words.add(word)

            entry_type = row.get("type", "")
            if entry_type not in EXCEPTION_TYPES:
                errors.append(f"line {line_no}: unknown type {entry_type!r}")

            suffix_behavior = row.get("suffix_behavior", "")
            if suffix_behavior not in SUFFIX_BEHAVIORS:
                errors.append(
                    f"line {line_no}: unknown suffix_behavior {suffix_behavior!r}"
                )

            status = row.get("status", "")
            if status not in STATUSES:
                errors.append(f"line {line_no}: unknown status {status!r}")

            for token in row.get("phonemes", "").split():
                if token in FORBIDDEN_SYMBOLS:
                    errors.append(f"line {line_no}: forbidden token {token!r}")
                if token not in ALL_PHONEMES:
                    errors.append(f"line {line_no}: unknown token {token!r}")

    return tuple(errors)
