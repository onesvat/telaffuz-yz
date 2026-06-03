"""Phone normalization and edit-distance helpers.

The G2P runtime uses the 49-phone inventory in :mod:`g2p.constants`. Evaluation
sources use several IPA/SAMPA-like encodings, so this module keeps normalization,
parsing, and edit-distance logic in one place.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

from g2p.constants import ALL_PHONEMES, CTC_SPC, STRESS


@dataclass(frozen=True)
class PhoneParse:
    phones: tuple[str, ...]
    unknown: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.unknown


_LONG_VOWELS: dict[str, str] = {
    "a:": "aː",
    "e:": "eː",
    "i:": "iː",
    "1:": "ɯː",
    "ɯ:": "ɯː",
    "o:": "oː",
    "2:": "œː",
    "œ:": "œː",
    "u:": "uː",
    "y:": "yː",
}

TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    **{phone: (phone,) for phone in ALL_PHONEMES},
    CTC_SPC: (CTC_SPC,),
    " ": (CTC_SPC,),
    "a:": ("aː",),
    "e:": ("eː",),
    "i:": ("iː",),
    "1:": ("ɯː",),
    "ɯ:": ("ɯː",),
    "o:": ("oː",),
    "2:": ("œː",),
    "œ:": ("œː",),
    "u:": ("uː",),
    "y:": ("yː",),
    "dZ": ("d͡ʒ",),
    "dʒ": ("d͡ʒ",),
    "ʤ": ("d͡ʒ",),
    "tS": ("t͡ʃ",),
    "tʃ": ("t͡ʃ",),
    "ʧ": ("t͡ʃ",),
    "S": ("ʃ",),
    "Z": ("ʒ",),
    "N": ("ŋ",),
    "gj": ("ɟ",),
    "g": ("ɡ",),
    "r": ("ɾ",),
    "R": ("ɾ",),
    "1": ("ɯ",),
    "2": ("œ",),
    "5": ("ɫ",),
    "L": ("ɫ",),
    "I": ("ɯ",),
    "ı": ("ɯ",),
    "ø": ("œ",),
    "ø̞": ("œ",),
    "ɛ": ("e",),
    "ɪ": ("i",),
    "ɨ": ("ɯ",),
    "ʊ": ("u",),
    "ʏ": ("y",),
    "ʎ": ("l",),
    "ɳ": ("ɲ",),
    "ł": ("ɫ",),
    "ɰ": ("j",),
}

_SCAN_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    sorted(TOKEN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
)

_SEPARATORS: frozenset[str] = frozenset(
    {
        "/",
        "[",
        "]",
        "(",
        ")",
        "{",
        "}",
        ".",
        ",",
        ";",
        "|",
        "_",
        "‿",
        "·",
        "‧",
        "\t",
        "\n",
        "\r",
    }
)
_DROPPED_DIACRITICS: frozenset[str] = frozenset(
    {
        "̞",
        "̪",
        "̥",
        "̟",
        "̠",
        "̚",
        "̝",
        "̩",
        "ʲ",
        "ˤ",
        "ˌ",
        "ː",
        "ʰ",
        "ᵝ",
        "˧",
        "˨",
        "˩",
        "˦",
        "˥",
    }
)


def parse_ipa_text(text: str) -> PhoneParse:
    """Parse an IPA transcription into inventory tokens.

    Unknown symbols are reported instead of raising. Known non-contrastive
    diacritics from imported data are ignored after their base segment has been
    normalized.
    """
    source = _prepare_text(text)
    phones: list[str] = []
    unknown: list[str] = []
    i = 0
    while i < len(source):
        char = source[i]
        if char in _SEPARATORS or char.isspace():
            i += 1
            continue

        matched = False
        for token, mapped in _SCAN_ALIASES:
            if source.startswith(token, i):
                phones.extend(mapped)
                i += len(token)
                matched = True
                break
        if matched:
            continue

        if char in _DROPPED_DIACRITICS or unicodedata.combining(char):
            i += 1
            continue

        unknown.append(char)
        i += 1
    return PhoneParse(tuple(phones), tuple(unknown))


def normalize_phone_text(text: str) -> PhoneParse:
    """Normalize a whitespace-delimited phone sequence or compact IPA string."""
    stripped = text.strip()
    if not stripped:
        return PhoneParse(())

    if any(ch.isspace() for ch in stripped):
        phones: list[str] = []
        unknown: list[str] = []
        for raw_token in stripped.split():
            parsed = normalize_phone_token(raw_token)
            phones.extend(parsed.phones)
            unknown.extend(parsed.unknown)
        return PhoneParse(tuple(phones), tuple(unknown))
    return parse_ipa_text(stripped)


def normalize_phone_token(token: str) -> PhoneParse:
    normalized = _prepare_text(token.strip())
    if not normalized:
        return PhoneParse(())
    if normalized in TOKEN_ALIASES:
        return PhoneParse(TOKEN_ALIASES[normalized])
    return parse_ipa_text(normalized)


def phones_to_text(phones: Iterable[str]) -> str:
    return " ".join(phones)


def strip_stress(phones: Iterable[str]) -> tuple[str, ...]:
    return tuple(phone for phone in phones if phone != STRESS)


def strip_meta(phones: Iterable[str]) -> tuple[str, ...]:
    """Stres marker ve ``<spc>`` separator'ünü çıkarır."""
    skip = {STRESS, CTC_SPC}
    return tuple(phone for phone in phones if phone not in skip)


def canonicalize_stress_position(phones: Iterable[str]) -> tuple[str, ...]:
    """Stres marker'ı ait olduğu hecenin başına taşır (Wikipedia/IPA standardı).

    Bazı değerlendirme kaynaklarında stres marker'ı nucleus önünde verilir
    (``e m i ɾ ˈ i``); bu fonksiyon onu hece-onset konumuna çeker
    (``e m i ˈ ɾ i``). ``<spc>`` ile ayrılmış çoklu sözcükleri ayrı ayrı işler.
    """
    from g2p.syllabifier import syllabify

    sequence = list(phones)
    if CTC_SPC in sequence:
        chunks: list[list[str]] = [[]]
        for phone in sequence:
            if phone == CTC_SPC:
                chunks.append([])
            else:
                chunks[-1].append(phone)
        out: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx:
                out.append(CTC_SPC)
            out.extend(canonicalize_stress_position(chunk))
        return tuple(out)

    if STRESS not in sequence:
        return tuple(sequence)

    plain_position = 0
    for phone in sequence:
        if phone == STRESS:
            break
        plain_position += 1

    plain = [p for p in sequence if p != STRESS]
    syllables = syllabify(plain)
    if not syllables:
        return tuple(p for p in sequence if p != STRESS)

    target = len(syllables) - 1
    cursor = 0
    for idx, syllable in enumerate(syllables):
        end = cursor + len(syllable)
        if cursor <= plain_position < end:
            target = idx
            break
        cursor = end

    out_phones: list[str] = []
    for idx, syllable in enumerate(syllables):
        if idx == target:
            out_phones.append(STRESS)
        out_phones.extend(syllable)
    return tuple(out_phones)


def levenshtein(left: Iterable[str], right: Iterable[str]) -> int:
    """Return Levenshtein edit distance for two token sequences."""
    a = tuple(left)
    b = tuple(right)
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, left_item in enumerate(a, start=1):
        curr = [i]
        for j, right_item in enumerate(b, start=1):
            cost = 0 if left_item == right_item else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def phone_error_rate(predicted: Iterable[str], gold: Iterable[str]) -> float:
    gold_tuple = tuple(gold)
    edits = levenshtein(tuple(predicted), gold_tuple)
    if not gold_tuple:
        return 0.0 if edits == 0 else 1.0
    return edits / len(gold_tuple)


def _prepare_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.strip())
    normalized = normalized.replace(":", "ː")
    for source_form, target_form in _LONG_VOWELS.items():
        normalized = normalized.replace(source_form.replace(":", "ː"), target_form)
    normalized = normalized.replace("d͜ʒ", "d͡ʒ").replace("t͜ʃ", "t͡ʃ")
    normalized = normalized.replace("ǧ", "ğ").replace("ǰ", "d͡ʒ")
    return normalized
