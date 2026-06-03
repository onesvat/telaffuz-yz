"""Ünlü uyumu yardımcıları.

Bu modül doğrudan token değiştirmez; morphology/stress ve ilerideki ek
realization kuralları için son ünlü ve uyumlu ek ünlüsü hesaplar.
"""

from __future__ import annotations

from g2p.constants import BACK_VOWELS, ROUNDED_VOWELS
from g2p.rules.common import base_vowel, is_vowel


def last_vowel(phonemes: list[str]) -> str | None:
    for phoneme in reversed(phonemes):
        if is_vowel(phoneme):
            return base_vowel(phoneme)
    return None


def two_way_suffix_vowel(phonemes: list[str]) -> str:
    vowel = last_vowel(phonemes)
    if vowel is None:
        return "e"
    return "a" if vowel in BACK_VOWELS else "e"


def four_way_suffix_vowel(phonemes: list[str]) -> str:
    vowel = last_vowel(phonemes)
    if vowel is None:
        return "i"
    if vowel in {"a", "ɯ"}:
        return "ɯ"
    if vowel in {"e", "i", "æ"}:
        return "i"
    return "u" if vowel in ROUNDED_VOWELS and vowel in {"o", "u"} else "y"


def apply(phonemes: list[str]) -> list[str]:
    """Pipeline uyumluluğu için no-op."""
    return list(phonemes)
