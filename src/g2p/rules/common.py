"""Rule modülleri için küçük fonolojik yardımcılar."""

from __future__ import annotations

from g2p.constants import (
    ALL_CONSONANTS,
    ALL_VOWELS,
    BACK_VOWELS,
    FRONT_VOWELS,
    LONG_TO_SHORT,
    SHORT_TO_LONG,
    STRESS,
)


def strip_stress(phonemes: list[str]) -> list[str]:
    return [p for p in phonemes if p != STRESS]


def base_vowel(phoneme: str) -> str:
    return LONG_TO_SHORT.get(phoneme, phoneme)


def is_vowel(phoneme: str) -> bool:
    return phoneme in ALL_VOWELS


def is_consonant(phoneme: str) -> bool:
    return phoneme in ALL_CONSONANTS


def is_front_vowel(phoneme: str) -> bool:
    return base_vowel(phoneme) in FRONT_VOWELS


def is_back_vowel(phoneme: str) -> bool:
    return base_vowel(phoneme) in BACK_VOWELS


def lengthen(phoneme: str) -> str:
    return SHORT_TO_LONG.get(phoneme, phoneme)


def previous_vowel_index(phonemes: list[str], index: int) -> int:
    for i in range(index - 1, -1, -1):
        if is_vowel(phonemes[i]):
            return i
    return -1


def next_vowel_index(phonemes: list[str], index: int) -> int:
    for i in range(index + 1, len(phonemes)):
        if is_vowel(phonemes[i]):
            return i
    return -1
