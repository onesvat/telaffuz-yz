"""V allofonisi."""

from __future__ import annotations

from g2p.constants import ROUNDED_VOWELS, V_APPROXIMANT, V_BILABIAL
from g2p.rules.common import base_vowel, is_vowel


def apply(phonemes: list[str]) -> list[str]:
    out = list(phonemes)
    for i, phoneme in enumerate(out):
        if phoneme != "v":
            continue
        prev_is_vowel = i > 0 and is_vowel(out[i - 1])
        next_is_vowel = i + 1 < len(out) and is_vowel(out[i + 1])
        adjacent = [
            base_vowel(out[j])
            for j in (i - 1, i + 1)
            if 0 <= j < len(out) and is_vowel(out[j])
        ]
        has_rounded = any(v in ROUNDED_VOWELS for v in adjacent)
        if prev_is_vowel and next_is_vowel and has_rounded:
            out[i] = V_APPROXIMANT
        elif has_rounded:
            out[i] = V_BILABIAL
    return out
