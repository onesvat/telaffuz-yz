"""L allofonisi: koyu [ɫ] ve açık [l]."""

from __future__ import annotations

from g2p.rules.common import is_front_vowel, is_vowel


THIN_L_WORDS: frozenset[str] = frozenset({"alkol", "rol", "gol", "hol", "lazım"})


def apply(
    phonemes: list[str], *, word: str = "", exception_type: str = ""
) -> list[str]:
    if exception_type == "thin_l" or word.lower() in THIN_L_WORDS:
        return ["l" if p == "ɫ" else p for p in phonemes]

    out = list(phonemes)
    for i, phoneme in enumerate(out):
        if phoneme not in {"l", "ɫ"}:
            continue
        prev_vowel = _nearest_vowel(out, i, -1)
        next_vowel = _nearest_vowel(out, i, 1)
        if (prev_vowel and is_front_vowel(prev_vowel)) or (
            next_vowel and is_front_vowel(next_vowel)
        ):
            out[i] = "l"
        else:
            out[i] = "ɫ"
    return out


def _nearest_vowel(phonemes: list[str], index: int, step: int) -> str | None:
    i = index + step
    while 0 <= i < len(phonemes):
        if is_vowel(phonemes[i]):
            return phonemes[i]
        i += step
    return None
