"""N nazal asimilasyonu."""

from __future__ import annotations

from g2p.constants import (
    NASAL_PALATAL,
    NASAL_PALATAL_TRIGGERS,
    NASAL_VELAR,
    NASAL_VELAR_TRIGGERS,
)


def apply(phonemes: list[str]) -> list[str]:
    out = list(phonemes)
    for i, phoneme in enumerate(out[:-1]):
        if phoneme != "n":
            continue
        nxt = out[i + 1]
        if nxt in NASAL_VELAR_TRIGGERS:
            out[i] = NASAL_VELAR
        elif nxt in NASAL_PALATAL_TRIGGERS:
            out[i] = NASAL_PALATAL
    return out
