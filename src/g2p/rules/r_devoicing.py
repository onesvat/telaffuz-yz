"""Sözcük sonunda veya izleyen sessiz ünsüz bağlamında /ɾ/ devoicing."""

from __future__ import annotations

from g2p.constants import R_DEVOICED, VOICELESS_CONS


def apply(phonemes: list[str]) -> list[str]:
    out = list(phonemes)
    for i, phoneme in enumerate(out):
        if phoneme != "ɾ":
            continue
        if i == len(out) - 1 or out[i + 1] in VOICELESS_CONS:
            out[i] = R_DEVOICED
    return out
