"""Hece-başı sessiz patlamalı aspirasyonu."""

from __future__ import annotations

from g2p.constants import ASPIRATED_MAP
from g2p.syllabifier import onset, syllabify


def apply(phonemes: list[str]) -> list[str]:
    out: list[str] = []
    for syllable in syllabify(phonemes):
        onset_len = len(onset(syllable))
        for idx, phoneme in enumerate(syllable):
            if idx < onset_len and phoneme in ASPIRATED_MAP:
                out.append(ASPIRATED_MAP[phoneme])
            else:
                out.append(phoneme)
    return out
