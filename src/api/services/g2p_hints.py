"""In-process G2P hint cache backed by ``g2p.pipeline.transcribe_text``."""

from __future__ import annotations

from dataclasses import dataclass

from g2p.pipeline import transcribe_text


@dataclass(frozen=True)
class Hint:
    ipa: str
    phones: tuple[str, ...]
    source: str
    exception_type: str | None


class HintCache:
    def __init__(self) -> None:
        self._by_word: dict[str, Hint] = {}

    def get(self, word: str) -> Hint:
        cached = self._by_word.get(word)
        if cached is not None:
            return cached
        hint = _run(word)
        self._by_word[word] = hint
        return hint


def _run(word: str) -> Hint:
    result = transcribe_text(word)
    return Hint(
        ipa=result.ipa,
        phones=tuple(result.phonemes),
        source=result.source,
        exception_type=result.exception_type,
    )
