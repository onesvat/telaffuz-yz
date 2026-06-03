"""soft_g rule tests for final Istanbul Turkish context rules.

Covered contexts:
- Word-final and consonant-followed coda ğ lengthens the vowel on its left
- Intervokalik VğV — front-front → [j]
- Intervokalik VğV — back-back → hiatus (uzatma YOK)
- /e/+coda ğ → [ej]
- Proper name C+ğ → /ɡ/
"""

import pytest

from g2p.grapheme_map import to_ipa
from g2p.rules.soft_g import apply


@pytest.mark.parametrize(
    "word,expected",
    [
        ("dağ", ["d", "aː"]),
        ("bağ", ["b", "aː"]),
        ("sağ", ["s", "aː"]),
        ("oğlan", ["oː", "ɫ", "a", "n"]),
        ("doğru", ["d", "oː", "ɾ", "u"]),
        ("yağmur", ["j", "aː", "m", "u", "ɾ"]),
        ("çığlık", ["t͡ʃ", "ɯː", "ɫ", "ɯ", "k"]),
        ("tuğla", ["t", "uː", "ɫ", "a"]),
        ("iğne", ["iː", "n", "e"]),
        ("sağlık", ["s", "aː", "ɫ", "ɯ", "k"]),
    ],
)
def test_coda_soft_g_lengthens_previous_vowel(word: str, expected: list[str]) -> None:
    assert apply(to_ipa(word)) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        ("değil", ["d", "e", "j", "i", "ɫ"]),
        ("eğitim", ["e", "j", "i", "t", "i", "m"]),
        ("beğeni", ["b", "e", "j", "e", "n", "i"]),
        ("meğer", ["m", "e", "j", "e", "ɾ"]),
    ],
)
def test_front_front_soft_g_becomes_j(word: str, expected: list[str]) -> None:
    assert apply(to_ipa(word)) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        # Yeni kural: back-back intervokalik VğV → hiatus, uzatma YOK
        ("oğul", ["o", "u", "ɫ"]),
        ("boğaz", ["b", "o", "a", "z"]),
        ("soğan", ["s", "o", "a", "n"]),
        ("mağara", ["m", "a", "a", "ɾ", "a"]),
        ("ağa", ["a", "a"]),
        ("doğa", ["d", "o", "a"]),
    ],
)
def test_back_back_intervocalic_soft_g_is_hiatus(
    word: str, expected: list[str]
) -> None:
    assert apply(to_ipa(word)) == expected


def test_coda_e_soft_g_becomes_ej() -> None:
    """e + coda ğ → [ej]: değnek → /dejnec/"""
    assert apply(to_ipa("eğlence"))[:2] == ["e", "j"]


def test_dagnek_coda_e_pattern() -> None:
    """değnek: d e ğ n e k → d e j n e k (e + coda ğ → ej)."""
    result = apply(to_ipa("değnek"))
    assert result == ["d", "e", "j", "n", "e", "k"]


def test_proper_name_consonant_soft_g_hardens() -> None:
    assert apply(to_ipa("Olğun"), word="Olğun", is_proper=True) == [
        "o", "ɫ", "ɡ", "u", "n",
    ]
