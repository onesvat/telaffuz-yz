"""Stress rule tests."""

import pytest

from g2p.grapheme_map import to_ipa
from g2p.rules.stress import apply


@pytest.mark.parametrize(
    "word,expected",
    [
        ("baba", ["b", "a", "b", "ˈ", "a"]),
        ("merhaba", ["m", "e", "ɾ", "h", "a", "b", "ˈ", "a"]),
        ("kitap", ["k", "i", "t", "ˈ", "a", "p"]),
    ],
)
def test_default_final_stress(word: str, expected: list[str]) -> None:
    assert apply(to_ipa(word), word=word) == expected


@pytest.mark.parametrize(
    "word", ["sinema", "lokanta", "bilgisayar", "simsiyah"],
)
def test_initial_stress_words(word: str) -> None:
    """INITIAL_STRESS_WORDS listesi: ilk hece stres."""
    result = apply(to_ipa(word), word=word)
    assert result.index("ˈ") <= 1


def test_explicit_ipa_detail_controls_stress() -> None:
    """Yer adı vb. lex girdileri ipa_detail üzerinden vurgu yerini sabitler."""
    result = apply(to_ipa("lazım"), word="lazım", ipa_detail="laː.ˈzɯm")
    assert result == ["ɫ", "a", "z", "ˈ", "ɯ", "m"]


def test_explicit_ipa_detail_without_dot_uses_vowel_count() -> None:
    """ipa_detail "." içermiyorsa "ˈ" solundaki ünlü sayısı = hece index'i."""
    # "faːˈɾe" prefix has one vowel; stress lands on the second syllable.
    result = apply(to_ipa("fare"), word="fare", ipa_detail="faːˈɾe")
    # 2 hece, idx 1 = "re"
    assert result.index("ˈ") > 0


def test_single_syllable_has_no_stress_token() -> None:
    assert apply(to_ipa("bir"), word="bir") == to_ipa("bir")
