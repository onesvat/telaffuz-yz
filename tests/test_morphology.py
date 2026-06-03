"""morphology.analyze için unit testler.

Starlang FSM analizörü ilk çağrıda yavaş yüklenir (~3-5 sn); sonraki
testler aynı singleton'u paylaşır.
"""

from __future__ import annotations

import pytest

from g2p.morphology import (
    AE_TRIGGERING_SUFFIXES,
    STRESS_ATTRACTING_SUFFIXES,
    STRESS_PREVENTING_SUFFIXES,
    analyze,
)


@pytest.fixture(scope="session")
def warm_analyzer() -> None:
    """Test session'ı başlamadan analizörü ısıt."""
    analyze("gel")


class TestBasicAnalysis:
    def test_simple_verb(self, warm_analyzer: None) -> None:
        result = analyze("gel")
        assert result is not None
        assert result.root == "gel"
        assert result.pos == "VERB"
        assert not result.is_proper

    def test_simple_noun(self, warm_analyzer: None) -> None:
        result = analyze("kitap")
        assert result is not None
        assert result.root == "kitap"
        assert result.pos == "NOUN"

    def test_inflected_verb(self, warm_analyzer: None) -> None:
        result = analyze("geliyordum")
        assert result is not None
        assert result.root == "gel"
        assert "Hyor" in result.suffixes
        assert result.pos == "VERB"

    def test_plural_noun(self, warm_analyzer: None) -> None:
        result = analyze("evler")
        assert result is not None
        assert result.root == "ev"
        assert result.pos == "NOUN"

    def test_consonant_mutation(self, warm_analyzer: None) -> None:
        # kitap+ı → kitabı (root kitap olarak kalır)
        result = analyze("kitabı")
        assert result is not None
        assert result.root == "kitap"


class TestProperNouns:
    def test_ankara_is_proper(self, warm_analyzer: None) -> None:
        result = analyze("Ankara")
        assert result is not None
        assert result.is_proper

    def test_lowercase_word_not_proper(self, warm_analyzer: None) -> None:
        result = analyze("kitap")
        assert result is not None
        assert not result.is_proper


class TestNegation:
    def test_yapma_negation(self, warm_analyzer: None) -> None:
        # yap+NEG+IMP+A2SG — kök yap, ek mA
        result = analyze("yapma")
        assert result is not None
        assert result.root == "yap"
        assert "mA" in result.suffixes


class TestUnknownWord:
    def test_random_string_returns_none_or_unparseable(
        self, warm_analyzer: None
    ) -> None:
        # Anlamsız harf dizisi parse alamaz
        result = analyze("zzzzzzz")
        assert result is None or result.root == "zzzzzzz"

    def test_empty_string(self, warm_analyzer: None) -> None:
        assert analyze("") is None


class TestSuffixSets:
    def test_stress_attracting_set_contains_known(self) -> None:
        # Stress-attracting suffix set used by the final stress rule.
        assert "Hyor" in STRESS_ATTRACTING_SUFFIXES
        assert "yArAk" in STRESS_ATTRACTING_SUFFIXES
        assert "HncA" in STRESS_ATTRACTING_SUFFIXES

    def test_stress_preventing_set_contains_known(self) -> None:
        # Stress-preventing suffix set used by the final stress rule.
        assert "mA" in STRESS_PREVENTING_SUFFIXES
        assert "ki" in STRESS_PREVENTING_SUFFIXES

    def test_ae_triggering_set(self) -> None:
        # -mez (gel+mez = /ɟælmæz/)
        assert "mAz" in AE_TRIGGERING_SUFFIXES

    def test_attracting_and_preventing_disjoint(self) -> None:
        assert STRESS_ATTRACTING_SUFFIXES & STRESS_PREVENTING_SUFFIXES == set()


class TestDataclass:
    def test_morph_analysis_is_frozen(self, warm_analyzer: None) -> None:
        result = analyze("gel")
        assert result is not None
        with pytest.raises(Exception):  # FrozenInstanceError
            result.root = "changed_root"  # type: ignore[misc]

    def test_morph_analysis_immutable_suffixes(self, warm_analyzer: None) -> None:
        result = analyze("geliyordum")
        assert result is not None
        assert isinstance(result.suffixes, tuple)
