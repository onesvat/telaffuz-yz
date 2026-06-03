"""Pipeline integration tests for final Istanbul Turkish G2P."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from g2p.constants import (
    ALL_PHONEMES,
    CTC_BLANK,
    CTC_PAD,
    CTC_SPC,
    CTC_UNK,
)
from g2p.pipeline import parse_ipa, transcribe_text, transcribe_training_text, transcribe_word


def _load_doc_examples() -> list[tuple[str, str]]:
    path = Path(__file__).resolve().parent / "fixtures" / "g2p" / "regression_50.csv"
    with path.open(encoding="utf-8") as f:
        return [(r["word"], r["expected_ipa"]) for r in csv.DictReader(f)]


DOC_EXAMPLES: list[tuple[str, str]] = _load_doc_examples()


@pytest.mark.parametrize("word,expected_ipa", DOC_EXAMPLES)
def test_doc_examples_match_expected_tokens(word: str, expected_ipa: str) -> None:
    result = transcribe_word(word)
    assert list(result.phonemes) == parse_ipa(expected_ipa)
    assert result.warnings == ()


@pytest.mark.parametrize("word,expected_ipa", DOC_EXAMPLES)
def test_doc_examples_stay_inside_inventory(word: str, expected_ipa: str) -> None:
    result = transcribe_word(word)
    allowed = ALL_PHONEMES | {CTC_SPC}
    assert set(result.phonemes) <= allowed
    assert set(parse_ipa(expected_ipa)) <= allowed


@pytest.mark.parametrize(
    "word,expected_ipa",
    [
        ("TBMM", "/teˈbeˈmeˈme/"),
        ("TRT", "/teˈɾeˈte/"),
        ("KDV", "/kaːˈdeˈve/"),
        ("AVM", "/aˈveˈme/"),
    ],
)
def test_acronym_reference_forms(word: str, expected_ipa: str) -> None:
    assert list(transcribe_word(word).phonemes) == parse_ipa(expected_ipa)


def test_unknown_word_uses_rule_pipeline() -> None:
    result = transcribe_word("yeni")
    assert result.source == "rules"
    assert result.ipa == "/je.ˈni/"
    assert result.warnings == ()


def test_reference_lookup_can_be_disabled_for_eval() -> None:
    result = transcribe_word("ekmek", use_reference=False)
    assert result.source == "rules"


def test_pedagogical_allophones_can_be_disabled() -> None:
    # --no-reference + --no-pedagogical: aspirasyon ve r-devoicing kapalı
    plain = transcribe_word("para", use_reference=False, pedagogical_allophones=False)
    assert "pʰ" not in plain.phonemes
    assert plain.phonemes[0] == "p"

    devoiced = transcribe_word("bir", use_reference=False, pedagogical_allophones=False)
    assert "ɾ̞̊" not in devoiced.phonemes
    assert devoiced.phonemes[-1] == "ɾ"


def test_pedagogical_allophones_enabled_by_default() -> None:
    aspirated = transcribe_word("para", use_reference=False)
    assert "pʰ" in aspirated.phonemes


def test_stress_marker_is_at_syllable_onset_in_flat_phones() -> None:
    # IPA stress marker is kept at the syllable onset.
    result = transcribe_word("Emir'i", use_reference=False)
    phones = list(result.phonemes)
    assert "ˈ" in phones
    stress_idx = phones.index("ˈ")
    # Stres'ten sonraki ilk fonem stresli hecenin onset'i (ɾ) olmalı
    assert phones[stress_idx + 1] == "ɾ"


def test_text_transcription_canonical_stress_per_word() -> None:
    result = transcribe_text("anonim şirket", use_reference=False)
    phones = list(result.phonemes)
    # İki sözcük, her biri kendi stres'ine sahip + <spc>
    assert phones.count("ˈ") == 2
    assert CTC_SPC in phones


def test_text_transcription_inserts_space_token() -> None:
    result = transcribe_text("anne baba")
    assert CTC_SPC in result.phonemes
    assert result.ipa == "/an.ˈne ba.ˈba/"


def test_training_text_single_word_emits_no_space_token() -> None:
    result = transcribe_training_text("anne")

    assert result.text == "anne"
    assert result.normalized == "anne"
    assert CTC_SPC not in result.tokens
    assert result.token_count == len(result.tokens)
    assert result.drop_reason is None
    assert [word.normalized for word in result.words] == ["anne"]


def test_training_text_multiword_emits_one_space_token_between_words() -> None:
    result = transcribe_training_text("anne baba")

    assert result.normalized == "anne baba"
    assert result.tokens.count(CTC_SPC) == 1
    assert result.tokens == result.words[0].tokens + (CTC_SPC,) + result.words[1].tokens


def test_training_text_punctuation_and_extra_spaces_normalize_predictably() -> None:
    result = transcribe_training_text("  Anne,   baba!  ")

    assert result.normalized == "Anne baba"
    assert [word.text for word in result.words] == ["Anne", "baba"]
    assert result.tokens.count(CTC_SPC) == 1


def test_training_text_rejects_forbidden_emitted_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    real = transcribe_word("anne")

    def fake_transcribe_word(*args: object, **kwargs: object):
        return G2PResultForTest(real, (CTC_UNK,))

    monkeypatch.setattr("g2p.pipeline.transcribe_word", fake_transcribe_word)

    result = transcribe_training_text("anne")

    assert result.tokens == (CTC_UNK,)
    assert result.drop_reason == f"forbidden_token:{CTC_UNK}"
    assert any("forbidden emitted token" in warning for warning in result.warnings)


@pytest.mark.parametrize("token", [CTC_BLANK, CTC_PAD, CTC_UNK, "<sil>"])
def test_training_text_policy_never_allows_infrastructure_tokens(token: str) -> None:
    result = transcribe_training_text("anne")
    allowed = ALL_PHONEMES | {CTC_SPC}

    assert token not in result.tokens
    assert set(result.tokens) <= allowed


def test_result_json_is_parseable() -> None:
    payload = json.loads(transcribe_word("ekmek").to_json())
    assert payload["word"] == "ekmek"
    assert payload["ipa"] == "/ek.ˈmec/"
    assert payload["source"] == "rules"


class G2PResultForTest:
    def __init__(self, source, phonemes: tuple[str, ...]) -> None:
        self.word = source.word
        self.normalized = source.normalized
        self.phonemes = phonemes
        self.syllables = source.syllables
        self.ipa = source.ipa
        self.source = source.source
        self.exception_type = source.exception_type
        self.warnings = source.warnings
        self.trace = source.trace
