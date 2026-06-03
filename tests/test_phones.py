"""Phone normalization and PER helpers."""

from __future__ import annotations

from g2p.phones import (
    canonicalize_stress_position,
    levenshtein,
    normalize_phone_text,
    parse_ipa_text,
    phone_error_rate,
)


def test_parse_ipa_normalizes_inventory_aliases() -> None:
    parsed = parse_ipa_text("/dɛɾs tʃodʒuk gyl/")
    assert parsed.phones == (
        "d",
        "e",
        "ɾ",
        "s",
        "t͡ʃ",
        "o",
        "d͡ʒ",
        "u",
        "k",
        "ɡ",
        "y",
        "l",
    )
    assert parsed.unknown == ()


def test_normalize_duygua_phone_text() -> None:
    parsed = normalize_phone_text("dZ 1 5 a r a")
    assert parsed.phones == ("d͡ʒ", "ɯ", "ɫ", "a", "ɾ", "a")
    assert parsed.unknown == ()


def test_levenshtein_and_per() -> None:
    assert levenshtein(("a", "b", "c"), ("a", "c")) == 1
    assert phone_error_rate(("a", "b", "c"), ("a", "c")) == 0.5


def test_canonicalize_moves_stress_to_syllable_onset() -> None:
    # Nucleus-position stress → hece-onset
    nucleus = ("e", "m", "i", "ɾ", "ˈ", "i")
    canonical = canonicalize_stress_position(nucleus)
    assert canonical == ("e", "m", "i", "ˈ", "ɾ", "i")


def test_canonicalize_idempotent_on_canonical_input() -> None:
    canonical = ("e", "m", "i", "ˈ", "ɾ", "i")
    assert canonicalize_stress_position(canonical) == canonical


def test_canonicalize_handles_pre_nucleus_stress_in_first_syllable() -> None:
    # Single-syllable: stress should attach to onset of the only syllable
    nucleus = ("b", "ˈ", "i", "ɾ")
    canonical = canonicalize_stress_position(nucleus)
    assert canonical == ("ˈ", "b", "i", "ɾ")


def test_canonicalize_preserves_spc_separator() -> None:
    phones = (
        "a",
        "n",
        "o",
        "n",
        "ˈ",
        "i",
        "m",
        "<spc>",
        "ʃ",
        "i",
        "ɾ",
        "ˈ",
        "c",
        "e",
        "t",
    )
    canonical = canonicalize_stress_position(phones)
    assert canonical == (
        "a",
        "n",
        "o",
        "ˈ",
        "n",
        "i",
        "m",
        "<spc>",
        "ʃ",
        "i",
        "ɾ",
        "ˈ",
        "c",
        "e",
        "t",
    )


def test_canonicalize_no_op_when_no_stress() -> None:
    phones = ("a", "b", "c")
    assert canonicalize_stress_position(phones) == phones
