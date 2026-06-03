"""Tests for the ``trace=True`` pipeline hook (Engine Inspector dependency)."""

from __future__ import annotations

from g2p.pipeline import G2PResult, TraceStep, transcribe_text, transcribe_word


def _step_names(trace: tuple[TraceStep, ...]) -> list[str]:
    return [step.name for step in trace]


def test_trace_disabled_by_default() -> None:
    result = transcribe_word("ekmek", use_reference=False)
    assert result.trace == ()


def test_trace_records_full_pipeline() -> None:
    result = transcribe_word(
        "karıcığım", use_reference=False, trace=True, pedagogical_allophones=True
    )

    names = _step_names(result.trace)
    expected = [
        "normalize",
        "reference_lookup",
        "exception_lookup",
        "grapheme_map",
        "morphology",
        "vowel_harmony",
        "soft_g",
        "palatalization",
        "l_allophony",
        "v_allophony",
        "n_assimilation",
        "e_open",
        "aspiration",
        "r_devoicing",
        "stress",
        "canonicalize_stress",
    ]
    assert names == expected


def test_trace_fired_flag_marks_changes() -> None:
    result = transcribe_word(
        "karıcığım", use_reference=False, trace=True
    )
    by_name = {step.name: step for step in result.trace}

    # Soft-g back-back lengthening must fire and aspiration must add /kʰ/.
    assert by_name["soft_g"].fired is True
    assert by_name["aspiration"].fired is True
    # Final fonem dizisi vurgulu ve /kʰ.../ ile başlamalı.
    assert result.phonemes[0] == "kʰ"
    assert "ˈ" in result.phonemes
    # Normalize untouched, e_open un-triggered (vowels back).
    assert by_name["normalize"].fired is False
    assert by_name["e_open"].fired is False


def test_trace_reference_hit_short_circuits_rules() -> None:
    # G2P final: REFERENCE_IPA minimal (sadece akronimler). TBMM hit verir.
    result = transcribe_word("TBMM", use_reference=True, trace=True)
    names = _step_names(result.trace)
    assert "reference_lookup" in names
    assert "parse_reference_ipa" in names
    # Rule chain must not run when a reference hit short-circuits.
    assert "vowel_harmony" not in names
    assert any(step.fired and step.name == "reference_lookup" for step in result.trace)


def test_trace_records_skipped_pedagogical_allophones() -> None:
    result = transcribe_word(
        "para",
        use_reference=False,
        trace=True,
        pedagogical_allophones=False,
    )
    by_name = {step.name: step for step in result.trace}
    assert "aspiration" in by_name
    assert by_name["aspiration"].fired is False
    assert "skipped" in by_name["aspiration"].note
    # Yes para should not become pʰa
    assert "pʰ" not in result.phonemes


def test_trace_morphology_step_notes_root_and_suffixes() -> None:
    result = transcribe_word("karıcığım", use_reference=False, trace=True)
    by_name = {step.name: step for step in result.trace}
    note = by_name["morphology"].note
    assert "root=" in note
    assert "suffixes=" in note


def test_text_mode_single_word_propagates_trace() -> None:
    result = transcribe_text("ekmek", use_reference=False, trace=True)
    assert isinstance(result, G2PResult)
    assert len(result.trace) > 0
    assert "grapheme_map" in _step_names(result.trace)


def test_text_mode_multiword_does_not_attach_trace() -> None:
    # Multi-word text path is a future enhancement; for now trace stays empty.
    result = transcribe_text(
        "ekmek ve su", use_reference=False, trace=True
    )
    assert result.trace == ()
