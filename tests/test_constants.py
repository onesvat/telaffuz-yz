"""constants.py için temel envanter ve kategori doğrulamaları."""

from g2p import constants as c


def test_phoneme_count_is_49() -> None:
    phonemes_without_stress = c.ALL_PHONEMES - {c.STRESS}
    assert len(phonemes_without_stress) == 49


def test_deployed_ctc_vocabulary_size_is_54() -> None:
    assert len(c.CTC_SPECIAL_TOKENS) + len(c.ALL_PHONEMES) == 54


def test_short_vowels_count() -> None:
    assert len(c.SHORT_VOWELS) == 8


def test_long_vowels_count() -> None:
    assert len(c.LONG_VOWELS) == 8


def test_vowel_allophones() -> None:
    assert c.VOWEL_ALLOPHONES == {"æ"}


def test_consonant_phoneme_count() -> None:
    assert len(c.CONSONANTS) == 20


def test_base_consonant_allophones_count() -> None:
    assert len(c.CONS_BASE_ALLOPHONES) == 7


def test_extra_consonant_allophones_count() -> None:
    assert len(c.CONS_EXTRA_ALLOPHONES) == 5


def test_short_to_long_round_trip() -> None:
    for short, long_ in c.SHORT_TO_LONG.items():
        assert c.LONG_TO_SHORT[long_] == short
    assert set(c.SHORT_TO_LONG) == c.SHORT_VOWELS
    assert set(c.LONG_TO_SHORT) == c.LONG_VOWELS


def test_front_back_partition_is_complete() -> None:
    short_plus_allo = c.SHORT_VOWELS | c.VOWEL_ALLOPHONES
    assert c.FRONT_VOWELS | c.BACK_VOWELS == short_plus_allo
    assert c.FRONT_VOWELS & c.BACK_VOWELS == set()


def test_rounded_unrounded_partition() -> None:
    short_plus_allo = c.SHORT_VOWELS | c.VOWEL_ALLOPHONES
    assert c.ROUNDED_VOWELS | c.UNROUNDED_VOWELS == short_plus_allo
    assert c.ROUNDED_VOWELS & c.UNROUNDED_VOWELS == set()


def test_high_vs_non_high_partition() -> None:
    short_plus_allo = c.SHORT_VOWELS | c.VOWEL_ALLOPHONES
    assert c.HIGH_VOWELS | c.NON_HIGH_VOWELS == short_plus_allo
    assert c.HIGH_VOWELS & c.NON_HIGH_VOWELS == set()


def test_aspirated_map_keys_are_plain_stops() -> None:
    assert set(c.ASPIRATED_MAP) == c.PLAIN_STOPS
    assert set(c.ASPIRATED_MAP.values()) == {"pʰ", "tʰ", "kʰ", "cʰ"}


def test_aspirated_outputs_in_extra_allophones() -> None:
    assert set(c.ASPIRATED_MAP.values()) <= c.CONS_EXTRA_ALLOPHONES


def test_nasal_triggers_are_disjoint() -> None:
    assert c.NASAL_PALATAL_TRIGGERS & c.NASAL_VELAR_TRIGGERS == set()


def test_voicing_partition_covers_obstruents() -> None:
    obstruents = c.ALL_CONSONANTS - c.SONORANTS
    assert c.VOICELESS_CONS | c.VOICED_OBSTRUENTS == obstruents
    assert c.VOICELESS_CONS & c.VOICED_OBSTRUENTS == set()


def test_circumflex_map() -> None:
    assert c.CIRCUMFLEX_MAP == {"â": "a", "î": "i", "û": "u"}


def test_special_tokens_match_deployed_ctc_vocabulary() -> None:
    assert c.CTC_SPECIAL_TOKENS == (c.CTC_PAD, c.CTC_UNK, c.CTC_SPC, c.CTC_BLANK)
