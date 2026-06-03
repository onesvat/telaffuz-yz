"""Core surface rule tests — G2P final (İstanbul Türkçesi)."""

import pytest

from g2p.grapheme_map import to_ipa
from g2p.rules import aspiration, e_open, l_allophony, n_assimilation
from g2p.rules import palatalization, r_devoicing, v_allophony, vowel_harmony
from g2p.rules.soft_g import apply as soft_g


@pytest.mark.parametrize(
    "word,expected",
    [
        # Onset palatalize:
        ("kedi", ["c", "e", "d", "i"]),
        ("gece", ["ɟ", "e", "d͡ʒ", "e"]),
        ("kötü", ["c", "œ", "t", "y"]),
        ("kara", ["k", "a", "ɾ", "a"]),
        # Koda palatalize with constrained e_open behavior:
        # - ekmek: ek.mek -> first k stays velar with nasal m to its right, final k -> c
        ("ekmek", ["e", "k", "m", "e", "c"]),
        # - erkek: er.kek → son k word-final → c
        ("erkek", ["e", "ɾ", "c", "e", "c"]),
        # - gerek: ge.rek → word-final k → c
        ("gerek", ["ɟ", "e", "ɾ", "e", "c"]),
        # - bekle: bek.le -> coda k followed by liquid l -> c
        ("bekle", ["b", "e", "c", "ɫ", "e"]),
        # - denk: Vnk blok → koru
        ("denk", ["d", "e", "n", "k"]),
        # - turkce: coda k followed by obstruent t͡ʃ stays velar
        ("türkçe", ["t", "y", "ɾ", "k", "t͡ʃ", "e"]),
        # - Akdeniz: a back vowel, k koda + d onset obstruent → velar koru
        ("Akdeniz", ["a", "k", "d", "e", "n", "i", "z"]),
    ],
)
def test_palatalization(word: str, expected: list[str]) -> None:
    assert palatalization.apply(to_ipa(word)) == expected


@pytest.mark.parametrize(
    "word,expected_l",
    [
        ("kulak", "ɫ"),
        ("ılık", "ɫ"),
        ("sol", "ɫ"),
        ("bilinç", "l"),
        ("el", "l"),
        ("lider", "l"),
    ],
)
def test_l_allophony(word: str, expected_l: str) -> None:
    assert expected_l in l_allophony.apply(to_ipa(word), word=word)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("vur", ["β", "u", "ɾ"]),
        ("kov", ["k", "o", "β"]),
        ("tavuk", ["t", "a", "β̞", "u", "k"]),
        ("ova", ["o", "β̞", "a"]),
        ("evet", ["e", "v", "e", "t"]),
    ],
)
def test_v_allophony(word: str, expected: list[str]) -> None:
    assert v_allophony.apply(to_ipa(word)) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        ("denk", ["d", "e", "ŋ", "k"]),
        ("yangın", ["j", "a", "ŋ", "ɡ", "ɯ", "n"]),
        ("inci", ["i", "ɲ", "d͡ʒ", "i"]),
        ("tencere", ["t", "e", "ɲ", "d͡ʒ", "e", "ɾ", "e"]),
        ("kanat", ["k", "a", "n", "a", "t"]),
    ],
)
def test_n_assimilation(word: str, expected: list[str]) -> None:
    assert n_assimilation.apply(to_ipa(word)) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        # Dar /æ/ kuralı: sadece sonorant koda bağlamı (l, ɫ, m, n, ɾ, ŋ, ɲ).
        ("el", ["æ", "ɫ"]),
        ("ders", ["d", "æ", "ɾ", "s"]),
        # ekmek: k coda OBSTRUENT → /e/ kalır
        ("ekmek", ["e", "k", "m", "e", "k"]),
        # bekledim: k coda followed by liquid; e_open alone keeps /e/.
        # e_open alone: ['b', 'e', 'k', 'l', 'e', 'd', 'i', 'm'] → bek hece k
        # obstruent koda; le açık. e tutulur.
        ("bekledim", ["b", "e", "k", "ɫ", "e", "d", "i", "m"]),
        # temel: te.mel; mel hecesi l sonorant koda → /æ/
        ("temel", ["t", "e", "m", "æ", "ɫ"]),
        # tek: k coda obstruent → /e/ kalır
        ("tek", ["t", "e", "k"]),
        # ben: n sonorant koda → /æ/
        ("ben", ["b", "æ", "n"]),
    ],
)
def test_e_open(word: str, expected: list[str]) -> None:
    assert e_open.apply(to_ipa(word), word=word) == expected


@pytest.mark.parametrize(
    "word",
    ["belli", "elli", "kendi", "pencere", "engin", "genç", "hem", "anne"],
)
def test_e_open_lex_exception_keeps_e(word: str) -> None:
    """no_e_open lex listesi: sonorant koda olsa bile /e/ tutulur."""
    result = e_open.apply(to_ipa(word), word=word)
    assert "æ" not in result, f"{word}: /æ/ produced but should stay /e/"


@pytest.mark.parametrize(
    "word,expected",
    [
        ("para", ["pʰ", "a", "ɾ", "a"]),
        ("tepe", ["tʰ", "e", "pʰ", "e"]),
        ("kapı", ["kʰ", "a", "pʰ", "ɯ"]),
        ("kitap", ["kʰ", "i", "tʰ", "a", "p"]),
        ("sap", ["s", "a", "p"]),
    ],
)
def test_aspiration(word: str, expected: list[str]) -> None:
    assert aspiration.apply(to_ipa(word)) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        ("bir", ["b", "i", "ɾ̞̊"]),
        ("var", ["v", "a", "ɾ̞̊"]),
        ("dert", ["d", "e", "ɾ̞̊", "t"]),
        ("kart", ["k", "a", "ɾ̞̊", "t"]),
        ("ara", ["a", "ɾ", "a"]),
    ],
)
def test_r_devoicing(word: str, expected: list[str]) -> None:
    assert r_devoicing.apply(to_ipa(word)) == expected


def test_vowel_harmony_helpers() -> None:
    assert vowel_harmony.two_way_suffix_vowel(to_ipa("kitap")) == "a"
    assert vowel_harmony.two_way_suffix_vowel(to_ipa("ev")) == "e"
    assert vowel_harmony.four_way_suffix_vowel(to_ipa("kol")) == "u"
    assert vowel_harmony.four_way_suffix_vowel(to_ipa("göz")) == "y"


def test_rule_chain_keeps_soft_g_length_for_dag() -> None:
    phones = soft_g(to_ipa("dağ"))
    assert r_devoicing.apply(phones) == ["d", "aː"]


def test_rule_chain_back_back_intervocalic_no_length() -> None:
    """oğul: back-back intervocalic VğV creates hiatus without lengthening."""
    phones = soft_g(to_ipa("oğul"))
    assert phones == ["o", "u", "ɫ"]


def test_mez_suffix_triggers_ae() -> None:
    """gelmez: -mez ek obstruent /z/ koda olmasına rağmen /æ/ tetikler."""
    phones = ["ɟ", "e", "l", "m", "e", "z"]
    result = e_open.apply(phones, word="gelmez")
    # Hem "gel" (l sonorant koda) hem "mez" /æ/ alır
    assert result == ["ɟ", "æ", "l", "m", "æ", "z"]


def test_rr_geminate_handled_via_lex() -> None:
    """cerrah: /rr/ geminate kapalı /e/ — lex exception ile çözülür.

    Surface /e/+/ɾ/+/ɾ/ pattern syllabify ile [d͡ʒeɾ][ɾah] olarak ayrılır;
    ilk hecenin coda'sı ['ɾ'] tek elemanlı olduğu için kural-tabanlı geminate
    algılama yapılamaz. Bu sözcükler ``no_e_open`` lex listesinde tutulur."""
    from g2p.pipeline import transcribe_word
    result = transcribe_word("cerrah")
    assert "æ" not in result.phonemes
    assert result.source == "exception"
