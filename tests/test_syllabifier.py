"""syllabifier için unit testler."""

from g2p.grapheme_map import to_ipa
from g2p.syllabifier import (
    coda,
    is_closed,
    is_open,
    is_vowel,
    nucleus,
    nucleus_index,
    onset,
    syllabify,
)


class TestIsVowel:
    def test_short_vowel_true(self) -> None:
        assert is_vowel("a")
        assert is_vowel("ɯ")
        assert is_vowel("œ")

    def test_long_vowel_true(self) -> None:
        assert is_vowel("aː")
        assert is_vowel("œː")

    def test_allophone_ae(self) -> None:
        assert is_vowel("æ")

    def test_consonant_false(self) -> None:
        assert not is_vowel("k")
        assert not is_vowel("d͡ʒ")
        assert not is_vowel("ɫ")

    def test_soft_g_placeholder_false(self) -> None:
        assert not is_vowel("ğ")


class TestSyllabify:
    def test_empty(self) -> None:
        assert syllabify([]) == []

    def test_single_vowel(self) -> None:
        assert syllabify(["a"]) == [["a"]]

    def test_only_consonants(self) -> None:
        # Ünlü yok → tek "hece" olarak döndür (placeholder davranışı)
        assert syllabify(["k", "s"]) == [["k", "s"]]

    def test_kara_two_syllables(self) -> None:
        # ka.ra
        assert syllabify(to_ipa("kara")) == [["k", "a"], ["ɾ", "a"]]

    def test_anne_two_syllables(self) -> None:
        # an.ne
        assert syllabify(to_ipa("anne")) == [["a", "n"], ["n", "e"]]

    def test_ekmek_two_syllables(self) -> None:
        # ek.mek
        assert syllabify(to_ipa("ekmek")) == [["e", "k"], ["m", "e", "k"]]

    def test_dogru_two_syllables(self) -> None:
        # doğ.ru — ğ koda
        assert syllabify(to_ipa("doğru")) == [["d", "o", "ğ"], ["ɾ", "u"]]

    def test_bogaz_two_syllables(self) -> None:
        # bo.ğaz — ğ ikinci hece onset (intervokalik)
        assert syllabify(to_ipa("boğaz")) == [["b", "o"], ["ğ", "a", "z"]]

    def test_araba_three_syllables(self) -> None:
        # a.ra.ba
        assert syllabify(to_ipa("araba")) == [["a"], ["ɾ", "a"], ["b", "a"]]

    def test_kitap_two_syllables(self) -> None:
        # ki.tap
        assert syllabify(to_ipa("kitap")) == [["k", "i"], ["t", "a", "p"]]

    def test_istanbul_three_syllables(self) -> None:
        # is.tan.bul
        assert syllabify(to_ipa("İstanbul")) == [
            ["i", "s"],
            ["t", "a", "n"],
            ["b", "u", "ɫ"],
        ]

    def test_perende_three_syllables(self) -> None:
        # pe.ren.de
        assert syllabify(to_ipa("perende")) == [
            ["p", "e"],
            ["ɾ", "e", "n"],
            ["d", "e"],
        ]

    def test_kar_one_syllable_long_vowel(self) -> None:
        # kâr → [k, aː, ɾ] tek hece
        assert syllabify(to_ipa("kâr")) == [["k", "aː", "ɾ"]]

    def test_dag_one_syllable(self) -> None:
        # dağ → tek hece (ğ coda)
        assert syllabify(to_ipa("dağ")) == [["d", "a", "ğ"]]

    def test_oglan_two_syllables(self) -> None:
        # oğ.lan — grapheme_map default /ɫ/, l_allophony rule sonra clear yapacak
        assert syllabify(to_ipa("oğlan")) == [["o", "ğ"], ["ɫ", "a", "n"]]

    def test_yagmur_two_syllables(self) -> None:
        # yağ.mur — yğmr arasında 'ğ' ve 'm' iki ünsüz
        assert syllabify(to_ipa("yağmur")) == [["j", "a", "ğ"], ["m", "u", "ɾ"]]

    def test_denk_one_syllable(self) -> None:
        # denk — CVCC tek hece
        assert syllabify(to_ipa("denk")) == [["d", "e", "n", "k"]]

    def test_dert_one_syllable(self) -> None:
        # dert — CVCC tek hece
        assert syllabify(to_ipa("dert")) == [["d", "e", "ɾ", "t"]]

    def test_ders_one_syllable(self) -> None:
        # ders — CVCC tek hece
        assert syllabify(to_ipa("ders")) == [["d", "e", "ɾ", "s"]]

    def test_var_one_syllable(self) -> None:
        # var — CVC tek hece
        assert syllabify(to_ipa("var")) == [["v", "a", "ɾ"]]


class TestSyllableHelpers:
    def test_nucleus_index_simple(self) -> None:
        assert nucleus_index(["k", "a"]) == 1
        assert nucleus_index(["a"]) == 0
        assert nucleus_index(["k", "ɾ", "a", "l"]) == 2

    def test_nucleus_index_no_vowel(self) -> None:
        assert nucleus_index(["k", "s"]) == -1

    def test_nucleus(self) -> None:
        assert nucleus(["k", "a"]) == "a"
        assert nucleus(["d", "e", "ɾ", "s"]) == "e"
        assert nucleus(["k", "s"]) is None

    def test_onset(self) -> None:
        assert onset(["k", "a"]) == ["k"]
        assert onset(["a"]) == []
        assert onset(["d", "e", "ɾ", "s"]) == ["d"]

    def test_coda(self) -> None:
        assert coda(["k", "a"]) == []
        assert coda(["d", "e", "ɾ", "s"]) == ["ɾ", "s"]
        assert coda(["e", "k"]) == ["k"]

    def test_is_open_closed(self) -> None:
        assert is_open(["k", "a"])
        assert not is_closed(["k", "a"])
        assert is_closed(["d", "e", "ɾ", "s"])
        assert not is_open(["d", "e", "ɾ", "s"])

    def test_is_open_empty_false(self) -> None:
        assert not is_open([])
        assert not is_closed([])


class TestEKmekClosedSyllableEnv:
    """E→æ kuralı için kritik: ek.mek hecelemesi her hecede /e/+koda göstermeli."""

    def test_ekmek_first_syllable_closed_with_e_coda_k(self) -> None:
        syllables = syllabify(to_ipa("ekmek"))
        first = syllables[0]
        assert is_closed(first)
        assert nucleus(first) == "e"
        assert coda(first) == ["k"]

    def test_ekmek_second_syllable_closed_with_e_coda_k(self) -> None:
        syllables = syllabify(to_ipa("ekmek"))
        second = syllables[1]
        assert is_closed(second)
        assert nucleus(second) == "e"
        assert coda(second) == ["k"]
