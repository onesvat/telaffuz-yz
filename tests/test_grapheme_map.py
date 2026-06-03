"""grapheme_map.to_ipa için unit testler."""

import pytest

from g2p.grapheme_map import GRAPHEME_TO_IPA, SOFT_G_PLACEHOLDER, _lower_tr, to_ipa


class TestLowerTr:
    def test_capital_i_becomes_dotless(self) -> None:
        assert _lower_tr("I") == "ı"

    def test_capital_dotted_i_becomes_dotted(self) -> None:
        assert _lower_tr("İ") == "i"

    def test_kiyi_word(self) -> None:
        assert _lower_tr("KIYI") == "kıyı"

    def test_mixed_case(self) -> None:
        assert _lower_tr("İSTANBUL") == "istanbul"

    def test_other_letters_lowercase(self) -> None:
        assert _lower_tr("ÇÖĞÜŞ") == "çöğüş"


class TestSingleLetters:
    @pytest.mark.parametrize(
        "letter,expected",
        [
            ("a", "a"),
            ("b", "b"),
            ("c", "d͡ʒ"),
            ("ç", "t͡ʃ"),
            ("d", "d"),
            ("e", "e"),
            ("f", "f"),
            ("g", "ɡ"),
            ("h", "h"),
            ("ı", "ɯ"),
            ("i", "i"),
            ("j", "ʒ"),
            ("k", "k"),
            ("l", "ɫ"),
            ("m", "m"),
            ("n", "n"),
            ("o", "o"),
            ("ö", "œ"),
            ("p", "p"),
            ("r", "ɾ"),
            ("s", "s"),
            ("ş", "ʃ"),
            ("t", "t"),
            ("u", "u"),
            ("ü", "y"),
            ("v", "v"),
            ("y", "j"),
            ("z", "z"),
        ],
    )
    def test_default_mapping(self, letter: str, expected: str) -> None:
        assert to_ipa(letter) == [expected]

    def test_all_29_letters_covered(self) -> None:
        letters = "abcçdefgğhıijklmnoöprsştuüvyz"
        for letter in letters:
            if letter == "ğ":
                continue
            assert letter in GRAPHEME_TO_IPA


class TestCircumflex:
    def test_a_circumflex_long(self) -> None:
        assert to_ipa("â") == ["aː"]

    def test_i_circumflex_long(self) -> None:
        assert to_ipa("î") == ["iː"]

    def test_u_circumflex_long(self) -> None:
        assert to_ipa("û") == ["uː"]

    def test_kar_circumflex(self) -> None:
        # 'kâr' loanword: k + â (long a) + r
        assert to_ipa("kâr") == ["k", "aː", "ɾ"]


class TestSoftG:
    def test_soft_g_returns_placeholder(self) -> None:
        assert to_ipa("ğ") == [SOFT_G_PLACEHOLDER]

    def test_dag_includes_placeholder(self) -> None:
        assert to_ipa("dağ") == ["d", "a", "ğ"]


class TestSimpleWords:
    def test_kara(self) -> None:
        assert to_ipa("kara") == ["k", "a", "ɾ", "a"]

    def test_anne(self) -> None:
        assert to_ipa("anne") == ["a", "n", "n", "e"]

    def test_baba(self) -> None:
        assert to_ipa("baba") == ["b", "a", "b", "a"]

    def test_turkce(self) -> None:
        assert to_ipa("Türkçe") == ["t", "y", "ɾ", "k", "t͡ʃ", "e"]

    def test_istanbul(self) -> None:
        assert to_ipa("İstanbul") == ["i", "s", "t", "a", "n", "b", "u", "ɫ"]


class TestUppercase:
    def test_kiyi_uppercase(self) -> None:
        assert to_ipa("KIYI") == ["k", "ɯ", "j", "ɯ"]

    def test_capital_i_dotted(self) -> None:
        assert to_ipa("İYİ") == ["i", "j", "i"]


class TestExtraLetters:
    def test_q_to_k(self) -> None:
        assert to_ipa("q") == ["k"]

    def test_w_to_v(self) -> None:
        assert to_ipa("w") == ["v"]

    def test_x_to_ks(self) -> None:
        assert to_ipa("x") == ["k", "s"]


class TestPunctuationAndWhitespace:
    def test_spaces_skipped(self) -> None:
        assert to_ipa("a b") == ["a", "b"]

    def test_punctuation_skipped(self) -> None:
        assert to_ipa("anne!") == ["a", "n", "n", "e"]

    def test_apostrophe_skipped(self) -> None:
        assert to_ipa("İstanbul'a") == ["i", "s", "t", "a", "n", "b", "u", "ɫ", "a"]


class TestEmpty:
    def test_empty_string(self) -> None:
        assert to_ipa("") == []

    def test_only_punctuation(self) -> None:
        assert to_ipa("...") == []
