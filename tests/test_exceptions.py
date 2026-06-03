"""exceptions yükleyicisi için unit testler."""

from __future__ import annotations

from g2p.exceptions import (
    DATA_PATH,
    EXPECTED_COLUMNS,
    ExceptionEntry,
    EXCEPTION_TYPES,
    _load,
    all_words,
    by_type,
    lookup,
    lookup_case_insensitive,
    validate_csv,
)


class TestLoading:
    def test_csv_path_exists(self) -> None:
        assert DATA_PATH.exists()

    def test_load_returns_non_empty(self) -> None:
        entries = _load()
        assert len(entries) > 0

    def test_expected_size_after_final_seed(self) -> None:
        # G2P final v3: seed büyütüldü (no_e_open + place_name + long_vowel).
        # Boyut bandı ileride seed gelişimini kabul eden geniş bir kontroldür.
        assert 320 <= len(_load()) <= 500

    def test_schema_and_inventory_are_valid(self) -> None:
        assert validate_csv() == ()

    def test_expected_columns_are_locked(self) -> None:
        assert EXPECTED_COLUMNS == (
            "word",
            "phonemes",
            "ipa_detail",
            "type",
            "suffix_behavior",
            "status",
            "source",
            "notes",
        )


class TestLookup:
    def test_geleceğim(self) -> None:
        entry = lookup("geleceğim")
        assert entry is not None
        assert entry.word == "geleceğim"
        assert entry.phonemes == ("ɟ", "e", "l", "i", "d͡ʒ", "eː", "m")
        assert entry.type == "narrowing"

    def test_sonra(self) -> None:
        entry = lookup("sonra")
        assert entry is not None
        assert entry.phonemes == ("s", "oː", "ɾ", "a")
        assert entry.type == "special"

    def test_unknown_word_returns_none(self) -> None:
        assert lookup("zzzznotaword") is None

    def test_case_sensitive(self) -> None:
        # AB ve ab farklı kayıtlar (CSV'de hem büyük hem küçük var)
        ab_entry = lookup("ab")
        AB_entry = lookup("AB")
        assert ab_entry is not None
        assert AB_entry is not None
        assert ab_entry.phonemes != AB_entry.phonemes

    def test_lookup_returns_frozen_dataclass(self) -> None:
        entry = lookup("sonra")
        assert entry is not None
        try:
            entry.word = "değişti"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("ExceptionEntry should be frozen")


class TestCaseInsensitive:
    def test_falls_back_to_lower(self) -> None:
        # SONRA → sonra fallback
        entry = lookup_case_insensitive("SONRA")
        assert entry is not None
        assert entry.word == "sonra"

    def test_direct_match_preferred(self) -> None:
        # AB direkt match — küçük harfli ab'a düşmemeli
        entry = lookup_case_insensitive("AB")
        assert entry is not None
        assert entry.word == "AB"

    def test_turkish_dotted_i_lowering(self) -> None:
        entry = lookup_case_insensitive("İSTANBUL")
        assert entry is not None
        assert entry.word == "istanbul"


class TestByType:
    def test_long_vowel_entries(self) -> None:
        entries = by_type("long_vowel")
        assert len(entries) > 0
        for entry in entries:
            assert entry.type == "long_vowel"

    def test_unknown_type_returns_empty(self) -> None:
        assert by_type("nonexistent_type_xyz") == ()

    def test_exception_type_whitelist_covers_data(self) -> None:
        for entry in _load().values():
            assert entry.type in EXCEPTION_TYPES


class TestAllWords:
    def test_returns_frozenset(self) -> None:
        words = all_words()
        assert isinstance(words, frozenset)
        assert len(words) > 0

    def test_contains_known_word(self) -> None:
        assert "sonra" in all_words()
        assert "geleceğim" in all_words()


class TestPhonemeSplitting:
    def test_affricates_kept_as_single_token(self) -> None:
        entry = lookup("geleceğim")
        assert entry is not None
        # /d͡ʒ/ tek token olarak korunmalı (boşlukla ayrılmamış)
        assert "d͡ʒ" in entry.phonemes

    def test_long_vowels_kept_as_single_token(self) -> None:
        entry = lookup("geleceğim")
        assert entry is not None
        # /eː/ tek token
        assert "eː" in entry.phonemes

    def test_cleaned_mutation_entries_do_not_use_gamma(self) -> None:
        entry = lookup("sokağa")
        assert entry is not None
        assert "ɣ" not in entry.phonemes
        assert "aː" in entry.phonemes


class TestEntryStructure:
    def test_entry_fields(self) -> None:
        entry = lookup("sonra")
        assert entry is not None
        assert hasattr(entry, "word")
        assert hasattr(entry, "phonemes")
        assert hasattr(entry, "ipa_detail")
        assert hasattr(entry, "type")
        assert hasattr(entry, "suffix_behavior")
        assert hasattr(entry, "status")
        assert hasattr(entry, "source")
        assert hasattr(entry, "notes")
        assert isinstance(entry, ExceptionEntry)
