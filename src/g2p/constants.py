"""49-fonem envanteri ve fonolojik kategori sabitleri.

Final fonem envanterinin runtime karşılığıdır. Tüm değişken adları fonolojik
kategorilere göredir; allofon → fonem dönüşümleri için ayrıca haritalar
tanımlanır.
"""

from __future__ import annotations

STRESS = "ˈ"

SHORT_VOWELS: frozenset[str] = frozenset({"a", "e", "i", "ɯ", "o", "œ", "u", "y"})
LONG_VOWELS: frozenset[str] = frozenset(
    {"aː", "eː", "iː", "ɯː", "oː", "œː", "uː", "yː"}
)
VOWEL_ALLOPHONES: frozenset[str] = frozenset({"æ"})
ALL_VOWELS: frozenset[str] = SHORT_VOWELS | LONG_VOWELS | VOWEL_ALLOPHONES

CONSONANTS: frozenset[str] = frozenset(
    {
        "p",
        "b",
        "t",
        "d",
        "k",
        "ɡ",
        "t͡ʃ",
        "d͡ʒ",
        "f",
        "v",
        "s",
        "z",
        "ʃ",
        "ʒ",
        "h",
        "m",
        "n",
        "l",
        "ɾ",
        "j",
    }
)
CONS_BASE_ALLOPHONES: frozenset[str] = frozenset({"c", "ɟ", "ɲ", "ŋ", "ɫ", "β", "β̞"})
CONS_EXTRA_ALLOPHONES: frozenset[str] = frozenset({"pʰ", "tʰ", "kʰ", "cʰ", "ɾ̞̊"})
ALL_CONSONANTS: frozenset[str] = (
    CONSONANTS | CONS_BASE_ALLOPHONES | CONS_EXTRA_ALLOPHONES
)

ALL_PHONEMES: frozenset[str] = ALL_VOWELS | ALL_CONSONANTS | {STRESS}

# /æ/ ön ünlü kategorisine dahildir; palatalizasyon ve l-allofonisi onu da tetikler.
FRONT_VOWELS: frozenset[str] = frozenset({"e", "i", "œ", "y", "æ"})
BACK_VOWELS: frozenset[str] = frozenset({"a", "ɯ", "o", "u"})
ROUNDED_VOWELS: frozenset[str] = frozenset({"o", "œ", "u", "y"})
UNROUNDED_VOWELS: frozenset[str] = frozenset({"a", "e", "i", "ɯ", "æ"})
HIGH_VOWELS: frozenset[str] = frozenset({"i", "ɯ", "u", "y"})
NON_HIGH_VOWELS: frozenset[str] = frozenset({"a", "e", "o", "œ", "æ"})

SHORT_TO_LONG: dict[str, str] = {
    "a": "aː",
    "e": "eː",
    "i": "iː",
    "ɯ": "ɯː",
    "o": "oː",
    "œ": "œː",
    "u": "uː",
    "y": "yː",
}
LONG_TO_SHORT: dict[str, str] = {v: k for k, v in SHORT_TO_LONG.items()}

SONORANTS: frozenset[str] = frozenset({"l", "ɫ", "m", "n", "ɲ", "ŋ", "ɾ", "ɾ̞̊", "j"})
VOICELESS_CONS: frozenset[str] = frozenset(
    {"p", "t", "k", "c", "t͡ʃ", "f", "s", "ʃ", "h", "pʰ", "tʰ", "kʰ", "cʰ"}
)
VOICED_OBSTRUENTS: frozenset[str] = frozenset(
    {"b", "d", "ɡ", "ɟ", "d͡ʒ", "v", "z", "ʒ", "β", "β̞"}
)

PLAIN_STOPS: frozenset[str] = frozenset({"p", "t", "k", "c"})
ASPIRATED_MAP: dict[str, str] = {"p": "pʰ", "t": "tʰ", "k": "kʰ", "c": "cʰ"}

PALATAL_K: str = "c"
PALATAL_G: str = "ɟ"
NASAL_VELAR: str = "ŋ"
NASAL_PALATAL: str = "ɲ"
L_DARK: str = "ɫ"
V_BILABIAL: str = "β"
V_APPROXIMANT: str = "β̞"
R_DEVOICED: str = "ɾ̞̊"

# Affricate'ler [ɲ] tetikler; bu ayrım telaffuz feedback yüzeyinde korunur.
NASAL_PALATAL_TRIGGERS: frozenset[str] = frozenset({"c", "ɟ", "t͡ʃ", "d͡ʒ"})
NASAL_VELAR_TRIGGERS: frozenset[str] = frozenset({"k", "ɡ"})

CIRCUMFLEX_MAP: dict[str, str] = {"â": "a", "î": "i", "û": "u"}
CIRCUMFLEX_LETTERS: frozenset[str] = frozenset(CIRCUMFLEX_MAP.keys())

CTC_BLANK: str = "<blank>"
CTC_PAD: str = "<pad>"
CTC_UNK: str = "<unk>"
CTC_SPC: str = "<spc>"
CTC_SPECIAL_TOKENS: tuple[str, ...] = (CTC_PAD, CTC_UNK, CTC_SPC, CTC_BLANK)
