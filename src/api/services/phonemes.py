"""Phoneme metadata catalog for the demo UI palette.

Single source of truth for tooltip text, examples, and category grouping. The
table mirrors ``g2p.constants.ALL_PHONEMES`` and additionally carries the
suprasegmental stress marker and the syllable boundary so the UI can render
them alongside segmental phones. End-user strings remain in Turkish since the
demo learner UI is Turkish-facing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SYLLABLE_BOUNDARY: str = "."


@dataclass(frozen=True)
class Phoneme:
    symbol: str
    name_tr: str
    example: str
    hint_tr: str
    category: str


CATEGORY_ORDER: tuple[str, ...] = (
    "vowel_short",
    "vowel_long",
    "vowel_allo",
    "cons",
    "cons_allo",
    "aspire",
    "supra",
)

CATEGORY_LABELS: dict[str, str] = {
    "vowel_short": "Ünlü — kısa",
    "vowel_long": "Ünlü — uzun",
    "vowel_allo": "Ünlü allofonu",
    "cons": "Ünsüz",
    "cons_allo": "Ünsüz allofonu",
    "aspire": "Aspire / sağırlaşmış",
    "supra": "Vurgu / hece",
}


PHONEMES: dict[str, Phoneme] = {
    # ---- Kısa ünlü (8) -------------------------------------------------
    "a": Phoneme("a", "açık arka düz ünlü", "ana → /ana/", "Türkçe a", "vowel_short"),
    "e": Phoneme(
        "e",
        "yarı-açık ön düz ünlü",
        "ev → /ev/",
        "Türkçe e (açık hece, kapanmayan)",
        "vowel_short",
    ),
    "i": Phoneme("i", "kapalı ön düz ünlü", "ip → /ip/", "Türkçe i", "vowel_short"),
    "ɯ": Phoneme(
        "ɯ", "kapalı arka düz ünlü", "ılık → /ɯlɯk/", "Türkçe ı", "vowel_short"
    ),
    "o": Phoneme(
        "o", "yarı-açık arka yuvarlak ünlü", "ok → /ok/", "Türkçe o", "vowel_short"
    ),
    "œ": Phoneme(
        "œ", "yarı-açık ön yuvarlak ünlü", "öl → /œl/", "Türkçe ö", "vowel_short"
    ),
    "u": Phoneme(
        "u", "kapalı arka yuvarlak ünlü", "un → /un/", "Türkçe u", "vowel_short"
    ),
    "y": Phoneme(
        "y", "kapalı ön yuvarlak ünlü", "üst → /yst/", "Türkçe ü", "vowel_short"
    ),
    # ---- Uzun ünlü (8) -------------------------------------------------
    "aː": Phoneme(
        "aː",
        "uzun a",
        "saat → /saːt/",
        "Uzun a (loanword veya soft-g uzaması)",
        "vowel_long",
    ),
    "eː": Phoneme(
        "eː", "uzun e", "memur → /meːmuɾ/", "Nadir; ödünç kelimelerde", "vowel_long"
    ),
    "iː": Phoneme(
        "iː", "uzun i", "tarih → /taːɾiː/", "Nadir; ödünç kelimelerde", "vowel_long"
    ),
    "ɯː": Phoneme(
        "ɯː",
        "uzun ı",
        "sığır → /sɯːɯɾ/",
        "Çoğunlukla soft-g uzamasından gelir (ığ)",
        "vowel_long",
    ),
    "oː": Phoneme(
        "oː",
        "uzun o",
        "boğaz → /boːaz/",
        "Back-back soft-g intervokalik uzaması",
        "vowel_long",
    ),
    "œː": Phoneme("œː", "uzun ö", "töre → /tœːɾe/", "Nadir; ödünç", "vowel_long"),
    "uː": Phoneme(
        "uː",
        "uzun u",
        "uğraş → /uːɾaʃ/",
        "Soft-g uzaması veya ödünç",
        "vowel_long",
    ),
    "yː": Phoneme(
        "yː", "uzun ü", "düğme → /dyːme/", "Soft-g uzaması veya ödünç", "vowel_long"
    ),
    # ---- Ünlü allofonu (1) ---------------------------------------------
    "æ": Phoneme(
        "æ",
        "açık ön düz ünlü (e-allofonu)",
        "el → /æl/",
        "Kapalı hecede /e/'nin geniş açılımı; 'el', 'denk' gibi",
        "vowel_allo",
    ),
    # ---- Ünsüz fonem (20) ----------------------------------------------
    "p": Phoneme("p", "sessiz çift-dudaksıl duraklı", "pek → /pek/", "Türkçe p", "cons"),
    "b": Phoneme(
        "b", "sesli çift-dudaksıl duraklı", "balık → /balɯk/", "Türkçe b", "cons"
    ),
    "t": Phoneme("t", "sessiz diş ardı duraklı", "tek → /tek/", "Türkçe t", "cons"),
    "d": Phoneme("d", "sesli diş ardı duraklı", "dut → /dut/", "Türkçe d", "cons"),
    "k": Phoneme(
        "k",
        "sessiz arka damak duraklı",
        "kek → /kek/",
        "Türkçe kalın k (velar)",
        "cons",
    ),
    "ɡ": Phoneme(
        "ɡ",
        "sesli arka damak duraklı",
        "gül → /ɡyl/",
        "Türkçe kalın g (velar); LATIN g değil — IPA ɡ",
        "cons",
    ),
    "t͡ʃ": Phoneme(
        "t͡ʃ",
        "sessiz post-alveolar yarı-kapantılı",
        "çay → /t͡ʃaj/",
        "Türkçe ç",
        "cons",
    ),
    "d͡ʒ": Phoneme(
        "d͡ʒ",
        "sesli post-alveolar yarı-kapantılı",
        "cam → /d͡ʒam/",
        "Türkçe c",
        "cons",
    ),
    "f": Phoneme("f", "sessiz dudak-diş sürtünmeli", "fil → /fil/", "Türkçe f", "cons"),
    "v": Phoneme(
        "v",
        "sesli dudak-diş sürtünmeli",
        "vakit → /vakit/",
        "Türkçe v (söz-başı)",
        "cons",
    ),
    "s": Phoneme("s", "sessiz alveolar sürtünmeli", "su → /su/", "Türkçe s", "cons"),
    "z": Phoneme("z", "sesli alveolar sürtünmeli", "zar → /zaɾ/", "Türkçe z", "cons"),
    "ʃ": Phoneme(
        "ʃ", "sessiz post-alveolar sürtünmeli", "şu → /ʃu/", "Türkçe ş", "cons"
    ),
    "ʒ": Phoneme(
        "ʒ",
        "sesli post-alveolar sürtünmeli",
        "jandarma → /ʒandaɾma/",
        "Türkçe j",
        "cons",
    ),
    "h": Phoneme(
        "h", "sessiz gırtlak sürtünmeli", "hava → /hava/", "Türkçe h", "cons"
    ),
    "m": Phoneme(
        "m", "çift-dudaksıl burunsu", "ev → /ev/, mum → /mum/", "Türkçe m", "cons"
    ),
    "n": Phoneme("n", "alveolar burunsu", "anne → /anne/", "Türkçe n", "cons"),
    "l": Phoneme(
        "l", "alveolar yan ünsüz (ince)", "el → /æl/", "İnce l (ön ünlü yanında)", "cons"
    ),
    "ɾ": Phoneme(
        "ɾ",
        "alveolar tek vuruş (tap)",
        "ara → /aɾa/",
        "Normal Türk r (kısa tap, İngilizce 'butter'daki t gibi)",
        "cons",
    ),
    "j": Phoneme("j", "ön damak yarı-ünlüsü", "yel → /jæl/", "Türkçe y", "cons"),
    # ---- Ünsüz allofonu (7) --------------------------------------------
    "c": Phoneme(
        "c",
        "sessiz ön damak duraklı (k-allofonu)",
        "kâğıt → /caːɯt/",
        "İnce k; 'kâ', 'kü' gibi ön ünlü yanında",
        "cons_allo",
    ),
    "ɟ": Phoneme(
        "ɟ",
        "sesli ön damak duraklı (g-allofonu)",
        "gül → /ɟyl/",
        "İnce g; ön ünlü yanında",
        "cons_allo",
    ),
    "ɲ": Phoneme(
        "ɲ",
        "ön damak burunsu (n-allofonu)",
        "anca → /aɲd͡ʒa/",
        "Palatal n; /c, ɟ, t͡ʃ, d͡ʒ/ öncesinde",
        "cons_allo",
    ),
    "ŋ": Phoneme(
        "ŋ",
        "arka damak burunsu (n-allofonu)",
        "denk → /dæŋk/",
        "Velar n; /k, ɡ/ öncesinde",
        "cons_allo",
    ),
    "ɫ": Phoneme(
        "ɫ",
        "koyu (velarize) alveolar yan (l-allofonu)",
        "kol → /kɫoɫ/, kalk → /kɫaɫk/",
        "Kalın l; arka ünlü yanında",
        "cons_allo",
    ),
    "β": Phoneme(
        "β",
        "sesli çift-dudaksıl sürtünmeli (v-allofonu)",
        "tava → /taβa/",
        "İntervokalik v",
        "cons_allo",
    ),
    "β̞": Phoneme(
        "β̞",
        "sesli çift-dudaksıl yarı-ünlü (v-allofonu)",
        "evvel → /eβ̞el/",
        "v yakınsayan zayıf allofon",
        "cons_allo",
    ),
    # ---- Aspire + r-sağırlaşma (5) -------------------------------------
    "pʰ": Phoneme(
        "pʰ",
        "aspire p",
        "para → /pʰaˈɾa/",
        "Hece-başı sessiz durakta hava üflemesi",
        "aspire",
    ),
    "tʰ": Phoneme(
        "tʰ", "aspire t", "taş → /tʰaʃ/", "Hece-başı /t/ allofonu", "aspire"
    ),
    "kʰ": Phoneme(
        "kʰ", "aspire k", "kabuk → /kʰabuk/", "Hece-başı kalın k allofonu", "aspire"
    ),
    "cʰ": Phoneme(
        "cʰ",
        "aspire palatal k",
        "kedi → /cʰedi/",
        "Hece-başı ince k allofonu",
        "aspire",
    ),
    "ɾ̞̊": Phoneme(
        "ɾ̞̊",
        "sağırlaşmış r",
        "bir → /biɾ̞̊/",
        "Sözcük-sonu /ɾ/ allofonu (sürtünmeli, sessizleşmiş)",
        "aspire",
    ),
    # ---- Suprasegmental (2) --------------------------------------------
    "ˈ": Phoneme(
        "ˈ",
        "vurgu işareti",
        "/kʰa.ɾɯ.d͡ʒɯː.ˈɯm/",
        "Sonraki hece vurguludur",
        "supra",
    ),
    SYLLABLE_BOUNDARY: Phoneme(
        SYLLABLE_BOUNDARY,
        "hece sınırı",
        "/kʰa.ɾa/",
        "İki hece arasında kullanılır",
        "supra",
    ),
}


def grouped_palette() -> list[tuple[str, str, list[Phoneme]]]:
    groups: dict[str, list[Phoneme]] = {cat: [] for cat in CATEGORY_ORDER}
    for ph in PHONEMES.values():
        groups[ph.category].append(ph)
    return [(cat, CATEGORY_LABELS[cat], groups[cat]) for cat in CATEGORY_ORDER]


def phonemes_as_dict() -> dict[str, dict[str, str]]:
    return {sym: asdict(ph) for sym, ph in PHONEMES.items()}
