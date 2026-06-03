"""Birincil vurgu yerleştirme.

İstanbul Türkçesi stres kuralları (Sezer 1981, Wikipedia Turkish phonology,
Göksel-Kerslake 2005):

- **Default**: son hece (word-final).
- **Pre-accent suffixleri**: vurgu ilgili ekin solundaki heceye yerleştirilir.
  Negation -mA, vasıta -lA, eşitlik -cA, sıfat türeten -ki, copula -(y)Im,
  -(y)Iz, -sIn, -sInIz, -DIr. Stres-attracting -Hyor, -(y)ArAk, -(y)IncA,
  -(I)vEr da pre-accent gibi davranır.
- **Yer adı (place_name) stresi**: exception_type ile gelir; ipa_detail
  alanında explicit vurgu kullanılır (Sezer cretic algoritması bireysel
  lex girdilerine kodlanmıştır).
- **Lex initial stress**: INITIAL_STRESS_WORDS listesi (bilgisayar, sinema,
  lokanta, başkomutan, simsiyah, …) → ilk hece.
- **-mA POS guard**: -mA suffix'i sonrasında nominal morfoloji
  (-sH, -lHk, -Hn, -nDA, -lAr, …) varsa türetme isim demektir → pre-accent
  uygulanmaz, default final stres kullanılır.
"""

from __future__ import annotations

from g2p.constants import STRESS
from g2p.morphology import MorphAnalysis
from g2p.syllabifier import is_closed, nucleus_index, syllabify


INITIAL_STRESS_WORDS: frozenset[str] = frozenset(
    {
        "lokanta",
        "gazete",
        "roman",
        "sinema",
        "masa",
        "bilgisayar",
        "türkçe",
        "birçok",
        "pazartesi",
        "başkomutan",
        "simsiyah",
        "bembeyaz",
        "apaçık",
        "kapkara",
        "yepyeni",
        # Konjonksiyon / interrogative sınıfı (Sezer 1981 Tablo 5; Wikipedia
        # Turkish phonology — Stress): initial-stress lexikal istisnalar.
        "çünkü",
        "sanki",
        "hangi",
    }
)
# Pre-accenting suffixes: vurgu ilgili ekin solundaki heceye gider.
PRE_ACCENT_ARCHIPHONEMES: frozenset[str] = frozenset(
    {
        "mA",  # negation (verbal -mA — POS-guarded below)
        "lA",  # -lA vasıta
        "ylA",  # -lA'nın y-li allomorph'u
        "cA",  # -cA eşitlik
        "ki",  # -ki sıfat türeten
        "yHm",  # 1sg copula
        "yHz",  # 1pl copula
        "SHn",  # 2sg copula
        "SHnHz",  # 2pl copula
        "DHr",  # 3sg copula -dIr
        # Stres-attracting suffix'ler (Sezer pre-accent davranışı):
        "Hyor",  # -iyor / -ıyor / -uyor / -üyor
        "yArAk",  # -erek / -arak
        "HncA",  # -ince / -ınca / -unca / -ünce
        "Hver",  # -iver / -ıver / -uver / -üver
    }
)
# Bu copula suffixleri sadece VERB POS'unda pre-accent davranır. NOUN/ADJ
# root + bu suffix durumunda default final stres uygulanır.
COPULA_PRE_ACCENT_ONLY_FOR_VERB: frozenset[str] = frozenset(
    {"yHm", "yHz", "SHn", "SHnHz"}
)
# -mA sonrası bu suffixlerden biri varsa, -mA negation değil nominal türetmedir
# (dökme, türetme, seçmen, yetmezlik, vb.). Pre-accent uygulanmaz.
NOMINAL_SUFFIXES_AFTER_MA: frozenset[str] = frozenset(
    {
        "sH",   # 3sg possessive
        "lHk",  # -lik nominal
        "Hn",   # -in agent / 2sg poss
        "nDA",  # locative
        "DAn",  # ablative
        "nHn",  # genitive
        "yA",   # dative
        "ki",   # -ki relative (separately pre-accents)
    }
)
# -mA'nın solunda yer alırsa, -mA verbal negation değil nominal türetme demektir
# (caustive/passive/reciprocal sonrası -mA = derivational nominal).
DERIVATIONAL_VOICE_SUFFIXES: frozenset[str] = frozenset(
    {
        "t",    # caustive (türet, soğut)
        "DHr",  # caustive (yatır, içir)
        "Hl",   # passive (yazıl, görül)
        "Hn",   # passive/reflexive (giyin)
        "Hş",   # reciprocal (görüş, çekiş)
    }
)
_ARCHI_VOWELS: frozenset[str] = frozenset("AHIUaeıioöuüâîû")


def apply(
    phonemes: list[str],
    *,
    word: str = "",
    exception_type: str = "",
    ipa_detail: str = "",
    morphology: MorphAnalysis | None = None,
) -> list[str]:
    phones = [p for p in phonemes if p != STRESS]
    syllables = syllabify(phones)
    if len(syllables) <= 1:
        return phones

    explicit = _stress_from_ipa_detail(ipa_detail, syllables)
    if explicit is not None:
        return _insert_stress(syllables, explicit)

    word_lc = _lower_tr(word)
    stress_idx = _stress_index(word_lc, syllables, exception_type, morphology)
    return _insert_stress(syllables, stress_idx)


def _stress_index(
    word_lc: str,
    syllables: list[list[str]],
    exception_type: str,
    morphology: MorphAnalysis | None,
) -> int:
    if word_lc in INITIAL_STRESS_WORDS or exception_type in {"stress", "thin_l"}:
        return 0

    if morphology is not None:
        suffix_tuple = morphology.suffixes
        # Pre-accent suffix bul (kelimenin sondan başına doğru tarar).
        for idx, suffix in enumerate(suffix_tuple):
            if suffix not in PRE_ACCENT_ARCHIPHONEMES:
                continue
            # Copula suffixleri yalnızca VERB POS'ta pre-accent davranır.
            if suffix in COPULA_PRE_ACCENT_ONLY_FOR_VERB and morphology.pos != "VERB":
                continue
            # -mA türetme isim ayrımı: sonrasında nominal morfoloji varsa,
            # veya solunda caustive/passive/reciprocal suffix varsa → türetme
            # nominal (türetme, ettirme, sürülme); pre-accent uygulama.
            if suffix == "mA":
                post = suffix_tuple[idx + 1 :]
                if any(s in NOMINAL_SUFFIXES_AFTER_MA for s in post):
                    continue
                prev_sfx = suffix_tuple[idx - 1] if idx > 0 else ""
                if prev_sfx in DERIVATIONAL_VOICE_SUFFIXES:
                    continue
            preceding_vowels = _archi_vowel_count(morphology.root) + sum(
                _archi_vowel_count(s) for s in suffix_tuple[:idx]
            )
            target = max(0, preceding_vowels - 1)
            return min(target, len(syllables) - 1)
        return len(syllables) - 1

    # Fallback: morfoloji yoksa string-end heuristic.
    return len(syllables) - 1


def _archi_vowel_count(text: str) -> int:
    return sum(1 for c in text if c in _ARCHI_VOWELS)


def _stress_from_ipa_detail(ipa_detail: str, syllables: list[list[str]]) -> int | None:
    """ipa_detail içindeki "ˈ" pozisyonuna göre hece index'i hesapla.

    İki format desteklenir:
    1. Açık hece sınırlı: ``pen.ˈd͡ʒeɾe`` — "." ile sayılır.
    2. Düz: ``penˈd͡ʒeɾe`` — "ˈ" solundaki ünlü sayısı = stres index.

    İkinci format için ünlü olarak: ``a, e, i, ɯ, o, œ, u, y, æ`` ve uzun
    varyantları. Stress mark hece çekirdeğinin solunda yer alır.
    """
    if "ˈ" not in ipa_detail:
        return None
    prefix = ipa_detail.split("ˈ", 1)[0]
    if "." in prefix:
        return min(
            len([part for part in prefix.split(".") if part]), len(syllables) - 1
        )
    if prefix == "":
        return 0
    # "." yoksa ünlü sayısına göre hece pozisyonunu hesapla.
    # Hem Türkçe ortografik ünlüler hem IPA ünlüleri sayılır.
    vowel_chars = set("aeıioöuüâîû") | set("aeiɯoœuyæ")
    vowel_count = 0
    i = 0
    while i < len(prefix):
        ch = prefix[i]
        if ch in vowel_chars:
            vowel_count += 1
            # Long vowel marker takip ediyorsa atla
            if i + 1 < len(prefix) and prefix[i + 1] == "ː":
                i += 1
        i += 1
    return min(vowel_count, len(syllables) - 1)


def _insert_stress(syllables: list[list[str]], stress_index: int) -> list[str]:
    out: list[str] = []
    for idx, syllable in enumerate(syllables):
        if idx == stress_index:
            nucleus = nucleus_index(syllable)
            if nucleus <= 0:
                out.append(STRESS)
            else:
                out.extend(syllable[:nucleus])
                out.append(STRESS)
                out.extend(syllable[nucleus:])
                continue
        out.extend(syllable)
    return out


def is_heavy(syllable: list[str]) -> bool:
    return is_closed(syllable)


def _lower_tr(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()
