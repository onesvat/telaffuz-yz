"""Context-free grapheme → base IPA dönüşümü.

Final G2P metodolojisindeki harf-fonem tablosunu uygular. Her Türkçe harf için
varsayılan (bağlam-bağımsız) IPA çıktısı üretir; bağlama duyarlı kurallar
palatalizasyon, soft-g, açma ve diğer kural modüllerinde uygulanır.

ğ harfinin tek başına IPA karşılığı yoktur (`SOFT_G_PLACEHOLDER` döner);
`rules/soft_g.py` bağlama göre uzun ünlüye, [j]'ye veya silmeye çevirir.
"""

from __future__ import annotations

from g2p.constants import CIRCUMFLEX_MAP, SHORT_TO_LONG

SOFT_G_PLACEHOLDER: str = "ğ"

GRAPHEME_TO_IPA: dict[str, str] = {
    "a": "a",
    "b": "b",
    "c": "d͡ʒ",
    "ç": "t͡ʃ",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "ı": "ɯ",
    "i": "i",
    "j": "ʒ",
    "k": "k",
    "l": "ɫ",
    "m": "m",
    "n": "n",
    "o": "o",
    "ö": "œ",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "u": "u",
    "ü": "y",
    "v": "v",
    "y": "j",
    "z": "z",
}

# Türkçe alfabesinde olmayan ama alıntı/kısaltmalarda görülen harfler.
EXTRA_LETTERS: dict[str, list[str]] = {
    "q": ["k"],
    "w": ["v"],
    "x": ["k", "s"],
}


def _lower_tr(text: str) -> str:
    """Türkçe-duyarlı küçük harf çevrimi: I → ı, İ → i, diğerleri standart."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def to_ipa(word: str) -> list[str]:
    """Sözcüğü bağlam-bağımsız IPA fonem listesine çevirir.

    Büyük harf girdileri otomatik küçük harfe çevrilir. Tanınmayan
    karakterler (boşluk, noktalama vb.) atlanır.

    Args:
        word: Türkçe sözcük (büyük/küçük harf farketmez).

    Returns:
        IPA fonem dizisi. ğ için ``SOFT_G_PLACEHOLDER`` korunur; downstream
        kural bağlamı çözer.
    """
    word_lc = _lower_tr(word)
    out: list[str] = []
    for char in word_lc:
        if char in CIRCUMFLEX_MAP:
            base = CIRCUMFLEX_MAP[char]
            out.append(SHORT_TO_LONG[base])
        elif char == SOFT_G_PLACEHOLDER:
            out.append(SOFT_G_PLACEHOLDER)
        elif char in GRAPHEME_TO_IPA:
            out.append(GRAPHEME_TO_IPA[char])
        elif char in EXTRA_LETTERS:
            out.extend(EXTRA_LETTERS[char])
    return out
