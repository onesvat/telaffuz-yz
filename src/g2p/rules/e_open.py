"""Kapalı hecede /e/ → /æ/ açılması (dar kural).

İstanbul Türkçesi standardı ve fonoloji literatürü (Kornfilt 1997,
Göksel-Kerslake 2005, Sezer 1981) bu kuralın kapsamını dar tutar: /e/ → /æ/
SADECE kapalı hecede VE koda'nın ilk ünsüzü sonorant ise uygulanır
(``l, ɫ, m, n, r``). Obstruent koda bağlamında (``k, t, p, s, z, …``) /e/
KAPALI kalır.

İki sabit istisna:

1. ``/rr/`` geminat koda — kapalı /e/: cerrah, berrak, zerrin, tekerrür.
2. Leksikal ``no_e_open`` istisnaları (belli, elli, kendi, mühendis, …):
   sonorant koda olmasına rağmen /e/ tutulur.

Ek kuralı: ``-mez/-maz`` olumsuzluk-aorist suffix'i obstruent /z/ koda olmasına
rağmen /æ/ tetikler (Wikipedia §3.3, Sezer 1981; ``gelmez /ɟælˈmæz/``). Bu
fonotaktik değil morfolojik bir kuraldır; surface ``word.endswith("mez"|"maz")``
ile yakalanır.
"""

from __future__ import annotations

from g2p.syllabifier import coda, is_closed, syllabify


SONORANT_CODA_TRIGGERS: frozenset[str] = frozenset(
    {"l", "ɫ", "m", "n", "ɾ", "ŋ", "ɲ"}
)


NO_E_OPEN_LEX: frozenset[str] = frozenset(
    {
        # Sonorant koda olmasına rağmen kapalı /e/ kalan leksikal istisnalar
        # Bu sözcüklerde kapalı /e/ final hedef olarak korunur.
        "anne",
        "bellek",
        "belli",
        "belki",
        "bencil",
        "bengi",
        "bengü",
        "denge",
        "elli",
        "engin",
        "genç",
        "hem",
        "kendi",
        "mehmet",
        "mühendis",
        "pencere",
        "tencere",
    }
)


def apply(
    phonemes: list[str],
    *,
    word: str = "",
    exception_type: str = "",
) -> list[str]:
    word_lc = _lower_tr(word)

    # Leksikal istisna: /e/ tamamen tutulur.
    if word_lc in NO_E_OPEN_LEX or exception_type == "no_e_open":
        return list(phonemes)

    out: list[str] = []
    for syllable in syllabify(phonemes):
        if not is_closed(syllable):
            out.extend(syllable)
            continue
        c = coda(syllable)
        if not c:
            out.extend(syllable)
            continue
        # /rr/ geminat koda istisnası: kapalı /e/.
        if c[0] == "ɾ" and len(c) >= 2 and c[1] == "ɾ":
            out.extend(syllable)
            continue
        if c[0] in SONORANT_CODA_TRIGGERS:
            out.extend("æ" if p == "e" else p for p in syllable)
        else:
            out.extend(syllable)

    # -mez/-maz suffix post-handler: surface tabanlı, morphology bağımsız.
    if word_lc.endswith(("mez", "maz")):
        out = _open_final_mez(out)

    return out


def _open_final_mez(phonemes: list[str]) -> list[str]:
    """Son ``m V z`` pattern'inde /e/ → /æ/.

    /-mAz/ suffix'i obstruent /z/ koda olmasına rağmen /æ/ alır (morfolojik
    istisna). Surface-driven: en sondaki ``m, e, z`` 3-token bloğunu bul.
    """
    if len(phonemes) < 3:
        return phonemes
    out = list(phonemes)
    for i in range(len(out) - 3, -1, -1):
        if out[i] == "m" and out[i + 1] == "e" and out[i + 2] == "z":
            # Yalnızca eki ekleyen yüzey form için son /mez/ bloğunu yakala.
            if i + 2 == len(out) - 1 or _is_suffix_tail(out[i + 3 :]):
                out[i + 1] = "æ"
                return out
            break
    return out


def _is_suffix_tail(tail: list[str]) -> bool:
    """Sonek kuyruğu mu (örn. ``-ler``, ``-im``, ``-sin``, ``-di``)?"""
    if not tail:
        return True
    # Pratik yaklaşım: kuyruk en fazla 4 fonem ve ünlü içeriyor → ek.
    return len(tail) <= 4 and any(t in {"a", "e", "i", "ɯ", "o", "œ", "u", "y", "æ"} for t in tail)


def _lower_tr(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()
