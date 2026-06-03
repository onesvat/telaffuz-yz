"""Starlang FSM morfolojik analiz wrapper'ı.

`NlpToolkit-MorphologicalAnalysis` paketini sarar; sözcüğü kök + archiphoneme
formundaki ek listesine ayrıştırır. Bu bilgi vurgu (vurgu çeken/önleyen ekler),
-yor bağlamındaki daralma (a/e → ı/i), -mez ekinin /æ/ tetiklemesi gibi G2P
kuralları için gereklidir.

Analizör tek bir singleton olarak yüklenir (~saniye mertebesinde başlangıç
maliyeti); sonraki çağrılar dahili LRU cache sayesinde hızlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class MorphAnalysis:
    """Tek bir sözcüğün morfolojik analizi.

    Attributes:
        word: Orijinal yüzey form.
        root: Sözcük kökü (lemmasyon sonrası).
        suffixes: Archiphoneme formunda ek listesi (örn. ``["Hyor", "yDH", "m"]``).
            Yüksek-ünlü H, dar-yuvarlak için U vb. Starlang konvansiyonları.
        pos: Kökün POS etiketi (``"VERB"``, ``"NOUN"``, ``"ADJ"``, ...).
        is_proper: Özel ad mı.
        raw_parse: Starlang'ın ham etiket dizgesi (inceleme ve hata ayıklama için).
    """

    word: str
    root: str
    suffixes: tuple[str, ...]
    pos: str
    is_proper: bool
    raw_parse: str


@cache
def _analyzer():  # type: ignore[no-untyped-def]
    from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer

    return FsmMorphologicalAnalyzer()


def analyze(word: str) -> MorphAnalysis | None:
    """Sözcüğü morfolojik olarak ayrıştırır.

    Birden fazla parse mümkünse en olası olanı (Starlang'ın 0. indeksi)
    döner. Parse bulunamazsa ``None`` döner; pipeline bu durumda kural-bazlı
    fallback'a yönlenir.

    Args:
        word: Türkçe sözcük (büyük harfler korunur — özel ad tespiti için).

    Returns:
        ``MorphAnalysis`` veya ``None``.
    """
    if not word:
        return None
    result = _analyzer().morphologicalAnalysis(word)
    if result.size() == 0:
        return None
    parse = result.getFsmParse(0)
    with_list = parse.withList()
    parts = with_list.split("+")
    root = parts[0]
    suffixes = tuple(parts[1:])
    pos = parse.getRootPos() or ""
    return MorphAnalysis(
        word=word,
        root=root,
        suffixes=suffixes,
        pos=pos,
        is_proper=parse.isProperNoun(),
        raw_parse=str(parse),
    )


# Vurgu kurallarında kullanılacak archiphoneme set'leri.
# Starlang konvansiyonları: H = high vowel (ı/i/u/ü), A = low vowel (a/e),
# D = d/t, C = c/ç. Büyük harfler ünlü uyumuyla değişen archiphoneme'ler.
STRESS_ATTRACTING_SUFFIXES: frozenset[str] = frozenset(
    {
        "Hyor",  # -iyor / -ıyor / -uyor / -üyor
        "yArAk",  # -erek / -arak
        "HncA",  # -ince / -ınca / -unca / -ünce
        "Hver",  # -iver / -ıver / -uver / -üver
    }
)

STRESS_PREVENTING_SUFFIXES: frozenset[str] = frozenset(
    {
        "mA",  # -me / -ma (olumsuzluk + isim-fiil)
        "lA",  # -le / -la (vasıta)
        "cA",  # -ce / -ca (eşitlik)
        "ki",  # -ki (sıfat türeten)
    }
)

# /æ/ açma kuralı için: -mez/-maz suffix'i her zaman /æ/ tetikler (leksikal sabit).
AE_TRIGGERING_SUFFIXES: frozenset[str] = frozenset({"mAz"})
