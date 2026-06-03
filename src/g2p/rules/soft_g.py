"""Yumuşak g (ğ) dönüşümleri.

İstanbul Türkçesi standardı için final bağlam kuralları:

1. **Word-final ve izleyen consonant bağlamındaki coda ğ**: soldaki ünlü uzar; ğ silinir.
   - dağ → /daː/, sağ → /saː/, bağ → /baː/
   - sağlık → /saːlɯk/, oğlan → /oːlan/, yağmur → /jaːmuɾ/
   - /e/ + coda ğ → [ej]: değnek → /dejnec/, teğmen → /tejmen/, eğlence → /ejlend͡ʒe/

2. **Intervokalik VğV — front-front**: ğ → [j].
   - değil → /dejil/, eğitim → /ejitim/, beğeni → /bejeni/

3. **Intervokalik VğV — back-back**: ğ silinir, hiatus kalır, **uzatma YOK**.
   - oğul → /o.ul/, boğaz → /bo.az/, mağara → /ma.a.ɾa/, doğa → /do.a/

   Tek-heceli word-final ``dağ /daː/`` uzar, ama intervokalik VğV uzatmaz;
   bu bağlamda yalnız ğ silinir.

4. **Intervokalik VğV — mixed (front/back)**: nadir; back-back kuralı gibi
   hiatus, uzatma yok (`şikâyetname` türü).

5. **Proper name C+ğ → /ɡ/**: özel adlarda ünsüz+ğ kombinasyonu sert /ɡ/
   olarak kalır (Olğun → /olɡun/, Erdoğan → varyant).

6. **Loanwords**: lex exception olarak ayrı işlenir (Iraq, hukuku, …).
"""

from __future__ import annotations

from g2p.grapheme_map import SOFT_G_PLACEHOLDER
from g2p.rules.common import (
    is_consonant,
    is_front_vowel,
    is_vowel,
    lengthen,
    previous_vowel_index,
)


def apply(
    phonemes: list[str], *, word: str = "", is_proper: bool = False
) -> list[str]:
    out = list(phonemes)
    i = 0
    while i < len(out):
        if out[i] != SOFT_G_PLACEHOLDER:
            i += 1
            continue

        prev_i = previous_vowel_index(out, i)
        prev = out[prev_i] if prev_i >= 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        prev_token = out[i - 1] if i > 0 else None

        # 5. Proper name C+ğ → /ɡ/
        if is_proper and prev_token is not None and is_consonant(prev_token):
            out[i] = "ɡ"
            i += 1
            continue

        # Solda ünlü yoksa ğ'yi sil (word-initial marjinal durum).
        if prev is None:
            out.pop(i)
            continue

        # 2 + 3 + 4. Intervokalik VğV
        if nxt is not None and is_vowel(nxt):
            if is_front_vowel(prev) and is_front_vowel(nxt):
                # Front-front → /j/
                out[i] = "j"
                i += 1
                continue
            # Back-back veya mixed → hiatus (uzatma YOK), ğ silinir
            out.pop(i)
            continue

        # 1. Coda ğ: word-final veya izleyen consonant bağlamı.
        if prev == "e":
            # /e/ + coda ğ → [ej]
            out[i] = "j"
            i += 1
            continue

        # Word-final / izleyen consonant coda ğ: soldaki ünlü uzar.
        out[prev_i] = lengthen(prev)
        out.pop(i)

    return out
