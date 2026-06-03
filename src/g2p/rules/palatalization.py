"""K/G palatalizasyonu.

K/G sesleri ön ünlü içeren hecelerde damaksıllaşır
(``/k/ → /c/``, ``/ɡ/ → /ɟ/``).

Onset pozisyonu: ön ünlü hecesinde koşulsuz palatalize
(`kedi /ˈcedi/`, `gece /ɟeˈd͡ʒe/`, `kötü /cœˈty/`).

Koda pozisyonu daha sınırlıdır. Koda palatalizasyonu yalnız şu bağlamlarda
uygulanır:
  - sözcük-sonu (`erkek /æɾˈcæc/`, `türk /tʰyɾ̞̊c/`, `gerçek /ɟæɾˈt͡ʃæc/`,
    `gitmek /ɟitˈmæc/`, `bebek /beˈbæc/`, `ilk /ilc/`), VEYA
  - sonraki onset bir liquid'tir (`l, ɫ, ɾ`) — geniş suffix sınırlarında geçer:
    `bekledim /bæcleˈdim/`, `pekliği /pʰæcliˈji/`, `desteklenemezse /dæstʰæcle.../`.

Koda KORUNUR (palatalize ETME):
  - soldaki fonem `n` ise (Vnk → Vŋk; `denk /dæŋk/`, `renk /ɾæŋk/`),
  - sonraki onset nasal/obstrüent (`ekmek /ækˈmæk/` k.m; `türkçe /tʰyɾ̞̊kt͡ʃe/`
    k.t͡ʃ).

Bu kural final envanter örneklerinde gözlenen açık vakalarla uyumludur ve
nasal/obstrüent onset bağlamında velar korumayı sürdürür.
"""

from __future__ import annotations

from g2p.rules.common import is_front_vowel
from g2p.syllabifier import onset, syllabify


_CODA_PALATALIZE_TRIGGERS: frozenset[str] = frozenset({"l", "ɫ", "ɾ"})


def apply(phonemes: list[str]) -> list[str]:
    syllables = syllabify(phonemes)
    # syllable_start_index — her hecenin global başlangıç indisi
    starts: list[int] = []
    cursor = 0
    for syll in syllables:
        starts.append(cursor)
        cursor += len(syll)

    out: list[str] = list(phonemes)
    for syll_idx, syllable in enumerate(syllables):
        has_front = any(is_front_vowel(p) for p in syllable)
        if not has_front:
            continue
        onset_len = len(onset(syllable))
        start = starts[syll_idx]
        for local_idx, phoneme in enumerate(syllable):
            if phoneme not in {"k", "ɡ"}:
                continue
            palatalized = "c" if phoneme == "k" else "ɟ"
            global_idx = start + local_idx
            if local_idx < onset_len:
                # Onset → koşulsuz palatalize
                out[global_idx] = palatalized
                continue
            # Koda pozisyonu
            # Vnk blokajı: soldaki fonem n ise velar koru.
            if local_idx > 0 and syllable[local_idx - 1] == "n":
                continue
            # Sonraki fonemi global listeden çek (sonraki hecenin onset'i veya sonraki coda fonemi)
            next_phoneme = (
                phonemes[global_idx + 1] if global_idx + 1 < len(phonemes) else None
            )
            if next_phoneme is None or next_phoneme in _CODA_PALATALIZE_TRIGGERS:
                out[global_idx] = palatalized
            # Aksi halde nasal/obstrüent onset bağlamında velar koru.
    return out
