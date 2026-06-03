"""Türkçe heceleme — maximal onset principle (CCV onset cluster destekli).

Yerli Türkçe sözcüklerde başlangıç ünsüz kümesi yoktur, hece sınırı VC.CV
şeklindedir. Alıntı sözcüklerde (elektronik, Slovenya, demokrasi, trafik)
``/tr, dr, kr, pr, br, fl, gl, sp, st, sk, …/`` gibi onset cluster'lar
izinlidir; bu durumda sınır maximal-onset prensibine göre CCV ikinci heceye
gider: `e.lek.tro.nik`, `de.mok.ra.si`.
"""

from __future__ import annotations

from g2p.constants import ALL_VOWELS


def is_vowel(phoneme: str) -> bool:
    return phoneme in ALL_VOWELS


_ASPIRATE_BASE: dict[str, str] = {"pʰ": "p", "tʰ": "t", "kʰ": "k", "cʰ": "c"}


def _strip_aspirate(phoneme: str) -> str:
    return _ASPIRATE_BASE.get(phoneme, phoneme)


# Türkçe alıntı sözcüklerinde izinli word-internal CCV onset cluster'lar.
# Sonority sequencing: stop/fricative + ɾ.
# Yerli Türkçede yok; alıntılarda (elektronik, demokrasi, trafik).
#
# NOT: stop + l/ɫ clusters (kl, ɡl, pl, bl, fl) word-internal onset DEĞİLDİR.
# Yerli pattern: /bekle/ → /bek.le/ (k coda, l onset), maximal-onset DEĞİL.
# Word-initial /pl, kl, fl/ (plan, klan, flama) otomatik onset olur:
# boundary algoritması ilk ünlünün solundaki tüm ünsüzleri ilk heceye verir.
#
# Word-initial /sp, st, sk/ (statik, spor) de aynı şekilde otomatik onset.
# Word-medial /VsC/ → /Vs.CV/ (İstanbul → is.tan.bul).
_LEGAL_ONSET_CLUSTERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("t", "ɾ"), ("d", "ɾ"), ("k", "ɾ"), ("ɡ", "ɾ"), ("p", "ɾ"), ("b", "ɾ"),
        ("f", "ɾ"), ("c", "ɾ"), ("ɟ", "ɾ"),
    }
)


def syllabify(phonemes: list[str]) -> list[list[str]]:
    """Fonem listesini hecelere böler.

    Args:
        phonemes: Bağlam-bağımsız IPA fonem dizisi (genelde
            ``grapheme_map.to_ipa`` çıktısı veya rule pipeline ara çıktısı).

    Returns:
        Her elemanı bir heceyi temsil eden iç-içe liste. Boş giriş için
        boş liste; ünlü içermeyen giriş için tek elemanlı liste döner.
    """
    if not phonemes:
        return []

    vowel_indices = [i for i, p in enumerate(phonemes) if is_vowel(p)]
    if not vowel_indices:
        return [list(phonemes)]

    boundaries: list[int] = [0]
    for i in range(len(vowel_indices) - 1):
        v_prev = vowel_indices[i]
        v_next = vowel_indices[i + 1]
        cluster_len = v_next - v_prev - 1

        if cluster_len == 0:
            # VV: hece sınırı ikinci ünlünün soluna yerleşir.
            boundary = v_next
        elif cluster_len == 1:
            # VCV — tek ünsüz ikinci hecenin onset'i
            boundary = v_next - 1
        else:
            # VCC…V — maximal onset: son iki ünsüz onset cluster oluşturuyorsa
            # ikinci heceye git. Aspirate varyantlar (pʰ, tʰ, kʰ, cʰ) base
            # forma indirgenip kontrol edilir.
            c_last = _strip_aspirate(phonemes[v_next - 1])
            c_prev_last = _strip_aspirate(phonemes[v_next - 2])
            if (c_prev_last, c_last) in _LEGAL_ONSET_CLUSTERS:
                boundary = v_next - 2
            else:
                boundary = v_next - 1
        boundaries.append(boundary)
    boundaries.append(len(phonemes))

    return [
        phonemes[boundaries[j] : boundaries[j + 1]] for j in range(len(boundaries) - 1)
    ]


def nucleus_index(syllable: list[str]) -> int:
    """Hecenin çekirdek (ilk ünlü) indisi; ünlü yoksa -1."""
    for i, p in enumerate(syllable):
        if is_vowel(p):
            return i
    return -1


def nucleus(syllable: list[str]) -> str | None:
    idx = nucleus_index(syllable)
    return syllable[idx] if idx >= 0 else None


def onset(syllable: list[str]) -> list[str]:
    idx = nucleus_index(syllable)
    return list(syllable[:idx]) if idx >= 0 else list(syllable)


def coda(syllable: list[str]) -> list[str]:
    idx = nucleus_index(syllable)
    return list(syllable[idx + 1 :]) if idx >= 0 else []


def is_open(syllable: list[str]) -> bool:
    """Hece açık mı (CV / V) — son fonem ünlü mü?"""
    return bool(syllable) and is_vowel(syllable[-1])


def is_closed(syllable: list[str]) -> bool:
    """Hece kapalı mı (CVC / VC / CVCC) — son fonem ünsüz mü?"""
    return bool(syllable) and not is_vowel(syllable[-1])
