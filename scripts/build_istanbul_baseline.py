"""Generate ``data/istanbul_baseline_20.csv``.

Istanbul Turkish pronunciation pilot set: 10 warm-up words + 10 sentences.
The prompts provide a compact manual smoke set for the demo and coach runtime.

Run::

    uv run python scripts/build_istanbul_baseline.py

Output: ``data/istanbul_baseline_20.csv`` (UTF-8, with header row).
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from g2p.pipeline import transcribe_text, transcribe_training_text  # noqa: E402


@dataclass(frozen=True)
class Item:
    text: str
    text_type: str  # 'word' | 'sentence'
    category: str
    notes: str = ""


# Words (10) — category-stratified
WORDS: list[Item] = [
    Item("su", "word", "daily", "1 hece, açık ünlü"),
    Item("kapı", "word", "daily", "2 hece, palatal /c/ önyok"),
    Item("evet", "word", "daily", "2 hece, son hece kapalı /e/"),
    Item("bakkal", "word", "daily", "geminat /kk/ + sonorant /ɫ/"),
    Item("para", "word", "allophone", "syllable-initial /pʰ/ aspirasyonu"),
    Item("kalın", "word", "allophone", "back vowel + dark /ɫ/"),
    Item("saat", "word", "long_vowel", "/aː/ uzun ünlü"),
    Item("şair", "word", "long_vowel", "hiatus /ʃa.iɾ/"),
    Item("lokanta", "word", "stress_loanword", "ilk hece stres (loanword)"),
    Item("spor", "word", "cluster", "word-initial CC kümesi"),
]

# Sentences (10) — four sub-categories
SENTENCES: list[Item] = [
    Item("Bugün hava çok güzel.", "sentence", "daily", "doğal tempo"),
    Item("Lütfen kapıyı kapatır mısın.", "sentence", "daily", "günlük rica kalıbı"),
    Item("Şair kâğıda şiir yazdı.", "sentence", "allophone_rich", "uzun ünlü + ʃ + ɾ"),
    Item(
        "Pencerenin önünde küçük bir kedi var.",
        "sentence",
        "allophone_rich",
        "palatal c+y, ɲ",
    ),
    Item("Para harcamayı sevmem.", "sentence", "allophone_rich", "pʰ + r-devoicing"),
    Item(
        "Şu an geliyor mu, gelmiyor mu?",
        "sentence",
        "stress_preaccent",
        "-iyor stres + -me pre-accent",
    ),
    Item(
        "Babam kapkara kaşlarıyla dikiliyordu.",
        "sentence",
        "stress_preaccent",
        "pekiştirme + -iyor",
    ),
    Item(
        "Eve gitmedim çünkü çalışıyordum.",
        "sentence",
        "stress_preaccent",
        "-me + -iyor",
    ),
    Item(
        "Şu köşe yaz köşesi, şu köşe kış köşesi.",
        "sentence",
        "tongue_twister",
        "ş + k tekrar",
    ),
    Item(
        "Saksıda saksağan suskunca dururdu.",
        "sentence",
        "tongue_twister",
        "s + ŋ + ɾ klaster",
    ),
]

OUTPUT = ROOT / "data" / "istanbul_baseline_20.csv"


def build_row(item_id: int, item: Item) -> dict[str, str]:
    if item.text_type == "word":
        result = transcribe_text(item.text)
        ipa = result.ipa
        phones = " ".join(p for p in result.phonemes)
    else:
        result = transcribe_training_text(item.text)
        ipa = transcribe_text(item.text).ipa
        phones = " ".join(result.tokens)
    return {
        "id": str(item_id),
        "text": item.text,
        "text_type": item.text_type,
        "category": item.category,
        "canonical_ipa": ipa,
        "canonical_phones": phones,
        "notes": item.notes,
    }


def main() -> None:
    items = WORDS + SENTENCES
    assert len(WORDS) == 10, f"warm-up word count {len(WORDS)}, expected 10"
    assert len(SENTENCES) == 10, f"sentence count {len(SENTENCES)}, expected 10"
    rows = [build_row(i + 1, item) for i, item in enumerate(items)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "text",
                "text_type",
                "category",
                "canonical_ipa",
                "canonical_phones",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"OK: {len(rows)} rows -> {OUTPUT.relative_to(ROOT)}")
    for row in rows:
        kind = "W" if row["text_type"] == "word" else "S"
        print(f"  [{kind}] {row['text']}  ->  {row['canonical_ipa']}")


if __name__ == "__main__":
    main()
