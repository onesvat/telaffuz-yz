#!/usr/bin/env python3
"""Generate the evidence-layer ablation figure in English and Turkish.

The figure decomposes the two-score coach into its cumulative decision layers
and traces what each layer contributes to the safety/detection trade-off,
measured on the canonical two-native-speaker studio set (onur + gamze) with the
frozen MMS-1B checkpoint. The numbers are re-derived offline from the cached
per-phone observations (``.smoke/eval_validation_obs.jsonl``); no model
inference is run, so the final-layer point reproduces the frozen canonical
operating point exactly.

Layers (cumulative):
  L1 Plain GOP   recognizer top-1 only (substitute on any mismatch)
  L2 +Abstain    add the usable-evidence gate (low_confidence when unusable)
  L3 +Atlas      add atlas quality band + atypicality (no authority ceiling)
  L4 Full coach  add the contrast-wise authority table = deployed system

A second figure breaks the deployed coach down by phone category, tracing the
mispronunciation-detection rate and the false-alarm rate (with Wilson 95 %
intervals) for vowels, base consonants and consonant allophones.

Output (figures/assessment/):
  fig_hybrid_gop_results_{en,tr}.png
  fig_mdd_by_category_{en,tr}.png

Run::

    uv run python scripts/plot_assessment_models.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Noto Sans"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from g2p.constants import (  # noqa: E402
    ALL_VOWELS,
    CONS_BASE_ALLOPHONES,
    CONS_EXTRA_ALLOPHONES,
)

OBS_PATH = REPO_ROOT / ".smoke" / "eval_validation_obs.jsonl"
AUTH_PATH = REPO_ROOT / "configs" / "decision_authority.json"
OUT_DIR = REPO_ROOT / "figures" / "assessment"

_CONSONANT_ALLOPHONES = CONS_BASE_ALLOPHONES | CONS_EXTRA_ALLOPHONES
CATEGORY_ORDER = ("vowel", "consonant", "allophone")

# Canonical evaluation set: native speakers only (anonymised K1/K2 in the text).
CANONICAL_SPEAKERS = {"onur", "gamze"}

# Frozen canonical operating point (full coach); the L4 layer must reproduce it.
CANONICAL = {"CTL_fp": 1.76, "WNAT_fp": 1.60, "detection": 55.56, "precision": 76.09}

COLORS = {"fp": "#E53935", "detection": "#1E88E5", "precision": "#43A047"}

STRINGS = {
    "en": {
        "title": "Contribution of each decision layer to the\n"
        "safety–detection trade-off (2 native speakers, MMS-1B)",
        "ylabel": "Rate (%)",
        "layers": [
            "Plain GOP\n(recognizer top-1)",
            "+ Abstain\ngate",
            "+ Atlas\nquality",
            "Full coach\n(+ authority table)",
        ],
        "series": {
            "fp": "Hard-FP (W_NAT)",
            "detection": "Detection (W_ERR)",
            "precision": "Precision (W_ERR)",
        },
        "note": "The authority-table layer cuts hard-FP from ~16% to 1.6% at the "
        "cost of ~6 detection points, while raising precision from 44% to 76%.",
        "mdd_title": "Mispronunciation detection and false-alarm rate by phone\n"
        "category (2 native speakers, MMS-1B; Wilson 95% CI)",
        "mdd_ylabel": "Rate (%)",
        "mdd_categories": ["Vowels", "Consonants", "Allophones"],
        "mdd_series": {
            "detection": "Detection (W_ERR)",
            "fp": "Hard-FP (correct speech)",
        },
        "mdd_note": "Vowels carry the front/back contrasts (ɯ–i, œ–o, y–u) and "
        "show the highest detection; allophones (aspirated stops, dark l, velar "
        "nasal) are deliberately not turned into hard substitutions.",
    },
    "tr": {
        "title": "Her karar katmanının güvenlik–tespit dengesine\n"
        "katkısı (2 anadili konuşmacı, MMS-1B)",
        "ylabel": "Oran (%)",
        "layers": [
            "Saf-GOP\n(tanıyıcı top-1)",
            "+ Belirsizlik\ngeçidi",
            "+ Atlas\nkalitesi",
            "Tam koç\n(+ otorite tablosu)",
        ],
        "series": {
            "fp": "Hard-FP (W_NAT)",
            "detection": "Tespit (W_ERR)",
            "precision": "Kesinlik (W_ERR)",
        },
        "note": "Otorite tablosu katmanı hard-FP'yi ~%16'dan %1,6'ya indirirken "
        "tespitten yalnızca ~6 puan verir ve kesinliği %44'ten %76'ya yükseltir.",
        "mdd_title": "Fonem kategorisine göre yanlış telaffuz tespiti ve\n"
        "yanlış-alarm oranı (2 anadili konuşmacı, MMS-1B; Wilson %95 GA)",
        "mdd_ylabel": "Oran (%)",
        "mdd_categories": ["Ünlüler", "Ünsüzler", "Allofonlar"],
        "mdd_series": {
            "detection": "Tespit (W_ERR)",
            "fp": "Hard-FP (doğru konuşma)",
        },
        "mdd_note": "Ünlüler ön/art kontrastlarını (ɯ–i, œ–o, y–u) taşır ve en "
        "yüksek tespiti gösterir; allofonlar (aspirasyonlu duraklar, kalın l, "
        "art damak nazalı) bilinçli olarak sert substitution'a çevrilmez.",
    },
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (rate, low, high) as percentages for a Wilson 95% interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p * 100, max(0.0, center - half) * 100, (center + half) * 100


def phone_category(phone: str) -> str:
    if phone in ALL_VOWELS:
        return "vowel"
    if phone in _CONSONANT_ALLOPHONES:
        return "allophone"
    return "consonant"


def mdd_by_category(obs: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    """Per-category (caught, intended) detection and (sub, total) false alarm."""
    detection = {c: [0, 0] for c in CATEGORY_ORDER}
    false_alarm = {c: [0, 0] for c in CATEGORY_ORDER}
    for o in obs:
        category = phone_category(o["target"])
        if o["kind"] == "W_ERR" and o["is_err"]:
            detection[category][1] += 1
            detection[category][0] += o["verdict_now"] == "substitution"
        if o["kind"] in ("CTL", "W_NAT"):
            false_alarm[category][1] += 1
            false_alarm[category][0] += o["verdict_now"] == "substitution"
    return {
        "detection": {c: (v[0], v[1]) for c, v in detection.items()},
        "false_alarm": {c: (v[0], v[1]) for c, v in false_alarm.items()},
    }


def load_observations() -> list[dict]:
    """Per-phone evidence for the canonical native speakers, mirroring decide()."""
    out: list[dict] = []
    with OBS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("speaker") not in CANONICAL_SPEAKERS:
                continue
            phone = row["phone"]
            atlas = phone.get("atlas") or {}
            wav2vec = phone.get("wav2vec") or {}
            out.append(
                {
                    "kind": row["kind"],
                    "is_err": bool(row["is_intended_error_pos"]),
                    "target": phone.get("target_phone"),
                    "verdict_now": (phone.get("decision") or {}).get("verdict"),
                    "match": wav2vec.get("observed_matches_target"),
                    "w2v_obs": wav2vec.get("observed_phone"),
                    "w2v_usable": (wav2vec.get("confidence") or 0) > 0,
                    "atlas_verdict": atlas.get("atlas_verdict", "low_confidence"),
                }
            )
    return out


def verdict_plain_gop(o: dict) -> str:
    """L1: trust the recogniser top-1; no abstain, atlas or authority."""
    return "correct" if o["match"] is True else "substitution"


def verdict_abstain(o: dict) -> str:
    """L2: add the usable-evidence gate."""
    if not o["w2v_usable"]:
        return "low_confidence"
    if o["match"] is True:
        return "correct"
    return "substitution" if o["w2v_obs"] is not None else "low_confidence"


def verdict_atlas(o: dict) -> str:
    """L3: add the atlas quality band, without the authority FP ceiling."""
    if o["verdict_now"] == "accepted_variant":
        return "accepted_variant"
    if not o["w2v_usable"]:
        return "low_confidence"
    if o["match"] is True:
        if o["atlas_verdict"] in ("fit_ok", "consonant_abstain", "low_confidence"):
            return "correct"
        if o["atlas_verdict"] == "likely_substitution":
            return "substitution"
        return "correct_atypical"
    return "substitution" if o["w2v_obs"] is not None else "low_confidence"


def verdict_full_coach(o: dict) -> str:
    """L4: the deployed decision = frozen canonical output."""
    return o["verdict_now"]


def layer_metrics(obs: list[dict], verdict_fn) -> dict[str, float]:
    counts: dict[str, Counter] = defaultdict(Counter)
    caught = total = werr_sub = 0
    for o in obs:
        v = verdict_fn(o)
        counts[o["kind"]][v] += 1
        if o["kind"] == "W_ERR":
            if o["is_err"]:
                total += 1
                caught += v == "substitution"
            werr_sub += v == "substitution"

    def hard_fp(kind: str) -> float:
        n = sum(counts[kind].values())
        return counts[kind]["substitution"] / n * 100 if n else 0.0

    return {
        "CTL_fp": hard_fp("CTL"),
        "WNAT_fp": hard_fp("W_NAT"),
        "detection": caught / total * 100 if total else 0.0,
        "precision": caught / werr_sub * 100 if werr_sub else 0.0,
    }


def render(lang: str, layer_results: list[dict[str, float]]) -> None:
    strings = STRINGS[lang]
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(layer_results))
    width = 0.27
    series = (
        ("fp", [m["WNAT_fp"] for m in layer_results]),
        ("detection", [m["detection"] for m in layer_results]),
        ("precision", [m["precision"] for m in layer_results]),
    )
    for offset, (key, values) in zip((-width, 0.0, width), series):
        bars = ax.bar(
            [i + offset for i in x],
            values,
            width,
            label=strings["series"][key],
            color=COLORS[key],
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(strings["layers"])
    ax.set_ylabel(strings["ylabel"])
    ax.set_ylim(0, 88)
    ax.set_title(strings["title"], fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, strings["note"], ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    out = OUT_DIR / f"fig_hybrid_gop_results_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"OK: {out.relative_to(REPO_ROOT)}")


def render_mdd(lang: str, category_results: dict) -> None:
    strings = STRINGS[lang]
    fig, ax = plt.subplots(figsize=(9, 6))

    x = range(len(CATEGORY_ORDER))
    width = 0.38
    panels = (
        ("detection", "detection", -width / 2, COLORS["detection"]),
        ("false_alarm", "fp", width / 2, COLORS["fp"]),
    )
    for data_key, series_key, offset, color in panels:
        rates, lows, highs, labels = [], [], [], []
        for category in CATEGORY_ORDER:
            caught, total = category_results[data_key][category]
            rate, low, high = wilson_ci(caught, total)
            rates.append(rate)
            lows.append(rate - low)
            highs.append(high - rate)
            labels.append(f"{rate:.1f}%\n({caught}/{total})")
        bars = ax.bar(
            [i + offset for i in x],
            rates,
            width,
            yerr=[lows, highs],
            capsize=4,
            label=strings["mdd_series"][series_key],
            color=color,
            error_kw={"alpha": 0.6},
        )
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(strings["mdd_categories"])
    ax.set_ylabel(strings["mdd_ylabel"])
    ax.set_ylim(0, 100)
    ax.set_title(strings["mdd_title"], fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.04, strings["mdd_note"], ha="center", fontsize=8.5, style="italic", wrap=True)

    fig.tight_layout()
    out = OUT_DIR / f"fig_mdd_by_category_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"OK: {out.relative_to(REPO_ROOT)}")


def main() -> None:
    obs = load_observations()
    layer_results = [
        layer_metrics(obs, verdict_plain_gop),
        layer_metrics(obs, verdict_abstain),
        layer_metrics(obs, verdict_atlas),
        layer_metrics(obs, verdict_full_coach),
    ]

    full = layer_results[-1]
    for metric, expected in CANONICAL.items():
        got = round(full[metric], 2)
        assert abs(got - expected) < 0.01, (
            f"L4 {metric}={got} drifted from frozen canonical {expected}; "
            "the figure must stay pinned to the frozen operating point."
        )

    category_results = mdd_by_category(obs)
    caught = sum(v[0] for v in category_results["detection"].values())
    total = sum(v[1] for v in category_results["detection"].values())
    assert (caught, total) == (35, 63), (
        f"MDD-by-category detection total {caught}/{total} drifted from the "
        "frozen canonical 35/63."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "tr"):
        render(lang, layer_results)
        render_mdd(lang, category_results)


if __name__ == "__main__":
    main()
