#!/usr/bin/env python3
"""Generate supplementary wav2vec training and comparison figures.

Outputs (figures/wav2vec/), in both English (_en) and Turkish (_tr):
    fig_training_curves_{en,tr}.png
    fig_model_comparison_{en,tr}.png
    fig_long_vowels_{en,tr}.png
    fig_phoneme_classes_{en,tr}.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts" / "wav2vec"
OUT_DIR = REPO_ROOT / "figures" / "wav2vec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Eval PER history from trainer_state.json (old training metric) ──────────
# These were recorded during training; absolute values differ from corrected
# test-time metric but show the convergence trajectory correctly.

XLSR_EVAL = [
    (400,0.9977),(800,0.7712),(1200,0.5183),(1600,0.4648),(2000,0.4146),
    (2400,0.3709),(2800,0.3674),(3200,0.3348),(3600,0.3301),(4000,0.3249),
    (4400,0.3023),(4800,0.3015),(5200,0.2906),(5600,0.2805),(6000,0.2758),
    (6400,0.2716),(6800,0.2642),(7200,0.2638),(7600,0.2522),(8000,0.2494),
    (8400,0.2541),(8800,0.2433),(9200,0.2390),(9600,0.2388),(10000,0.2338),
    (10400,0.2358),(10800,0.2351),(11200,0.2325),(11600,0.2302),(12000,0.2288),
    (12400,0.2273),(12800,0.2281),(13200,0.2207),(13600,0.2176),(14000,0.2174),
    (14400,0.2140),(14800,0.2137),(15200,0.2143),(15600,0.2126),(16000,0.2159),
    (16400,0.2104),(16800,0.2083),(17200,0.2112),(17600,0.2040),(18000,0.2067),
    (18400,0.2029),(18800,0.2035),(19200,0.2011),(19600,0.2013),(20000,0.2036),
    (20400,0.1995),(20800,0.1962),(21200,0.1938),(21600,0.1982),(22000,0.1959),
    (22400,0.1915),(22800,0.1904),(23200,0.1894),(23600,0.1920),(24000,0.1893),
    (24400,0.1896),(24800,0.1914),(25200,0.1871),(25600,0.1871),(26000,0.1867),
    (26400,0.1867),(26800,0.1841),(27200,0.1810),(27600,0.1850),(28000,0.1838),
    (28400,0.1820),(28800,0.1818),(29200,0.1799),(29600,0.1774),(30000,0.1846),
    (30400,0.1807),(30800,0.1799),(31200,0.1787),(31600,0.1800),(32000,0.1787),
    (32400,0.1797),(32800,0.1788),(33200,0.1749),(33600,0.1736),(34000,0.1702),
    (34400,0.1744),(34800,0.1732),(35200,0.1703),(35600,0.1751),(36000,0.1729),
    (36400,0.1715),(36800,0.1714),(37200,0.1704),(37600,0.1691),(38000,0.1687),
    (38400,0.1674),(38800,0.1686),(39200,0.1651),(39600,0.1641),(40000,0.1658),
    (40400,0.1615),(40800,0.1624),(41200,0.1625),(41600,0.1641),(42000,0.1650),
    (42400,0.1609),(42800,0.1613),(43200,0.1634),(43600,0.1612),(44000,0.1623),
    (44400,0.1622),(44800,0.1580),(45200,0.1573),(45600,0.1592),(46000,0.1564),
    (46400,0.1605),(46800,0.1578),(47200,0.1553),(47600,0.1547),(48000,0.1560),
    (48400,0.1546),(48800,0.1556),(49200,0.1552),(49600,0.1530),(50000,0.1518),
    (50400,0.1539),(50800,0.1525),(51200,0.1536),(51600,0.1531),(52000,0.1530),
    (52400,0.1539),(52800,0.1515),(53200,0.1509),(53600,0.1523),(54000,0.1500),
    (54400,0.1522),(54800,0.1515),(55200,0.1489),(55600,0.1477),(56000,0.1479),
    (56400,0.1469),(56800,0.1461),(58000,0.1461),(58400,0.1461),(58800,0.1459),
]

MMS_EVAL = [
    (500,0.103),(1000,0.1108),(1500,0.1017),(2000,0.1053),(2500,0.0956),
    (3000,0.0876),(3500,0.0885),(4000,0.0846),(4500,0.0843),(5000,0.0832),
    (5500,0.0802),(6000,0.0795),(6500,0.0766),(7000,0.0778),(7500,0.075),
    (8000,0.0766),(8500,0.0726),(9000,0.073),(9500,0.0711),(10000,0.0713),
    (10500,0.0698),(11000,0.0698),(11500,0.068),(12000,0.067),(12500,0.0672),
    (13000,0.0664),(13500,0.0668),(14000,0.0649),(14500,0.0628),(15000,0.0649),
    (15500,0.064),(16000,0.0624),(16500,0.0614),(17000,0.0605),(17500,0.0602),
    (18000,0.0628),(18500,0.0611),(19000,0.06),(19500,0.0584),(20000,0.0585),
    (20500,0.0589),(21000,0.0579),(21500,0.0575),(22000,0.0567),(22500,0.0568),
    (23000,0.0547),(23500,0.0555),(24000,0.0551),(24500,0.0548),(25000,0.057),
    (25500,0.0536),(26000,0.0535),(26500,0.0532),(27000,0.0521),(27500,0.0532),
    (28000,0.0527),(28500,0.0522),(29000,0.0523),(29500,0.0507),(30000,0.052),
    (30500,0.0516),(31000,0.0508),(31500,0.0508),(32000,0.0499),(32500,0.0496),
    (33000,0.0498),(33500,0.0486),(34000,0.0487),(34500,0.0484),(35000,0.0472),
    (35500,0.0484),(36000,0.0469),(36500,0.047),(37000,0.0479),(37500,0.046),
    (38000,0.0459),(38500,0.046),(39000,0.0461),(39500,0.0454),(40000,0.046),
    (40500,0.0465),(41000,0.0459),(41500,0.0449),(42000,0.0438),(42500,0.0432),
    (43000,0.0444),(43500,0.0435),(44000,0.0429),(44500,0.0427),(45000,0.0423),
    (45500,0.0416),(46000,0.042),(46500,0.0418),(47000,0.0414),(47500,0.0415),
    (48000,0.0418),(48500,0.0408),(49000,0.0402),(49500,0.0405),(50000,0.0403),
    (50500,0.0397),(51000,0.0399),(51500,0.0396),(52000,0.0397),(52500,0.0391),
    (53000,0.0393),(53500,0.0388),(54000,0.0387),(54500,0.0386),(55000,0.0385),
    (55500,0.0383),(56000,0.0382),(56500,0.0383),(57000,0.0381),(57500,0.0376),
    (58000,0.0377),(58500,0.0373),(59000,0.0372),(59500,0.0368),(60000,0.0367),
    (60500,0.037),(61000,0.0363),(61500,0.0369),(62000,0.0364),(62500,0.0364),
]

# ─── Per-phone data ────────────────────────────────────────────────────────────

def load_per_phone(stem: str) -> dict:
    path = ARTIFACTS / f"{stem}_per_phone.csv"
    data = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["phone"]] = {
                "per": float(row["per"]),
                "ref": int(row["ref_count"]),
            }
    return data

# ─── Figure 1: Training convergence curves ────────────────────────────────────

def plot_training_curves(lang: str = "en") -> Path:
    labels = {
        "en": {
            "title": "Dev-set PER During Fine-tuning",
            "ylabel": "Phoneme Error Rate (dev, training metric)",
            "xlsr": "XLS-R-300M (8 epochs, 59,888 steps)",
            "mms": "MMS-1B (7.4 epochs, 62,500 steps)",
            "xlabel": "Optimizer Step",
            "note": "Note: values computed with training-loop decoder; absolute values\ndiffer from corrected test-time metric by ~1.1×.",
            "best": "Best checkpoint",
        },
        "tr": {
            "title": "İnce Ayar Sürecinde Doğrulama Seti FHO",
            "ylabel": "Fonem Hata Oranı (doğrulama, eğitim metriği)",
            "xlsr": "XLS-R-300M (8 epok, 59.888 adım)",
            "mms": "MMS-1B (7,4 epok, 62.500 adım)",
            "xlabel": "Optimizasyon Adımı",
            "note": "Not: değerler eğitim-döngüsü kod çözücüsüyle hesaplanmıştır;\nmutlak değerler, düzeltilmiş test metriğinden ~1,1× farklıdır.",
            "best": "En iyi kontrol noktası",
        },
    }[lang]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    xs_x = [s for s, _ in XLSR_EVAL]
    ys_x = [p * 100 for _, p in XLSR_EVAL]
    xs_m = [s for s, _ in MMS_EVAL]
    ys_m = [p * 100 for _, p in MMS_EVAL]

    ax.plot(xs_x, ys_x, color="#1f77b4", lw=1.8, label=labels["xlsr"])
    ax.plot(xs_m, ys_m, color="#ff7f0e", lw=1.8, linestyle="--", label=labels["mms"])

    # mark best checkpoints
    ax.axvline(58800, color="#1f77b4", lw=0.8, linestyle=":", alpha=0.7)
    ax.axvline(61000, color="#ff7f0e", lw=0.8, linestyle=":", alpha=0.7)
    ax.scatter([58800], [14.59], color="#1f77b4", zorder=5, s=60)
    ax.scatter([61000], [3.63], color="#ff7f0e", zorder=5, s=60)

    # epoch markers for XLS-R (7486 steps/epoch)
    STEPS_PER_EPOCH = 7486
    for ep in range(1, 9):
        ax.axvline(ep * STEPS_PER_EPOCH, color="gray", lw=0.4, linestyle=":", alpha=0.4)
        ax.text(ep * STEPS_PER_EPOCH, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 105,
                f"E{ep}", fontsize=6.5, ha="center", color="gray", va="top")

    ax.set_xlabel(labels["xlabel"], fontsize=11)
    ax.set_ylabel(labels["ylabel"], fontsize=11)
    ax.set_title(labels["title"], fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"%{y:.0f}"))
    ax.set_xlim(0, 63500)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.04, labels["note"], ha="center", fontsize=8, style="italic", color="gray")
    plt.tight_layout()

    suffix = f"_{lang}"
    out = OUT_DIR / f"fig_training_curves{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 2: Model comparison (XLS-R vs MMS-1B, top-20 worst phonemes) ─────

def plot_model_comparison(lang: str = "en") -> Path:
    labels = {
        "en": {
            "title": "Per-Phoneme PER: XLS-R-300M vs MMS-1B",
            "ylabel": "Phoneme Error Rate (%)",
            "xlabel": "Phoneme",
            "xlsr": "XLS-R-300M (4.05% overall)",
            "mms": "MMS-1B (4.13% overall)",
        },
        "tr": {
            "title": "Fonem Bazında FHO: XLS-R-300M ve MMS-1B Karşılaştırması",
            "ylabel": "Fonem Hata Oranı (%)",
            "xlabel": "Fonem",
            "xlsr": "XLS-R-300M (genel: %4,05)",
            "mms": "MMS-1B (genel: %4,13)",
        },
    }[lang]

    xlsr = load_per_phone("xlsr_test")
    mms = load_per_phone("mms1b_test")

    # Common phones, sorted by XLS-R PER descending
    phones = sorted(
        [p for p in xlsr if p in mms and xlsr[p]["ref"] >= 100],
        key=lambda p: xlsr[p]["per"],
        reverse=True,
    )[:25]

    x = np.arange(len(phones))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - w/2, [xlsr[p]["per"]*100 for p in phones],
           width=w, color="#1f77b4", label=labels["xlsr"], alpha=0.85)
    ax.bar(x + w/2, [mms[p]["per"]*100 for p in phones],
           width=w, color="#ff7f0e", label=labels["mms"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(phones, fontsize=9)
    ax.set_xlabel(labels["xlabel"], fontsize=11)
    ax.set_ylabel(labels["ylabel"], fontsize=11)
    ax.set_title(labels["title"], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"%{y:.0f}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    suffix = f"_{lang}"
    out = OUT_DIR / f"fig_model_comparison{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 3: Long vowel analysis ────────────────────────────────────────────

def plot_long_vowels(lang: str = "en") -> Path:
    labels = {
        "en": {
            "title": "Long Vowel PER: XLS-R-300M vs MMS-1B",
            "ylabel": "Phoneme Error Rate (%)",
            "xlabel": "Long Vowel Phoneme",
            "xlsr": "XLS-R-300M",
            "mms": "MMS-1B",
            "note": "Reference counts shown above bars. Dashed line = overall test PER.",
        },
        "tr": {
            "title": "Uzun Ünlü FHO: XLS-R-300M ve MMS-1B Karşılaştırması",
            "ylabel": "Fonem Hata Oranı (%)",
            "xlabel": "Uzun Ünlü Fonem",
            "xlsr": "XLS-R-300M",
            "mms": "MMS-1B",
            "note": "Sütun üzerindeki sayılar referans fonem sayısını göstermektedir. Kesikli çizgi = genel test FHO.",
        },
    }[lang]

    xlsr = load_per_phone("xlsr_test")
    mms = load_per_phone("mms1b_test")

    long_vowels = ["aː", "eː", "iː", "oː", "uː", "yː", "ɯː", "œː"]
    present = [v for v in long_vowels if v in xlsr]

    x = np.arange(len(present))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_x = ax.bar(x - w/2, [xlsr[v]["per"]*100 for v in present],
                    width=w, color="#1f77b4", label=labels["xlsr"], alpha=0.85)
    bars_m = ax.bar(x + w/2, [mms[v]["per"]*100 for v in present],
                    width=w, color="#ff7f0e", label=labels["mms"], alpha=0.85)

    # ref counts above xlsr bars
    for bar, v in zip(bars_x, present):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"n={xlsr[v]['ref']}", ha="center", va="bottom", fontsize=7.5, color="#1f77b4")

    # overall PER reference lines
    ax.axhline(4.05, color="#1f77b4", lw=1.0, linestyle="--", alpha=0.6)
    ax.axhline(4.13, color="#ff7f0e", lw=1.0, linestyle="--", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=11)
    ax.set_xlabel(labels["xlabel"], fontsize=11)
    ax.set_ylabel(labels["ylabel"], fontsize=11)
    ax.set_title(labels["title"], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"%{y:.0f}"))
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.04, labels["note"], ha="center", fontsize=8.5, style="italic", color="gray")
    plt.tight_layout()

    suffix = f"_{lang}"
    out = OUT_DIR / f"fig_long_vowels{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 4: Phoneme class breakdown ────────────────────────────────────────

PHONEME_CLASSES = {
    "Plosives\nPatlamalar": ["b", "d", "ɡ", "k", "p", "t", "c", "ɟ"],
    "Asp. Stops\nÜflemeli": ["pʰ", "tʰ", "kʰ", "cʰ"],
    "Affricates\nAfrikatlar": ["t͡ʃ", "d͡ʒ"],
    "Fricatives\nSürtünmeliler": ["f", "s", "z", "ʃ", "ʒ", "h", "v", "β", "β̞"],
    "Nasals\nGenizsilier": ["m", "n", "ɲ", "ŋ"],
    "Liquids\nAkıcılar": ["l", "ɫ", "ɾ", "ɾ̞̊"],
    "Glides\nYaklaşıcılar": ["j"],
    "Short Vowels\nKısa Ünlüler": ["a", "e", "æ", "i", "o", "œ", "u", "y", "ɯ"],
    "Long Vowels\nUzun Ünlüler": ["aː", "eː", "iː", "oː", "uː", "yː", "ɯː", "œː"],
    "Stress ˈ": ["ˈ"],
}


def plot_phoneme_classes(lang: str = "en") -> Path:
    labels_map = {
        "en": {
            "title": "Mean PER by Phoneme Class",
            "ylabel": "Mean PER (%, weighted by ref count)",
            "xlabel": "Phoneme Class",
            "xlsr": "XLS-R-300M",
            "mms": "MMS-1B",
        },
        "tr": {
            "title": "Fonem Sınıfına Göre Ortalama FHO",
            "ylabel": "Ağırlıklı Ortalama FHO (%)",
            "xlabel": "Fonem Sınıfı",
            "xlsr": "XLS-R-300M",
            "mms": "MMS-1B",
        },
    }[lang]

    xlsr = load_per_phone("xlsr_test")
    mms = load_per_phone("mms1b_test")

    class_names_en = [k.split("\n")[0] for k in PHONEME_CLASSES]
    class_names_tr = [k.split("\n")[1] if "\n" in k else k for k in PHONEME_CLASSES]
    class_names = class_names_tr if lang == "tr" else class_names_en

    def weighted_mean(data, phones):
        total_ref = total_err = 0
        for p in phones:
            if p in data:
                total_ref += data[p]["ref"]
                total_err += data[p]["per"] * data[p]["ref"]
        return (total_err / total_ref * 100) if total_ref > 0 else 0

    means_x = [weighted_mean(xlsr, phones) for phones in PHONEME_CLASSES.values()]
    means_m = [weighted_mean(mms, phones) for phones in PHONEME_CLASSES.values()]

    x = np.arange(len(class_names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - w/2, means_x, width=w, color="#1f77b4",
           label=labels_map["xlsr"], alpha=0.85)
    ax.bar(x + w/2, means_m, width=w, color="#ff7f0e",
           label=labels_map["mms"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=8.5, rotation=15, ha="right")
    ax.set_xlabel(labels_map["xlabel"], fontsize=11)
    ax.set_ylabel(labels_map["ylabel"], fontsize=11)
    ax.set_title(labels_map["title"], fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"%{y:.1f}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    suffix = f"_{lang}"
    out = OUT_DIR / f"fig_phoneme_classes{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for lang in ("en", "tr"):
        for fn in (plot_training_curves, plot_model_comparison, plot_long_vowels, plot_phoneme_classes):
            out = fn(lang)
            print(f"  {out}")
    print("Done.")
