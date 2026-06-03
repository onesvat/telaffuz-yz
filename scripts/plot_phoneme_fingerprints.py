#!/usr/bin/env python3
"""Generate three types of phoneme fingerprint figures in English and Turkish.

Figures (figures/audio/):
  fig_fingerprint_melgrid_{en,tr}.png  — average mel spectrogram per phone (7×7 grid)
  fig_fingerprint_spectrum_{en,tr}.png — average power spectrum per phone (small multiples)
  fig_fingerprint_mfcc_{en,tr}.png     — mean MFCC profile heatmap (phones × coefficients)

Data sources (read-only, from source project):
  phone_instances.parquet — timing boundaries per aligned phone window
  audio.sqlite            — segment_id → audio file path mapping
"""
from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

matplotlib.rcParams["font.family"] = "Noto Sans"

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = Path("/home/onur/Code/telaffuz-yz-v2")
PARQUET   = SRC_ROOT / "data" / "phoneme_atlas" / "phone_instances.parquet"
DB_PATH   = SRC_ROOT / "data" / "audio.sqlite"
AUDIO_ROOT = Path("/media/onur/shared/telaffuz-yz-audio")
OUT_DIR   = REPO_ROOT / "figures" / "audio"

# ── parameters ───────────────────────────────────────────────────────────────
N_SAMPLES   = 50        # instances per phone
SR          = 16_000
WIN_MS      = 80        # ms window centred on phone midpoint
WIN_SAMPLES = int(WIN_MS / 1000 * SR)
N_MELS      = 64
N_MFCC      = 13
HOP         = 80        # 5 ms at 16 kHz
N_FFT       = 512       # ~32 ms
SEED        = 42

PHONES_ORDER = [
    "a", "e", "i", "ɯ", "o", "u", "y", "œ",
    "aː", "eː", "iː", "ɯː", "oː", "uː", "yː", "œː",
    "p", "pʰ", "b", "t", "tʰ", "d", "k", "kʰ", "c", "cʰ", "ɟ", "ɡ",
    "t͡ʃ", "d͡ʒ",
    "f", "v", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ɲ", "ŋ", "l", "ɫ", "ɾ", "ɾ̞̊", "j",
    "æ", "β", "β̞",
]

STRINGS = {
    "en": {
        "mel_title":      "Average mel spectrogram per phone",
        "mel_time":       "Time",
        "mel_freq":       "Frequency",
        "mel_cbar":       "Energy (dB, normalised)",
        "spec_title":     "Average power spectrum per phone",
        "spec_xlabel":    "Frequency (kHz)",
        "spec_ylabel":    "Magnitude (dB, normalised)",
        "mfcc_title":     "Mean MFCC profile per phone",
        "mfcc_xlabel":    "MFCC coefficient",
        "mfcc_ylabel":    "Phone",
        "mfcc_cbar":      "Mean coefficient value (normalised)",
        "n_label":        lambda n: f"n={n}",
    },
    "tr": {
        "mel_title":      "Fonem başına ortalama mel spektrogram",
        "mel_time":       "Zaman",
        "mel_freq":       "Frekans",
        "mel_cbar":       "Enerji (dB, normalleştirilmiş)",
        "spec_title":     "Fonem başına ortalama güç spektrumu",
        "spec_xlabel":    "Frekans (kHz)",
        "spec_ylabel":    "Genlik (dB, normalleştirilmiş)",
        "mfcc_title":     "Fonem başına ortalama MFCC profili",
        "mfcc_xlabel":    "MFCC katsayısı",
        "mfcc_ylabel":    "Fonem",
        "mfcc_cbar":      "Ortalama katsayı değeri (normalleştirilmiş)",
        "n_label":        lambda n: f"n={n}",
    },
}


# ── data loading ─────────────────────────────────────────────────────────────

def load_samples() -> dict[str, list[np.ndarray]]:
    """Return {phone: [audio_chunk, ...]} for N_SAMPLES instances per phone."""
    random.seed(SEED)

    print("Reading parquet …")
    table = pq.read_table(PARQUET, columns=["segment_id", "expected_phone", "feature_json"])
    df = table.to_pandas()
    df["_fj"]      = df["feature_json"].apply(json.loads)
    df["start_ms"] = df["_fj"].apply(lambda x: x["start_ms"])
    df["end_ms"]   = df["_fj"].apply(lambda x: x["end_ms"])
    df["conf"]     = df["_fj"].apply(lambda x: x.get("confidence", 0))
    df = df[df["conf"] > 0.6].copy()

    print("Reading segment paths …")
    conn = sqlite3.connect(str(DB_PATH))
    seg_df = pd.read_sql("SELECT id, path FROM segments", conn)
    conn.close()
    path_map: dict[int, Path] = {
        int(r["id"]): AUDIO_ROOT / r["path"]
        for _, r in seg_df.iterrows()
    }

    # sample per phone
    wanted: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    phones = [p for p in PHONES_ORDER if p in df["expected_phone"].values]
    for phone in phones:
        sub = df[df["expected_phone"] == phone]
        rows = sub.sample(min(N_SAMPLES, len(sub)), random_state=SEED)
        for _, r in rows.iterrows():
            seg_id = int(r["segment_id"])
            if seg_id in path_map:
                wanted[seg_id].append((phone, int(r["start_ms"]), int(r["end_ms"])))

    # load audio, group by file to minimise I/O
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    total_segs = len(wanted)
    for idx, (seg_id, entries) in enumerate(wanted.items()):
        path = path_map[seg_id]
        if not path.exists():
            continue
        if idx % 500 == 0:
            print(f"  loading segments {idx}/{total_segs} …")
        try:
            audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
            if file_sr != SR:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
        except Exception:
            continue
        for phone, start_ms, end_ms in entries:
            mid = (start_ms + end_ms) / 2 / 1000
            s   = max(0, int((mid - WIN_MS / 2000) * SR))
            e   = s + WIN_SAMPLES
            if e > len(audio):
                e = len(audio)
                s = max(0, e - WIN_SAMPLES)
            chunk = audio[s:e]
            if len(chunk) < WIN_SAMPLES // 2:
                continue
            # zero-pad to WIN_SAMPLES
            if len(chunk) < WIN_SAMPLES:
                chunk = np.pad(chunk, (0, WIN_SAMPLES - len(chunk)))
            chunks[phone].append(chunk)

    return chunks


# ── feature extraction ────────────────────────────────────────────────────────

def _mel(chunk: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS,
                                        n_fft=N_FFT, hop_length=HOP)
    return librosa.power_to_db(S, ref=np.max)


def _spectrum(chunk: np.ndarray) -> np.ndarray:
    S = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=N_FFT))
    db = 20 * np.log10(S + 1e-9)
    return db


def _mfcc(chunk: np.ndarray) -> np.ndarray:
    return librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP).mean(axis=1)


def compute_features(
    chunks: dict[str, list[np.ndarray]],
) -> tuple[dict, dict, dict, dict]:
    mel_avg: dict[str, np.ndarray] = {}
    spec_avg: dict[str, np.ndarray] = {}
    mfcc_avg: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    phones = [p for p in PHONES_ORDER if p in chunks]
    total = len(phones)
    for idx, phone in enumerate(phones):
        clist = chunks[phone]
        if not clist:
            continue
        print(f"  features {idx+1}/{total}: /{phone}/ ({len(clist)} clips)")

        # fixed-length mel frames by resampling time axis
        n_frames = WIN_SAMPLES // HOP + 1
        mels  = [librosa.util.fix_length(
                     _mel(c), size=n_frames, axis=1) for c in clist]
        specs = [_spectrum(c) for c in clist]
        mfccs = [_mfcc(c) for c in clist]

        mel_stack  = np.stack(mels, axis=0)
        spec_stack = np.stack(specs, axis=0)
        mfcc_stack = np.stack(mfccs, axis=0)

        mel_avg[phone]  = mel_stack.mean(0)
        spec_avg[phone] = spec_stack.mean(0)
        mfcc_avg[phone] = mfcc_stack.mean(0)
        counts[phone]   = len(clist)

    return mel_avg, spec_avg, mfcc_avg, counts


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


# ── plotting ──────────────────────────────────────────────────────────────────

NCOLS = 7

def plot_mel_grid(mel_avg: dict, counts: dict, lang: str) -> None:
    s = STRINGS[lang]
    phones = [p for p in PHONES_ORDER if p in mel_avg]
    n = len(phones)
    nrows = (n + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 1.8, nrows * 1.9))
    axes = axes.flatten()

    for i, phone in enumerate(phones):
        ax = axes[i]
        img = _norm01(mel_avg[phone])
        ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"/{phone}/\n{s['n_label'](counts[phone])}", fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(s["mel_title"], fontsize=13)
    fig.tight_layout(rect=[0.04, 0.0, 1.0, 0.97])

    # shared axis labels (applied after tight_layout so they don't interfere)
    fig.text(0.5, 0.005, s["mel_time"], ha="center", fontsize=10)
    fig.text(0.01, 0.5, s["mel_freq"], va="center", rotation="vertical", fontsize=10)

    # colour bar: use only visible axes to avoid invisible panel conflicts
    visible_axes = [axes[k] for k in range(n)]
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=visible_axes, orientation="horizontal",
                        shrink=0.35, pad=0.06, aspect=40, location="bottom")
    cbar.set_label(s["mel_cbar"], fontsize=8)
    out = OUT_DIR / f"fig_fingerprint_melgrid_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_spectrum_grid(spec_avg: dict, counts: dict, lang: str) -> None:
    s = STRINGS[lang]
    phones = [p for p in PHONES_ORDER if p in spec_avg]
    n = len(phones)
    nrows = (n + NCOLS - 1) // NCOLS

    freqs = np.fft.rfftfreq(N_FFT, 1 / SR) / 1000  # kHz

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 1.8, nrows * 1.9),
                              sharey=False)
    axes = axes.flatten()

    for i, phone in enumerate(phones):
        ax = axes[i]
        spec = _norm01(spec_avg[phone])
        ax.fill_between(freqs, spec, alpha=0.7, color="#1E88E5")
        ax.plot(freqs, spec, lw=0.6, color="#0D47A1")
        ax.set_title(f"/{phone}/\n{s['n_label'](counts[phone])}", fontsize=8, pad=2)
        ax.set_xlim(0, SR / 2000)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(s["spec_title"], fontsize=13, y=1.01)
    fig.text(0.5, -0.01, s["spec_xlabel"], ha="center", fontsize=10)
    fig.text(-0.01, 0.5, s["spec_ylabel"], va="center", rotation="vertical", fontsize=10)

    fig.tight_layout()
    out = OUT_DIR / f"fig_fingerprint_spectrum_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_mfcc_heatmap(mfcc_avg: dict, lang: str) -> None:
    s = STRINGS[lang]
    phones = [p for p in PHONES_ORDER if p in mfcc_avg]
    matrix = np.stack([mfcc_avg[p] for p in phones], axis=0)

    # normalise each coefficient column independently
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    matrix_n = (matrix - col_min) / (col_max - col_min + 1e-9)

    fig, ax = plt.subplots(figsize=(8, len(phones) * 0.32 + 1.5))
    im = ax.imshow(matrix_n, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(phones)))
    ax.set_yticklabels([f"/{p}/" for p in phones], fontsize=7.5)
    ax.set_xticks(range(N_MFCC))
    ax.set_xticklabels([str(i + 1) for i in range(N_MFCC)], fontsize=8)
    ax.set_xlabel(s["mfcc_xlabel"], fontsize=11)
    ax.set_ylabel(s["mfcc_ylabel"], fontsize=11)
    ax.set_title(s["mfcc_title"], fontsize=13, pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(s["mfcc_cbar"], fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / f"fig_fingerprint_mfcc_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading audio chunks ===")
    chunks = load_samples()
    print(f"Loaded clips for {len(chunks)} phones\n")

    print("=== Computing features ===")
    mel_avg, spec_avg, mfcc_avg, counts = compute_features(chunks)
    print()

    for lang in ("en", "tr"):
        print(f"=== Plotting [{lang}] ===")
        plot_mel_grid(mel_avg, counts, lang)
        plot_spectrum_grid(spec_avg, counts, lang)
        plot_mfcc_heatmap(mfcc_avg, lang)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
