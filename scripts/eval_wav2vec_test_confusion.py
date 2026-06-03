#!/usr/bin/env python3
"""Evaluate a wav2vec phoneme-CTC model on a manifest split and emit confusion reports.

This is the large-set companion to ``eval_wav2vec_test_split.py``.  It keeps
the same model/data loading path, but also writes per-segment predictions and
phone-level confusion artifacts so we can answer where the acoustic PER comes
from.

Example:
    uv run python scripts/eval_wav2vec_test_confusion.py \\
      --model-dir data/models/xlsr300m_phoneme \\
      --manifest data/wav2vec/manifest_v5.csv \\
      --split test \\
      --report-stem xlsr_test

Outputs under ``artifacts/wav2vec``:
    <stem>_predictions.csv
    <stem>_confusion.json
    <stem>_confusion.md
    <stem>_per_phone.csv
    <stem>_substitutions.csv
    <stem>_insertions.csv
    <stem>_stress.csv
    <stem>_long_vowels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from train_wav2vec import (  # noqa: E402
    DEFAULT_MANIFEST,
    DataCollatorCTCWithPadding,
    ManifestDataset,
    _edit_distance,
    load_manifest,
)
from g2p.constants import LONG_TO_SHORT, LONG_VOWELS, STRESS  # noqa: E402

REPORTS_DIR = REPO_ROOT / "artifacts" / "wav2vec"
SEGMENTAL_EXCLUDED_PHONES = frozenset({STRESS})


def format_percent(value: float | None) -> str:
    return f"%{value * 100:.2f}" if value is not None else "—"


def decode_pred_ids(pred_ids: list[int], ids_to_tokens: dict[int, str], drop_ids: set[int]) -> list[str]:
    """CTC argmax ids -> collapsed phone-token sequence.

    ``processor.batch_decode`` concatenates multi-character IPA tokens inside a
    word, which is useful for reading but wrong for phone-level confusion.  This
    function decodes ids directly through the tokenizer vocabulary so tokens
    such as ``t͡ʃ`` and ``ɾ̞̊`` remain atomic phones.
    """
    collapsed: list[int] = []
    prev = -1
    for idx in pred_ids:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return [ids_to_tokens[idx] for idx in collapsed if idx not in drop_ids and idx in ids_to_tokens]


def decode_label_ids(label_ids: list[int], ids_to_tokens: dict[int, str], drop_ids: set[int]) -> list[str]:
    """Label ids -> phone-token sequence, preserving repeated phones."""
    return [ids_to_tokens[idx] for idx in label_ids if idx != -100 and idx not in drop_ids and idx in ids_to_tokens]


def align(ref: list[str], pred: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Needleman-Wunsch alignment with unit edit cost."""
    n, m = len(ref), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == pred[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", ref[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, pred[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def read_manifest_metadata(manifest: Path, split: str) -> dict[str, dict[str, str]]:
    """Return metadata by segment id for the requested split."""
    meta: dict[str, dict[str, str]] = {}
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != split:
                continue
            meta[row["segment_id"]] = {
                "provider": row.get("provider") or "unknown",
                "path": row.get("path") or "",
                "speaker_id": row.get("speaker_id") or "",
                "duration_s": row.get("duration_s") or "",
            }
    return meta


def choose_processor_dir(model_dir: Path, processor_dir: Path | None) -> Path:
    """Prefer explicit processor dir; otherwise fall back from checkpoint to run dir."""
    if processor_dir is not None:
        return processor_dir
    tokenizer_files = ("vocab.json", "tokenizer_config.json", "preprocessor_config.json")
    if all((model_dir / name).exists() for name in tokenizer_files):
        return model_dir
    parent = model_dir.parent
    if all((parent / name).exists() for name in tokenizer_files):
        return parent
    return model_dir


def analyze_predictions(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Build overall, per-provider, per-phone and substitution summaries."""
    per_phone = defaultdict(lambda: {"ref_count": 0, "correct": 0, "substituted": 0, "deleted": 0})
    sub_matrix: Counter[tuple[str, str]] = Counter()
    insertions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()
    provider_agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # edits, ref, n
    segmental_provider_agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # edits, ref, n
    provider_subs: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    total_edits = 0
    total_ref = 0
    total_pred = 0
    segmental_edits = 0
    segmental_ref = 0
    segmental_pred = 0

    for row in rows:
        ref = row["ref_phones"].split()
        pred = row["pred_phones"].split()
        provider = row.get("provider") or "unknown"
        edits = int(row["edit_distance"])
        seg_ref = [phone for phone in ref if phone not in SEGMENTAL_EXCLUDED_PHONES]
        seg_pred = [phone for phone in pred if phone not in SEGMENTAL_EXCLUDED_PHONES]
        seg_edits = _edit_distance(seg_pred, seg_ref)

        total_edits += edits
        total_ref += len(ref)
        total_pred += len(pred)
        segmental_edits += seg_edits
        segmental_ref += len(seg_ref)
        segmental_pred += len(seg_pred)
        provider_agg[provider][0] += edits
        provider_agg[provider][1] += len(ref)
        provider_agg[provider][2] += 1
        segmental_provider_agg[provider][0] += seg_edits
        segmental_provider_agg[provider][1] += len(seg_ref)
        segmental_provider_agg[provider][2] += 1

        for op, rp, pp in align(ref, pred):
            if op == "match":
                per_phone[rp]["ref_count"] += 1
                per_phone[rp]["correct"] += 1
            elif op == "sub":
                per_phone[rp]["ref_count"] += 1
                per_phone[rp]["substituted"] += 1
                sub_matrix[(rp, pp)] += 1
                provider_subs[provider][(rp, pp)] += 1
            elif op == "del":
                per_phone[rp]["ref_count"] += 1
                per_phone[rp]["deleted"] += 1
                deletions[rp] += 1
            elif op == "ins":
                insertions[pp] += 1

    phone_rows: list[dict[str, Any]] = []
    for phone, stats in per_phone.items():
        ref_count = stats["ref_count"]
        errors = stats["substituted"] + stats["deleted"]
        top_confusions = [
            {"pred": pred, "count": count}
            for (ref, pred), count in sub_matrix.most_common()
            if ref == phone
        ][:5]
        phone_rows.append(
            {
                "phone": phone,
                "ref_count": ref_count,
                "correct": stats["correct"],
                "substituted": stats["substituted"],
                "deleted": stats["deleted"],
                "per": round(errors / ref_count, 4) if ref_count else None,
                "top_confusions": top_confusions,
            }
        )
    phone_rows.sort(key=lambda r: (-(r["per"] or 0), -r["ref_count"], r["phone"]))

    providers = {
        provider: {
            "n_utterances": values[2],
            "ref_phones": values[1],
            "edits": values[0],
            "per": round(values[0] / values[1], 4) if values[1] else None,
            "top_substitutions": [
                {"ref": ref, "pred": pred, "count": count}
                for (ref, pred), count in provider_subs[provider].most_common(15)
            ],
        }
        for provider, values in sorted(provider_agg.items())
    }

    segmental_providers = {
        provider: {
            "n_utterances": values[2],
            "ref_phones": values[1],
            "pred_phones": sum(
                len([phone for phone in row["pred_phones"].split() if phone not in SEGMENTAL_EXCLUDED_PHONES])
                for row in rows
                if (row.get("provider") or "unknown") == provider
            ),
            "edits": values[0],
            "per": round(values[0] / values[1], 4) if values[1] else None,
        }
        for provider, values in sorted(segmental_provider_agg.items())
    }

    empty_phone = {"ref_count": 0, "correct": 0, "substituted": 0, "deleted": 0}
    stress_stats = per_phone.get(STRESS, empty_phone)
    stress_errors = stress_stats["substituted"] + stress_stats["deleted"]
    stress_diagnostic = {
        "phone": STRESS,
        "ref_count": stress_stats["ref_count"],
        "correct": stress_stats["correct"],
        "substituted": stress_stats["substituted"],
        "deleted": stress_stats["deleted"],
        "inserted": insertions.get(STRESS, 0),
        "per": round(stress_errors / stress_stats["ref_count"], 4) if stress_stats["ref_count"] else None,
    }

    long_vowel_rows: list[dict[str, Any]] = []
    for phone in sorted(LONG_VOWELS):
        stats = per_phone.get(phone, empty_phone)
        ref_count = stats["ref_count"]
        errors = stats["substituted"] + stats["deleted"]
        short_phone = LONG_TO_SHORT.get(phone, "")
        shortened = sub_matrix.get((phone, short_phone), 0) if short_phone else 0
        top_confusions = [
            {"pred": pred, "count": count}
            for (ref, pred), count in sub_matrix.most_common()
            if ref == phone
        ][:5]
        if ref_count or shortened:
            long_vowel_rows.append(
                {
                    "phone": phone,
                    "short_phone": short_phone,
                    "ref_count": ref_count,
                    "correct": stats["correct"],
                    "substituted": stats["substituted"],
                    "deleted": stats["deleted"],
                    "per": round(errors / ref_count, 4) if ref_count else None,
                    "shortened": shortened,
                    "shortened_rate": round(shortened / ref_count, 4) if ref_count else None,
                    "top_confusions": top_confusions,
                }
            )
    long_vowel_rows.sort(
        key=lambda r: (-(r["per"] or 0), -(r["shortened_rate"] or 0), -r["ref_count"], r["phone"])
    )
    long_vowel_totals = {
        "ref_count": sum(row["ref_count"] for row in long_vowel_rows),
        "correct": sum(row["correct"] for row in long_vowel_rows),
        "substituted": sum(row["substituted"] for row in long_vowel_rows),
        "deleted": sum(row["deleted"] for row in long_vowel_rows),
        "shortened": sum(row["shortened"] for row in long_vowel_rows),
    }
    long_vowel_errors = long_vowel_totals["substituted"] + long_vowel_totals["deleted"]
    long_vowel_totals["per"] = (
        round(long_vowel_errors / long_vowel_totals["ref_count"], 4)
        if long_vowel_totals["ref_count"]
        else None
    )
    long_vowel_totals["shortened_rate"] = (
        round(long_vowel_totals["shortened"] / long_vowel_totals["ref_count"], 4)
        if long_vowel_totals["ref_count"]
        else None
    )

    return {
        "n_utterances": len(rows),
        "ref_phones": total_ref,
        "pred_phones": total_pred,
        "edits": total_edits,
        "per": round(total_edits / total_ref, 4) if total_ref else None,
        "segmental": {
            "excluded_phones": sorted(SEGMENTAL_EXCLUDED_PHONES),
            "ref_phones": segmental_ref,
            "pred_phones": segmental_pred,
            "edits": segmental_edits,
            "per": round(segmental_edits / segmental_ref, 4) if segmental_ref else None,
            "providers": segmental_providers,
        },
        "diagnostics": {
            "stress": stress_diagnostic,
            "long_vowels": {
                "totals": long_vowel_totals,
                "phones": long_vowel_rows,
            },
        },
        "providers": providers,
        "per_phone": phone_rows,
        "top_substitutions": [
            {"ref": ref, "pred": pred, "count": count}
            for (ref, pred), count in sub_matrix.most_common(50)
        ],
        "top_insertions": [{"phone": phone, "count": count} for phone, count in insertions.most_common(30)],
        "top_deletions": [{"phone": phone, "count": count} for phone, count in deletions.most_common(30)],
    }


def write_prediction_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "segment_id",
        "provider",
        "speaker_id",
        "duration_s",
        "audio_path",
        "ref_phones",
        "pred_phones",
        "edit_distance",
        "ref_len",
        "pred_len",
        "per",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_prediction_csv(path: Path) -> list[dict[str, str]]:
    required = {"ref_phones", "pred_phones", "edit_distance"}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"prediction CSV missing required columns: {sorted(missing)}")
    return rows


def write_summary_csvs(prefix: Path, stats: dict[str, Any]) -> None:
    per_phone_path = prefix.parent / f"{prefix.name}_per_phone.csv"
    with per_phone_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phone", "ref_count", "correct", "substituted", "deleted", "per", "top_confusions"])
        for row in stats["per_phone"]:
            confusions = " | ".join(f"{c['pred']}:{c['count']}" for c in row["top_confusions"])
            writer.writerow([
                row["phone"],
                row["ref_count"],
                row["correct"],
                row["substituted"],
                row["deleted"],
                row["per"],
                confusions,
            ])

    sub_path = prefix.parent / f"{prefix.name}_substitutions.csv"
    with sub_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ref_phone", "pred_phone", "count"])
        for row in stats["top_substitutions"]:
            writer.writerow([row["ref"], row["pred"], row["count"]])

    ins_path = prefix.parent / f"{prefix.name}_insertions.csv"
    with ins_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["inserted_phone", "count"])
        for row in stats["top_insertions"]:
            writer.writerow([row["phone"], row["count"]])

    stress_path = prefix.parent / f"{prefix.name}_stress.csv"
    with stress_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phone", "ref_count", "correct", "substituted", "deleted", "inserted", "per"])
        row = stats["diagnostics"]["stress"]
        writer.writerow([
            row["phone"],
            row["ref_count"],
            row["correct"],
            row["substituted"],
            row["deleted"],
            row["inserted"],
            row["per"],
        ])

    long_path = prefix.parent / f"{prefix.name}_long_vowels.csv"
    with long_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "phone",
            "short_phone",
            "ref_count",
            "correct",
            "substituted",
            "deleted",
            "per",
            "shortened",
            "shortened_rate",
            "top_confusions",
        ])
        for row in stats["diagnostics"]["long_vowels"]["phones"]:
            confusions = " | ".join(f"{c['pred']}:{c['count']}" for c in row["top_confusions"])
            writer.writerow([
                row["phone"],
                row["short_phone"],
                row["ref_count"],
                row["correct"],
                row["substituted"],
                row["deleted"],
                row["per"],
                row["shortened"],
                row["shortened_rate"],
                confusions,
            ])


def write_markdown(path: Path, *, model_dir: Path, manifest: Path, split: str, stats: dict[str, Any], elapsed_s: float) -> None:
    segmental = stats["segmental"]
    stress = stats["diagnostics"]["stress"]
    long_vowels = stats["diagnostics"]["long_vowels"]
    lines = [
        f"# Test-Split Confusion — `{model_dir.name}` ({model_dir.parent.name})",
        "",
        "> Otomatik üretildi: `scripts/eval_wav2vec_test_confusion.py`. "
        "Full IPA PER = Levenshtein(predicted phones, reference phones) / reference phone count. "
        "Segmental PER, suprasegmental stres token'ı `ˈ` hariç hesaplanır. "
        "Varsayılan olarak `<spc>` word-boundary token'ı hariç tutulur.",
        "",
        f"- Model: `{model_dir}`",
        f"- Manifest / split: `{manifest}` / `{split}`",
        f"- Değerlendirilen: **{stats['n_utterances']:,}** segment / {stats['ref_phones']:,} referans fonem",
        f"- Full IPA PER: **{format_percent(stats['per'])}**",
        f"- Segmental PER (`ˈ` hariç): **{format_percent(segmental['per'])}**",
        f"- Süre: {elapsed_s / 60:.1f} dk",
        "",
        "## Metrik Ayrımı",
        "",
        "| Metrik | ref fonem | pred fonem | edit | PER |",
        "|---|---:|---:|---:|---:|",
        f"| Full IPA | {stats['ref_phones']:,} | {stats['pred_phones']:,} | {stats['edits']:,} | {format_percent(stats['per'])} |",
        f"| Segmental (`ˈ` hariç) | {segmental['ref_phones']:,} | {segmental['pred_phones']:,} | {segmental['edits']:,} | {format_percent(segmental['per'])} |",
        "",
        "## Provider Kırılımı",
        "",
        "| Provider | n | ref fonem | edit | Full PER | Segmental PER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for provider, row in stats["providers"].items():
        seg = segmental["providers"].get(provider, {})
        lines.append(
            f"| {provider} | {row['n_utterances']:,} | {row['ref_phones']:,} | {row['edits']:,} | "
            f"{format_percent(row['per'])} | {format_percent(seg.get('per'))} |"
        )

    lines += [
        "",
        "## Stress Diagnostic",
        "",
        "Stres token'ı model kalitesi için ayrı izlenir; ürün feedback'inde nihai karar prosodi/GOP katmanından verilmelidir.",
        "",
        "| phone | ref | correct | sub | del | inserted | PER |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `{stress['phone']}` | {stress['ref_count']} | {stress['correct']} | {stress['substituted']} | "
        f"{stress['deleted']} | {stress['inserted']} | {format_percent(stress['per'])} |",
        "",
        "## Uzun Ünlü Diagnostic",
        "",
        "Kısa okuma oranı, uzun ünlünün karşılık gelen kısa ünlüye substitution edilme oranıdır.",
        "",
        "| Phone | kısa | ref | correct | sub | del | PER | kısa okuma | kısa oranı | Top confusions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    totals = long_vowels["totals"]
    lines.append(
        f"| **Toplam** | — | {totals['ref_count']} | {totals['correct']} | {totals['substituted']} | "
        f"{totals['deleted']} | {format_percent(totals['per'])} | {totals['shortened']} | "
        f"{format_percent(totals['shortened_rate'])} | — |"
    )
    for row in long_vowels["phones"]:
        confusions = " · ".join(f"`{c['pred']}`×{c['count']}" for c in row["top_confusions"]) or "—"
        lines.append(
            f"| `{row['phone']}` | `{row['short_phone']}` | {row['ref_count']} | {row['correct']} | "
            f"{row['substituted']} | {row['deleted']} | {format_percent(row['per'])} | "
            f"{row['shortened']} | {format_percent(row['shortened_rate'])} | {confusions} |"
        )

    lines += [
        "",
        "## En Kötü Fonemler",
        "",
        "Phone-level PER = (substitution + deletion) / ref_count; insertions ayrıca raporlanır.",
        "",
        "| Phone | ref | correct | sub | del | PER | Top confusions |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in stats["per_phone"][:25]:
        confusions = " · ".join(f"`{c['pred']}`×{c['count']}" for c in row["top_confusions"]) or "—"
        per_txt = f"{row['per']:.4f}" if row["per"] is not None else "—"
        lines.append(
            f"| `{row['phone']}` | {row['ref_count']} | {row['correct']} | "
            f"{row['substituted']} | {row['deleted']} | {per_txt} | {confusions} |"
        )

    lines += [
        "",
        "## En Sık Substitution Çiftleri",
        "",
        "| ref → pred | count |",
        "|---|---:|",
    ]
    for row in stats["top_substitutions"][:25]:
        lines.append(f"| `{row['ref']}` → `{row['pred']}` | {row['count']} |")

    lines += [
        "",
        "## En Sık Insertions",
        "",
        "| inserted | count |",
        "|---|---:|",
    ]
    for row in stats["top_insertions"][:20]:
        lines.append(f"| `{row['phone']}` | {row['count']} |")

    lines += [
        "",
        "## En Sık Deletions",
        "",
        "| deleted | count |",
        "|---|---:|",
    ]
    for row in stats["top_deletions"][:20]:
        lines.append(f"| `{row['phone']}` | {row['count']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report_paths(report_stem: str, pred_path: Path, json_path: Path, md_path: Path) -> None:
    print("Reports:")
    print(f"  {pred_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {REPORTS_DIR / (report_stem + '_per_phone.csv')}")
    print(f"  {REPORTS_DIR / (report_stem + '_substitutions.csv')}")
    print(f"  {REPORTS_DIR / (report_stem + '_insertions.csv')}")
    print(f"  {REPORTS_DIR / (report_stem + '_stress.csv')}")
    print(f"  {REPORTS_DIR / (report_stem + '_long_vowels.csv')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True, help="Fine-tuned model/checkpoint directory.")
    ap.add_argument("--processor-dir", type=Path, default=None, help="Optional processor/tokenizer directory.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--split", default="test")
    ap.add_argument("--report-stem", required=True, help="Output basename under docs/reports.")
    ap.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Reuse an existing <stem>_predictions.csv and only rebuild summary reports.",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=15.0, help="<=0 disables max-duration filter.")
    ap.add_argument("--limit", type=int, default=None, help="Debug: cap rows after filters.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--include-spc", action="store_true", help="Include <spc> word-boundary tokens in PER/confusion.")
    args = ap.parse_args()

    if args.predictions_csv is not None:
        if not args.predictions_csv.exists():
            print(f"ERROR: predictions CSV not found: {args.predictions_csv}", file=sys.stderr)
            return 1
        t0 = time.time()
        try:
            prediction_rows = read_prediction_csv(args.predictions_csv)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if not prediction_rows:
            print("ERROR: no prediction rows found", file=sys.stderr)
            return 1
        elapsed_s = time.time() - t0
        stats = analyze_predictions(prediction_rows)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = REPORTS_DIR / args.report_stem
        pred_path = REPORTS_DIR / f"{args.report_stem}_predictions.csv"
        json_path = REPORTS_DIR / f"{args.report_stem}_confusion.json"
        md_path = REPORTS_DIR / f"{args.report_stem}_confusion.md"

        if args.predictions_csv.resolve() != pred_path.resolve():
            write_prediction_csv(pred_path, prediction_rows)
        json_path.write_text(
            json.dumps(
                {
                    "model_dir": str(args.model_dir),
                    "processor_dir": None,
                    "manifest": str(args.manifest),
                    "split": args.split,
                    "source_predictions_csv": str(args.predictions_csv),
                    "eval_runtime_s": round(elapsed_s, 1),
                    "device": "not_used",
                    **stats,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_summary_csvs(prefix, stats)
        write_markdown(
            md_path,
            model_dir=args.model_dir,
            manifest=args.manifest,
            split=args.split,
            stats=stats,
            elapsed_s=elapsed_s,
        )

        print()
        print(
            f"=== Rebuilt PER: full={stats['per']:.4f} "
            f"segmental_no_stress={stats['segmental']['per']:.4f} "
            f"({stats['n_utterances']} utt) ==="
        )
        print_report_paths(args.report_stem, pred_path, json_path, md_path)
        return 0

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if not (args.model_dir / "model.safetensors").exists():
        print(f"ERROR: no model.safetensors in {args.model_dir}", file=sys.stderr)
        return 1
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    processor_dir = choose_processor_dir(args.model_dir, args.processor_dir)
    max_duration = None if args.max_duration <= 0 else args.max_duration

    print(f"Device           : {args.device}")
    print(f"Model dir        : {args.model_dir}")
    print(f"Processor dir    : {processor_dir}")
    print(f"Manifest / split : {args.manifest} / {args.split}")

    rows = load_manifest(
        args.manifest,
        split=args.split,
        limit=args.limit,
        max_duration_s=max_duration,
        min_duration_s=args.min_duration,
        skip_file_check=False,
    )
    if not rows:
        print("ERROR: no rows after filtering", file=sys.stderr)
        return 1

    manifest_meta = read_manifest_metadata(args.manifest, args.split)

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(args.model_dir))
    model.to(args.device).eval()
    if args.device == "cuda":
        model.half()

    dataset = ManifestDataset(rows=rows, processor=processor)
    collator = DataCollatorCTCWithPadding(processor=processor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    pad_id = processor.tokenizer.pad_token_id
    vocab = processor.tokenizer.get_vocab()
    ids_to_tokens = {idx: token for token, idx in vocab.items()}
    drop_ids = {pad_id}
    blank_id = vocab.get("<blank>")
    if blank_id is not None:
        drop_ids.add(blank_id)
    spc_id = vocab.get("<spc>")
    if spc_id is not None and not args.include_spc:
        drop_ids.add(spc_id)

    prediction_rows: list[dict[str, str]] = []
    row_idx = 0
    t0 = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_values = batch["input_values"].to(args.device)
            if args.device == "cuda":
                input_values = input_values.half()
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)

            logits = model(input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

            labels = batch["labels"].cpu().numpy()

            for pred_id_row, label_id_row in zip(pred_ids, labels):
                row = rows[row_idx]
                ref = decode_label_ids(label_id_row.tolist(), ids_to_tokens, drop_ids)
                pred = decode_pred_ids(pred_id_row.tolist(), ids_to_tokens, drop_ids)
                edit_distance = _edit_distance(pred, ref)
                meta = manifest_meta.get(row["segment_id"], {})
                ref_len = len(ref)
                pred_len = len(pred)
                prediction_rows.append(
                    {
                        "segment_id": row["segment_id"],
                        "provider": meta.get("provider", "unknown"),
                        "speaker_id": row.get("speaker_id", ""),
                        "duration_s": str(row.get("duration_s", "")),
                        "audio_path": meta.get("path", row.get("path", "")),
                        "ref_phones": " ".join(ref),
                        "pred_phones": " ".join(pred),
                        "edit_distance": str(edit_distance),
                        "ref_len": str(ref_len),
                        "pred_len": str(pred_len),
                        "per": f"{edit_distance / ref_len:.4f}" if ref_len else "",
                    }
                )
                row_idx += 1

            if (bi + 1) % 200 == 0:
                done = row_idx
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                edits = sum(int(r["edit_distance"]) for r in prediction_rows)
                refs = sum(int(r["ref_len"]) for r in prediction_rows)
                per = edits / refs if refs else 0.0
                eta = (len(rows) - done) / rate if rate else 0.0
                print(f"  {done:>6}/{len(rows)}  PER~{per:.4f}  {rate:.0f} utt/s  ETA {eta / 60:.1f}m")

    elapsed_s = time.time() - t0
    stats = analyze_predictions(prediction_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = REPORTS_DIR / args.report_stem
    pred_path = REPORTS_DIR / f"{args.report_stem}_predictions.csv"
    json_path = REPORTS_DIR / f"{args.report_stem}_confusion.json"
    md_path = REPORTS_DIR / f"{args.report_stem}_confusion.md"

    write_prediction_csv(pred_path, prediction_rows)
    json_path.write_text(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "processor_dir": str(processor_dir),
                "manifest": str(args.manifest),
                "split": args.split,
                "filter": {"min_duration_s": args.min_duration, "max_duration_s": max_duration},
                "eval_runtime_s": round(elapsed_s, 1),
                "device": args.device,
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_csvs(prefix, stats)
    write_markdown(md_path, model_dir=args.model_dir, manifest=args.manifest, split=args.split, stats=stats, elapsed_s=elapsed_s)

    print()
    print(
        f"=== Test PER: full={stats['per']:.4f} "
        f"segmental_no_stress={stats['segmental']['per']:.4f} "
        f"({stats['n_utterances']} utt, {elapsed_s / 60:.1f} min) ==="
    )
    print_report_paths(args.report_stem, pred_path, json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
