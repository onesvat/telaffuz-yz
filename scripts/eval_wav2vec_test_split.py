#!/usr/bin/env python3
"""Held-out manifest test-split PER evaluation for a fine-tuned wav2vec model.

The training loop reports only dev PER (model-selection split). This script
evaluates the held-out test split for an unbiased generalisation estimate using
the exact same PER metric as training (Levenshtein / gold phone count, same
duration filters, same vocab decoding).

Usage:
    uv run python scripts/eval_wav2vec_test_split.py \\
        --model-dir data/models/xlsr300m_phoneme \\
        --report-stem xlsr_test

Outputs (artifacts/wav2vec/):
    <stem>.json   overall PER + per-provider breakdown
    <stem>.md     human-readable summary table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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
    build_decode_maps,
    decode_label_ids,
    decode_pred_ids,
    load_manifest,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "wav2vec"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--processor-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--split", default="test")
    ap.add_argument("--report-stem", default="wav2vec_test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-duration", type=float, default=15.0)
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if not (args.model_dir / "model.safetensors").exists():
        print(f"ERROR: no model.safetensors in {args.model_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device           : {device}")
    print(f"Model dir        : {args.model_dir}")
    print(f"Manifest / split : {args.manifest} / {args.split}")

    if args.processor_dir is not None:
        processor_dir = args.processor_dir
    elif (args.model_dir / "vocab.json").exists():
        processor_dir = args.model_dir
    else:
        processor_dir = args.model_dir.parent

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(args.model_dir))
    model.to(device).eval()
    if device == "cuda":
        model.half()

    max_dur = None if args.max_duration <= 0 else args.max_duration
    rows = load_manifest(
        args.manifest, split=args.split,
        limit=args.limit, max_duration_s=max_dur,
        min_duration_s=args.min_duration, skip_file_check=False,
    )
    if not rows:
        print("ERROR: no rows after filtering", file=sys.stderr)
        return 1

    src_by_seg: dict[str, str] = {}
    with args.manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == args.split:
                src_by_seg[row["segment_id"]] = row.get("provider") or "unknown"

    dataset = ManifestDataset(rows=rows, processor=processor)
    collator = DataCollatorCTCWithPadding(processor=processor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=4,
    )

    ids_to_tokens, drop_ids = build_decode_maps(processor)
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    overall = [0, 0, 0]
    t0 = time.time()
    row_idx = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_values = batch["input_values"].to(device)
            if device == "cuda":
                input_values = input_values.half()
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = model(input_values, attention_mask=attn).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for pred_row, label_row in zip(pred_ids, labels):
                p = decode_pred_ids(pred_row.tolist(), ids_to_tokens, drop_ids)
                r = decode_label_ids(label_row.tolist(), ids_to_tokens, drop_ids)
                ed = _edit_distance(p, r)
                src = src_by_seg.get(rows[row_idx]["segment_id"], "unknown")
                agg[src][0] += ed
                agg[src][1] += len(r)
                agg[src][2] += 1
                overall[0] += ed
                overall[1] += len(r)
                overall[2] += 1
                row_idx += 1

            if (bi + 1) % 200 == 0:
                done = row_idx
                rate = done / (time.time() - t0)
                eta = (len(rows) - done) / rate if rate else 0
                cur = overall[0] / overall[1] if overall[1] else 0
                print(f"  {done:>6}/{len(rows)}  PER~{cur:.4f}  {rate:.0f} utt/s  ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    overall_per = overall[0] / overall[1] if overall[1] else 0.0

    by_source = {
        src: {
            "per": round(v[0] / v[1], 4) if v[1] else None,
            "n_utt": v[2],
            "ref_phones": v[1],
            "edits": v[0],
        }
        for src, v in sorted(agg.items())
    }

    result: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "manifest": str(args.manifest),
        "split": args.split,
        "filter": {"min_duration_s": args.min_duration, "max_duration_s": max_dur},
        "n_utterances": overall[2],
        "ref_phones": overall[1],
        "edits": overall[0],
        "per": round(overall_per, 4),
        "eval_runtime_s": round(elapsed, 1),
        "device": device,
        "by_source": by_source,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACTS_DIR / f"{args.report_stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Test-split PER — `{args.model_dir.name}`",
        "",
        f"- Model: `{args.model_dir}`",
        f"- Split: `{args.split}` ({args.manifest.name})",
        f"- Filter: {args.min_duration}s ≤ dur ≤ {max_dur}s",
        f"- Segments evaluated: **{overall[2]:,}** / {overall[1]:,} reference phones",
        f"- Runtime: {elapsed/60:.1f} min ({device})",
        "",
        f"## Overall test PER: **{overall_per*100:.2f}%**",
        "",
        "| Provider | Segments | Ref phones | PER |",
        "|---|---:|---:|---:|",
    ]
    for src, v in by_source.items():
        per_txt = f"{v['per']*100:.2f}%" if v["per"] is not None else "—"
        lines.append(f"| {src} | {v['n_utt']:,} | {v['ref_phones']:,} | {per_txt} |")
    lines.append(f"| **Total** | **{overall[2]:,}** | **{overall[1]:,}** | **{overall_per*100:.2f}%** |")
    lines.append("")
    (ARTIFACTS_DIR / f"{args.report_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"=== Test PER: {overall_per:.4f} ({overall[2]} utt, {elapsed/60:.1f} min) ===")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
