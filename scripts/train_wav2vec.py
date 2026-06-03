#!/usr/bin/env python3
"""Wav2Vec2 / MMS phoneme-CTC fine-tuning script.

Models supported:
  - facebook/mms-1b-fl102           (1B params, multilingual, includes Turkish)
  - facebook/wav2vec2-xls-r-300m    (315M params, multilingual)

Usage:
    # Fresh run
    uv run python scripts/train_wav2vec.py --model xls-r-300m --run-name xlsr_run1

    # Fresh run under a custom output root
    uv run python scripts/train_wav2vec.py --model mms-1b --run-name mms_run1 \\
        --output-dir /home/onur/Code/telaffuz-yz-runs

    # Resume from last checkpoint
    uv run python scripts/train_wav2vec.py --run-name xlsr_run1 \\
        --resume-from-checkpoint auto

    # Warm-start weights from a checkpoint (fresh optimizer/scheduler)
    uv run python scripts/train_wav2vec.py --run-name xlsr_run2 \\
        --init-from-checkpoint runs/xlsr_run1/checkpoint-10000

    # Smoke test
    uv run python scripts/train_wav2vec.py --dry-run

Training output: runs/<run_name>/  (checkpoints, vocab.json, tensorboard)

Methodology: docs/wav2vec/methodology-wav2vec.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from audio.db import AUDIO_ROOT  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = REPO_ROOT / "data" / "wav2vec" / "manifest_v5.csv"
RUNS_DIR = REPO_ROOT / "runs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "wav2vec"

MODEL_ALIASES: dict[str, str] = {
    "mms-1b":     "facebook/mms-1b-fl102",
    "xls-r-300m": "facebook/wav2vec2-xls-r-300m",
}

# 49-phoneme inventory + CTC specials — must match build_dataset.py PHONEME_INVENTORY
PHONEME_INVENTORY: list[str] = [
    # 8 short vowels
    "a", "e", "i", "ɯ", "o", "œ", "u", "y",
    # 8 long vowels
    "aː", "eː", "iː", "ɯː", "oː", "œː", "uː", "yː",
    # 1 vowel allophone
    "æ",
    # 20 consonant phonemes
    "p", "b", "t", "d", "k", "ɡ", "t͡ʃ", "d͡ʒ", "f", "v",
    "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "ɾ", "j",
    # 7 base consonant allophones
    "c", "ɟ", "ɲ", "ŋ", "ɫ", "β", "β̞",
    # 5 pedagogical allophones
    "pʰ", "tʰ", "kʰ", "cʰ", "ɾ̞̊",
    # 1 suprasegmental
    "ˈ",
    # Word boundary
    "<spc>",
]

# CTC vocab: pad=0, unk=1, then phonemes, then blank last
VOCAB: dict[str, int] = {"<pad>": 0, "<unk>": 1}
for _p in PHONEME_INVENTORY:
    VOCAB[_p] = len(VOCAB)
VOCAB["<blank>"] = len(VOCAB)  # CTC blank (must be last for HF Wav2Vec2)

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Vocab / processor helpers
# ---------------------------------------------------------------------------


def write_vocab(run_dir: Path, *, overwrite: bool = True) -> Path:
    """Write vocab.json to run dir. Verifies equality on resume."""
    vocab_path = run_dir / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    new_payload = json.dumps(VOCAB, ensure_ascii=False, indent=2)
    if vocab_path.exists() and not overwrite:
        existing = vocab_path.read_text(encoding="utf-8")
        if json.loads(existing) != VOCAB:
            print(
                f"ERROR: Existing vocab.json at {vocab_path} differs from current VOCAB. "
                "Refusing to resume with mismatched vocab.",
                file=sys.stderr,
            )
            sys.exit(1)
        return vocab_path
    vocab_path.write_text(new_payload, encoding="utf-8")
    return vocab_path


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the checkpoint dir with the highest step number that has a trainer_state.json."""
    if not run_dir.exists():
        return None
    candidates = [
        p for p in run_dir.glob("checkpoint-*")
        if p.is_dir() and (p / "trainer_state.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("-", 1)[1]))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_manifest(
    manifest_path: Path,
    *,
    split: str,
    audio_root: Path = AUDIO_ROOT,
    limit: int | None = None,
    max_duration_s: float | None = None,
    min_duration_s: float = 0.5,
    skip_file_check: bool = False,
) -> list[dict[str, Any]]:
    """Load manifest CSV rows for a given split, with duration filtering."""
    import csv

    rows: list[dict[str, Any]] = []
    skipped_long = skipped_short = skipped_missing = 0
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["split"] != split:
                continue
            dur = float(row["duration_s"])
            if dur < min_duration_s:
                skipped_short += 1
                continue
            if max_duration_s is not None and dur > max_duration_s:
                skipped_long += 1
                continue
            audio_path = audio_root / row["path"]
            if not skip_file_check and not audio_path.exists():
                skipped_missing += 1
                continue
            rows.append({
                "segment_id": row["segment_id"],
                "path": str(audio_path),
                "phonemes": row["phonemes"],
                "duration_s": dur,
                "speaker_id": row["speaker_id"],
            })
            if limit and len(rows) >= limit:
                break

    summary = f"  {split}: {len(rows):,} rows"
    extras = []
    if skipped_long:
        extras.append(f"{skipped_long:,} > {max_duration_s}s")
    if skipped_short:
        extras.append(f"{skipped_short:,} < {min_duration_s}s")
    if skipped_missing:
        extras.append(f"{skipped_missing:,} missing")
    if extras:
        summary += "  (filtered: " + ", ".join(extras) + ")"
    print(summary)
    return rows


@dataclass
class ManifestDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    rows: list[dict[str, Any]]
    processor: Any  # Wav2Vec2Processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import soundfile as sf

        row = self.rows[idx]
        audio, sr = sf.read(row["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        phonemes = row["phonemes"].split()
        label_ids = [VOCAB.get(p, VOCAB["<unk>"]) for p in phonemes]

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "input_length": int(inputs.input_values.shape[-1]),
        }


@dataclass
class DataCollatorCTCWithPadding:
    """Pad input_values and labels to batch max length."""
    processor: Any
    padding: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        max_label_len = max(len(l) for l in label_features)
        labels_padded = torch.full(
            (len(label_features), max_label_len), fill_value=-100, dtype=torch.long
        )
        for i, lbl in enumerate(label_features):
            labels_padded[i, : len(lbl)] = lbl
        batch["labels"] = labels_padded
        return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two phone sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def build_decode_maps(processor: Any, include_spc: bool = False) -> tuple[dict[int, str], set[int]]:
    """Return ``(ids_to_tokens, drop_ids)`` for atomic phone-level decoding.

    ``processor.batch_decode`` glues multi-character IPA tokens (``t͡ʃ``,
    ``ɾ̞̊``, ``aː``) inside a word into one string, so a whitespace ``split``
    massively under-counts phones and inflates PER. Decoding ids directly
    through the vocabulary keeps every phone atomic. Drops pad, ``<blank>``
    and (by default) the ``<spc>`` word-boundary token so the canonical PER
    matches the phone-level confusion eval byte-for-byte.
    """
    vocab = processor.tokenizer.get_vocab()
    ids_to_tokens = {idx: token for token, idx in vocab.items()}
    drop_ids = {processor.tokenizer.pad_token_id}
    blank_id = vocab.get("<blank>")
    if blank_id is not None:
        drop_ids.add(blank_id)
    spc_id = vocab.get("<spc>")
    if spc_id is not None and not include_spc:
        drop_ids.add(spc_id)
    return ids_to_tokens, drop_ids


def decode_pred_ids(pred_ids: list[int], ids_to_tokens: dict[int, str], drop_ids: set[int]) -> list[str]:
    """CTC argmax ids -> collapsed atomic phone sequence (multi-char IPA stays atomic)."""
    collapsed: list[int] = []
    prev = -1
    for idx in pred_ids:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return [ids_to_tokens[idx] for idx in collapsed if idx not in drop_ids and idx in ids_to_tokens]


def decode_label_ids(label_ids: list[int], ids_to_tokens: dict[int, str], drop_ids: set[int]) -> list[str]:
    """Label ids -> atomic phone sequence, preserving repeated phones."""
    return [
        ids_to_tokens[idx]
        for idx in label_ids
        if idx != -100 and idx not in drop_ids and idx in ids_to_tokens
    ]


def compute_metrics_fn(processor: Any):
    import numpy as np

    ids_to_tokens, drop_ids = build_decode_maps(processor)

    def compute_metrics(pred: Any) -> dict[str, float]:
        predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        pred_ids = predictions if predictions.ndim == 2 else np.argmax(predictions, axis=-1)
        pred_ids = np.asarray(pred_ids)
        label_ids = pred.label_ids

        total_edits = total_len = 0
        for pred_row, label_row in zip(pred_ids, label_ids):
            pred_phones = decode_pred_ids(pred_row.tolist(), ids_to_tokens, drop_ids)
            ref_phones = decode_label_ids(label_row.tolist(), ids_to_tokens, drop_ids)
            total_len += len(ref_phones)
            total_edits += _edit_distance(pred_phones, ref_phones)

        per = total_edits / total_len if total_len > 0 else 0.0
        return {"per": round(per, 4)}

    return compute_metrics


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:  # noqa: ARG001
    """Keep eval RAM bounded: store argmax ids, not full (batch,time,vocab) logits."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Model + processor setup
# ---------------------------------------------------------------------------


def build_processor_and_model(
    model_id: str,
    vocab_path: Path,
    run_dir: Path,
    *,
    init_from_checkpoint: Path | None = None,
    save_processor: bool = True,
):
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        Wav2Vec2ForCTC,
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="<spc>",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    if save_processor:
        processor.save_pretrained(str(run_dir))

    source = str(init_from_checkpoint) if init_from_checkpoint else model_id
    print(f"  Loading weights from: {source}")
    model = Wav2Vec2ForCTC.from_pretrained(
        source,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(VOCAB),
        ignore_mismatched_sizes=True,
    )
    # Freeze the conv feature encoder (standard wav2vec2 fine-tuning recipe)
    model.freeze_feature_encoder()

    return processor, model


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def make_training_args(run_dir: Path, args: argparse.Namespace):
    from transformers import TrainingArguments

    use_cuda = torch.cuda.is_available()
    # Prefer bf16 when explicitly requested and supported; fp16 otherwise
    bf16 = bool(args.bf16 and use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)

    return TrainingArguments(
        output_dir=str(run_dir),
        group_by_length=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=50,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        remove_unused_columns=False,
        ignore_data_skip=True,
        seed=args.seed,
        data_seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mms-1b",
        choices=list(MODEL_ALIASES.keys()),
        help="Model alias (default: mms-1b)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest_v5.csv",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run directory name under runs/ (default: <model>_<YYYYMMDD_HHMM>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_DIR,
        help="Root directory for run outputs (default: runs/)",
    )

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default 10; MMS-1B 30→10 per CLAUDE.md)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size (default 8)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default 4 → effective batch 32)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="LR warmup steps (default 1000 for 10-epoch run)")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)

    # Data filtering
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="Drop train/dev rows longer than this (seconds). Prevents OOM on long clips. 0 = no filter")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Drop train/dev rows shorter than this (seconds)")

    # Resume / warm-start
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="'auto' to resume from latest checkpoint in run_dir, or explicit path. "
                             "Restores model+optimizer+scheduler+rng state.")
    parser.add_argument("--init-from-checkpoint", type=Path, default=None,
                        help="Warm-start model WEIGHTS from this checkpoint dir (optimizer/scheduler reset). "
                             "Mutually exclusive with --resume-from-checkpoint.")

    # Regularization / early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=6,
                        help="Stop if eval_per does not improve for this many evals (default 6; 0 = disabled)")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0005,
                        help="Minimum absolute eval_per improvement to reset patience (default 0.0005)")
    parser.add_argument("--save-total-limit", type=int, default=5,
                        help="Maximum checkpoints to keep per run (default 5)")

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default 4)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 instead of fp16 (if GPU supports it)")
    parser.add_argument("--skip-file-check", action="store_true", default=False,
                        help="Skip per-file exists() check on manifest load (faster restarts)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 100 train / 50 dev samples, 2 steps")

    args = parser.parse_args()

    if args.resume_from_checkpoint and args.init_from_checkpoint:
        print("ERROR: --resume-from-checkpoint and --init-from-checkpoint are mutually exclusive",
              file=sys.stderr)
        return 1

    # Lazy imports
    from transformers import Trainer, TrainerCallback, set_seed
    set_seed(args.seed)

    model_id = MODEL_ALIASES[args.model]

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = args.run_name or f"{args.model}_{ts}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve resume target
    resume_target: bool | str | None = None
    if args.resume_from_checkpoint == "auto":
        latest = find_latest_checkpoint(run_dir)
        if latest is None:
            print(f"WARN: --resume-from-checkpoint=auto but no checkpoint found in {run_dir}; starting fresh.")
        else:
            resume_target = str(latest)
            print(f"Auto-resume      : {latest}")
    elif args.resume_from_checkpoint:
        rp = Path(args.resume_from_checkpoint)
        if not rp.exists():
            print(f"ERROR: --resume-from-checkpoint path not found: {rp}", file=sys.stderr)
            return 1
        resume_target = str(rp)

    is_resuming = resume_target is not None

    print(f"Model            : {model_id}")
    print(f"Run dir          : {run_dir}")
    print(f"Manifest         : {args.manifest}")
    print(f"Device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"VRAM             : {gpu_props.total_memory // 1024**3} GB")
        print(f"bf16 supported   : {torch.cuda.is_bf16_supported()}")
    print(f"Effective batch  : {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Epochs           : {args.epochs}")
    print(f"Seed             : {args.seed}")
    print(f"Dry run          : {args.dry_run}")
    if args.init_from_checkpoint:
        print(f"Init checkpoint  : {args.init_from_checkpoint}")
    print(f"Early stopping   : patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")
    print(f"Checkpoint limit : {args.save_total_limit}")
    print()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run: uv run python scripts/build_dataset.py", file=sys.stderr)
        return 1

    if args.init_from_checkpoint and not args.init_from_checkpoint.exists():
        print(f"ERROR: --init-from-checkpoint path not found: {args.init_from_checkpoint}",
              file=sys.stderr)
        return 1

    # Build / verify vocab
    vocab_path = write_vocab(run_dir, overwrite=not is_resuming)
    print(f"Vocab            : {vocab_path} ({len(VOCAB)} tokens)")

    # Load manifest rows
    train_limit = 100 if args.dry_run else None
    dev_limit = 50 if args.dry_run else None
    max_dur = None if args.max_duration <= 0 else args.max_duration

    print("Loading splits...")
    train_rows = load_manifest(
        args.manifest, split="train",
        limit=train_limit,
        max_duration_s=max_dur, min_duration_s=args.min_duration,
        skip_file_check=args.skip_file_check,
    )
    dev_rows = load_manifest(
        args.manifest, split="dev",
        limit=dev_limit,
        max_duration_s=max_dur, min_duration_s=args.min_duration,
        skip_file_check=args.skip_file_check,
    )

    if not train_rows:
        print("ERROR: No train rows found (check AUDIO_ROOT symlink and filters)", file=sys.stderr)
        return 1

    # Build processor + model
    print(f"\nLoading model {model_id} ...")
    processor, model = build_processor_and_model(
        model_id, vocab_path, run_dir,
        init_from_checkpoint=args.init_from_checkpoint,
        save_processor=not is_resuming,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Trainable:  {n_trainable:,}")

    train_dataset = ManifestDataset(rows=train_rows, processor=processor)
    eval_dataset = ManifestDataset(rows=dev_rows, processor=processor)
    collator = DataCollatorCTCWithPadding(processor=processor)

    t_args = make_training_args(run_dir, args)
    if args.dry_run:
        t_args.num_train_epochs = 1
        t_args.eval_steps = 1
        t_args.save_steps = 1
        t_args.max_steps = 2

    callbacks = []

    class StatelessEarlyStoppingCallback(TrainerCallback):
        """Stop on eval_per plateau without requiring serialized callback state.

        HuggingFace's built-in EarlyStoppingCallback can fail when resuming from
        checkpoints saved before the callback state existed. This callback keeps
        only in-process patience state, so resume is safe; it uses
        trainer_state.best_metric as the baseline when available.
        """

        def __init__(self, patience: int, threshold: float) -> None:
            self.patience = patience
            self.threshold = threshold
            self.best_metric: float | None = None
            self.bad_evals = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            if metrics is None:
                return control
            metric = metrics.get("eval_per")
            if metric is None:
                return control
            metric = float(metric)
            baseline = self.best_metric
            if baseline is None and state.best_metric is not None:
                baseline = float(state.best_metric)

            if baseline is None or metric < (baseline - self.threshold):
                self.best_metric = metric
                self.bad_evals = 0
                return control

            self.best_metric = baseline
            self.bad_evals += 1
            print(
                f"Early stopping watch: eval_per={metric:.4f}, "
                f"best={baseline:.4f}, bad_evals={self.bad_evals}/{self.patience}"
            )
            if self.bad_evals >= self.patience:
                print("Early stopping: eval_per plateau reached.")
                control.should_training_stop = True
            return control

    if args.early_stopping_patience > 0:
        callbacks.append(
            StatelessEarlyStoppingCallback(
                patience=args.early_stopping_patience,
                threshold=args.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=t_args,
        compute_metrics=compute_metrics_fn(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    print("\nStarting training...\n")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_target)
    elapsed = time.time() - t0

    print("\nSaving final model...")
    trainer.save_model(str(run_dir / "final"))
    processor.save_pretrained(str(run_dir / "final"))

    # Summary
    state = trainer.state
    print("\n=== Training complete ===")
    print(f"  Wall time      : {elapsed/3600:.2f} h")
    print(f"  Global step    : {state.global_step}")
    print(f"  Epoch          : {state.epoch:.3f}")
    print(f"  Best metric    : {state.best_metric}")
    print(f"  Best ckpt      : {state.best_model_checkpoint}")
    print(f"  Run dir        : {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
