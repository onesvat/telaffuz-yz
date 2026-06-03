# Telaffuz-YZ Thesis Repository

This repository contains the final research code, small validation data,
figures, and the compiled thesis, paper, and slides for a Turkish pronunciation
feedback pipeline.

The repository is intentionally lean. Large generated artifacts such as raw
audio, full studio assessment caches, SQLite databases, parquet manifests, and
model checkpoints are treated as external artifacts unless they are small
enough and necessary for local smoke tests.

## Pipeline

```text
Turkish text
  -> rule-based G2P
  -> expected IPA phones
  -> wav2vec2 phone recognizer
  -> phone-level coach assessment (status + continuous score)
  -> conservative pronunciation feedback
```

The final system has four main parts:

| Area | Purpose |
|---|---|
| G2P | Rule-based Istanbul Turkish grapheme-to-phoneme engine and IPA inventory |
| Audio atlas | Corpus metadata, transcript policy, forced-alignment schema, and phoneme acoustic summaries |
| wav2vec | XLS-R and MMS-1B phoneme recognizer fine-tuning/evaluation scripts and reports |
| Assessment | Two-score coach runtime, atlas/GMM quality evidence, validation set, and feedback analysis |

## Quick Start

```bash
uv sync
uv run g2p "merhaba"
uv run g2p --phones "ekmek"
uv run pytest
```

Focused checks:

```bash
PYTHONPATH=src uv run pytest tests/assess -q
uv run python scripts/eval_validation_recordings.py --limit 3
uv run python scripts/plot_wav2vec_per.py
```

## Audio Pipeline CLI

The `uv run audio <subcommand>` command exposes the internal audio corpus
pipeline (atlas building, forced alignment, dataset export, etc.). It is a
**research-only tool** and requires a desktop GPU, external audio files, and
the SQLite atlas database — none of which are committed to this repository.
Run `uv run audio --help` for a full subcommand listing.

## Reproducibility Scripts

Research scripts are run directly with `uv run python scripts/...`; they are
not exposed as public console commands.

| Script | Role |
|---|---|
| `scripts/build_dataset.py` | Build wav2vec training/evaluation datasets from prepared manifests |
| `scripts/train_wav2vec.py` | Fine-tune XLS-R or MMS-1B phoneme recognizers |
| `scripts/eval_wav2vec_test_split.py` | Evaluate wav2vec test-set PER |
| `scripts/eval_wav2vec_test_confusion.py` | Export wav2vec per-phone confusion analysis |
| `scripts/build_istanbul_baseline.py` | Generate the 20-item Istanbul pronunciation pilot set |
| `scripts/eval_validation_recordings.py` | End-to-end coach validation from raw audio (2 native speakers × 75 prompts) |
| `scripts/plot_*` | Generate committed thesis figures |

The coach endpoint is `/api/v1/assess`. For each target phone it returns a
`status` (`correct`/`incorrect`/`missing`/`extra`), a continuous phonetic-distance
`score`, separate non-gating `quality` and `length` channels, and diagnostics for
the free decode.

## Key Results

| Component | Main result |
|---|---|
| G2P Blind 75 | 1.48% PER, 98.52% symbol accuracy |
| wav2vec XLS-R | 4.05% Full IPA PER |
| wav2vec MMS-1B | 4.13% Full IPA PER |
| Coach validation | 65.62% intended-error detection at 5.29%/5.54% flagged FP (CTL/W_NAT) on the 2-speaker native set; see the safety–detection frontier in the thesis |

### Why These Numbers Matter

- **G2P at 1.48% PER** establishes a reliable target-pronunciation producer. The
  remaining errors are concentrated in long-vowel lex gaps and loanword
  palatalization. These are documented blind spots: K1 only emits a non-punitive
  long-vowel cue, while palatal/velar evidence is handled by the K3 F2 rule.
- **wav2vec PER** is reported separately from coach feedback quality. The coach
  runtime treats recognizer identity evidence and atlas quality evidence as
  different signals.
- **The validation set** separates canonical, natural, and intended-wrong prompts
  so false positives and caught intended errors can be reported directly.
- **MMS-1B is the default** model alias for the runtime, with XLS-R kept as an
  alternate recognizer for comparison and diagnostics.

## Repository Layout

```text
configs/              coach runtime configs and model registry
data/g2p/             small G2P benchmark and lexical data
data/validation/      small assessment prompt and annotation tables
deliverables/         compiled thesis (PDF/DOCX), paper (PDF), and slides (HTML)
figures/              generated PNG figures
frontend/             single-page Vite + React demo frontend
scripts/              reproducibility and figure-generation scripts
src/g2p/              Turkish G2P package
src/audio/            audio corpus, alignment, and atlas helpers
src/assess/           phone-level coach assessment package (status + score)
src/api/              FastAPI demo backend (5 endpoints under /api/v1)
tests/                focused regression tests
```

## Artifact Policy

Committed data is limited to small tables and final-facing documentation.
Large or machine-generated artifacts stay outside Git:

| Artifact type | Policy |
|---|---|
| raw audio and recordings | external |
| SQLite audio atlas databases | external |
| full studio assessment JSON caches | external |
| training manifests and parquet datasets | external |
| model checkpoints | external unless explicitly staged as a small local smoke artifact |

## Citation

```bibtex
@software{telaffuz_yz,
  title = {Telaffuz-YZ: Turkish Pronunciation Feedback Pipeline},
  author = {Onur Neşvat},
  year = {2026},
  note = {Master's thesis research repository}
}
```
