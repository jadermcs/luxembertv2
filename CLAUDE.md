# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A controlled experiment for a data-efficient Luxembourgish (LB) encoder. The first
ablation, **CaseOps**, tests whether re-encoding casing as an explicit token helps a
low-resource MLM. The whole point is a *fair head-to-head*: two pipeline variants
(`baseline` vs `caseops`), each its own config, that are byte-for-byte identical
except for the one change under test, judged by **bits-per-byte (BPB)**. Preserve
that symmetry when extending.

- `IDEAS.md` — research motivation and the broader roadmap.
- `EVALUATION.md` — the downstream benchmark suite (POS, NER, SIB-200, NLI, Belebele,
  etc.) for evaluating a finished encoder. Not yet wired into code.

## Commands

Always use `uv` — never bare `python`/`pip`.

```bash
uv sync                                     # install deps into .venv
uv run --extra dev pytest                    # run tests (pytest is in the dev extra)
uv run --extra dev pytest tests/test_caseops.py::test_roundtrip_simple   # single test

scripts/run.sh                              # full pipeline locally (poc pair)
sbatch scripts/slurm.sh                       # full pipeline on the GPU cluster (base pair)
```

Individual stages (all under `uv run python -m luxbert.<module>`): `data` →
`train_tokenizer` → `pretrain` → `eval_bpb`. **Every stage takes one `--config
configs/<exp>.yml`** and nothing else — the variant is read from the config, not
a flag. See `scripts/run.sh` for the order; it runs the stages once per config.

## Experiments and configs

An experiment+variant is one YAML file in `configs/` (e.g. `poc-baseline.yml`,
`poc-caseops.yml`, `base-*.yml`, `modernbert-*.yml`) holding all hyperparameters
(vocab size, layers/hidden/heads, optimizer, lr, epochs, `pretrain.arch`, …) plus a
**required `variant:`** (`baseline` or `caseops`). `name:` defaults to the file stem.
One config = one variant, so a fair comparison is a **pair** of configs that are
identical except for `name` and `variant` — keeping that pair in sync by hand is what
keeps the comparison fair. `experiment.py` parses a config into frozen dataclasses
(`DataCfg`, `TokenizerCfg`, `PretrainCfg`, `EvalCfg`); a missing/invalid `variant`
raises, unknown YAML keys raise, omitted keys use the dataclass defaults. To add an
experiment, copy a config *pair* and change `name:`, `variant:`, + the fields.

All artifacts are **namespaced by experiment name** (which encodes the variant) so
experiments don't clobber each other: `data/<exp>/{raw,caseops}/`, `tokenizers/<exp>/`,
`runs/<exp>/` (path helpers in `config.py`; only the text dir branches on the variant,
since `data.py` writes both transforms). `eval_bpb` appends one row per run to
`results/bpb_summary.tsv` (tab-separated; header in `eval_bpb.RESULT_COLUMNS`) tagging the
experiment, variant, key config fields, and metrics. `data/`, `tokenizers/`, `runs/` are
gitignored and regenerated; `configs/` and `results/` are tracked.

## Pipeline architecture

Four stages, each a standalone CLI module in `src/luxbert/`, chained by the scripts.
State passes between stages as **files on disk** at the namespaced paths above.

1. **`data.py`** — streams fineweb-2 `ltz_Latn`, NFC-normalizes and one-doc-per-line
   cleans it, and writes *both* `data/<exp>/raw/{train,eval}.txt` and
   `data/<exp>/caseops/{train,eval}.txt` from the **same source docs**. This shared origin
   is what makes BPB comparable. It verifies CaseOps round-trips on every doc.
2. **`train_tokenizer.py`** — trains an identical-config BPE tokenizer per variant; the
   *only* input difference is raw vs CaseOps text. NFC normalize, no lowercasing.
3. **`pretrain.py`** — trains a small MLM from scratch. The encoder architecture is
   chosen by `pretrain.arch` (`bert` → `BertForMaskedLM`, `modernbert` →
   `ModernBertForMaskedLM`; default `bert`); see `ARCHITECTURES` / `build_model`.
   Architecture, objective, and hyperparameters are identical across variants — only the
   tokenizer and text differ. PoC-sized defaults; the SLURM script overrides with larger
   model/data.
4. **`eval_bpb.py`** — scores held-out text with pseudo-log-likelihood (mask one token
   at a time), normalized by **original UTF-8 byte count** (not token count). Both
   variants are scored against the same original bytes, so CaseOps' extra marker tokens
   are paid for in the numerator. **Lower BPB wins.**

Supporting modules: `caseops.py` (the reversible transform, the core idea),
`experiment.py` (config loading), `config.py` (namespaced path helpers, the `VARIANTS`
tuple, dataset constants), `hf_utils.py` (tokenizer loading + text-file path resolution
shared by stages 3–4).

## CaseOps transform (`caseops.py`)

Each uppercase letter `X` → `marker + x` (marker is `↑`, U+2191). So `"Hello"` →
`"↑hello"`, `"HELLO"` → `"↑h↑e↑l↑l↑o"`. Cased variants then share lowercase sub-words
and embeddings. It is a **bijection**: literal markers are escaped by doubling, and only
letters that round-trip exactly through `.lower()/.upper()` are folded. `decode(encode(x))
== x` is an invariant the tests and `data.py` both check — keep it true.

## Conventions and gotchas

- **Keep the two variants symmetric.** Any new stage or hyperparameter must apply
  identically to `baseline` and `caseops`, or the BPB comparison is invalid. Since
  each variant is now its own config, a new field must be added to *both* configs of
  a pair with the same value.
- **transformers is v5.x** — `overwrite_output_dir` is gone; `warmup_ratio` is
  deprecated. Don't reintroduce removed args.
- **Real training runs on the SLURM GPU cluster, not the login node.** The login node's
  NVIDIA driver is too old for the installed `cu130` torch wheel — CUDA only works on GPU
  nodes (or pin a wheel matching the local driver, e.g. `--index-url
  .../whl/cu128`). `eval_bpb.py` falls back to CPU automatically.
