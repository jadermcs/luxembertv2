# luxbertv2 — CaseOps tokenizer experiment

First experiment toward a data-efficient Luxembourgish encoder (see
[`IDEAS.md`](IDEAS.md), [`EVALUATION.md`](EVALUATION.md)).

## CaseOps

**CaseOps** removes uppercase letters from the character stream and re-encodes
casing as an explicit operation: every uppercase letter `X` becomes
`<marker>x` (a marker char `↑` followed by the lowercase letter).

```
"Lëtzebuerg" -> "↑lëtzebuerg"
"HELLO"      -> "↑h↑e↑l↑l↑o"
```

It is a reversible transform (`luxbert.caseops.CaseOps.encode/decode`), so
`"hello"`, `"Hello"` and `"HELLO"` share the same lowercase sub-words and
embeddings — concentrating scarce LB data on fewer, better-estimated pieces.

## The comparison

Train the **same** tiny BERT (MLM) twice — once on raw text + a raw-text BPE
tokenizer, once on CaseOps text + a CaseOps BPE tokenizer — then compare with
**bits-per-byte (BPB)**.

BPB is normalized by the **original** UTF-8 byte count (identical for both
variants), so it is tokenizer-agnostic: CaseOps' extra marker tokens are paid
for in the numerator. **Lower BPB wins.**

## Run it

> **Cluster GPU note:** `uv sync` installs the default CUDA wheel of torch
> (`cu130` here). If your GPU node's NVIDIA driver is older, pin a matching
> wheel, e.g. `uv pip install "torch" --index-url https://download.pytorch.org/whl/cu128`.

An experiment is fully described by a config file in [`configs/`](configs)
(hyperparameters: vocab size, layers, optimizer, …) plus a required `variant:`
(`baseline` or `caseops`). One config = one variant, so a fair comparison is a
**pair** of configs identical except for `name`/`variant`
(e.g. `poc-baseline.yml` / `poc-caseops.yml`). Each `eval_bpb` run appends a row
to [`results/bpb_summary.tsv`](results/bpb_summary.tsv).

```bash
uv sync

# quick PoC (CPU-ok; GPU recommended) — runs the poc baseline+caseops pair
scripts/run.sh

# on the cluster (larger config) — runs the base pair
sbatch scripts/slurm.sh
```

Individual steps (each takes a single `--config`):

```bash
uv run python -m luxbert.data            --config configs/poc-baseline.yml
uv run python -m luxbert.train_tokenizer --config configs/poc-baseline.yml
uv run python -m luxbert.pretrain        --config configs/poc-baseline.yml
uv run python -m luxbert.eval_bpb        --config configs/poc-baseline.yml
# …then repeat with configs/poc-caseops.yml for the other half of the comparison
```

To add an experiment, copy a config *pair*, change `name:` and the
hyperparameters in both (keeping `variant:` distinct), and run them — artifacts
and results are namespaced by `name`.

## Layout

| Path | What |
|---|---|
| `configs/*.yml` | experiment hyperparameters (one file = one experiment+variant) |
| `src/luxbert/experiment.py` | loads a config into typed dataclasses |
| `src/luxbert/caseops.py` | reversible CaseOps transform (+ `tests/`) |
| `src/luxbert/data.py` | fineweb-2 `ltz_Latn` -> raw + caseops text |
| `src/luxbert/train_tokenizer.py` | BPE tokenizer per variant |
| `src/luxbert/pretrain.py` | small BERT MLM trainer |
| `src/luxbert/eval_bpb.py` | pseudo-log-likelihood BPB -> `results/bpb_summary.tsv` |
| `scripts/run.sh` / `slurm.sh` | end-to-end runners (take config paths) |
