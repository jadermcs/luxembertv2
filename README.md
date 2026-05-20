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

```bash
uv sync

# quick PoC (CPU-ok if small; GPU recommended)
scripts/run_poc.sh 50000 2000 16000      # train_docs eval_docs vocab

# on the cluster
sbatch scripts/slurm_poc.sh
```

Individual steps:

```bash
uv run python -m luxbert.data --train-docs 50000 --eval-docs 2000
uv run python -m luxbert.train_tokenizer --variant both --vocab-size 16000
uv run python -m luxbert.pretrain --variant baseline
uv run python -m luxbert.pretrain --variant caseops
uv run python -m luxbert.eval_bpb --variant baseline
uv run python -m luxbert.eval_bpb --variant caseops
```

## Layout

| Path | What |
|---|---|
| `src/luxbert/caseops.py` | reversible CaseOps transform (+ `tests/`) |
| `src/luxbert/data.py` | fineweb-2 `ltz_Latn` -> raw + caseops text |
| `src/luxbert/train_tokenizer.py` | BPE tokenizer per variant |
| `src/luxbert/pretrain.py` | small BERT MLM trainer |
| `src/luxbert/eval_bpb.py` | pseudo-log-likelihood BPB |
| `scripts/run_poc.sh` / `slurm_poc.sh` | end-to-end runners |
