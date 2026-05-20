#!/usr/bin/env bash
# End-to-end CaseOps vs. baseline comparison (PoC). Runs with `uv`.
# Usage: scripts/run_poc.sh [train_docs] [eval_docs] [vocab_size]
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_DOCS="${1:-50000}"
EVAL_DOCS="${2:-2000}"
VOCAB="${3:-16000}"

echo "== 1. data prep =="
uv run python -m luxbert.data --train-docs "$TRAIN_DOCS" --eval-docs "$EVAL_DOCS"

echo "== 2. train tokenizers =="
uv run python -m luxbert.train_tokenizer --variant both --vocab-size "$VOCAB"

for V in baseline caseops; do
  echo "== 3. pretrain ($V) =="
  uv run python -m luxbert.pretrain --variant "$V"
done

echo "== 4. evaluate BPB =="
for V in baseline caseops; do
  uv run python -m luxbert.eval_bpb --variant "$V" | tee "runs/$V/bpb.txt"
done

echo "== summary =="
grep -H "BPB" runs/*/bpb.txt
