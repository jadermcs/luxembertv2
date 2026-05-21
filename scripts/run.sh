#!/usr/bin/env bash
# End-to-end CaseOps vs. baseline comparison for one experiment config. Runs with `uv`.
# Usage: scripts/run_poc.sh [configs/<experiment>.yml]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/poc.yml}"

echo "== 1. data prep ($CONFIG) =="
uv run python -m luxbert.data --config "$CONFIG"

echo "== 2. train tokenizers =="
uv run python -m luxbert.train_tokenizer --config "$CONFIG" --variant both

for V in baseline caseops; do
  echo "== 3. pretrain ($V) =="
  uv run python -m luxbert.pretrain --config "$CONFIG" --variant "$V"
done

echo "== 4. evaluate BPB =="
for V in baseline caseops; do
  uv run python -m luxbert.eval_bpb --config "$CONFIG" --variant "$V"
done

echo "== summary (results/bpb_summary.tsv) =="
column -t -s$'\t' results/bpb_summary.tsv
