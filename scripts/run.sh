#!/usr/bin/env bash
# End-to-end CaseOps vs. baseline comparison. Each config is one tokenizer kind, so a
# comparison is a pair of configs run through every stage. Runs with `uv`.
# Usage: scripts/run.sh [configs/<exp>.yml ...]  (default: the poc pair)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  CONFIGS=(configs/poc-baseline.yml configs/poc-caseops.yml)
fi

for CONFIG in "${CONFIGS[@]}"; do
  echo "== 1. data prep ($CONFIG) =="
  uv run python -m luxbert.data --config "$CONFIG"

  echo "== 2. train tokenizer ($CONFIG) =="
  uv run python -m luxbert.train_tokenizer --config "$CONFIG"

  echo "== 3. pretrain ($CONFIG) =="
  uv run python -m luxbert.pretrain --config "$CONFIG"

  echo "== 4. evaluate BPB ($CONFIG) =="
  uv run python -m luxbert.eval_bpb --config "$CONFIG"
done

echo "== summary (results/bpb_summary.tsv) =="
column -t -s$'\t' results/bpb_summary.tsv
