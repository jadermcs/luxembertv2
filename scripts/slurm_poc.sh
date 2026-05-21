#!/usr/bin/env bash
#SBATCH --job-name=caseops-poc
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Submit with: sbatch scripts/slurm_poc.sh [configs/<experiment>.yml]
# Adjust --partition / account flags for your cluster as needed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda 2>/dev/null || true   # cluster-specific; ignore if absent

CONFIG="${1:-configs/base.yml}"

uv run python -m luxbert.data --config "$CONFIG"
uv run python -m luxbert.train_tokenizer --config "$CONFIG" --variant both

for V in baseline caseops; do
  uv run python -m luxbert.pretrain --config "$CONFIG" --variant "$V"
done

for V in baseline caseops; do
  uv run python -m luxbert.eval_bpb --config "$CONFIG" --variant "$V"
done

column -t -s$'\t' results/bpb_summary.tsv
