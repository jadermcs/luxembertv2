#!/usr/bin/env bash
#SBATCH --job-name=caseops-poc
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Submit with: sbatch scripts/slurm_poc.sh
# Adjust --partition / account flags for your cluster as needed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda 2>/dev/null || true   # cluster-specific; ignore if absent

# Heavier-than-PoC defaults now that we have a GPU. Tune freely.
TRAIN_DOCS="${TRAIN_DOCS:-200000}"
EVAL_DOCS="${EVAL_DOCS:-3000}"
VOCAB="${VOCAB:-32000}"

uv run python -m luxbert.data --train-docs "$TRAIN_DOCS" --eval-docs "$EVAL_DOCS"
uv run python -m luxbert.train_tokenizer --variant both --vocab-size "$VOCAB"

for V in baseline caseops; do
  uv run python -m luxbert.pretrain --variant "$V" \
    --hidden 512 --layers 8 --heads 8 --intermediate 2048 \
    --batch-size 128 --epochs 5
done

for V in baseline caseops; do
  uv run python -m luxbert.eval_bpb --variant "$V" --max-lines 1000 | tee "runs/$V/bpb.txt"
done

grep -H "BPB" runs/*/bpb.txt
