"""Bits-per-byte (BPB) evaluation, comparable across tokenizers.

We score each model with the pseudo-log-likelihood (PLL) of Salazar et al. 2020:
mask one position at a time and read the model's log-prob of the true token,
summed over all positions. Crucially we normalize by the number of bytes in the
*original* text (identical for both kinds), not by token count -- so CaseOps'
extra marker tokens are paid for in the numerator and the comparison is fair:

    BPB = sum_positions(-log p(token)) / ln(2) / original_utf8_bytes

Lower is better. A model whose markers cost more bits than the shared lowercase
embeddings save will show a *higher* BPB than the baseline.

Each run appends one row to ``results/bpb_summary.tsv``, tagged with the
experiment name and key config fields so experiments can be compared later.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from luxbert import config, experiment
from luxbert.caseops import CaseOps
from luxbert.hf_utils import load_tokenizer

# Column order for results/bpb_summary.tsv (also the header written on creation).
RESULT_COLUMNS = [
    "timestamp",
    "experiment",
    "tokenizer",
    "arch",
    "vocab_size",
    "layers",
    "hidden",
    "heads",
    "intermediate",
    "optim",
    "lr",
    "epochs",
    "train_docs",
    "lines",
    "bytes",
    "tokens",
    "tokens_per_byte",
    "nats_per_token",
    "bpb",
]


def append_result(row: dict) -> None:
    """Append one TSV row to results/bpb_summary.tsv, writing the header once."""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_TSV
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(RESULT_COLUMNS) + "\n")
        f.write("\t".join(str(row[c]) for c in RESULT_COLUMNS) + "\n")


@torch.no_grad()
def seq_pll_nats(model, mask_id, input_ids, device, micro_batch) -> float:
    """Pseudo-log-likelihood (nats) of one token sequence under MLM scoring."""
    ids = torch.tensor(input_ids, dtype=torch.long)
    L = ids.size(0)
    if L == 0:
        return 0.0
    total = 0.0
    for start in range(0, L, micro_batch):
        positions = list(range(start, min(start + micro_batch, L)))
        b = len(positions)
        batch = ids.unsqueeze(0).repeat(b, 1).clone()
        for r, pos in enumerate(positions):
            batch[r, pos] = mask_id
        batch = batch.to(device)
        logits = model(input_ids=batch).logits  # [b, L, V]
        logp = F.log_softmax(logits, dim=-1)
        for r, pos in enumerate(positions):
            total += -logp[r, pos, ids[pos]].item()
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yml file")
    ap.add_argument("--model-dir", default=None, help="defaults to runs/<exp>")
    ap.add_argument("--eval-file", default=None, help="defaults to data/<data_key>/raw/eval.txt")
    args = ap.parse_args()

    cfg = experiment.load(args.config)
    ec = cfg.eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = args.model_dir or str(config.run_dir(cfg.name))
    eval_file = args.eval_file or str(config.raw_dir(cfg.data_key) / "eval.txt")

    tokenizer = load_tokenizer(cfg.name)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device).eval()
    mask_id = tokenizer.mask_token_id
    co = CaseOps(marker=cfg.marker)

    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    n = 0
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            total_bytes += len(line.encode("utf-8"))  # ORIGINAL bytes, both kinds
            model_text = line if cfg.tokenizer.kind == "baseline" else co.encode(line)
            ids = tokenizer(model_text, add_special_tokens=False)["input_ids"]
            total_tokens += len(ids)
            # Score EVERY token (chunked into context windows of block_size) so the
            # full line's bytes correspond to the full PLL -> fair per-byte ratio.
            for start in range(0, len(ids), ec.block_size):
                chunk = ids[start : start + ec.block_size]
                total_nats += seq_pll_nats(model, mask_id, chunk, device, ec.micro_batch)
            n += 1
            if n >= ec.max_lines:
                break

    tokens_per_byte = total_tokens / total_bytes
    nats_per_token = total_nats / max(total_tokens, 1)
    bpb = total_nats / math.log(2) / total_bytes

    print(f"experiment       : {cfg.name}")
    print(f"tokenizer        : {cfg.tokenizer.kind}")
    print(f"lines scored     : {n}")
    print(f"original bytes    : {total_bytes}")
    print(f"tokens            : {total_tokens}")
    print(f"tokens/byte       : {tokens_per_byte:.4f}  (lower = denser)")
    print(f"nats/token        : {nats_per_token:.4f}")
    print(f"BPB               : {bpb:.4f}   <-- lower is better")

    append_result(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment": cfg.name,
            "tokenizer": cfg.tokenizer.kind,
            "arch": cfg.pretrain.arch,
            "vocab_size": cfg.tokenizer.vocab_size,
            "layers": cfg.pretrain.layers,
            "hidden": cfg.pretrain.hidden,
            "heads": cfg.pretrain.heads,
            "intermediate": cfg.pretrain.intermediate,
            "optim": cfg.pretrain.optim,
            "lr": cfg.pretrain.lr,
            "epochs": cfg.pretrain.epochs,
            "train_docs": cfg.data.train_docs,
            "lines": n,
            "bytes": total_bytes,
            "tokens": total_tokens,
            "tokens_per_byte": f"{tokens_per_byte:.6f}",
            "nats_per_token": f"{nats_per_token:.6f}",
            "bpb": f"{bpb:.6f}",
        }
    )
    print(f"appended -> {config.RESULTS_TSV}")


if __name__ == "__main__":
    main()
