"""Prepare fineweb-2 ``ltz_Latn`` text: raw + CaseOps-transformed train/eval.

Both tokenizer kinds are derived from the *same* original text so that
bits-per-byte (normalized by original utf-8 bytes) is a fair comparison later.

The corpus is written under ``data/<data_key>/`` (see
``ExperimentConfig.data_key``), so experiments that request the same data share
one copy. If that copy already exists this stage is a no-op unless ``--force``.
"""

from __future__ import annotations

import argparse
import unicodedata

from datasets import load_dataset

from luxbert import config, experiment
from luxbert.caseops import CaseOps


def clean(text: str) -> str:
    """Collapse a document to a single normalized line.

    NFC-normalize, replace newlines/tabs with spaces, squeeze whitespace.
    Keeping one doc per line makes downstream file reading trivial; the MLM
    trainer re-chunks by tokens anyway.
    """
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yml file")
    ap.add_argument(
        "--force",
        action="store_true",
        help="regenerate even if the shared corpus already exists",
    )
    args = ap.parse_args()

    cfg = experiment.load(args.config)
    co = CaseOps(marker=cfg.marker)
    raw_root = config.raw_dir(cfg.data_key)
    caseops_root = config.caseops_dir(cfg.data_key)

    splits = {
        "train": (cfg.data.train_docs, "train"),
        "eval": (cfg.data.eval_docs, "test"),
    }

    expected = [
        root / f"{name}.txt" for root in (raw_root, caseops_root) for name in splits
    ]
    if not args.force and all(p.exists() for p in expected):
        print(f"[data] reusing existing corpus -> data/{cfg.data_key}/ (use --force to rebuild)")
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    caseops_root.mkdir(parents=True, exist_ok=True)

    rt_fail = 0
    for name, (n_docs, hf_split) in splits.items():
        ds = load_dataset(
            config.HF_DATASET, config.HF_SUBSET, split=hf_split, streaming=True
        )
        raw_path = raw_root / f"{name}.txt"
        co_path = caseops_root / f"{name}.txt"
        kept = 0
        with raw_path.open("w", encoding="utf-8") as fr, co_path.open(
            "w", encoding="utf-8"
        ) as fc:
            for row in ds:
                line = clean(row["text"])
                if len(line) < cfg.data.min_chars:
                    continue
                enc = co.encode(line)
                if co.decode(enc) != line:
                    rt_fail += 1
                fr.write(line + "\n")
                fc.write(enc + "\n")
                kept += 1
                if kept >= n_docs:
                    break
        print(f"[{name}] wrote {kept} docs -> {raw_path.name}, {co_path.name}")

    if rt_fail:
        print(f"WARNING: {rt_fail} docs failed CaseOps round-trip (unicode quirks)")
    else:
        print("CaseOps round-trip exact on all kept docs")


if __name__ == "__main__":
    main()
