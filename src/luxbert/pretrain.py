"""Pretrain a small BERT (MLM) for one variant.

Defaults are PoC-sized so the pipeline runs quickly; pass larger values on GPU
for a real comparison. The architecture/objective is identical across variants
to isolate the tokenizer effect.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from luxbert import config
from luxbert.hf_utils import load_tokenizer, text_file


def build_dataset(tokenizer, train_file: str, block_size: int, num_proc: int):
    ds = load_dataset("text", data_files={"train": train_file})["train"]

    def tok_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)

    def group(batch):
        concat = sum(batch["input_ids"], [])
        total = (len(concat) // block_size) * block_size
        ids = [concat[i : i + block_size] for i in range(0, total, block_size)]
        return {"input_ids": ids, "attention_mask": [[1] * block_size for _ in ids]}

    ds = ds.map(group, batched=True, num_proc=num_proc)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=config.VARIANTS, required=True)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--intermediate", type=int, default=1024)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.variant)
    ds = build_dataset(
        tokenizer, text_file(args.variant, "train"), args.block_size, args.num_proc
    )
    print(f"[{args.variant}] {len(ds)} blocks of {args.block_size} tokens")

    model_cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate,
        max_position_embeddings=args.block_size + 2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(model_cfg)
    print(f"[{args.variant}] params: {model.num_parameters()/1e6:.1f}M")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    out_dir = config.variant_dir(config.RUNS, args.variant)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        bf16=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[{args.variant}] saved model -> {out_dir}")


if __name__ == "__main__":
    main()
