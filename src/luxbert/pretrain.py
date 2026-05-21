"""Pretrain a small BERT (MLM) for one variant of an experiment.

All hyperparameters come from the experiment config (``configs/*.yml``); the
architecture/objective is identical across variants to isolate the tokenizer
effect. Use a small config for a CPU PoC, a larger one on GPU.
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

from luxbert import config, experiment
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
    ap.add_argument("--config", required=True, help="path to a configs/*.yml file")
    ap.add_argument("--variant", choices=config.VARIANTS, required=True)
    args = ap.parse_args()

    cfg = experiment.load(args.config)
    pc = cfg.pretrain

    tokenizer = load_tokenizer(cfg.name, args.variant)
    ds = build_dataset(
        tokenizer, text_file(cfg.name, args.variant, "train"), pc.block_size, pc.num_proc
    )
    tag = f"{cfg.name}/{args.variant}"
    print(f"[{tag}] {len(ds)} blocks of {pc.block_size} tokens")

    model_cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=pc.hidden,
        num_hidden_layers=pc.layers,
        num_attention_heads=pc.heads,
        intermediate_size=pc.intermediate,
        max_position_embeddings=pc.block_size + 2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(model_cfg)
    print(f"[{tag}] params: {model.num_parameters()/1e6:.1f}M")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=pc.mlm_prob
    )

    out_dir = config.run_dir(cfg.name, args.variant)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=pc.batch_size,
        learning_rate=pc.lr,
        num_train_epochs=pc.epochs,
        max_steps=pc.max_steps,
        optim=pc.optim,
        warmup_ratio=pc.warmup_ratio,
        weight_decay=pc.weight_decay,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=pc.seed,
        bf16=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[{tag}] saved model -> {out_dir}")


if __name__ == "__main__":
    main()
