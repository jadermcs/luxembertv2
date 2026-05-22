"""Pretrain a small BERT (MLM) for one variant of an experiment.

All hyperparameters come from the experiment config (``configs/*.yml``); the
architecture/objective is identical across variants to isolate the tokenizer
effect. Use a small config for a CPU PoC, a larger one on GPU.
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    ModernBertConfig,
    ModernBertForMaskedLM,
    Trainer,
    TrainingArguments,
)
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
    find_packed_sequence_indices,
    packed_sequence_mask_function,
)

from luxbert import config, experiment
from luxbert.hf_utils import load_tokenizer, text_file


class PackedModernBertForMaskedLM(ModernBertForMaskedLM):
    """ModernBERT MLM that confines attention to within each packed document.

    When ``position_ids`` restart per document (see ``build_dataset``), unrelated
    docs share a block but must not attend to each other. We detect the packed
    segments and build per-layer-type masks (full + sliding-window) that forbid
    cross-document attention, so packing removes pad waste *without* the
    cross-doc contamination plain concatenation would introduce. Falls back to
    the stock behavior when the batch isn't packed.
    """

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is not None and not isinstance(attention_mask, dict):
            packed = find_packed_sequence_indices(position_ids)
            if packed is not None:
                and_fn = packed_sequence_mask_function(packed)
                # create_*_mask only reads shape/dtype/device off this tensor.
                embeds = torch.empty(
                    input_ids.shape[0], input_ids.shape[1], 1,
                    dtype=self.dtype, device=input_ids.device,
                )
                attention_mask = {
                    "full_attention": create_bidirectional_mask(
                        self.config, embeds, None, and_mask_function=and_fn
                    ),
                    "sliding_attention": create_bidirectional_sliding_window_mask(
                        self.config, embeds, None, and_mask_function=and_fn
                    ),
                }
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


# Encoder architectures, selected by pretrain.arch in the config. Both variants of
# an experiment always use the same arch -- only the tokenizer/text differs -- so
# the BPB comparison stays fair. Each entry is (config class, MLM model class).
ARCHITECTURES = {
    "bert": (BertConfig, BertForMaskedLM),
    "modernbert": (ModernBertConfig, PackedModernBertForMaskedLM),
}


def build_model(pc, tokenizer):
    """Construct a from-scratch MLM whose arch is chosen by ``pc.arch``.

    The hyperparameters below map onto the shared keyword names accepted by every
    supported config class, so an experiment's sizing is identical across arches.
    """
    try:
        config_cls, model_cls = ARCHITECTURES[pc.arch]
    except KeyError:
        raise ValueError(
            f"unknown arch {pc.arch!r}; choose one of {sorted(ARCHITECTURES)}"
        ) from None
    kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        hidden_size=pc.hidden,
        num_hidden_layers=pc.layers,
        num_attention_heads=pc.heads,
        intermediate_size=pc.intermediate,
        max_position_embeddings=pc.block_size + 2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # ModernBERT's config carries extra special-token ids that default to the
    # original 50k-vocab ModernBERT (overflowing ours); align them with our
    # tokenizer so they stay valid indices.
    if pc.arch == "modernbert":
        kwargs["cls_token_id"] = tokenizer.cls_token_id
        kwargs["sep_token_id"] = tokenizer.sep_token_id
    return model_cls(config_cls(**kwargs))


def build_dataset(tokenizer, train_file: str, block_size: int, num_proc: int, arch: str):
    """Tokenize and pack the corpus into fixed ``block_size`` blocks (no padding).

    Documents are concatenated and chunked, which avoids pad waste but lets a
    block straddle document boundaries. How we keep those boundaries from leaking
    across attention depends on the architecture:

    * ``modernbert`` -- emit ``position_ids`` that restart at 0 at every block
      start and every doc boundary within a block; ``PackedModernBertForMaskedLM``
      turns those segments into block-diagonal attention masks (true isolation).
    * ``bert`` -- insert one ``[SEP]`` between documents (RoBERTa FULL-SENTENCES);
      a learned boundary signal rather than hard isolation.

    Either way the treatment is identical across the two variants, so BPB stays
    a fair comparison.
    """
    ds = load_dataset("text", data_files={"train": train_file})["train"]
    pack = arch == "modernbert"
    sep_id = tokenizer.sep_token_id

    def tok_fn(batch):
        out = tokenizer(batch["text"], add_special_tokens=False)
        if not pack:
            out["input_ids"] = [ids + [sep_id] for ids in out["input_ids"]]
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)

    def group(batch):
        concat, docs = [], []
        for doc_idx, ids in enumerate(batch["input_ids"]):
            concat.extend(ids)
            docs.extend([doc_idx] * len(ids))
        total = (len(concat) // block_size) * block_size
        starts = range(0, total, block_size)
        out = {"input_ids": [concat[i : i + block_size] for i in starts]}
        if pack:
            position_ids = []
            for i in starts:
                seg = docs[i : i + block_size]
                pos, p = [], 0
                for j in range(block_size):
                    if j and seg[j] != seg[j - 1]:
                        p = 0
                    pos.append(p)
                    p += 1
                position_ids.append(pos)
            out["position_ids"] = position_ids
        else:
            out["attention_mask"] = [[1] * block_size for _ in out["input_ids"]]
        return out

    ds = ds.map(group, batched=True, num_proc=num_proc)
    return ds


class PackedMLMCollator(DataCollatorForLanguageModeling):
    """MLM collator that carries per-token ``position_ids`` through to the model.

    The base collator would drop the dataset's ``position_ids`` and add an
    all-ones 2D ``attention_mask``; the latter suppresses the packed-sequence
    mask path. We keep the position ids (which encode doc boundaries) and drop
    the redundant 2D mask -- blocks are already full ``block_size``, so there is
    nothing to pad.
    """

    def torch_call(self, examples):
        positions = [example.pop("position_ids") for example in examples]
        batch = super().torch_call(examples)
        batch["position_ids"] = torch.tensor(positions, dtype=torch.long)
        batch.pop("attention_mask", None)
        return batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yml file")
    args = ap.parse_args()

    cfg = experiment.load(args.config)
    pc = cfg.pretrain

    tokenizer = load_tokenizer(cfg.name)
    ds = build_dataset(
        tokenizer,
        text_file(cfg.name, cfg.variant, "train"),
        pc.block_size,
        pc.num_proc,
        pc.arch,
    )
    tag = f"{cfg.name}/{cfg.variant}"
    print(f"[{tag}] {len(ds)} blocks of {pc.block_size} tokens")

    model = build_model(pc, tokenizer)
    print(f"[{tag}] arch: {pc.arch}  params: {model.num_parameters()/1e6:.1f}M")

    # modernbert packs documents and isolates them via position_ids; bert relies
    # on the [SEP] separator build_dataset inserted, so it uses the stock collator.
    collator_cls = (
        PackedMLMCollator if pc.arch == "modernbert" else DataCollatorForLanguageModeling
    )
    collator = collator_cls(tokenizer=tokenizer, mlm=True, mlm_probability=pc.mlm_prob)

    out_dir = config.run_dir(cfg.name)
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
        report_to="wandb",
        seed=pc.seed,
        bf16=False,
        dataloader_num_workers=4,
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
