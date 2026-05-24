"""Pretrain a small BERT (MLM) for one tokenizer kind of an experiment.

All hyperparameters come from the experiment config (``configs/*.yml``); the
architecture/objective is identical across kinds to isolate the tokenizer
effect. Use a small config for a CPU PoC, a larger one on GPU.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
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


# Encoder architectures, selected by pretrain.arch in the config. Both kinds of
# an experiment always use the same arch -- only the tokenizer/text differs -- so
# the BPB comparison stays fair. Each entry is (config class, MLM model class).
ARCHITECTURES = {
    "bert": (BertConfig, BertForMaskedLM),
    "modernbert": (ModernBertConfig, PackedModernBertForMaskedLM),
    "deberta": (DebertaV2Config, DebertaV2ForMaskedLM),
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
    # DeBERTa's distinguishing feature is disentangled attention over relative
    # positions, which is off by the config's defaults; turn it on with the
    # standard DeBERTa-v3 settings so the arch is actually DeBERTa, not a BERT
    # twin. Identical across kinds, so the BPB comparison stays fair.
    if pc.arch == "deberta":
        kwargs["relative_attention"] = True
        kwargs["position_buckets"] = 256
        kwargs["pos_att_type"] = ["p2c", "c2p"]
        kwargs["norm_rel_ebd"] = "layer_norm"
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

    Either way the treatment is identical across the two kinds, so BPB stays
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


class NoiseSchedule:
    """Absorbing-diffusion noise schedule, shared by the collator and the loss.

    A schedule maps a diffusion time ``t`` to two consistent quantities:

    * ``mask_prob(t) = 1 - alpha_t`` -- the per-token probability of being absorbed
      to ``[MASK]`` (used by ``DiffusionCollator`` to draw the corruption), and
    * ``hazard(t) = -alpha'_t / (1 - alpha_t)`` -- the continuous-time NELBO weight
      (used by ``diffusion_loss``).

    Drawing ``t`` and the mask here keeps the two in lockstep, so a schedule change
    is a single, self-consistent swap. ``sample_t`` draws ``t ~ U(eps, 1)``; the
    ``eps`` floor caps the otherwise-unbounded hazard as ``t -> 0``.
    """

    def __init__(self, eps: float):
        self.eps = eps

    def sample_t(self, n: int) -> torch.Tensor:
        return torch.empty(n).uniform_(self.eps, 1.0)

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:  # 1 - alpha_t
        raise NotImplementedError

    def hazard(self, t: torch.Tensor) -> torch.Tensor:  # -alpha'_t / (1 - alpha_t)
        raise NotImplementedError


class LinearSchedule(NoiseSchedule):
    """``alpha_t = 1 - t``: mask probability ``t`` and hazard ``1/t`` (capped by eps)."""

    def mask_prob(self, t):
        return t

    def hazard(self, t):
        return 1.0 / t


class LogLinearSchedule(NoiseSchedule):
    """Log-survival-linear schedule: ``sigma(t) = sigma_max * t``, ``alpha_t = exp(-sigma)``.

    Here ``log alpha_t`` is linear in ``t`` (hence "log-linear"). The mask
    probability ``1 - exp(-sigma_max * t)`` rises gently then saturates toward 1 at
    ``t = 1`` (set ``sigma_max`` so ``exp(-sigma_max) ~ 0``, e.g. 7 -> 0.999), and
    the hazard ``w(t) = sigma_max * alpha_t / (1 - alpha_t)`` *decays* across ``t``
    instead of the linear schedule's flat-in-magnitude ``1/t`` -- shifting loss
    weight away from the heavily-masked, high-``t`` regime.
    """

    def __init__(self, eps: float, sigma_max: float):
        super().__init__(eps)
        self.sigma_max = sigma_max

    def _alpha(self, t):
        return torch.exp(-self.sigma_max * t)

    def mask_prob(self, t):
        return 1.0 - self._alpha(t)

    def hazard(self, t):
        a = self._alpha(t)
        return self.sigma_max * a / (1.0 - a)


def build_schedule(pc) -> NoiseSchedule:
    """Construct the diffusion noise schedule selected by ``pc.diffusion_schedule``."""
    if pc.diffusion_schedule == "linear":
        return LinearSchedule(pc.diffusion_eps)
    if pc.diffusion_schedule == "loglinear":
        return LogLinearSchedule(pc.diffusion_eps, pc.diffusion_sigma_max)
    raise ValueError(
        f"unknown diffusion_schedule {pc.diffusion_schedule!r}; "
        "choose 'linear' or 'loglinear'"
    )


class DiffusionCollator:
    """Absorbing-state (masked) diffusion noising under a ``NoiseSchedule``.

    For each example we draw a diffusion time ``t`` and mask each non-special token
    independently with probability ``schedule.mask_prob(t) = 1 - alpha_t``. Every
    chosen token is replaced by ``[MASK]`` -- a *pure* absorbing state, with none of
    MLM's 80/10/10 random/keep corruption -- and we keep ``t`` per example so the
    loss can apply ``schedule.hazard(t)``. Special tokens are never masked, and at
    least one token is masked per example so every row contributes a gradient.

    ``position_ids`` (the doc-packing signal for modernbert) are carried through
    unchanged; rows without them (e.g. the bert ``[SEP]`` path) just omit the key.
    """

    def __init__(self, tokenizer, schedule: NoiseSchedule):
        self.mask_id = tokenizer.mask_token_id
        self.special_ids = torch.tensor(sorted(tokenizer.all_special_ids))
        self.schedule = schedule

    def __call__(self, examples):
        input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        B, L = input_ids.shape
        special = torch.isin(input_ids, self.special_ids)

        t = self.schedule.sample_t(B)
        probs = self.schedule.mask_prob(t)
        mask = (torch.rand(B, L) < probs.unsqueeze(1)) & ~special
        # Guarantee >=1 masked token per example: where nothing was selected, mask
        # one uniformly chosen non-special position.
        empty = ~mask.any(dim=1)
        for i in empty.nonzero(as_tuple=True)[0].tolist():
            cand = (~special[i]).nonzero(as_tuple=True)[0]
            if len(cand):
                mask[i, cand[torch.randint(len(cand), (1,))]] = True

        labels = input_ids.clone()
        labels[~mask] = -100  # score only the masked (denoised) positions
        noised = input_ids.clone()
        noised[mask] = self.mask_id

        batch = {"input_ids": noised, "labels": labels, "t": t}
        if "position_ids" in examples[0]:
            batch["position_ids"] = torch.tensor(
                [e["position_ids"] for e in examples], dtype=torch.long
            )
        return batch


def diffusion_loss(logits, labels, t, schedule: NoiseSchedule):
    """Masked-diffusion NELBO (Monte-Carlo over ``t``) for the given schedule.

    Each example's cross-entropy is *summed* over its masked positions, scaled by
    the absorbing-diffusion hazard ``schedule.hazard(t) = -alpha'_t / (1 - alpha_t)``,
    and averaged over the batch. This is the MDLM / MD4 objective; standard MLM
    would instead take an unweighted token mean. The schedule sets the hazard
    (``1/t`` for linear, ``sigma_max * alpha_t / (1 - alpha_t)`` for loglinear).
    """
    vocab = logits.size(-1)
    ce = F.cross_entropy(
        logits.view(-1, vocab), labels.view(-1), ignore_index=-100, reduction="none"
    ).view(labels.shape)  # [B, L]; zero at ignored (-100) positions
    per_example = ce.sum(dim=1)  # sum over masked tokens
    weight = schedule.hazard(t).to(per_example.device)
    return (weight * per_example).mean()


class DiffusionTrainer(Trainer):
    """Trainer that swaps MLM's token-mean loss for the hazard-weighted diffusion NELBO.

    The model is the same ``ModernBertForMaskedLM`` denoiser; only the loss differs.
    We pop the per-example time ``t`` and ``labels`` (so the model does not compute
    its own MLM loss) before delegating to :func:`diffusion_loss`. The schedule is
    read off the collator so it stays identical to the one used to draw the masks.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        t = inputs.pop("t")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = diffusion_loss(outputs.logits, labels, t, self.data_collator.schedule)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yml file")
    args = ap.parse_args()

    cfg = experiment.load(args.config)
    pc = cfg.pretrain

    tokenizer = load_tokenizer(cfg.name)
    ds = build_dataset(
        tokenizer,
        text_file(cfg.data_key, cfg.tokenizer.kind, "train"),
        pc.block_size,
        pc.num_proc,
        pc.arch,
    )
    tag = f"{cfg.name}/{cfg.tokenizer.kind}"
    print(f"[{tag}] {len(ds)} blocks of {pc.block_size} tokens")

    model = build_model(pc, tokenizer)
    print(f"[{tag}] arch: {pc.arch}  params: {model.num_parameters()/1e6:.1f}M")

    # The objective picks both the noising (collator) and the loss (trainer). For
    # "diffusion" the time-dependent collator carries position_ids itself, so it
    # works under both the packed (modernbert) and [SEP] (bert) data layouts. For
    # "mlm", modernbert packs documents and isolates them via position_ids while
    # bert relies on the [SEP] separator build_dataset inserted (stock collator).
    if pc.objective == "diffusion":
        schedule = build_schedule(pc)
        collator = DiffusionCollator(tokenizer, schedule)
        trainer_cls = DiffusionTrainer
        print(f"[{tag}] diffusion schedule: {pc.diffusion_schedule}")
    elif pc.objective == "mlm":
        collator_cls = (
            PackedMLMCollator if pc.arch == "modernbert" else DataCollatorForLanguageModeling
        )
        collator = collator_cls(tokenizer=tokenizer, mlm=True, mlm_probability=pc.mlm_prob)
        trainer_cls = Trainer
    else:
        raise ValueError(
            f"unknown objective {pc.objective!r}; choose 'mlm' or 'diffusion'"
        )
    print(f"[{tag}] objective: {pc.objective}")

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
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=ds, data_collator=collator
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[{tag}] saved model -> {out_dir}")


if __name__ == "__main__":
    main()
