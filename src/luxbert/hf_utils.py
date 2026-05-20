"""Helpers shared by pretraining and evaluation."""

from __future__ import annotations

from transformers import PreTrainedTokenizerFast

from luxbert import config


def load_tokenizer(variant: str) -> PreTrainedTokenizerFast:
    path = config.variant_dir(config.TOKENIZERS, variant) / "tokenizer.json"
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def text_file(variant: str, split: str) -> str:
    text_dir = config.RAW if variant == "baseline" else config.CASEOPS
    return str(text_dir / f"{split}.txt")
