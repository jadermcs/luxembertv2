"""Helpers shared by pretraining and evaluation."""

from __future__ import annotations

from transformers import PreTrainedTokenizerFast

from luxbert import config


def load_tokenizer(experiment: str, variant: str) -> PreTrainedTokenizerFast:
    path = config.tokenizer_dir(experiment, variant) / "tokenizer.json"
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def text_file(experiment: str, variant: str, split: str) -> str:
    return str(config.text_dir(experiment, variant) / f"{split}.txt")
