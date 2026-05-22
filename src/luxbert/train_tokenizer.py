"""Train a BPE tokenizer for one variant (baseline raw text or CaseOps text).

Both variants use the *identical* algorithm/config — the only difference is the
input text — so any downstream gap is attributable to CaseOps, not tokenizer
hyper-parameters.
"""

from __future__ import annotations

import argparse

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from luxbert import config, experiment

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_one(exp: str, variant: str, vocab_size: int) -> None:
    train_file = str(config.text_dir(exp, variant) / "train.txt")

    tok = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    # NFC only: do NOT lowercase. For CaseOps, casing is already encoded as the
    # marker char; for baseline, the model must learn case from cased pieces.
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)]
    )
    tok.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2,
        show_progress=True,
    )
    tok.train([train_file], trainer)

    out_dir = config.tokenizer_dir(exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    tok.save(str(out_path))
    print(f"[{exp}/{variant}] vocab={tok.get_vocab_size()} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yml file")
    args = ap.parse_args()

    cfg = experiment.load(args.config)
    train_one(cfg.name, cfg.variant, cfg.tokenizer.vocab_size)


if __name__ == "__main__":
    main()
