"""Experiment configuration loaded from ``configs/*.yml``.

One config file describes one experiment *and one tokenizer kind* (``baseline``
or ``caseops``); the kind is a required field (``tokenizer.kind``), not a CLI
flag. A fair bits-per-byte comparison is therefore a pair of configs that are
byte-for-byte identical except for ``tokenizer.kind`` (and ``name``), e.g.
``poc-baseline.yml`` / ``poc-caseops.yml`` — keep that pair in sync.

Schema (all sections optional except ``tokenizer.kind``; omitted fields fall
back to the defaults below)::

    name: poc-baseline        # defaults to the file stem
    marker: "↑"
    data:      {train_docs, eval_docs, min_chars}
    tokenizer: {kind, vocab_size}   # kind is required: baseline | caseops
    pretrain:  {arch, objective, block_size, hidden, layers, heads, intermediate,
                mlm_prob, diffusion_eps, diffusion_schedule, diffusion_sigma_max,
                electra_generator_layer_ratio, electra_loss_weight,
                electra_sample_temperature, electra_finetune_epochs,
                electra_finetune_steps,
                optim, lr, weight_decay, batch_size, epochs, max_steps,
                warmup_ratio, num_proc, seed}
    eval:      {max_lines, block_size, micro_batch}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from luxbert.caseops import DEFAULT_MARKER
from luxbert.config import TOKENIZER_KINDS


@dataclass(frozen=True)
class DataCfg:
    train_docs: int = 50_000
    eval_docs: int = 2_000
    min_chars: int = 64


@dataclass(frozen=True)
class TokenizerCfg:
    kind: str  # required: baseline | caseops (was the top-level `variant`)
    vocab_size: int = 16_000


@dataclass(frozen=True)
class PretrainCfg:
    arch: str = "bert"  # encoder architecture: "bert" | "modernbert" | "deberta"
    objective: str = "mlm"  # training objective: "mlm" | "diffusion" | "electra"
    block_size: int = 128
    hidden: int = 256
    layers: int = 4
    heads: int = 4
    intermediate: int = 1024
    mlm_prob: float = 0.15  # fixed corruption rate for the "mlm" objective
    diffusion_eps: float = 1e-3  # min diffusion time t (time floor; caps the hazard)
    diffusion_schedule: str = "linear"  # "linear" (1/t) | "loglinear" (exp(-sigma_max*t))
    diffusion_sigma_max: float = 7.0  # loglinear total noise at t=1 (mask prob 1-e^-x)
    # ELECTRA-only knobs. The generator shares hidden/heads/intermediate with the
    # discriminator (so embeddings can be shared at the same dim) but uses fewer
    # layers, set by max(1, round(layers * electra_generator_layer_ratio)).
    # electra_loss_weight is the lambda multiplier on the discriminator BCE loss
    # (ELECTRA paper uses 50). After ELECTRA pretraining, the discriminator
    # backbone is fitted with an MLM head and trained under MLM for
    # electra_finetune_epochs (or electra_finetune_steps if >= 0) so its output
    # can be scored with BPB on equal footing with the other objectives.
    electra_generator_layer_ratio: float = 1 / 3
    electra_loss_weight: float = 50.0
    electra_sample_temperature: float = 1.0
    electra_finetune_epochs: float = 1.0
    electra_finetune_steps: int = -1
    optim: str = "adamw_torch"
    lr: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.05
    num_proc: int = 4
    seed: int = 42


@dataclass(frozen=True)
class EvalCfg:
    max_lines: int = 300
    block_size: int = 128
    micro_batch: int = 64


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    tokenizer: TokenizerCfg
    marker: str = DEFAULT_MARKER
    data: DataCfg = field(default_factory=DataCfg)
    pretrain: PretrainCfg = field(default_factory=PretrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @property
    def data_key(self) -> str:
        """Stable key for the shared corpus: identical data fields => one copy.

        The corpus content is fully determined by the doc counts, ``min_chars``,
        and ``marker`` (the marker only affects the CaseOps text). Experiments
        that agree on these reuse the same ``data/<key>/`` tree rather than each
        regenerating a byte-identical copy. ``tokenizer.kind`` is deliberately
        excluded: ``data.py`` writes both raw and CaseOps text every time.
        """
        d = self.data
        return f"train{d.train_docs}-eval{d.eval_docs}-min{d.min_chars}-m{self.marker}"


def load(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment config, validating that no unknown keys slip in."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tok = raw.get("tokenizer", {})
    kind = tok.get("kind")
    if kind not in TOKENIZER_KINDS:
        raise ValueError(
            f"{path}: 'tokenizer.kind' must be one of {TOKENIZER_KINDS}, got {kind!r}"
        )
    return ExperimentConfig(
        name=raw.get("name", path.stem),
        marker=raw.get("marker", DEFAULT_MARKER),
        data=DataCfg(**raw.get("data", {})),
        tokenizer=TokenizerCfg(**tok),
        pretrain=PretrainCfg(**raw.get("pretrain", {})),
        eval=EvalCfg(**raw.get("eval", {})),
    )
