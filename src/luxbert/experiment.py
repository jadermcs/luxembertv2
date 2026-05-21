"""Experiment configuration loaded from ``configs/*.yml``.

One config file describes one experiment and is applied *identically* to both
the ``baseline`` and ``caseops`` variants — the variant is chosen at run time,
not stored in the config — which is what keeps the bits-per-byte comparison fair.

Schema (all sections optional; omitted fields fall back to the defaults below)::

    name: poc                 # defaults to the file stem
    marker: "↑"
    data:      {train_docs, eval_docs, min_chars}
    tokenizer: {vocab_size}
    pretrain:  {block_size, hidden, layers, heads, intermediate, mlm_prob,
                optim, lr, weight_decay, batch_size, epochs, max_steps,
                warmup_ratio, num_proc, seed}
    eval:      {max_lines, block_size, micro_batch}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from luxbert.caseops import DEFAULT_MARKER


@dataclass(frozen=True)
class DataCfg:
    train_docs: int = 50_000
    eval_docs: int = 2_000
    min_chars: int = 64


@dataclass(frozen=True)
class TokenizerCfg:
    vocab_size: int = 16_000


@dataclass(frozen=True)
class PretrainCfg:
    block_size: int = 128
    hidden: int = 256
    layers: int = 4
    heads: int = 4
    intermediate: int = 1024
    mlm_prob: float = 0.15
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
    marker: str = DEFAULT_MARKER
    data: DataCfg = field(default_factory=DataCfg)
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    pretrain: PretrainCfg = field(default_factory=PretrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


def load(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment config, validating that no unknown keys slip in."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        name=raw.get("name", path.stem),
        marker=raw.get("marker", DEFAULT_MARKER),
        data=DataCfg(**raw.get("data", {})),
        tokenizer=TokenizerCfg(**raw.get("tokenizer", {})),
        pretrain=PretrainCfg(**raw.get("pretrain", {})),
        eval=EvalCfg(**raw.get("eval", {})),
    )
