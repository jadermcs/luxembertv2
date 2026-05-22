"""Shared dataset constants and per-experiment path resolution.

Every artifact is namespaced by the experiment name (the config file's ``name``)
so distinct experiments never clobber each other's data/tokenizers/models, and
``results/bpb_summary.tsv`` can attribute each row to a specific config. Since
each config carries its own ``variant`` and the ``name`` encodes it (e.g.
``poc-baseline`` / ``poc-caseops``), tokenizer/run paths key on ``name`` alone;
only the text dir still branches on the variant (raw vs CaseOps text).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TOKENIZERS = ROOT / "tokenizers"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"

RESULTS_TSV = RESULTS / "bpb_summary.tsv"

HF_DATASET = "HuggingFaceFW/fineweb-2"
HF_SUBSET = "ltz_Latn"

VARIANTS = ("baseline", "caseops")


def _check_variant(variant: str) -> None:
    assert variant in VARIANTS, variant


def raw_dir(experiment: str) -> Path:
    return DATA / experiment / "raw"


def caseops_dir(experiment: str) -> Path:
    return DATA / experiment / "caseops"


def text_dir(experiment: str, variant: str) -> Path:
    """Where a variant's train/eval text lives (raw for baseline, caseops else)."""
    _check_variant(variant)
    return raw_dir(experiment) if variant == "baseline" else caseops_dir(experiment)


def tokenizer_dir(experiment: str) -> Path:
    return TOKENIZERS / experiment


def run_dir(experiment: str) -> Path:
    return RUNS / experiment
