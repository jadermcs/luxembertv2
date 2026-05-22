"""Shared dataset constants and path resolution.

Tokenizers and runs are namespaced by the experiment name (the config file's
``name``) so distinct experiments never clobber each other's tokenizers/models,
and ``results/bpb_summary.tsv`` can attribute each row to a specific config.

The corpus, however, is namespaced by a **dataset key** derived purely from the
data-shaping fields (doc counts, ``min_chars``, ``marker``) — see
``ExperimentConfig.data_key``. Experiments that ask for the same data therefore
share one ``data/<key>/`` tree instead of each regenerating an identical copy.
Within that tree only the text dir branches on the tokenizer kind (raw for
``baseline``, CaseOps text otherwise).
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

TOKENIZER_KINDS = ("baseline", "caseops")


def _check_kind(kind: str) -> None:
    assert kind in TOKENIZER_KINDS, kind


def raw_dir(data_key: str) -> Path:
    return DATA / data_key / "raw"


def caseops_dir(data_key: str) -> Path:
    return DATA / data_key / "caseops"


def text_dir(data_key: str, kind: str) -> Path:
    """Where a tokenizer kind's train/eval text lives (raw for baseline, caseops else)."""
    _check_kind(kind)
    return raw_dir(data_key) if kind == "baseline" else caseops_dir(data_key)


def tokenizer_dir(experiment: str) -> Path:
    return TOKENIZERS / experiment


def run_dir(experiment: str) -> Path:
    return RUNS / experiment
