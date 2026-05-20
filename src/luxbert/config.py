"""Shared paths and small PoC defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CASEOPS = DATA / "caseops"
TOKENIZERS = ROOT / "tokenizers"
RUNS = ROOT / "runs"

HF_DATASET = "HuggingFaceFW/fineweb-2"
HF_SUBSET = "ltz_Latn"

VARIANTS = ("baseline", "caseops")

for _p in (RAW, CASEOPS, TOKENIZERS, RUNS):
    _p.mkdir(parents=True, exist_ok=True)


def variant_dir(base: Path, variant: str) -> Path:
    assert variant in VARIANTS, variant
    return base / variant
