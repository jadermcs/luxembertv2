"""Unit tests for the masked-diffusion noising, schedules, and hazard-weighted loss.

These exercise the pieces that make the diffusion objective differ from MLM:
the NoiseSchedule (linear vs log-linear), DiffusionCollator (time-dependent
absorbing-state masking), and diffusion_loss (the hazard-weighted NELBO). None
need a trained tokenizer or a real model -- a stub tokenizer and hand-built
logits suffice.
"""

import math

import torch

from luxbert.pretrain import (
    DiffusionCollator,
    LinearSchedule,
    LogLinearSchedule,
    diffusion_loss,
)


class _StubTokenizer:
    """Minimal stand-in: the collator only reads these two attributes."""

    mask_token_id = 1
    all_special_ids = [0, 1, 2]  # pad, mask, [SEP]-like


def _collate(schedule=None, n=64, length=12, seed=0):
    torch.manual_seed(seed)
    tok = _StubTokenizer()
    col = DiffusionCollator(tok, schedule or LinearSchedule(1e-3))
    # Rows of plain (non-special) ids 10..10+length, plus one trailing special id.
    examples = [{"input_ids": list(range(10, 10 + length)) + [2]} for _ in range(n)]
    return tok, col(examples)


# --- schedules -------------------------------------------------------------

def test_linear_schedule_formulas():
    s = LinearSchedule(1e-3)
    t = torch.tensor([0.25, 0.5, 1.0])
    assert torch.allclose(s.mask_prob(t), t)
    assert torch.allclose(s.hazard(t), 1.0 / t)


def test_loglinear_schedule_formulas():
    s = LogLinearSchedule(1e-3, sigma_max=7.0)
    t = torch.tensor([0.1, 0.5, 1.0])
    alpha = torch.exp(-7.0 * t)
    assert torch.allclose(s.mask_prob(t), 1.0 - alpha)
    assert torch.allclose(s.hazard(t), 7.0 * alpha / (1.0 - alpha))


def test_loglinear_mask_prob_monotonic_and_endpoints():
    s = LogLinearSchedule(1e-3, sigma_max=7.0)
    t = torch.linspace(0.0, 1.0, 50)
    p = s.mask_prob(t)
    assert (p.diff() >= 0).all()              # rises with t
    assert math.isclose(p[0].item(), 0.0, abs_tol=1e-6)
    assert p[-1].item() > 0.99                 # ~fully masked at t=1


def test_loglinear_hazard_decays():
    # The defining contrast with linear: the hazard falls across t (vs 1/t's shape).
    s = LogLinearSchedule(1e-3, sigma_max=7.0)
    t = torch.linspace(0.05, 1.0, 50)
    assert (s.hazard(t).diff() <= 0).all()


def test_time_sampling_respects_eps_floor():
    for s in (LinearSchedule(0.05), LogLinearSchedule(0.05, 7.0)):
        t = s.sample_t(5000)
        assert (t >= 0.05).all() and (t <= 1.0).all()


# --- collator (schedule-agnostic invariants) -------------------------------

def test_collator_shapes_and_keys():
    _, batch = _collate()
    assert set(batch) == {"input_ids", "labels", "t"}
    assert batch["input_ids"].shape == batch["labels"].shape == (64, 13)
    assert batch["t"].shape == (64,)


def test_special_tokens_never_masked():
    _, batch = _collate()
    # The trailing id (2) is special and must survive everywhere.
    assert (batch["input_ids"][:, -1] == 2).all()
    assert (batch["labels"][:, -1] == -100).all()


def test_at_least_one_masked_per_example():
    for schedule in (LinearSchedule(1e-3), LogLinearSchedule(1e-3, 7.0)):
        _, batch = _collate(schedule=schedule)
        assert (batch["labels"] != -100).any(dim=1).all()


def test_masked_positions_are_mask_id_and_labels_align():
    tok, batch = _collate()
    masked = batch["labels"] != -100
    # Every masked position holds the mask id in the input...
    assert (batch["input_ids"][masked] == tok.mask_token_id).all()
    # ...and the label there recovers the original token (10 + column index).
    _, cols = masked.nonzero(as_tuple=True)
    assert torch.equal(batch["labels"][masked], cols.to(batch["labels"].dtype) + 10)


def test_higher_t_masks_more():
    # Averaged over many rows, the realized mask fraction tracks t.
    _, batch = _collate(n=4000, length=20, seed=1)
    t = batch["t"]
    frac = (batch["labels"] != -100).float().sum(dim=1) / 20
    assert frac[t > 0.7].mean() > frac[t < 0.3].mean()


# --- loss ------------------------------------------------------------------

def test_diffusion_loss_linear_hazard_weight():
    # Uniform logits -> per-token CE = ln(V) exactly; with that we can predict the
    # hazard-weighted batch mean in closed form: mean_i w(t_i) * n_masked_i * ln V.
    vocab = 5
    # Example 0: 2 masked tokens at t=0.5 -> linear weight 2.0
    # Example 1: 1 masked token  at t=0.25 -> linear weight 4.0
    logits = torch.zeros(2, 3, vocab)
    labels = torch.tensor([[3, 4, -100], [2, -100, -100]])
    t = torch.tensor([0.5, 0.25])
    expected = ((1 / 0.5) * 2 + (1 / 0.25) * 1) / 2 * math.log(vocab)
    got = diffusion_loss(logits, labels, t, LinearSchedule(1e-3)).item()
    assert math.isclose(got, expected, rel_tol=1e-5)


def test_diffusion_loss_uses_schedule_hazard():
    # Same setup, log-linear schedule: the weights become the loglinear hazard.
    vocab = 5
    logits = torch.zeros(2, 3, vocab)
    labels = torch.tensor([[3, 4, -100], [2, -100, -100]])
    t = torch.tensor([0.5, 0.25])
    s = LogLinearSchedule(1e-3, sigma_max=7.0)
    w = s.hazard(t)
    expected = (w[0] * 2 + w[1] * 1).item() / 2 * math.log(vocab)
    got = diffusion_loss(logits, labels, t, s).item()
    assert math.isclose(got, expected, rel_tol=1e-5)
