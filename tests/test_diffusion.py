"""Unit tests for the masked-diffusion noising and hazard-weighted loss.

These exercise the pieces that make the diffusion objective differ from MLM:
DiffusionCollator (time-dependent absorbing-state masking) and diffusion_loss
(the linear-schedule 1/t hazard weight). Neither needs a trained tokenizer or a
real model -- a stub tokenizer and hand-built logits suffice.
"""

import math

import torch

from luxbert.pretrain import DiffusionCollator, diffusion_loss


class _StubTokenizer:
    """Minimal stand-in: the collator only reads these two attributes."""

    mask_token_id = 1
    all_special_ids = [0, 1, 2]  # pad, mask, [SEP]-like


def _collate(eps=1e-3, n=64, length=12, seed=0):
    torch.manual_seed(seed)
    tok = _StubTokenizer()
    col = DiffusionCollator(tok, eps)
    # Rows of plain (non-special) ids 10..10+length, plus one trailing special id.
    examples = [{"input_ids": list(range(10, 10 + length)) + [2]} for _ in range(n)]
    return tok, col(examples)


def test_collator_shapes_and_keys():
    tok, batch = _collate()
    assert set(batch) == {"input_ids", "labels", "t"}
    assert batch["input_ids"].shape == batch["labels"].shape == (64, 13)
    assert batch["t"].shape == (64,)


def test_time_in_range():
    _, batch = _collate(eps=0.01)
    t = batch["t"]
    assert (t >= 0.01).all() and (t <= 1.0).all()


def test_special_tokens_never_masked():
    tok, batch = _collate()
    # The trailing id (2) is special and must survive everywhere.
    assert (batch["input_ids"][:, -1] == 2).all()
    assert (batch["labels"][:, -1] == -100).all()


def test_at_least_one_masked_per_example():
    _, batch = _collate()
    masked = batch["labels"] != -100
    assert masked.any(dim=1).all()


def test_masked_positions_are_mask_id_and_labels_align():
    tok, batch = _collate()
    masked = batch["labels"] != -100
    # Every masked position holds the mask id in the input...
    assert (batch["input_ids"][masked] == tok.mask_token_id).all()
    # ...and the label there recovers the original token (10 + column index).
    rows, cols = masked.nonzero(as_tuple=True)
    assert torch.equal(batch["labels"][masked], cols.to(batch["labels"].dtype) + 10)


def test_higher_t_masks_more():
    # Averaged over many rows, the realized mask fraction tracks t.
    _, batch = _collate(n=4000, length=20, seed=1)
    t = batch["t"]
    frac = (batch["labels"] != -100).float().sum(dim=1) / 20
    lo = frac[t < 0.3].mean()
    hi = frac[t > 0.7].mean()
    assert hi > lo


def test_diffusion_loss_hazard_weight():
    # Uniform logits -> per-token CE = ln(V) exactly; with that we can predict the
    # hazard-weighted batch mean in closed form: mean_i (1/t_i) * n_masked_i * ln V.
    vocab = 5
    # Example 0: 2 masked tokens at t=0.5 -> weight 2.0
    # Example 1: 1 masked token  at t=0.25 -> weight 4.0
    logits = torch.zeros(2, 3, vocab)
    labels = torch.tensor([[3, 4, -100], [2, -100, -100]])
    t = torch.tensor([0.5, 0.25])
    expected = ((1 / 0.5) * 2 + (1 / 0.25) * 1) / 2 * math.log(vocab)
    got = diffusion_loss(logits, labels, t).item()
    assert math.isclose(got, expected, rel_tol=1e-5)
