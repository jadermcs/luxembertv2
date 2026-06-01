"""Unit tests for the Barlow-Twins decorrelation loss and its helpers.

These exercise the pieces that make the "barlow" objective differ from plain MLM:
the off-diagonal extraction helper and the redundancy-reduction loss itself. They
need neither a trained tokenizer nor a real model -- hand-built representation
matrices suffice.
"""

import torch

from luxbert.pretrain import _off_diagonal, barlow_twins_loss


def test_off_diagonal_extracts_non_diagonal_entries():
    m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    got = set(_off_diagonal(m).tolist())
    assert got == {2.0, 3.0, 4.0, 6.0, 7.0, 8.0}  # everything but 1, 5, 9


def test_decorrelated_dims_give_zero_loss():
    # Two orthogonal, zero-mean, unit-variance columns: C = I exactly, so both the
    # on- and off-diagonal terms vanish regardless of lambda.
    z = torch.tensor(
        [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    )
    assert torch.isclose(
        barlow_twins_loss(z, lambd=0.1), torch.tensor(0.0), atol=1e-5
    )


def test_perfectly_correlated_dims_pay_off_diagonal():
    # Two identical columns standardize to the same unit vector, so C = [[1,1],[1,1]]:
    # on-diagonal is 0 (population std => unit variance) and the two off-diagonal 1s
    # contribute lambd * 2.
    z = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    lambd = 0.05
    got = barlow_twins_loss(z, lambd=lambd).item()
    assert abs(got - 2 * lambd) < 1e-5


def test_off_diagonal_weight_scales_redundancy_penalty():
    # More redundancy (correlation) costs more, and a larger lambda amplifies it.
    z = torch.tensor([[0.0, 0.1], [1.0, 0.9], [2.0, 2.1], [3.0, 2.9]])
    small = barlow_twins_loss(z, lambd=1e-3).item()
    large = barlow_twins_loss(z, lambd=1.0).item()
    assert large > small > 0
