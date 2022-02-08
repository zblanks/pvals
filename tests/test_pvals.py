import numpy as np
import pytest

from pvals.pvals import PVALS


def test_rank_one_tensor():
    a = np.array([1, 1])
    b = np.array([2, 2])
    c = np.array([3, 3])
    X = np.einsum("i, j, k", a, b, c).astype(float)
    als = PVALS(X, r=1, n_restart=1)
    A, B, C = als.pvals()
    obj_val = als._compute_objective(A, B, C)

    assert obj_val == pytest.approx(0, abs=1e-7)


def test_gaussian_2_2_2_tensor():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(2, 2, 2))
    als = PVALS(X, r=2, n_restart=5)
    A, B, C = als.pvals()
    obj_val = als._compute_objective(A, B, C)

    assert obj_val == pytest.approx(0, abs=1e-7)


def test_monotone_decrease():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(2, 2, 2))
    als = PVALS(X, r=2, n_restart=5)
    _, _, _ = als.pvals()

    for i in range(1, len(als.best_losses)):
        assert als.best_losses[i] <= als.best_losses[i - 1]
