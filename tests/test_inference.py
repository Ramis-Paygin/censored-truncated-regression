"""Tests for the standalone likelihood-ratio test."""

from __future__ import annotations

import numpy as np
import pytest

from censtrunc import CensoredRegression, lr_test


def test_lr_test_detects_significant_restriction(sim_two_sided):
    """If we wrongly drop a relevant regressor, LR should reject."""
    full = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    # Drop the second slope (which is truly non-zero)
    restricted = CensoredRegression(
        left=sim_two_sided.left, right=sim_two_sided.right
    ).fit(sim_two_sided.X[:, :1], sim_two_sided.y)

    res = lr_test(full, restricted)
    assert res.df == 1
    assert res.statistic > 0
    assert res.p_value < 1e-6
    assert res.n_obs == full.n_obs_


def test_lr_test_accepts_true_restriction():
    """If we drop a truly zero coefficient, LR should not reject."""
    rng = np.random.default_rng(1)
    n = 2500
    X = rng.normal(size=(n, 3))
    beta = np.array([1.0, 0.5, -0.3, 0.0])  # last slope is zero
    y_star = beta[0] + X @ beta[1:] + rng.normal(size=n)
    y = np.clip(y_star, 0.0, 2.5)

    full = CensoredRegression(left=0.0, right=2.5).fit(X, y)
    restricted = CensoredRegression(left=0.0, right=2.5).fit(X[:, :2], y)
    res = lr_test(full, restricted)
    assert res.df == 1
    # With the true coefficient at 0, p-value should not be small at any reasonable level
    assert res.p_value > 0.01


def test_lr_test_rejects_unfitted_model(sim_two_sided):
    full = CensoredRegression(left=0.0, right=2.5).fit(sim_two_sided.X, sim_two_sided.y)
    unfit = CensoredRegression(left=0.0, right=2.5)
    with pytest.raises(ValueError, match="not been fitted"):
        lr_test(full, unfit)
    with pytest.raises(ValueError, match="not been fitted"):
        lr_test(unfit, full)


def test_lr_test_rejects_mismatched_sample_sizes(sim_two_sided):
    full = CensoredRegression(left=0.0, right=2.5).fit(sim_two_sided.X, sim_two_sided.y)
    smaller = CensoredRegression(left=0.0, right=2.5).fit(
        sim_two_sided.X[:-100], sim_two_sided.y[:-100]
    )
    with pytest.raises(ValueError, match="Sample sizes differ"):
        lr_test(full, smaller)


def test_lr_test_rejects_inverted_nesting(sim_two_sided):
    """Passing the restricted model as 'full' should error out."""
    full = CensoredRegression(left=0.0, right=2.5).fit(sim_two_sided.X, sim_two_sided.y)
    restricted = CensoredRegression(left=0.0, right=2.5).fit(
        sim_two_sided.X[:, :1], sim_two_sided.y
    )
    with pytest.raises(ValueError, match="strictly more parameters"):
        lr_test(restricted, full)
