"""Tests for the truncated regression estimator."""

from __future__ import annotations

import numpy as np
import pytest

from censtrunc import TruncatedRegression


def test_truncated_recovery(sim_truncated):
    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    truth = sim_truncated.beta_true
    assert np.all(np.abs(model.coef_ - truth) < 4 * model.bse_[1:])
    assert abs(model.sigma_ - sim_truncated.sigma_true) < 0.15
    assert model.diagnostics_.converged


def test_truncated_rejects_out_of_range_y():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    # We deliberately leave some y outside (0, 2); should fail validation.
    with pytest.raises(ValueError, match="y > left|y < right"):
        TruncatedRegression(left=0.0, right=2.0).fit(X, y)


def test_truncated_overall_lr(sim_truncated):
    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    assert model.llr_ > 0
    assert model.llr_pvalue_ < 1e-6


def test_truncated_predict_inside_interval(sim_truncated):
    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    X_new = sim_truncated.X[:50]
    y_lat = model.predict(X_new, kind="latent")
    y_trunc = model.predict(X_new, kind="truncated")
    assert y_lat.shape == (50,)
    # truncated mean must lie strictly inside the interval
    assert np.all((y_trunc > sim_truncated.left) & (y_trunc < sim_truncated.right))


def test_truncated_predict_default_two_columns(sim_truncated):
    import pandas as pd

    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    out = model.predict(sim_truncated.X[:10])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["latent", "truncated"]


def test_truncated_predict_rejects_prob_letters(sim_truncated):
    """Probability kinds make no sense for a truncated model -> ValueError."""
    import pytest

    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    for bad in ("l", "lmr", "ht l".replace(" ", ""), "c"):
        with pytest.raises(ValueError, match="not available"):
            model.predict(sim_truncated.X[:5], kind=bad)
