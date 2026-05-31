"""Tests for the censored normal regression estimator.

Strategy:
- Smoke tests: API shape and presence of fitted attributes.
- Recovery tests: estimates should lie within ~3 standard errors of the true
  parameters on large simulated samples. We use a fixed seed and assert
  reasonable tolerances rather than exact values.
- Compare classical (left-only) Tobit against ``statsmodels`` OLS on the
  uncensored subsample to confirm that the censoring correction shrinks the
  bias OLS would suffer.
"""

from __future__ import annotations

import numpy as np
import pytest

from censtrunc import CensoredRegression


# ----------------------------------------------------------------------
# Recovery tests on simulated data
# ----------------------------------------------------------------------


def test_two_sided_recovery(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    # beta_hat[0] is the intercept (added internally), then slopes
    estimated = model.coef_
    truth = sim_two_sided.beta_true
    # Within 4 SE — generous to account for finite-sample noise
    assert np.all(np.abs(estimated - truth) < 4 * model.bse_[1:]), (
        f"recovered={estimated}, true={truth}, se={model.bse_[1:]}"
    )
    # Sigma within ~10%
    assert abs(model.sigma_ - sim_two_sided.sigma_true) < 0.1
    # Optimiser should converge
    assert model.diagnostics_.converged
    # Both censoring regions populated
    assert model.n_left_censored_ > 50
    assert model.n_right_censored_ > 50
    assert model.n_uncensored_ > 1000


def test_left_only_recovery(sim_left_only):
    model = CensoredRegression(left=sim_left_only.left).fit(sim_left_only.X, sim_left_only.y)
    truth = sim_left_only.beta_true
    assert np.all(np.abs(model.coef_ - truth) < 4 * model.bse_[1:])
    assert abs(model.sigma_ - sim_left_only.sigma_true) < 0.1
    assert model.n_right_censored_ == 0
    assert model.n_left_censored_ > 100


def test_right_only_recovery(sim_right_only):
    model = CensoredRegression(right=sim_right_only.right).fit(
        sim_right_only.X, sim_right_only.y
    )
    truth = sim_right_only.beta_true
    assert np.all(np.abs(model.coef_ - truth) < 4 * model.bse_[1:])
    assert model.n_left_censored_ == 0
    assert model.n_right_censored_ > 50


# ----------------------------------------------------------------------
# Correctness checks vs. OLS bias
# ----------------------------------------------------------------------


def test_tobit_beats_ols_in_presence_of_censoring(sim_left_only):
    """OLS on the censored sample is biased toward zero (Greene 1981 result).

    Tobit should be closer to the true beta than OLS."""
    X = sim_left_only.X
    y = sim_left_only.y
    truth = sim_left_only.beta_true

    # OLS with intercept
    X_with_int = np.column_stack([np.ones(X.shape[0]), X])
    ols_beta, *_ = np.linalg.lstsq(X_with_int, y, rcond=None)

    model = CensoredRegression(left=sim_left_only.left).fit(X, y)
    tobit_beta = model.params_[1:]  # skip sigma

    # Tobit should be closer in MSE to truth than OLS
    mse_ols = np.mean((ols_beta - truth) ** 2)
    mse_tobit = np.mean((tobit_beta - truth) ** 2)
    assert mse_tobit < mse_ols, f"Tobit MSE {mse_tobit:g} should beat OLS MSE {mse_ols:g}"


# ----------------------------------------------------------------------
# Likelihood-ratio and pseudo-R^2 sanity checks
# ----------------------------------------------------------------------


def test_overall_lr_is_significant(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    assert model.llr_ > 0
    assert 0 <= model.llr_pvalue_ <= 1
    assert model.llr_pvalue_ < 1e-6  # truth has non-zero slopes
    assert 0 < model.prsquared_ < 1


# ----------------------------------------------------------------------
# Predictions and predict_proba
# ----------------------------------------------------------------------


def test_prediction_shapes_and_bounds(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    X_new = sim_two_sided.X[:50]
    y_latent = model.predict(X_new, kind="latent")
    y_censored = model.predict(X_new, kind="censored")
    y_truncated = model.predict(X_new, kind="truncated")
    assert y_latent.shape == (50,)
    assert y_censored.shape == (50,)
    assert y_truncated.shape == (50,)
    # E[Y|X] should lie within [L, R] (it's a censored mean)
    assert np.all(y_censored >= sim_two_sided.left - 1e-9)
    assert np.all(y_censored <= sim_two_sided.right + 1e-9)
    # Truncated mean should be strictly inside (L, R) for plausible X
    assert np.all((y_truncated > sim_two_sided.left) & (y_truncated < sim_two_sided.right))


def test_predict_proba_sums_to_one(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    proba = model.predict_proba(sim_two_sided.X[:100])
    total = proba["left"] + proba["interior"] + proba["right"]
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_predict_unfitted_raises(sim_two_sided):
    model = CensoredRegression(left=0.0, right=2.5)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(sim_two_sided.X)


# ----------------------------------------------------------------------
# Letter-string predict() API
# ----------------------------------------------------------------------


def test_predict_default_returns_all_six_columns(sim_two_sided):
    import pandas as pd

    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    out = model.predict(sim_two_sided.X[:10])  # no kind argument
    assert isinstance(out, pd.DataFrame)
    expected = ["latent", "censored", "truncated", "prob_left", "prob_interior", "prob_right"]
    assert list(out.columns) == expected
    assert out.shape == (10, 6)


def test_predict_hctlmr_equals_default(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    a = model.predict(sim_two_sided.X[:5])
    b = model.predict(sim_two_sided.X[:5], kind="hctlmr")
    import pandas as pd
    pd.testing.assert_frame_equal(a, b)


def test_predict_lmr_three_probabilities_sum_to_one(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    out = model.predict(sim_two_sided.X[:50], kind="lmr")
    assert list(out.columns) == ["prob_left", "prob_interior", "prob_right"]
    total = out["prob_left"] + out["prob_interior"] + out["prob_right"]
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-12)


def test_predict_single_letter_returns_1d_ndarray(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    out = model.predict(sim_two_sided.X[:5], kind="h")
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1


def test_predict_h_matches_latent_backward_compat(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    np.testing.assert_allclose(
        model.predict(sim_two_sided.X[:20], kind="h"),
        model.predict(sim_two_sided.X[:20], kind="latent"),
    )


def test_predict_letter_order_preserved(sim_two_sided):
    """The output columns should appear in the order the user requested."""
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    out = model.predict(sim_two_sided.X[:5], kind="rml")
    assert list(out.columns) == ["prob_right", "prob_interior", "prob_left"]


def test_predict_invalid_letter_raises(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    with pytest.raises(ValueError, match="Unknown kind letter"):
        model.predict(sim_two_sided.X[:5], kind="hxyz")


def test_predict_duplicate_letters_deduplicated(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    out = model.predict(sim_two_sided.X[:5], kind="hhhh")
    # one letter de-duplicates to a single column -> 1-D ndarray
    assert isinstance(out, np.ndarray) and out.ndim == 1


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


def test_requires_at_least_one_threshold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="no censoring"):
        CensoredRegression().fit(X, y)


def test_invalid_thresholds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="strictly less than"):
        CensoredRegression(left=1.0, right=0.5).fit(X, y)


def test_all_observations_censored():
    """Pathological case: every y is at the threshold."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = np.zeros(50)  # all at left threshold
    with pytest.raises(ValueError, match="undefined"):
        CensoredRegression(left=0.0).fit(X, y)


def test_nan_input_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    X[0, 0] = np.nan
    y = rng.normal(size=50)
    with pytest.raises(ValueError, match="NaN"):
        CensoredRegression(left=0.0).fit(X, y)
