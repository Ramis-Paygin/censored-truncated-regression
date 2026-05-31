"""Tests for the Heckman sample-selection regression estimator.

We simulate from the full Heckman model — jointly normal errors with a known
correlation ``rho`` and a known exclusion restriction (a regressor that enters
the selection equation but not the outcome equation) — then verify that:

- the two-step and joint-MLE estimators both recover the true ``(beta, gamma,
  rho, sigma)`` within a few standard errors;
- OLS on the selected subsample exhibits clear selection bias that Heckit
  corrects;
- prediction methods return arrays of the right shape for all three ``kind``s.
"""

from __future__ import annotations

import numpy as np
import pytest

statsmodels = pytest.importorskip("statsmodels.api")
import statsmodels.api as sm  # noqa: E402

from censtrunc import HeckitRegression  # noqa: E402


@pytest.fixture
def heckit_data():
    rng = np.random.default_rng(42)
    n = 4000
    shared = rng.normal(size=n)
    x_only = rng.normal(size=n)
    z_only = rng.normal(size=n)
    beta_true = np.array([1.0, 0.5, -0.3])  # const, shared, x_only
    gamma_true = np.array([0.0, 0.3, 0.6])  # const, shared, z_only (exclusion)
    rho_true, sigma_true = 0.6, 1.0
    cov = np.array(
        [[sigma_true**2, rho_true * sigma_true], [rho_true * sigma_true, 1.0]]
    )
    errs = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    e, u = errs[:, 0], errs[:, 1]
    X = np.column_stack([shared, x_only])
    Z = np.column_stack([shared, z_only])
    S = (gamma_true[0] + Z @ gamma_true[1:] + u > 0).astype(int)
    y_full = beta_true[0] + X @ beta_true[1:] + e
    y = np.where(S == 1, y_full, np.nan)
    return {
        "y": y, "X": X, "Z": Z, "S": S,
        "beta_true": beta_true, "gamma_true": gamma_true,
        "rho_true": rho_true, "sigma_true": sigma_true,
    }


def test_twostep_recovery(heckit_data):
    m = HeckitRegression(method="twostep").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    np.testing.assert_allclose(m.coef_, heckit_data["beta_true"], atol=0.1)
    np.testing.assert_allclose(m.gamma_, heckit_data["gamma_true"], atol=0.1)
    assert abs(m.sigma_ - heckit_data["sigma_true"]) < 0.1
    assert abs(m.rho_ - heckit_data["rho_true"]) < 0.15
    assert m.converged_


def test_mle_recovery(heckit_data):
    m = HeckitRegression(method="mle").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    np.testing.assert_allclose(m.coef_, heckit_data["beta_true"], atol=0.1)
    np.testing.assert_allclose(m.gamma_, heckit_data["gamma_true"], atol=0.1)
    assert abs(m.sigma_ - heckit_data["sigma_true"]) < 0.1
    assert abs(m.rho_ - heckit_data["rho_true"]) < 0.15
    assert np.isfinite(m.llf_)
    assert m.converged_


def test_heckit_corrects_selection_bias(heckit_data):
    """OLS on the selected subsample is biased; Heckit recovers the truth."""
    y, X = heckit_data["y"], heckit_data["X"]
    sel = ~np.isnan(y)
    X_full = sm.add_constant(X)
    ols_beta = np.asarray(sm.OLS(y[sel], X_full[sel]).fit().params)
    truth = heckit_data["beta_true"]
    heck = HeckitRegression(method="mle").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    # OLS slope on the shared regressor should be biased; Heckit should be
    # closer to the truth in MSE.
    mse_ols = np.mean((ols_beta - truth) ** 2)
    mse_heck = np.mean((heck.coef_ - truth) ** 2)
    assert mse_heck < mse_ols, (
        f"Heckit MSE {mse_heck:.4f} should be smaller than OLS MSE {mse_ols:.4f}"
    )


def test_twostep_and_mle_agree(heckit_data):
    """The two estimators target the same MLE; their point estimates should
    not disagree by more than a few SEs on a large sample."""
    ts = HeckitRegression(method="twostep").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    ml = HeckitRegression(method="mle").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    diff = np.max(np.abs(ts.coef_ - ml.coef_))
    assert diff < 0.15, f"|two-step - MLE|_max = {diff:.3f}, expected < 0.15"


def test_predict_kinds(heckit_data):
    m = HeckitRegression(method="twostep").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    n = 10
    X_new = heckit_data["X"][:n]
    Z_new = heckit_data["Z"][:n]
    out_uncond = m.predict(X=X_new, kind="outcome")
    out_prob = m.predict(Z=Z_new, kind="selection_prob")
    out_cond = m.predict(X=X_new, Z=Z_new, kind="conditional")
    assert out_uncond.shape == (n,)
    assert out_prob.shape == (n,)
    assert out_cond.shape == (n,)
    assert np.all((0 <= out_prob) & (out_prob <= 1))


def test_predict_invalid_kind_raises(heckit_data):
    m = HeckitRegression(method="twostep").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    with pytest.raises(ValueError, match="Unknown kind"):
        m.predict(X=heckit_data["X"][:5], kind="nonsense")


def test_unfitted_predict_raises():
    m = HeckitRegression()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(X=np.zeros((2, 2)), Z=np.zeros((2, 2)))


def test_summary_renders_two_blocks(heckit_data):
    m = HeckitRegression(method="twostep").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )
    text = m.summary()
    assert "Outcome equation" in text
    assert "Selection equation" in text
    assert "rho" in text.lower()
