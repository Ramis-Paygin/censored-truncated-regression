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


# ----------------------------------------------------------------------
# Letter-string predict: six quantities via one-letter codes
# ----------------------------------------------------------------------


@pytest.fixture
def fitted_for_predict(heckit_data):
    return HeckitRegression(method="mle").fit(
        heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    )


def test_predict_default_returns_all_six(fitted_for_predict, heckit_data):
    pd = pytest.importorskip("pandas")
    m = fitted_for_predict
    n = 6
    df = m.predict(X=heckit_data["X"][:n], Z=heckit_data["Z"][:n])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "prob_selected", "prob_not_selected", "propensity",
        "observed", "hidden", "unobserved",
    ]
    assert df.shape == (n, 6)


def test_predict_single_letter_returns_1d(fitted_for_predict, heckit_data):
    m = fitted_for_predict
    for letter in "snpohu":
        kw = {}
        if letter in "ohu":
            kw["X"] = heckit_data["X"][:5]
        if letter in "snpou":
            kw["Z"] = heckit_data["Z"][:5]
        arr = m.predict(kind=letter, **kw)
        assert isinstance(arr, np.ndarray) and arr.shape == (5,), letter


def test_predict_subset_returns_dataframe(fitted_for_predict, heckit_data):
    pd = pytest.importorskip("pandas")
    m = fitted_for_predict
    df = m.predict(Z=heckit_data["Z"][:4], kind="sn")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["prob_selected", "prob_not_selected"]
    # the two probabilities must sum to 1
    np.testing.assert_allclose(df.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_predict_legacy_names_back_compat(fitted_for_predict, heckit_data):
    m = fitted_for_predict
    X = heckit_data["X"][:5]
    Z = heckit_data["Z"][:5]
    # Legacy names should equal the corresponding letter-kind 1-D arrays.
    np.testing.assert_allclose(
        m.predict(X=X, kind="outcome"), m.predict(X=X, kind="h"), atol=1e-12,
    )
    np.testing.assert_allclose(
        m.predict(Z=Z, kind="selection_prob"), m.predict(Z=Z, kind="s"), atol=1e-12,
    )
    np.testing.assert_allclose(
        m.predict(X=X, Z=Z, kind="conditional"), m.predict(X=X, Z=Z, kind="o"),
        atol=1e-12,
    )


def test_predict_formulas_against_direct_calc(fitted_for_predict, heckit_data):
    """Verify each letter formula matches a direct computation."""
    from scipy.stats import norm as _norm

    m = fitted_for_predict
    X = heckit_data["X"][:8]
    Z = heckit_data["Z"][:8]
    Xd = np.column_stack([np.ones(len(X)), X])
    Zd = np.column_stack([np.ones(len(Z)), Z])
    Xb = Xd @ m.coef_
    Zg = Zd @ m.gamma_
    Phi = _norm.cdf(Zg)
    phi = _norm.pdf(Zg)

    np.testing.assert_allclose(m.predict(Z=Z, kind="s"), Phi, atol=1e-12)
    np.testing.assert_allclose(m.predict(Z=Z, kind="n"), 1 - Phi, atol=1e-12)
    np.testing.assert_allclose(m.predict(Z=Z, kind="p"), Zg, atol=1e-12)
    np.testing.assert_allclose(m.predict(X=X, kind="h"), Xb, atol=1e-12)
    # observed conditional: X'b + rho*sigma * phi/Phi
    np.testing.assert_allclose(
        m.predict(X=X, Z=Z, kind="o"), Xb + m.sigma_eu_ * phi / Phi, atol=1e-12,
    )
    # unobserved conditional: X'b - rho*sigma * phi/(1 - Phi)
    np.testing.assert_allclose(
        m.predict(X=X, Z=Z, kind="u"), Xb - m.sigma_eu_ * phi / (1 - Phi), atol=1e-9,
    )


def test_predict_requires_X_for_outcome_letters(fitted_for_predict, heckit_data):
    m = fitted_for_predict
    Z = heckit_data["Z"][:5]
    # Asking for any of o/h/u without X must fail.
    with pytest.raises(ValueError, match="requires X"):
        m.predict(Z=Z, kind="h")
    with pytest.raises(ValueError, match="requires X"):
        m.predict(Z=Z, kind="ohu")


def test_predict_requires_Z_for_selection_letters(fitted_for_predict, heckit_data):
    m = fitted_for_predict
    X = heckit_data["X"][:5]
    with pytest.raises(ValueError, match="requires Z"):
        m.predict(X=X, kind="s")
    with pytest.raises(ValueError, match="requires Z"):
        m.predict(X=X, kind="ou")  # o and u both touch Z


def test_predict_unknown_letter_raises(fitted_for_predict, heckit_data):
    m = fitted_for_predict
    with pytest.raises(ValueError, match="Unknown kind letter"):
        m.predict(X=heckit_data["X"][:3], Z=heckit_data["Z"][:3], kind="xyz")


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


# ----------------------------------------------------------------------
# Cross-validation against `py4etrics.Heckit`.
#
# py4etrics is the Heckit reference recommended during review (`py4etrics`,
# Hasebe et al., https://py4etrics-github-io.translate.goog/ ). It only
# implements the two-step estimator -- the `method='mle'` kwarg is a no-op
# silently dispatched to two-step. We compare *both* the outcome equation
# (beta) and the selection equation (gamma) on the same simulated data; the
# numbers must match to many decimals since both packages solve the same
# closed-form Heckman expressions.
# ----------------------------------------------------------------------

py4etrics = pytest.importorskip("py4etrics.heckit")


def test_twostep_matches_py4etrics(heckit_data):
    """censtrunc.HeckitRegression(method='twostep') should reproduce
    py4etrics.Heckit to machine precision on the outcome coefficients,
    selection coefficients, sigma and rho.
    """
    import warnings

    y, X, Z = heckit_data["y"], heckit_data["X"], heckit_data["Z"]
    # py4etrics expects design matrices that include the intercept column.
    X_with_const = sm.add_constant(X)
    Z_with_const = sm.add_constant(Z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = py4etrics.Heckit(y, X_with_const, Z_with_const).fit(method="twostep")

    m = HeckitRegression(method="twostep", fit_intercept=True).fit(y, X, Z)

    # outcome beta -- py4etrics stores them in `.params`
    np.testing.assert_allclose(m.coef_, ref.params, atol=1e-6)
    # selection gamma -- py4etrics stores them on the probit results object
    np.testing.assert_allclose(m.gamma_, ref.select_res.params, atol=1e-6)
    # variance and correlation of residuals (py4etrics' attribute names differ)
    assert abs(m.sigma_ - np.sqrt(ref.var_reg_error)) < 1e-6
    assert abs(m.rho_ - ref.corr_eqnerrors) < 1e-6
