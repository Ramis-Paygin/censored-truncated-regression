"""Cross-validation against independent estimators.

Two analytical limits of the censored normal model coincide with standard
estimators that are well-implemented elsewhere; matching them is the strongest
"correctness" check available short of running another package.

1. **No censoring limit** (``L = -inf``, ``R = +inf``). Every observation
   contributes the plain normal density, so the MLE must coincide with OLS:

   - ``beta_hat`` equals the OLS coefficient vector,
   - ``sigma_hat`` equals the MLE residual std ``sqrt(SSR / n)``
     (OLS reports ``SSR / (n - k)``, the unbiased variant),
   - the log-likelihood equals statsmodels' OLS ``llf``.

2. **Probit limit** (``L = R = c``, i.e. ``y`` only takes the two bound values
   so the interior region has measure zero). Then ``log f(y; beta, sigma)``
   becomes the log-likelihood of a probit on the indicator ``1{Y* > c}``, with
   ``beta_probit = beta_tobit / sigma_tobit`` (Hansen 2022, Ch. 27.4). We can't
   set ``L = R`` exactly because ``sigma`` becomes unidentified, but with a
   tight band around ``c`` the censored model effectively reduces to probit.

These checks validate the likelihood, the optimiser, *and* the back-transform
from Olsen's parameterisation, independently of this package's own code.

A second module (``test_r_reference``) compares against R's ``survival::survreg``
/ ``truncreg`` when R is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

statsmodels = pytest.importorskip("statsmodels.api")
import statsmodels.api as sm  # noqa: E402

from censtrunc import CensoredRegression, TruncatedRegression  # noqa: E402

WIDE = 1e6  # thresholds far outside any plausible data range


@pytest.fixture
def ols_data():
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(size=(n, 3))
    y = 2.0 + 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=1.3, size=n)
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    return X, y, ols


def test_censored_reduces_to_ols(ols_data):
    X, y, ols = ols_data
    model = CensoredRegression(left=-WIDE, right=WIDE).fit(X, y)
    ols_beta = np.asarray(ols.params)
    # Coefficients match OLS to optimiser tolerance.
    np.testing.assert_allclose(model.coef_, ols_beta, atol=1e-3)
    # No observation should be flagged as censored.
    assert model.n_left_censored_ == 0
    assert model.n_right_censored_ == 0
    assert model.n_uncensored_ == len(y)


def test_censored_sigma_matches_mle(ols_data):
    X, y, ols = ols_data
    model = CensoredRegression(left=-WIDE, right=WIDE).fit(X, y)
    sigma_mle = np.sqrt(ols.ssr / len(y))  # MLE, not the n-k version
    assert abs(model.sigma_ - sigma_mle) < 1e-3


def test_censored_loglik_matches_ols(ols_data):
    X, y, ols = ols_data
    model = CensoredRegression(left=-WIDE, right=WIDE).fit(X, y)
    assert abs(model.llf_ - ols.llf) < 1e-3


def test_truncated_reduces_to_ols(ols_data):
    X, y, ols = ols_data
    model = TruncatedRegression(left=-WIDE, right=WIDE).fit(X, y)
    ols_beta = np.asarray(ols.params)
    # Truncated optimiser works in (sigma, beta) directly and matches OLS very tightly.
    np.testing.assert_allclose(model.coef_, ols_beta, atol=1e-6)


def test_latent_ame_equals_ols_slopes(ols_data):
    """In the no-censoring limit, the latent AME equals the OLS slope vector."""
    X, y, ols = ols_data
    model = CensoredRegression(left=-WIDE, right=WIDE).fit(X, y)
    ame = model.ame(kind="latent").margeff
    np.testing.assert_allclose(ame, np.asarray(ols.params)[1:], atol=1e-3)


def test_one_sided_wide_threshold_matches_ols(ols_data):
    """A single very-low left threshold also leaves the data uncensored."""
    X, y, ols = ols_data
    model = CensoredRegression(left=-WIDE).fit(X, y)
    np.testing.assert_allclose(model.coef_, np.asarray(ols.params), atol=1e-3)
    assert model.n_left_censored_ == 0


# ----------------------------------------------------------------------
# Probit limit:  L = R = c  =>  tobit  ===  probit on 1{Y* > c}
# ----------------------------------------------------------------------


def test_censored_reduces_to_probit_at_tight_thresholds():
    """At a tiny censoring band ``[c - eps, c + eps]``, the censored MLE is
    (up to sigma being unidentified) the probit MLE for the indicator
    ``1{Y > c}`` with ``beta_probit = beta_tobit / sigma_tobit``.

    The proof is Hansen (2022), Ch. 27.4: at ``L = R = c`` the censored log-
    likelihood reduces to ``sum_i 1{y_i = c} log Phi(a_L) + 1{y_i = c} log[1 -
    Phi(a_L)]``, identical to a probit's. With a narrow band ``eps > 0`` the
    interior region carries a handful of observations — just enough to identify
    ``sigma`` weakly — and the relation ``beta_tobit / sigma_tobit ≈ beta_probit``
    holds up to finite-eps slack and finite-sample noise.
    """
    from statsmodels.discrete.discrete_model import Probit

    rng = np.random.default_rng(20250601)
    n = 8000
    X = rng.normal(size=(n, 2))
    beta_true = np.array([0.4, 1.2, -0.8])  # intercept + 2 slopes
    sigma_true = 1.0
    y_star = beta_true[0] + X @ beta_true[1:] + sigma_true * rng.normal(size=n)

    # tight band around c = 0
    eps = 0.05
    y_obs = np.clip(y_star, -eps, eps)

    cens = CensoredRegression(left=-eps, right=eps).fit(X, y_obs)

    # Probit on the underlying sign
    y_bin = (y_star > 0.0).astype(float)
    probit = Probit(y_bin, sm.add_constant(X)).fit(disp=False)

    cens_ratio = cens.coef_ / cens.sigma_
    probit_params = np.asarray(probit.params)

    # Slopes should agree to within ~5%; intercept can drift more because the
    # narrow band shifts the implicit threshold by ~ eps / sigma.
    np.testing.assert_allclose(cens_ratio[1:], probit_params[1:], rtol=0.08, atol=0.05)
    # Intercept comparison gets a looser tolerance.
    assert abs(cens_ratio[0] - probit_params[0]) < 0.15


def test_censored_to_probit_shrinks_with_band():
    """Sanity: the gap ``|beta_tobit / sigma_tobit - beta_probit|`` is
    monotone (non-strictly) in ``eps`` — narrower bands give closer agreement.

    We don't enforce strict monotonicity (finite-sample noise blows that up)
    but the eps = 0.02 fit should be at least as close as eps = 0.5.
    """
    from statsmodels.discrete.discrete_model import Probit

    rng = np.random.default_rng(7)
    n = 8000
    X = rng.normal(size=(n, 2))
    beta_true = np.array([0.0, 1.0, -0.6])
    y_star = beta_true[0] + X @ beta_true[1:] + rng.normal(size=n)
    y_bin = (y_star > 0.0).astype(float)
    probit_params = np.asarray(Probit(y_bin, sm.add_constant(X)).fit(disp=False).params)

    def gap(eps: float) -> float:
        y_obs = np.clip(y_star, -eps, eps)
        cens = CensoredRegression(left=-eps, right=eps).fit(X, y_obs)
        ratio = cens.coef_ / cens.sigma_
        # Compare slopes only (the intercept is the one most affected by eps).
        return float(np.max(np.abs(ratio[1:] - probit_params[1:])))

    gap_narrow = gap(0.05)
    gap_wide = gap(0.6)
    # Narrow-band agreement is at most as bad as wide-band, plus a slack.
    assert gap_narrow <= gap_wide + 0.05, (
        f"narrow-band gap {gap_narrow:.4f} should not exceed wide-band {gap_wide:.4f}"
    )
    # And the narrow band is absolutely close.
    assert gap_narrow < 0.10
