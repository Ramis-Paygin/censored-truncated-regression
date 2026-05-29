"""Cross-validation against independent estimators.

The most important sanity check (explicitly requested during review): with the
censoring/truncation thresholds pushed beyond the data range, *no* observation is
censored or truncated, so the model must reduce to an ordinary normal regression.
In that limit the maximum-likelihood estimates must coincide with OLS:

- ``beta_hat`` equals the OLS coefficient vector;
- ``sigma_hat`` equals the MLE residual standard deviation ``sqrt(SSR / n)``
  (note: OLS reports ``SSR / (n - k)``, the unbiased variant);
- the log-likelihood equals statsmodels' OLS log-likelihood.

These checks confirm the likelihood, the optimiser, and the back-transform from
Olsen's parameterisation are all wired up correctly, independently of this
package's own code.

A second module (``test_r_reference``) compares against R's ``AER::tobit`` /
``truncreg`` when R is available.
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
