"""Low-level checks on the log-likelihood and gradient.

The most rigorous sanity check we can perform on a custom MLE implementation
is to verify that the analytical gradient matches the numerical (finite-difference)
gradient on simulated data. This catches sign errors, missing terms, and other
subtle bugs in the closed-form derivative expressions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import approx_fprime

from censtrunc._likelihood import (
    neg_loglik_censored_olsen,
    neg_loglik_grad_censored_olsen,
)
from censtrunc._utils import _classify_observations


def _make_data(*, n=400, left=0.0, right=2.5, seed=11):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    beta = np.array([1.0, 0.5, -0.3])
    sigma = 1.0
    y_star = X @ beta + rng.normal(scale=sigma, size=n)
    y = np.clip(y_star, left, right)
    return X, y, beta, sigma


def test_olsen_gradient_matches_numerical():
    X, y, beta, sigma = _make_data()
    left, right = 0.0, 2.5
    mL, mR, mF = _classify_observations(y, left, right, True, True)

    # Evaluate at a point near (but not exactly at) the truth
    nu = 1.0 / sigma * 1.05
    gamma = beta / sigma + np.array([0.1, -0.05, 0.07])
    params = np.concatenate([[nu], gamma])

    args = (X, y, left, right, mL, mR, mF)

    analytic = neg_loglik_grad_censored_olsen(params, *args)
    numeric = approx_fprime(
        params,
        lambda p: neg_loglik_censored_olsen(p, *args),
        epsilon=1e-6,
    )
    # Tolerance reflects finite-difference accuracy
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-3)


def test_olsen_gradient_at_mle_is_zero():
    """At the MLE the gradient should be ~0 (KKT for the unconstrained problem)."""
    from censtrunc import CensoredRegression

    X_raw, y, _, _ = _make_data()
    # drop intercept column from X_raw because CensoredRegression adds one
    X = X_raw[:, 1:]
    model = CensoredRegression(left=0.0, right=2.5).fit(X, y)
    # Reconstruct the gradient at MLE in Olsen coordinates
    mL, mR, mF = _classify_observations(y, 0.0, 2.5, True, True)
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    nu = 1.0 / model.sigma_
    gamma = model.coef_ / model.sigma_
    params = np.concatenate([[nu], gamma])
    grad = neg_loglik_grad_censored_olsen(
        params, X_design, y, 0.0, 2.5, mL, mR, mF
    )
    # Norm of gradient should be small (the optimiser stops when it falls below `tol`)
    assert np.linalg.norm(grad) < 1e-3
