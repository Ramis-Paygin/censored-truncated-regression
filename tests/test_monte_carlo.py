"""Monte Carlo consistency check.

We simulate many independent samples under a known data-generating process,
fit the model on each, and verify that the average of the estimates is close
to the true parameter values (consistency / approximate unbiasedness in large
samples). This is the empirical analogue of the asymptotic property that
``E[beta_hat] -> beta`` as ``n -> infinity``.

The test is marked ``slow``; pytest still runs it by default but users can
exclude it with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from censtrunc import CensoredRegression, TruncatedRegression


@pytest.mark.slow
def test_censored_monte_carlo_unbiased():
    n_reps = 60
    n = 2000
    beta_true = np.array([1.0, 0.5, -0.3])
    sigma_true = 1.0
    L, R = 0.0, 2.5

    estimates = np.zeros((n_reps, len(beta_true)))
    sigma_hats = np.zeros(n_reps)

    base_rng = np.random.default_rng(2024)
    for r in range(n_reps):
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        X = rng.normal(size=(n, 2))
        y_star = beta_true[0] + X @ beta_true[1:] + rng.normal(scale=sigma_true, size=n)
        y = np.clip(y_star, L, R)
        model = CensoredRegression(left=L, right=R).fit(X, y)
        estimates[r] = model.coef_
        sigma_hats[r] = model.sigma_

    mean_estimates = estimates.mean(axis=0)
    # Each coefficient should be within a small bias band of truth.
    # Monte Carlo SE on the mean is roughly SE/sqrt(n_reps); allow 3x that as
    # a conservative tolerance.
    se_per_rep = estimates.std(axis=0, ddof=1)
    se_mean = se_per_rep / np.sqrt(n_reps)
    deviation = np.abs(mean_estimates - beta_true)
    assert np.all(deviation < 3 * se_mean + 0.02), (
        f"Bias too large: mean={mean_estimates}, true={beta_true}, "
        f"3*SE={3 * se_mean}, deviation={deviation}"
    )
    # Sigma should also be close
    assert abs(sigma_hats.mean() - sigma_true) < 0.02


@pytest.mark.slow
def test_truncated_monte_carlo_unbiased():
    n_reps = 40
    n_target = 1500  # observations after truncation
    beta_true = np.array([1.0, 0.5, -0.3])
    sigma_true = 1.0
    L, R = 0.0, 2.5

    estimates = []
    base_rng = np.random.default_rng(2025)
    for _ in range(n_reps):
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        # oversample to ensure at least n_target observations after truncation
        n_raw = int(n_target / 0.6)
        X = rng.normal(size=(n_raw, 2))
        y_star = beta_true[0] + X @ beta_true[1:] + rng.normal(scale=sigma_true, size=n_raw)
        mask = (y_star > L) & (y_star < R)
        X_t = X[mask][:n_target]
        y_t = y_star[mask][:n_target]
        model = TruncatedRegression(left=L, right=R).fit(X_t, y_t)
        estimates.append(model.coef_)
    estimates = np.asarray(estimates)
    mean_est = estimates.mean(axis=0)
    se_mean = estimates.std(axis=0, ddof=1) / np.sqrt(n_reps)
    deviation = np.abs(mean_est - beta_true)
    assert np.all(deviation < 3 * se_mean + 0.05)
