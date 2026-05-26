"""Shared fixtures for the test suite.

Provides simulated datasets with known parameters, plus a helper that performs
the censoring step. Tests across the suite reuse these fixtures so that bias
checks have a common baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class SimData:
    """Container for simulated data."""

    X: np.ndarray
    y: np.ndarray
    y_star: np.ndarray  # latent values (uncensored)
    beta_true: np.ndarray
    sigma_true: float
    left: float | None
    right: float | None


def _simulate(
    *,
    n: int,
    beta: np.ndarray,
    sigma: float,
    left: float | None,
    right: float | None,
    seed: int,
) -> SimData:
    rng = np.random.default_rng(seed)
    k = beta.shape[0] - 1
    X = rng.normal(size=(n, k))
    y_star = beta[0] + X @ beta[1:] + rng.normal(scale=sigma, size=n)
    y = y_star.copy()
    if left is not None:
        y = np.where(y < left, left, y)
    if right is not None:
        y = np.where(y > right, right, y)
    return SimData(
        X=X, y=y, y_star=y_star, beta_true=beta, sigma_true=sigma, left=left, right=right
    )


@pytest.fixture
def sim_two_sided() -> SimData:
    """Two-sided censoring with about 20% / 10% left/right censoring."""
    return _simulate(
        n=3000,
        beta=np.array([1.0, 0.5, -0.3]),
        sigma=1.0,
        left=0.0,
        right=2.5,
        seed=42,
    )


@pytest.fixture
def sim_left_only() -> SimData:
    """Classical Tobit: only left censoring at zero."""
    return _simulate(
        n=3000,
        beta=np.array([1.0, 0.5, -0.3]),
        sigma=1.0,
        left=0.0,
        right=None,
        seed=43,
    )


@pytest.fixture
def sim_right_only() -> SimData:
    """Right-only censoring (rare in practice but tests symmetry)."""
    return _simulate(
        n=3000,
        beta=np.array([1.0, 0.5, -0.3]),
        sigma=1.0,
        left=None,
        right=2.5,
        seed=44,
    )


@pytest.fixture
def sim_truncated() -> SimData:
    """Truncated dataset: latent values outside (L, R) are dropped entirely."""
    full = _simulate(
        n=8000,
        beta=np.array([1.0, 0.5, -0.3]),
        sigma=1.0,
        left=0.0,
        right=2.5,
        seed=45,
    )
    mask = (full.y_star > 0.0) & (full.y_star < 2.5)
    return SimData(
        X=full.X[mask],
        y=full.y_star[mask],  # observed = latent because we kept only the interior
        y_star=full.y_star[mask],
        beta_true=full.beta_true,
        sigma_true=full.sigma_true,
        left=0.0,
        right=2.5,
    )
