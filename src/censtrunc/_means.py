"""Conditional means, region probabilities, and their derivatives.

These helpers are shared by the prediction methods (`CensoredRegression.predict`,
`TruncatedRegression.predict`) and by the marginal-effects machinery
(`effects.get_margeff`). Centralising the formulae here guarantees that a
prediction and the marginal effect of that same prediction stay mutually
consistent.

Notation (per observation):
    mu     = X'beta
    a_L    = (L - mu) / sigma
    a_R    = (R - mu) / sigma
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Valid prediction "kinds" per model family.
CENSORED_KINDS = ("latent", "censored", "truncated")
TRUNCATED_KINDS = ("latent", "truncated")


def _alphas(
    mu: np.ndarray,
    sigma: float,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
):
    """Return (a_L, a_R, Phi_L, Phi_R, phi_L, phi_R) with sensible no-bound defaults."""
    a_L = (left - mu) / sigma if has_left else None
    a_R = (right - mu) / sigma if has_right else None
    Phi_L = norm.cdf(a_L) if a_L is not None else np.zeros_like(mu)
    Phi_R = norm.cdf(a_R) if a_R is not None else np.ones_like(mu)
    phi_L = norm.pdf(a_L) if a_L is not None else np.zeros_like(mu)
    phi_R = norm.pdf(a_R) if a_R is not None else np.zeros_like(mu)
    return a_L, a_R, Phi_L, Phi_R, phi_L, phi_R


def conditional_mean(
    beta: np.ndarray,
    sigma: float,
    X: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    kind: str,
) -> np.ndarray:
    """Conditional mean of the chosen kind at each row of ``X``.

    - ``latent``    : E[Y* | X] = X'beta
    - ``censored``  : E[Y  | X]
    - ``truncated`` : E[Y  | X, L < Y < R]
    """
    mu = X @ beta
    if kind == "latent":
        return mu

    _, _, Phi_L, Phi_R, phi_L, phi_R = _alphas(mu, sigma, left, right, has_left, has_right)

    if kind == "censored":
        interior = mu * (Phi_R - Phi_L) + sigma * (phi_L - phi_R)
        left_mass = left * Phi_L if has_left else 0.0
        right_mass = right * (1.0 - Phi_R) if has_right else 0.0
        return left_mass + interior + right_mass

    if kind == "truncated":
        denom = Phi_R - Phi_L
        denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        return mu + sigma * (phi_L - phi_R) / denom_safe

    raise ValueError(f"Unknown kind: {kind!r}")


def region_probabilities(
    beta: np.ndarray,
    sigma: float,
    X: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
) -> dict[str, np.ndarray]:
    """Probabilities of the left / interior / right regions for each row."""
    mu = X @ beta
    _, _, Phi_L, Phi_R, _, _ = _alphas(mu, sigma, left, right, has_left, has_right)
    return {
        "left": Phi_L if has_left else np.zeros_like(mu),
        "interior": Phi_R - Phi_L,
        "right": (1.0 - Phi_R) if has_right else np.zeros_like(mu),
    }


def dmean_dx(
    beta: np.ndarray,
    sigma: float,
    X: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    kind: str,
    slope_beta: np.ndarray,
) -> np.ndarray:
    """Derivative of the conditional mean w.r.t. each slope regressor.

    Returns an ``(n_rows, n_slopes)`` array. ``slope_beta`` are the coefficients
    of the regressors whose marginal effects we want (excluding the intercept).

    - latent    : d/dx_j E[Y*|X]              = beta_j
    - censored  : d/dx_j E[Y|X]               = beta_j * [Phi(a_R) - Phi(a_L)]
    - truncated : d/dx_j E[Y|X, L<Y<R]        = beta_j * delta(X), where
        delta = 1 - (a_R phi_R - a_L phi_L)/D - ((phi_L - phi_R)/D)^2,  D = Phi_R - Phi_L
    """
    mu = X @ beta
    n = mu.shape[0]
    k = slope_beta.shape[0]

    if kind == "latent":
        return np.broadcast_to(slope_beta, (n, k)).copy()

    a_L, a_R, Phi_L, Phi_R, phi_L, phi_R = _alphas(
        mu, sigma, left, right, has_left, has_right
    )

    if kind == "censored":
        scale = Phi_R - Phi_L  # (n,)
        return np.outer(scale, slope_beta)

    if kind == "truncated":
        denom = Phi_R - Phi_L
        denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        term_alpha_phi = (
            (a_R * phi_R if a_R is not None else 0.0)
            - (a_L * phi_L if a_L is not None else 0.0)
        )
        mills_sq = ((phi_L - phi_R) / denom_safe) ** 2
        delta = 1.0 - term_alpha_phi / denom_safe - mills_sq  # (n,)
        return np.outer(delta, slope_beta)

    raise ValueError(f"Unknown kind: {kind!r}")
