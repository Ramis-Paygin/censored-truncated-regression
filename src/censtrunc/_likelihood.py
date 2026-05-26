"""Log-likelihood functions and gradients for censored and truncated normal regression.

Both standard ``(beta, sigma)`` and Olsen's reparameterised ``(gamma, nu)`` formulations
are provided. Olsen's reparameterisation is preferred for optimisation because the
censored-regression log-likelihood is globally concave in ``(gamma, nu)`` (Olsen, 1978).

References
----------
- Olsen, R. J. (1978). "Note on the Uniqueness of the Maximum Likelihood Estimator
  for the Tobit Model". Econometrica, 46(5), 1211-1215.
- Hansen, B. E. (2022). *Econometrics*. Chapter 27, "Censoring and Selection".
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr  # numerically stable log(Phi(z))
from scipy.stats import norm

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _inverse_mills(z: np.ndarray) -> np.ndarray:
    """Inverse Mills ratio: phi(z) / Phi(z), computed in a numerically stable way."""
    # exp(log(phi(z)) - log(Phi(z))) avoids underflow in the tails
    return np.exp(norm.logpdf(z) - log_ndtr(z))


def _log_phi_diff(a_lo: np.ndarray, a_hi: np.ndarray) -> np.ndarray:
    """log(Phi(a_hi) - Phi(a_lo)), stable for the cases that arise in practice.

    Used by the truncated log-likelihood. The arguments satisfy ``a_hi >= a_lo``
    (otherwise the truncation interval is degenerate and likelihood is -inf).
    """
    # Most stable form: use log_ndtr and logsumexp-like trick.
    # log(Phi(a_hi) - Phi(a_lo)) = log_ndtr(a_hi) + log(1 - exp(log_ndtr(a_lo) - log_ndtr(a_hi)))
    log_hi = log_ndtr(a_hi)
    log_lo = log_ndtr(a_lo)
    diff = log_lo - log_hi
    # If the interval is degenerate (round-off makes log_lo >= log_hi), the
    # probability is effectively zero and the corresponding log-likelihood is
    # -inf. We return a large negative number to keep the optimiser away from
    # this region without raising a runtime warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = log_hi + np.log1p(-np.exp(np.minimum(diff, -1e-30)))
    return np.where(np.isfinite(result), result, -1e30)


# ----------------------------------------------------------------------
# Censored regression: Olsen (gamma, nu) parameterisation
# ----------------------------------------------------------------------


def neg_loglik_censored_olsen(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    left: float,
    right: float,
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    mask_free: np.ndarray,
) -> float:
    """Negative log-likelihood for two-sided censored regression in Olsen parameterisation.

    ``params = [nu, gamma_1, ..., gamma_k]`` where ``nu = 1/sigma`` and ``gamma = beta/sigma``.
    """
    nu = params[0]
    gamma = params[1:]
    if nu <= 0:
        return np.inf  # outside feasible region

    Xg = X @ gamma
    ll = 0.0

    if mask_free.any():
        w = nu * y[mask_free] - Xg[mask_free]
        n_free = int(mask_free.sum())
        ll_free = n_free * np.log(nu) - 0.5 * n_free * np.log(2 * np.pi) - 0.5 * np.sum(w * w)
        ll += ll_free

    if mask_left.any():
        m_left = nu * left - Xg[mask_left]
        ll += float(np.sum(log_ndtr(m_left)))

    if mask_right.any():
        m_right = Xg[mask_right] - nu * right
        ll += float(np.sum(log_ndtr(m_right)))

    return -ll


def neg_loglik_grad_censored_olsen(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    left: float,
    right: float,
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    mask_free: np.ndarray,
) -> np.ndarray:
    """Gradient of ``neg_loglik_censored_olsen`` w.r.t. ``[nu, gamma]``.

    Closed-form expressions; see Hansen (2022), eq. (27.5) and surrounding derivation.
    """
    nu = params[0]
    gamma = params[1:]
    n_params = params.shape[0]
    grad = np.zeros(n_params)

    Xg = X @ gamma

    if mask_free.any():
        Xf = X[mask_free]
        yf = y[mask_free]
        w = nu * yf - Xg[mask_free]
        n_free = int(mask_free.sum())
        grad[0] += n_free / nu - np.sum(w * yf)  # d/d nu
        grad[1:] += Xf.T @ w  # d/d gamma

    if mask_left.any():
        Xl = X[mask_left]
        m_left = nu * left - Xg[mask_left]
        lam = _inverse_mills(m_left)
        grad[0] += np.sum(lam) * left
        grad[1:] -= Xl.T @ lam

    if mask_right.any():
        Xr = X[mask_right]
        m_right = Xg[mask_right] - nu * right
        lam = _inverse_mills(m_right)
        grad[0] -= np.sum(lam) * right
        grad[1:] += Xr.T @ lam

    return -grad


# ----------------------------------------------------------------------
# Censored regression: standard (beta, sigma) parameterisation
# ----------------------------------------------------------------------


def neg_loglik_censored(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    left: float,
    right: float,
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    mask_free: np.ndarray,
) -> float:
    """Negative log-likelihood for two-sided censored regression in (beta, sigma) form.

    ``params = [sigma, beta_1, ..., beta_k]``. Used for computing the Hessian (and
    hence standard errors) in the natural parameterisation after Olsen-based optimisation.
    """
    sigma = params[0]
    beta = params[1:]
    if sigma <= 0:
        return np.inf
    Xb = X @ beta
    ll = 0.0

    if mask_free.any():
        z = (y[mask_free] - Xb[mask_free]) / sigma
        n_free = int(mask_free.sum())
        ll += -n_free * np.log(sigma) - 0.5 * n_free * np.log(2 * np.pi) - 0.5 * np.sum(z * z)

    if mask_left.any():
        a_left = (left - Xb[mask_left]) / sigma
        ll += float(np.sum(log_ndtr(a_left)))

    if mask_right.any():
        a_right = (Xb[mask_right] - right) / sigma
        ll += float(np.sum(log_ndtr(a_right)))

    return -ll


# ----------------------------------------------------------------------
# Truncated regression: standard parameterisation
# ----------------------------------------------------------------------


def neg_loglik_truncated(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
) -> float:
    """Negative log-likelihood for truncated regression with arbitrary L, R.

    All ``y_i`` are assumed to satisfy ``left < y_i < right`` (truncation removes
    observations outside this interval entirely).

    Density: ``f(y | L < y* < R) = (1/sigma) phi((y - X'beta)/sigma) / [Phi(a_R) - Phi(a_L)]``
    where ``a_L = (L - X'beta)/sigma`` and ``a_R = (R - X'beta)/sigma``.
    """
    sigma = params[0]
    beta = params[1:]
    if sigma <= 0:
        return np.inf

    Xb = X @ beta
    z = (y - Xb) / sigma
    n = y.shape[0]

    ll = -n * np.log(sigma) - 0.5 * n * np.log(2 * np.pi) - 0.5 * np.sum(z * z)

    # subtract the log-probability of being in the truncation interval
    if has_left and has_right:
        a_left = (left - Xb) / sigma
        a_right = (right - Xb) / sigma
        ll -= float(np.sum(_log_phi_diff(a_left, a_right)))
    elif has_left:  # only left truncation: P(y* > L) = Phi((X'beta - L)/sigma)
        ll -= float(np.sum(log_ndtr((Xb - left) / sigma)))
    elif has_right:  # only right truncation: P(y* < R) = Phi((R - X'beta)/sigma)
        ll -= float(np.sum(log_ndtr((right - Xb) / sigma)))
    # else: no truncation, plain OLS log-likelihood

    return -ll
