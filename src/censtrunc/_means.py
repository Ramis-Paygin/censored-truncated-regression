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

# Region-probability "kinds" — marginal effects on the probability of landing
# below the left threshold, inside (L, R), or above the right threshold. These
# extend the marginal-effects API for the censored model (the three probabilities
# sum to 1, so their marginal effects sum to 0).
PROB_KINDS = ("prob-left", "prob-interior", "prob-right")
MARGEFF_CENSORED_KINDS = CENSORED_KINDS + PROB_KINDS

# Compact single-letter aliases for predict(): h/c/t for the three conditional
# means, l/m/r for the three region probabilities. Output column labels are
# the more readable underscore_separated names.
LETTER_TO_KIND = {
    "h": "latent",
    "c": "censored",
    "t": "truncated",
    "l": "prob-left",
    "m": "prob-interior",
    "r": "prob-right",
}
LETTER_TO_COLUMN = {
    "h": "latent",
    "c": "censored",
    "t": "truncated",
    "l": "prob_left",
    "m": "prob_interior",
    "r": "prob_right",
}
ALL_LETTERS_CENSORED = "hctlmr"   # default for CensoredRegression
ALL_LETTERS_TRUNCATED = "ht"      # default for TruncatedRegression


def dispatch_predict(model, X_design: np.ndarray, kind: str | None, default: str):
    """Shared ``predict()`` dispatcher for both model classes.

    Handles three input modes for ``kind``:

    1. ``None`` -> expand to ``default`` (a letter string).
    2. A *long* kind name (``'latent'``, ``'censored'``, ``'truncated'``,
       ``'prob-left'``, ...) -> return a 1-D ``ndarray``. Kept for backward
       compatibility.
    3. A *letter* string (any subset of ``'hctlmr'``). One letter -> 1-D array;
       multiple letters -> ``pandas.DataFrame`` (or a 2-D ``ndarray`` if pandas
       is not installed).
    """
    valid_kinds = getattr(model, "_valid_margeff_kinds", MARGEFF_CENSORED_KINDS)
    valid_letters = {ltr for ltr in default}

    # Long-name backward-compat path
    if isinstance(kind, str) and kind in MARGEFF_CENSORED_KINDS:
        if kind not in valid_kinds:
            raise ValueError(
                f"kind={kind!r} is not available for {type(model).__name__}; "
                f"valid kinds: {valid_kinds}"
            )
        return value_for_kind(
            model.coef_, model.sigma_, X_design,
            model._left, model._right, model._has_left, model._has_right, kind,
        )

    if kind is None:
        kind = default

    if not isinstance(kind, str):
        raise TypeError(f"kind must be a string or None; got {type(kind).__name__}")
    if not kind:
        raise ValueError("kind is empty")

    unknown = [c for c in kind if c not in LETTER_TO_KIND]
    if unknown:
        raise ValueError(
            f"Unknown kind letter(s): {unknown}; valid letters: {sorted(LETTER_TO_KIND)}"
        )
    unavailable = [c for c in kind if c not in valid_letters]
    if unavailable:
        raise ValueError(
            f"kind letter(s) {unavailable} not available for {type(model).__name__}; "
            f"valid for this model: {sorted(valid_letters)}"
        )

    # De-duplicate while preserving order.
    seen: set[str] = set()
    ordered = [c for c in kind if not (c in seen or seen.add(c))]

    if len(ordered) == 1:
        return value_for_kind(
            model.coef_, model.sigma_, X_design,
            model._left, model._right, model._has_left, model._has_right,
            LETTER_TO_KIND[ordered[0]],
        )

    cols: dict[str, np.ndarray] = {}
    for c in ordered:
        cols[LETTER_TO_COLUMN[c]] = value_for_kind(
            model.coef_, model.sigma_, X_design,
            model._left, model._right, model._has_left, model._has_right,
            LETTER_TO_KIND[c],
        )
    try:
        import pandas as pd

        return pd.DataFrame(cols)
    except ImportError:  # pragma: no cover
        return np.column_stack(list(cols.values()))


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


# ----------------------------------------------------------------------
# Region-probability values and their derivatives
# ----------------------------------------------------------------------


def region_prob_value(
    beta: np.ndarray,
    sigma: float,
    X: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    which: str,
) -> np.ndarray:
    """Probability of the requested region (``'left'``/``'interior'``/``'right'``)."""
    probs = region_probabilities(beta, sigma, X, left, right, has_left, has_right)
    return probs[which]


def dprob_dx(
    beta: np.ndarray,
    sigma: float,
    X: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    which: str,
    slope_beta: np.ndarray,
) -> np.ndarray:
    """Derivative of a region probability w.r.t. each slope regressor.

    With ``a_L = (L - X'beta)/sigma`` (so ``d a_L / d x_j = -beta_j / sigma``):

    - d P(Y=L) / dx_j      = -beta_j * phi(a_L) / sigma
    - d P(L<Y<R) / dx_j    =  beta_j * (phi(a_L) - phi(a_R)) / sigma
    - d P(Y=R) / dx_j      =  beta_j * phi(a_R) / sigma

    The three derivatives sum to zero, mirroring the constraint that the three
    probabilities sum to one.
    """
    mu = X @ beta
    _, _, _, _, phi_L, phi_R = _alphas(mu, sigma, left, right, has_left, has_right)
    if which == "left":
        factor = -phi_L / sigma
    elif which == "interior":
        factor = (phi_L - phi_R) / sigma
    elif which == "right":
        factor = phi_R / sigma
    else:
        raise ValueError(f"Unknown region: {which!r}")
    return np.outer(factor, slope_beta)


# ----------------------------------------------------------------------
# Dispatchers used by the marginal-effects machinery
# ----------------------------------------------------------------------


def _split_prob_kind(kind: str) -> str:
    """'prob-left' -> 'left', etc."""
    return kind.split("-", 1)[1]


def value_for_kind(
    beta: np.ndarray,
    sigma: float,
    X: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    kind: str,
) -> np.ndarray:
    """Per-row value of the conditional mean or region probability for ``kind``."""
    if kind in CENSORED_KINDS:
        return conditional_mean(beta, sigma, X, left, right, has_left, has_right, kind)
    if kind in PROB_KINDS:
        return region_prob_value(
            beta, sigma, X, left, right, has_left, has_right, _split_prob_kind(kind)
        )
    raise ValueError(f"Unknown kind: {kind!r}")


def dydx_for_kind(
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
    """Per-row, per-slope derivative of the conditional mean or region probability."""
    if kind in CENSORED_KINDS:
        return dmean_dx(beta, sigma, X, left, right, has_left, has_right, kind, slope_beta)
    if kind in PROB_KINDS:
        return dprob_dx(
            beta, sigma, X, left, right, has_left, has_right, _split_prob_kind(kind), slope_beta
        )
    raise ValueError(f"Unknown kind: {kind!r}")
