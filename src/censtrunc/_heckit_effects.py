"""Marginal effects for :class:`HeckitRegression`.

Heckman's selection model has four standard marginal-effect quantities
(Greene, *Econometric Analysis*, 8th ed., Section 19.5):

================ =============================================================
``kind``         Quantity differentiated w.r.t. ``v``
================ =============================================================
``'latent'``       ``E[Y* | X] = X'beta``                       (X-side only)
``'conditional'``  ``E[Y | X, Z, S=1] = X'beta + rho sigma lambda(Z'gamma)``
``'unconditional'````E[Y * S | X, Z] = Phi(Z'gamma) X'beta + rho sigma phi(Z'gamma)``
``'prob-selected'````P(S=1 | Z) = Phi(Z'gamma)``                (Z-side only)
================ =============================================================

Notation: ``lambda(t) = phi(t)/Phi(t)`` is the inverse Mills ratio and
``delta(t) = lambda(t) (t + lambda(t))`` is its negated derivative, so
``d lambda / dt = -delta(t)``.

A regressor can appear in ``X`` only, in ``Z`` only, or in both (matched by
feature name). The combined effect on a variable that lives in both equations
is the algebraic sum of its X-side derivative (coming from ``X'beta``) and its
Z-side derivative (coming from the Mills-ratio / selection-probability terms).

Closed-form per-row derivatives
-------------------------------
With shorthand ``Phi = Phi(Z'gamma)``, ``phi = phi(Z'gamma)``,
``delta = lambda (Z'gamma + lambda)``, and indicators ``[in X]``, ``[in Z]``:

``'latent'``::

    d E[Y*]/d v = beta_v  [in X]

``'conditional'``::

    d E[Y|S=1]/d v = beta_v  [in X]   -   gamma_v * rho * sigma * delta  [in Z]

``'unconditional'``::

    d E[Y*S]/d v = beta_v * Phi  [in X]   +   gamma_v * phi * (X'beta - rho*sigma*Z'gamma)  [in Z]

``'prob-selected'``::

    d Phi(Z'gamma)/d v = gamma_v * phi  [in Z]

Standard errors use the delta method with a numerical Jacobian over the full
parameter vector ``theta = (sigma, rho, beta..., gamma...)`` and the inverse-
Hessian covariance ``cov_params_`` from the joint MLE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

from .effects import MarginalEffects, _delta_method_se

if TYPE_CHECKING:
    from .heckit import HeckitRegression


HECKIT_KINDS = ("conditional", "unconditional", "prob-selected", "latent")
_VALID_AT = ("overall", "mean", "median", "zero")


def _inverse_mills(t: np.ndarray) -> np.ndarray:
    """``phi(t) / Phi(t)`` — numerically stable for large negative ``t``."""
    return np.exp(norm.logpdf(t) - log_ndtr(t))


# ----------------------------------------------------------------------
# Variable resolution
# ----------------------------------------------------------------------


def _slope_names(all_names: list[str], fit_intercept: bool) -> list[str]:
    """Drop the intercept (named ``'const'``) from a list of design columns.

    The intercept may be there either because ``fit_intercept=True`` prepended
    one *or* because a formula like ``'y ~ 1 + x'`` parsed an explicit one
    (``from_formula`` sets ``fit_intercept=False`` but still produces a
    leading ``'const'`` column). We strip whenever the first name is
    ``'const'``, regardless of the flag.
    """
    if not all_names:
        return []
    if all_names[0] == "const":
        return list(all_names[1:])
    return list(all_names)


def _resolve_variables(
    model: "HeckitRegression", kind: str,
) -> tuple[list[str], list[int | None], list[int | None]]:
    """Return ``(slope_names, beta_idx, gamma_idx)``.

    For each slope variable returned, ``beta_idx[j]`` is its position in
    ``model.coef_`` (or ``None`` if the variable is not in the outcome
    equation), and ``gamma_idx[j]`` is its position in ``model.gamma_`` (or
    ``None`` if not in the selection equation).
    """
    x_names = list(model.outcome_feature_names_)
    z_names = list(model.selection_feature_names_)
    x_slopes = _slope_names(x_names, model.fit_intercept)
    z_slopes = _slope_names(z_names, model.fit_intercept)

    if kind == "latent":
        slopes = x_slopes
    elif kind == "prob-selected":
        slopes = z_slopes
    elif kind in ("conditional", "unconditional"):
        slopes = list(x_slopes)
        for n in z_slopes:
            if n not in slopes:
                slopes.append(n)
    else:
        raise ValueError(f"Unknown kind: {kind!r}; expected one of {HECKIT_KINDS}.")

    beta_idx: list[int | None] = [
        (x_names.index(n) if n in x_names else None) for n in slopes
    ]
    gamma_idx: list[int | None] = [
        (z_names.index(n) if n in z_names else None) for n in slopes
    ]
    return slopes, beta_idx, gamma_idx


# ----------------------------------------------------------------------
# Per-row, per-variable derivative
# ----------------------------------------------------------------------


def _row_effects(
    sigma: float,
    rho: float,
    beta: np.ndarray,
    gamma: np.ndarray,
    X_eval: np.ndarray,
    Z_eval: np.ndarray,
    beta_idx: list[int | None],
    gamma_idx: list[int | None],
    kind: str,
) -> np.ndarray:
    """Return an ``(n_eval, n_vars)`` matrix of pointwise marginal effects."""
    Zg = Z_eval @ gamma
    n_eval = Z_eval.shape[0]
    n_vars = len(beta_idx)
    out = np.zeros((n_eval, n_vars))

    if kind == "latent":
        for j in range(n_vars):
            if beta_idx[j] is not None:
                out[:, j] = beta[beta_idx[j]]
        return out

    if kind == "prob-selected":
        phi = norm.pdf(Zg)
        for j in range(n_vars):
            if gamma_idx[j] is not None:
                out[:, j] = gamma[gamma_idx[j]] * phi
        return out

    if kind == "conditional":
        lam = _inverse_mills(Zg)
        delta = lam * (Zg + lam)  # = -d lambda / d t
        for j in range(n_vars):
            if beta_idx[j] is not None:
                out[:, j] += beta[beta_idx[j]]
            if gamma_idx[j] is not None:
                out[:, j] -= gamma[gamma_idx[j]] * rho * sigma * delta
        return out

    if kind == "unconditional":
        Phi = norm.cdf(Zg)
        phi = norm.pdf(Zg)
        Xb = X_eval @ beta
        common = Xb - rho * sigma * Zg
        for j in range(n_vars):
            if beta_idx[j] is not None:
                out[:, j] += beta[beta_idx[j]] * Phi
            if gamma_idx[j] is not None:
                out[:, j] += gamma[gamma_idx[j]] * phi * common
        return out

    raise ValueError(f"Unknown kind: {kind!r}; expected one of {HECKIT_KINDS}.")


# ----------------------------------------------------------------------
# Evaluation-point construction
# ----------------------------------------------------------------------


def _eval_rows(model: "HeckitRegression", at: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the (X_eval, Z_eval) row pair to evaluate effects at.

    At ``'mean'`` / ``'median'`` / ``'zero'`` we return a single-row pair (the
    effects are then a single number per variable); at ``'overall'`` we return
    the full training-data design matrices so the caller can average over rows.
    """
    X, Z = model._X_train_, model._Z_train_
    if at == "overall":
        return X.copy(), Z.copy()
    if at == "mean":
        return X.mean(axis=0, keepdims=True), Z.mean(axis=0, keepdims=True)
    if at == "median":
        return (
            np.median(X, axis=0, keepdims=True),
            np.median(Z, axis=0, keepdims=True),
        )
    if at == "zero":
        # Keep the intercept column at 1 if there is one -- "all *slopes* at zero".
        Xz = np.zeros((1, X.shape[1]))
        Zz = np.zeros((1, Z.shape[1]))
        if model.fit_intercept:
            if model.outcome_feature_names_ and model.outcome_feature_names_[0] == "const":
                Xz[:, 0] = 1.0
            if model.selection_feature_names_ and model.selection_feature_names_[0] == "const":
                Zz[:, 0] = 1.0
        return Xz, Zz
    raise ValueError(f"Unknown at: {at!r}; expected one of {_VALID_AT}.")


# ----------------------------------------------------------------------
# Top-level API
# ----------------------------------------------------------------------


def get_heckit_margeff(
    model: "HeckitRegression",
    at: str = "overall",
    kind: str = "conditional",
    atexog: dict | None = None,
) -> MarginalEffects:
    """Compute marginal effects for a Heckman selection model.

    Parameters mirror the convention used by the censored / truncated
    ``get_margeff`` so the user-facing API is consistent across the package.

    Notes
    -----
    SEs require a joint covariance of ``(sigma, rho, beta, gamma)`` which is
    only available after a ``method='mle'`` fit. For ``method='twostep'``
    point estimates are returned with ``NaN`` standard errors. (A bootstrap
    routine is available via ``HeckitRegression.bootstrap``; tunneling those
    samples through the marginal-effect formula and returning a sampling SE
    is left as a follow-up.)
    """
    if not model._fitted:
        raise RuntimeError("Model not fitted. Call .fit(...) first.")
    if at not in _VALID_AT:
        raise ValueError(f"Unknown at: {at!r}; expected one of {_VALID_AT}.")
    if atexog is not None:
        raise NotImplementedError(
            "atexog is not yet supported for HeckitRegression.get_margeff"
        )

    slope_names, beta_idx, gamma_idx = _resolve_variables(model, kind)
    if not slope_names:
        raise ValueError(
            f"No slope variables for kind={kind!r}; the relevant equation "
            "has only an intercept."
        )

    k_x = model.coef_.shape[0]
    k_z = model.gamma_.shape[0]
    params = np.concatenate(
        [[float(model.sigma_), float(model.rho_)], model.coef_, model.gamma_]
    )
    X_eval, Z_eval = _eval_rows(model, at)

    def effect_fn(theta: np.ndarray) -> np.ndarray:
        sigma_, rho_ = float(theta[0]), float(theta[1])
        beta_ = theta[2 : 2 + k_x]
        gamma_ = theta[2 + k_x : 2 + k_x + k_z]
        row = _row_effects(
            sigma_, rho_, beta_, gamma_, X_eval, Z_eval, beta_idx, gamma_idx, kind,
        )
        return row.mean(axis=0)

    effects = effect_fn(params)

    cov = getattr(model, "cov_params_", None)
    if cov is not None and model.method == "mle":
        se = _delta_method_se(effect_fn, params, cov)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, effects / se, np.nan)
        p = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    else:
        se = np.full_like(effects, np.nan)
        z = np.full_like(effects, np.nan)
        p = np.full_like(effects, np.nan)

    return MarginalEffects(
        margeff=effects,
        margeff_se=se,
        tvalues=z,
        pvalues=p,
        names=slope_names,
        at=at,
        method="dydx",
        kind=kind,
        model_name="Heckit Regression",
    )
