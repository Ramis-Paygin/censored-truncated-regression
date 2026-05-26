"""Marginal effects (AME and MEM) with delta-method standard errors.

For the censored model, three notions of marginal effect are natural, one for each
conditional mean discussed in Hansen (2022), Section 27.3:

- ``latent``     :  d E[Y* | X] / dX_j = beta_j
- ``censored``   :  d E[Y  | X] / dX_j = beta_j * [Phi(alpha_R) - Phi(alpha_L)]
- ``truncated``  :  d E[Y  | X, L<Y<R] / dX_j  — see ``_truncated_marginal``

For each effect, the package returns:

- **AME** (Average Marginal Effect): the effect evaluated at each observation in
  the training sample, then averaged.
- **MEM** (Marginal Effect at the Mean): the effect evaluated once at the sample
  mean of the regressors.

Standard errors are computed by the delta method: if ``g(theta)`` is the marginal
effect and ``Var(theta_hat)`` the parameter covariance, then
``Var(g) ~= (dg/dtheta)' Var(theta_hat) (dg/dtheta)``. We use numerical gradients
to avoid lengthy closed-form derivatives for the censored/truncated cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from .censored import CensoredRegression


@dataclass
class MarginalEffects:
    """Container for marginal-effect estimates.

    Attributes
    ----------
    effects : ndarray of shape (k,)
        Point estimates of the marginal effect for each slope coefficient
        (excluding the intercept).
    se : ndarray of shape (k,)
        Delta-method standard errors.
    z_values : ndarray
        ``effects / se``.
    p_values : ndarray
        Two-sided Wald p-values.
    names : list of str
        Names of the corresponding variables.
    kind : str
        One of ``'latent'``, ``'censored'``, ``'truncated'``.
    at : str
        One of ``'ame'``, ``'mem'``.
    """

    effects: np.ndarray
    se: np.ndarray
    z_values: np.ndarray
    p_values: np.ndarray
    names: list[str]
    kind: str
    at: str

    def to_dataframe(self):
        """Return a pandas DataFrame view (requires pandas)."""
        import pandas as pd

        z_crit = norm.ppf(0.975)
        return pd.DataFrame(
            {
                "dy/dx": self.effects,
                "std err": self.se,
                "z": self.z_values,
                "P>|z|": self.p_values,
                "[0.025": self.effects - z_crit * self.se,
                "0.975]": self.effects + z_crit * self.se,
            },
            index=self.names,
        )


# ----------------------------------------------------------------------
# Censored-regression marginal effects
# ----------------------------------------------------------------------


def _slope_indices(model: "CensoredRegression") -> tuple[np.ndarray, list[str]]:
    """Indices of slope coefficients in ``params_`` (excluding sigma and intercept)
    and their human-readable names."""
    # params_ = [sigma, beta_0, beta_1, ...]; beta_0 corresponds to feature_names_[0].
    names = list(model.feature_names_)
    # Drop intercept if present
    if model.fit_intercept and names and names[0] == "const":
        slope_names = names[1:]
        slope_offsets = np.arange(2, 2 + len(slope_names))  # skip sigma (0) and intercept (1)
    else:
        slope_names = names
        slope_offsets = np.arange(1, 1 + len(slope_names))
    return slope_offsets, slope_names


def _design_matrix_for_eval(model: "CensoredRegression", X: np.ndarray | None) -> np.ndarray:
    """Build the design matrix at which marginal effects are evaluated."""
    from ._utils import _prepare_design_matrix  # local import to avoid cycle

    if X is None:
        raise ValueError("X must be supplied for marginal effects.")
    X_design, _ = _prepare_design_matrix(
        X,
        fit_intercept=model.fit_intercept,
        feature_names=model._user_feature_names(),  # type: ignore[attr-defined]
    )
    return X_design


def _marginal_effect_value(
    params: np.ndarray,
    X_rows: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    kind: str,
    slope_idx_in_params: np.ndarray,
    n_features_with_intercept: int,
) -> np.ndarray:
    """Compute marginal effects for each slope at a set of evaluation rows.

    The output is an averaged-over-rows vector (length k), so this function works
    for both AME (rows = all training X) and MEM (rows = mean row).
    """
    sigma = params[0]
    beta = params[1 : n_features_with_intercept + 1]
    Xb = X_rows @ beta  # length m

    if kind == "latent":
        # ME = beta_j for each slope; broadcast to (m, k) then mean -> (k,)
        slope_beta = beta[slope_idx_in_params - 1]  # offset by 1 (params[0] is sigma)
        return np.broadcast_to(slope_beta, (X_rows.shape[0], slope_beta.shape[0])).mean(axis=0)

    a_L = (left - Xb) / sigma if has_left else None
    a_R = (right - Xb) / sigma if has_right else None
    Phi_L = norm.cdf(a_L) if a_L is not None else np.zeros_like(Xb)
    Phi_R = norm.cdf(a_R) if a_R is not None else np.ones_like(Xb)

    if kind == "censored":
        scale = Phi_R - Phi_L  # (m,)
        slope_beta = beta[slope_idx_in_params - 1]  # (k,)
        # ME_{m,k} = beta_k * scale_m
        ME = np.outer(scale, slope_beta)  # (m, k)
        return ME.mean(axis=0)

    if kind == "truncated":
        phi_L = norm.pdf(a_L) if a_L is not None else np.zeros_like(Xb)
        phi_R = norm.pdf(a_R) if a_R is not None else np.zeros_like(Xb)
        denom = Phi_R - Phi_L
        denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        # d E[Y | L<Y<R] / dX_j = beta_j * [ 1 - (alpha_R * phi_R - alpha_L * phi_L) / denom
        #                                    - ((phi_L - phi_R) / denom)^2 ]
        term_alpha_phi = (
            (a_R * phi_R if a_R is not None else 0.0)
            - (a_L * phi_L if a_L is not None else 0.0)
        )
        mills_sq = ((phi_L - phi_R) / denom_safe) ** 2
        adjustment = 1.0 - term_alpha_phi / denom_safe - mills_sq  # (m,)
        slope_beta = beta[slope_idx_in_params - 1]
        ME = np.outer(adjustment, slope_beta)
        return ME.mean(axis=0)

    raise ValueError(f"Unknown kind: {kind!r}")


def _delta_method_se(
    f: callable,  # type: ignore[valid-type]
    params: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Delta-method standard errors for a vector-valued function ``f(params)``.

    Uses central finite differences for the Jacobian.
    """
    base = f(params)
    n_params = params.shape[0]
    n_out = base.shape[0]
    J = np.zeros((n_out, n_params))
    step = np.sqrt(np.finfo(float).eps)
    h = step * np.maximum(np.abs(params), 1.0)
    for p in range(n_params):
        x_p = params.copy(); x_p[p] += h[p]
        x_m = params.copy(); x_m[p] -= h[p]
        J[:, p] = (f(x_p) - f(x_m)) / (2 * h[p])
    var = np.einsum("ip,pq,iq->i", J, cov, J)
    return np.sqrt(np.maximum(var, 0.0))


# ----------------------------------------------------------------------
# Public API attached as methods to CensoredRegression
# ----------------------------------------------------------------------


def compute_marginal_effects(
    model: "CensoredRegression",
    X_eval: np.ndarray | None,
    kind: str,
    at: str,
) -> MarginalEffects:
    """Compute AME or MEM for a fitted CensoredRegression.

    Parameters
    ----------
    model : CensoredRegression
        A fitted model.
    X_eval : array-like or None
        Sample of regressors at which to evaluate. If ``None``, this function
        cannot proceed (no training X is cached); pass the training matrix
        (or any sample of interest) explicitly.
    kind : {'latent', 'censored', 'truncated'}
    at : {'ame', 'mem'}
    """
    model._check_fitted()  # type: ignore[attr-defined]

    X_design = _design_matrix_for_eval(model, X_eval)
    slope_idx, slope_names = _slope_indices(model)
    if not slope_names:
        raise ValueError("Model has no slope coefficients (intercept-only); marginal effects are empty.")

    if at == "ame":
        rows = X_design
    elif at == "mem":
        rows = X_design.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown 'at': {at!r}; expected 'ame' or 'mem'.")

    n_features_with_intercept = X_design.shape[1]

    def f(params: np.ndarray) -> np.ndarray:
        return _marginal_effect_value(
            params,
            rows,
            left=model._left,  # type: ignore[attr-defined]
            right=model._right,  # type: ignore[attr-defined]
            has_left=model._has_left,  # type: ignore[attr-defined]
            has_right=model._has_right,  # type: ignore[attr-defined]
            kind=kind,
            slope_idx_in_params=slope_idx,
            n_features_with_intercept=n_features_with_intercept,
        )

    effects = f(model.params_)
    se = _delta_method_se(f, model.params_, model.cov_params_)
    z = effects / se
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    return MarginalEffects(
        effects=effects,
        se=se,
        z_values=z,
        p_values=p,
        names=slope_names,
        kind=kind,
        at=at,
    )
