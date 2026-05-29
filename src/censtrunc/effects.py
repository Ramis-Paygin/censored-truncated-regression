"""Marginal effects via a single ``get_margeff`` entry point.

The API deliberately mirrors :meth:`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff`:
one function with options selecting *where* the effect is evaluated and *what kind*
of effect (derivative or elasticity) is reported.

Parameters of :func:`get_margeff`
---------------------------------
at : {'overall', 'mean', 'median', 'zero'}
    Point(s) at which to evaluate the effect.
    - ``'overall'`` — average of the per-observation effects (a.k.a. **AME**).
    - ``'mean'``    — effect at the sample mean of the regressors (a.k.a. **MEM**).
    - ``'median'``  — effect at the sample median of the regressors.
    - ``'zero'``    — effect with all regressors set to zero.
method : {'dydx', 'eyex', 'dyex', 'eydx'}
    - ``'dydx'`` — derivative  dE[y]/dx_j.
    - ``'eyex'`` — elasticity  (dE[y]/dx_j)(x_j / E[y]).
    - ``'dyex'`` — semi-elasticity  (dE[y]/dx_j) x_j.
    - ``'eydx'`` — semi-elasticity  (dE[y]/dx_j)/E[y].
kind : {'latent', 'censored', 'truncated'}
    Which conditional mean the effect refers to (``'censored'`` = E[Y|X] is the
    usual choice; this dimension has no analogue in the probit case but is
    essential for censored/truncated models).
dummy : bool
    If ``True``, binary (0/1) regressors are given a **discrete difference**
    E[y | x=1] - E[y | x=0] instead of a derivative.
count : bool
    If ``True``, integer-valued regressors are given a discrete difference
    E[y | x = round(x)+1] - E[y | x = round(x)].
atexog : dict or None
    Optional ``{design_column_index: value}`` overrides for the evaluation point.

Standard errors are obtained by the delta method with a numerical Jacobian, which
handles all four ``method`` choices and the discrete-difference cases uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from . import _means

if TYPE_CHECKING:
    from .censored import CensoredRegression

_METHOD_LABELS = {"dydx": "dy/dx", "eyex": "eyex", "dyex": "dyex", "eydx": "eydx"}
_VALID_AT = ("overall", "mean", "median", "zero")
_VALID_METHOD = ("dydx", "eyex", "dyex", "eydx")


@dataclass
class MarginalEffects:
    """Marginal-effects results, mirroring statsmodels' margins object.

    Attributes
    ----------
    margeff : ndarray
        Point estimates of the marginal effect for each slope regressor.
    margeff_se : ndarray
        Delta-method standard errors.
    tvalues, pvalues : ndarray
        Wald z-statistics and two-sided p-values.
    names : list of str
        Regressor names (slopes only; the intercept has no marginal effect).
    at, method, kind : str
        The options used to compute the effects.
    model_name : str
        Name of the originating model class (for the summary header).

    Notes
    -----
    Backward-compatible aliases ``effects`` and ``se`` are provided.
    """

    margeff: np.ndarray
    margeff_se: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    names: list[str]
    at: str
    method: str
    kind: str
    model_name: str = "Censored Regression"

    # ---- backward-compatible aliases -------------------------------------
    @property
    def effects(self) -> np.ndarray:
        return self.margeff

    @property
    def se(self) -> np.ndarray:
        return self.margeff_se

    @property
    def z_values(self) -> np.ndarray:
        return self.tvalues

    @property
    def p_values(self) -> np.ndarray:
        return self.pvalues

    # ---- views -----------------------------------------------------------
    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        z = norm.ppf(1 - alpha / 2)
        return np.column_stack(
            [self.margeff - z * self.margeff_se, self.margeff + z * self.margeff_se]
        )

    def summary_frame(self, alpha: float = 0.05):
        """Return a pandas DataFrame of the marginal effects."""
        import pandas as pd

        ci = self.conf_int(alpha)
        label = _METHOD_LABELS[self.method]
        return pd.DataFrame(
            {
                label: self.margeff,
                "std err": self.margeff_se,
                "z": self.tvalues,
                "P>|z|": self.pvalues,
                "[0.025": ci[:, 0],
                "0.975]": ci[:, 1],
            },
            index=self.names,
        )

    # legacy name kept so older example code / tests keep working
    def to_dataframe(self, alpha: float = 0.05):
        return self.summary_frame(alpha)

    def summary(self, alpha: float = 0.05) -> str:
        """Text summary in the style of statsmodels' marginal-effects table."""
        label = _METHOD_LABELS[self.method]
        ci = self.conf_int(alpha)
        title = f"{self.model_name} Marginal Effects"
        head = [
            f"{title:^78}",
            "=" * 37,
            f"{'Dep. Variable:':<20}{'y':>17}",
            f"{'Method:':<20}{self.method:>17}",
            f"{'At:':<20}{self.at:>17}",
            "=" * 78,
            f"{'':<14}{label:>10}{'std err':>11}{'z':>11}{'P>|z|':>11}{'[0.025':>11}{'0.975]':>11}",
            "-" * 78,
        ]
        rows = [
            f"{nm:<14}{m:>10.4f}{s:>11.3f}{t:>11.3f}{p:>11.3f}{lo:>11.3f}{hi:>11.3f}"
            for nm, m, s, t, p, (lo, hi) in zip(
                self.names, self.margeff, self.margeff_se, self.tvalues, self.pvalues, ci
            )
        ]
        return "\n".join(head + rows + ["=" * 78])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _slope_indices(model: "CensoredRegression") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (param indices, design-column indices, names) for slope regressors.

    ``params_ = [sigma, beta_0, beta_1, ...]``; design columns line up with
    ``feature_names_``. The intercept (if present) is excluded.
    """
    names = list(model.feature_names_)
    if model.fit_intercept and names and names[0] == "const":
        slope_names = names[1:]
        col_idx = np.arange(1, 1 + len(slope_names))  # design columns (skip const at 0)
        param_idx = np.arange(2, 2 + len(slope_names))  # params (skip sigma at 0, const at 1)
    else:
        slope_names = names
        col_idx = np.arange(0, len(slope_names))
        param_idx = np.arange(1, 1 + len(slope_names))
    return param_idx, col_idx, slope_names


def _eval_rows(model: "CensoredRegression", at: str, atexog: dict | None) -> np.ndarray:
    """Construct the design rows at which to evaluate effects."""
    X = model._X_train_  # cached at fit time  # type: ignore[attr-defined]
    if at == "overall":
        rows = X.copy()
    elif at == "mean":
        rows = X.mean(axis=0, keepdims=True)
    elif at == "median":
        rows = np.median(X, axis=0, keepdims=True)
    elif at == "zero":
        rows = np.zeros((1, X.shape[1]))
    else:
        raise ValueError(f"Unknown at: {at!r}; expected one of {_VALID_AT}.")
    if atexog:
        rows = rows.copy()
        for col, val in atexog.items():
            rows[:, col] = val
    return rows


def _detect_discrete_columns(
    X: np.ndarray, col_idx: np.ndarray, dummy: bool, count: bool
) -> dict[int, str]:
    """Map design-column index -> 'dummy' or 'count' for discrete regressors."""
    discrete: dict[int, str] = {}
    for col in col_idx:
        values = X[:, col]
        uniq = np.unique(values)
        if dummy and set(uniq.tolist()).issubset({0.0, 1.0}):
            discrete[int(col)] = "dummy"
        elif count and np.allclose(values, np.round(values)):
            discrete[int(col)] = "count"
    return discrete


def _margeff_vector(
    params: np.ndarray,
    rows: np.ndarray,
    model: "CensoredRegression",
    param_idx: np.ndarray,
    col_idx: np.ndarray,
    kind: str,
    method: str,
    discrete: dict[int, str],
) -> np.ndarray:
    """Compute the aggregated marginal-effect vector as a function of ``params``.

    Written so that a numerical Jacobian over ``params`` yields delta-method SEs.
    """
    sigma = params[0]
    beta = params[1:]
    slope_beta = params[param_idx]  # coefficients of the slope regressors
    left, right = model._left, model._right  # type: ignore[attr-defined]
    has_left, has_right = model._has_left, model._has_right  # type: ignore[attr-defined]

    # base derivative effects (n_rows, k_slopes) — works for both conditional
    # means (latent/censored/truncated) and region probabilities (prob-*).
    effect = _means.dydx_for_kind(
        beta, sigma, rows, left, right, has_left, has_right, kind, slope_beta
    )

    # override discrete columns with finite differences
    if discrete:
        for local_j, col in enumerate(col_idx):
            mode = discrete.get(int(col))
            if mode is None:
                continue
            rows_hi = rows.copy()
            rows_lo = rows.copy()
            if mode == "dummy":
                rows_hi[:, col] = 1.0
                rows_lo[:, col] = 0.0
            else:  # count
                base = np.round(rows[:, col])
                rows_hi[:, col] = base + 1.0
                rows_lo[:, col] = base
            m_hi = _means.value_for_kind(
                beta, sigma, rows_hi, left, right, has_left, has_right, kind
            )
            m_lo = _means.value_for_kind(
                beta, sigma, rows_lo, left, right, has_left, has_right, kind
            )
            effect[:, local_j] = m_hi - m_lo

    # elasticity / semi-elasticity transforms
    if method != "dydx":
        ypred = _means.value_for_kind(
            beta, sigma, rows, left, right, has_left, has_right, kind
        )
        xvals = rows[:, col_idx]  # (n_rows, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            if method == "eyex":
                effect = effect * xvals / ypred[:, None]
            elif method == "dyex":
                effect = effect * xvals
            elif method == "eydx":
                effect = effect / ypred[:, None]
            else:  # pragma: no cover
                raise ValueError(f"Unknown method: {method!r}")

    return effect.mean(axis=0)


def _delta_method_se(f: Any, params: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Delta-method SEs for a vector function via central finite differences."""
    base = f(params)
    n_params = params.shape[0]
    n_out = base.shape[0]
    J = np.zeros((n_out, n_params))
    step = np.sqrt(np.finfo(float).eps)
    h = step * np.maximum(np.abs(params), 1.0)
    for p in range(n_params):
        xp = params.copy(); xp[p] += h[p]
        xm = params.copy(); xm[p] -= h[p]
        J[:, p] = (f(xp) - f(xm)) / (2 * h[p])
    var = np.einsum("ip,pq,iq->i", J, cov, J)
    return np.sqrt(np.maximum(var, 0.0))


def get_margeff(
    model: "CensoredRegression",
    at: str = "overall",
    method: str = "dydx",
    kind: str = "censored",
    atexog: dict | None = None,
    dummy: bool = False,
    count: bool = False,
) -> MarginalEffects:
    """Compute marginal effects for a fitted censored/truncated model.

    See the module docstring for the meaning of each option.
    """
    model._check_fitted()  # type: ignore[attr-defined]
    if at not in _VALID_AT:
        raise ValueError(f"Unknown at: {at!r}; expected one of {_VALID_AT}.")
    if method not in _VALID_METHOD:
        raise ValueError(f"Unknown method: {method!r}; expected one of {_VALID_METHOD}.")
    valid_kinds = getattr(model, "_valid_margeff_kinds", _means.CENSORED_KINDS)
    if kind not in valid_kinds:
        raise ValueError(f"Unknown kind: {kind!r}; expected one of {valid_kinds}.")

    param_idx, col_idx, slope_names = _slope_indices(model)
    if not slope_names:
        raise ValueError("Model has no slope coefficients; marginal effects are empty.")

    rows = _eval_rows(model, at, atexog)
    discrete = _detect_discrete_columns(model._X_train_, col_idx, dummy, count)  # type: ignore[attr-defined]

    def f(params: np.ndarray) -> np.ndarray:
        return _margeff_vector(
            params, rows, model, param_idx, col_idx, kind, method, discrete
        )

    effects = f(model.params_)
    se = _delta_method_se(f, model.params_, model.cov_params_)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, effects / se, np.nan)
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    return MarginalEffects(
        margeff=effects,
        margeff_se=se,
        tvalues=z,
        pvalues=p,
        names=slope_names,
        at=at,
        method=method,
        kind=kind,
        model_name=type(model).__name__.replace("Regression", " Regression"),
    )
