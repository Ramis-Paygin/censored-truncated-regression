"""Heckman's sample-selection regression ("Heckit").

The model has two equations:

    Y* = X'beta + e,                  outcome equation (observed when S = 1)
    S* = Z'gamma + u,                 selection equation (latent utility)
    S  = 1{S* > 0},                   selection indicator
    Y  = Y* if S = 1, else NaN,

with the error pair distributed jointly normal,

    (e, u) ~ N(0, Sigma),    Sigma = [[sigma_e^2, sigma_eu],
                                     [sigma_eu,   1       ]].

The variance of ``u`` is normalised to one for identification (the usual probit
convention). The model is therefore parameterised by ``(beta, gamma, sigma_e, rho)``
with ``rho = sigma_eu / sigma_e`` the correlation between the two errors.

Two estimators are provided:

- ``method='twostep'`` (Heckman 1979): first a probit of ``S`` on ``Z`` yields
  ``gamma_hat``; then OLS of ``Y`` on ``(X, lambda_hat)`` on the selected
  subsample gives ``beta_hat`` and the coefficient ``rho * sigma_e`` on the
  inverse Mills ratio ``lambda(z_i'gamma_hat)``.
- ``method='mle'``: numerical maximisation of the joint log-likelihood

      l = sum_{S=0} log[1 - Phi(z'gamma)]
        + sum_{S=1} { log Phi[(z'gamma + (rho/sigma_e)(y - x'beta)) / sqrt(1 - rho^2)]
                      - (1/2) log(2 pi sigma_e^2) - (y - x'beta)^2 / (2 sigma_e^2) }.

The MLE is asymptotically efficient; the two-step is robust and fast but its
naive second-stage standard errors understate uncertainty (the inverse Mills
ratio is a *generated regressor*). For a Heckman-correct two-step SE, use the
optional paired-bootstrap helper :meth:`HeckitRegression.bootstrap`.

References
----------
- Heckman, J. (1979). "Sample Selection Bias as a Specification Error."
  Econometrica, 47(1), 153-161.
- Hansen, B. E. (2022). *Econometrics*. Section 27.10, "Heckman's Model".
- Greene, W. H. (2018). *Econometric Analysis*, 8th ed., Section 19.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from ._utils import _prepare_design_matrix, _prepare_predict_design, _prepare_y


# ----------------------------------------------------------------------
# Log-likelihood helpers
# ----------------------------------------------------------------------


def _inverse_mills(z: np.ndarray) -> np.ndarray:
    """phi(z) / Phi(z), numerically stable."""
    return np.exp(norm.logpdf(z) - log_ndtr(z))


def _neg_loglik_probit(gamma: np.ndarray, Z: np.ndarray, s: np.ndarray) -> float:
    """Standard probit negative log-likelihood (with var(u) = 1)."""
    Zg = Z @ gamma
    # P(S=1) = Phi(Zg); use log_ndtr for numerical stability.
    ll = log_ndtr(np.where(s == 1, Zg, -Zg)).sum()
    return -float(ll)


def _neg_loglik_heckit(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    s: np.ndarray,
    k_x: int,
    k_z: int,
) -> float:
    """Negative joint log-likelihood for Heckman's model.

    ``params = [sigma_e, rho, beta_0, ..., beta_{k_x-1}, gamma_0, ..., gamma_{k_z-1}]``.
    """
    sigma_e = params[0]
    rho = params[1]
    beta = params[2 : 2 + k_x]
    gamma = params[2 + k_x : 2 + k_x + k_z]
    if sigma_e <= 0 or not (-0.999 < rho < 0.999):
        return np.inf

    Zg = Z @ gamma
    ll = 0.0

    # S = 0 part
    mask0 = s == 0
    if mask0.any():
        ll += float(log_ndtr(-Zg[mask0]).sum())

    # S = 1 part
    mask1 = s == 1
    if mask1.any():
        e = y[mask1] - X[mask1] @ beta
        z_inner = (Zg[mask1] + (rho / sigma_e) * e) / np.sqrt(1 - rho**2)
        ll += float(log_ndtr(z_inner).sum())
        ll += -0.5 * mask1.sum() * np.log(2 * np.pi * sigma_e**2)
        ll += -0.5 * float(np.sum(e**2)) / sigma_e**2

    return -ll


# ----------------------------------------------------------------------
# Probit (own implementation to avoid statsmodels as a hard dep)
# ----------------------------------------------------------------------


def _fit_probit(Z: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """MLE probit. Returns ``(gamma_hat, vcov_gamma, loglik)``."""
    # OLS warm start
    gamma0, *_ = np.linalg.lstsq(Z, 2 * s - 1, rcond=None)
    res = minimize(
        fun=_neg_loglik_probit,
        x0=gamma0,
        args=(Z, s),
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 500},
    )
    gamma_hat = res.x
    # Observed-information Hessian via finite differences for cov.
    step = np.cbrt(np.finfo(float).eps)
    p = gamma_hat.shape[0]
    h = step * np.maximum(np.abs(gamma_hat), 1.0)
    H = np.zeros((p, p))
    f0 = _neg_loglik_probit(gamma_hat, Z, s)
    for i in range(p):
        x_p = gamma_hat.copy(); x_p[i] += h[i]
        x_m = gamma_hat.copy(); x_m[i] -= h[i]
        H[i, i] = (_neg_loglik_probit(x_p, Z, s) - 2 * f0 + _neg_loglik_probit(x_m, Z, s)) / h[i] ** 2
    for i in range(p):
        for j in range(i + 1, p):
            xpp = gamma_hat.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = gamma_hat.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = gamma_hat.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = gamma_hat.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = (
                _neg_loglik_probit(xpp, Z, s)
                - _neg_loglik_probit(xpm, Z, s)
                - _neg_loglik_probit(xmp, Z, s)
                + _neg_loglik_probit(xmm, Z, s)
            ) / (4 * h[i] * h[j])
            H[j, i] = H[i, j]
    H = 0.5 * (H + H.T)
    try:
        vcov = np.linalg.inv(H)
    except np.linalg.LinAlgError:  # pragma: no cover
        vcov = np.linalg.pinv(H)
    return gamma_hat, vcov, -float(res.fun)


# ----------------------------------------------------------------------
# HeckitRegression
# ----------------------------------------------------------------------


@dataclass
class HeckitRegression:
    """Heckman sample-selection regression (two-step or joint MLE).

    Parameters
    ----------
    method : {'twostep', 'mle'}, default ``'twostep'``
        ``'twostep'`` runs Heckman's two-step estimator: probit on ``Z``, then
        OLS of ``y`` on ``(X, lambda_hat)``. Fast and robust. ``'mle'`` jointly
        maximises the log-likelihood; asymptotically efficient.
    fit_intercept : bool, default ``True``
        Prepend a constant column to both ``X`` and ``Z``.
    optimizer : str, default ``'BFGS'``
        Optimiser for MLE / probit (only used by gradient-free probit fit and MLE).
    max_iter : int, default ``500``
    tol : float, default ``1e-8``

    After ``fit`` the following attributes are populated:

    coef_ : ndarray
        Outcome-equation coefficients ``beta``.
    gamma_ : ndarray
        Selection-equation coefficients ``gamma``.
    sigma_ : float
        Residual standard deviation of the outcome equation.
    rho_ : float
        Correlation between the outcome and selection error terms.
    sigma_eu_ : float
        Covariance ``rho * sigma`` (the coefficient on the inverse Mills ratio
        in the two-step second stage).
    bse_, bse_gamma_ : ndarray
        Standard errors.
    llf_ : float (MLE only)
        Joint log-likelihood at the MLE.
    n_obs_, n_selected_ : int
        Total observations and number with ``S = 1``.
    """

    method: str = "twostep"
    fit_intercept: bool = True
    optimizer: str = "BFGS"
    max_iter: int = 500
    tol: float = 1e-8

    coef_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    gamma_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    sigma_: float = field(default=np.nan, init=False, repr=False)
    rho_: float = field(default=np.nan, init=False, repr=False)
    sigma_eu_: float = field(default=np.nan, init=False, repr=False)
    bse_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    bse_gamma_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    pvalues_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    pvalues_gamma_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    # Full covariance matrix of [sigma, rho, beta..., gamma...] (MLE only). Used
    # by ``get_margeff`` for the delta-method standard errors. ``None`` after a
    # two-step fit because the two-step covariance of the joint parameter is not
    # available in closed form -- bootstrap if you need it.
    cov_params_: np.ndarray | None = field(default=None, init=False, repr=False)
    llf_: float = field(default=np.nan, init=False, repr=False)
    n_obs_: int = field(default=0, init=False, repr=False)
    n_selected_: int = field(default=0, init=False, repr=False)
    outcome_feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    selection_feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    converged_: bool = field(default=False, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _pending_formula_data: Any = field(default=None, init=False, repr=False)
    # Cached design matrices (with intercept column if present) and selection
    # indicator -- used by ``get_margeff`` to evaluate at='overall'/'mean'/
    # 'median' without re-asking the user for the training data.
    _X_train_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _Z_train_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _s_train_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y: Any | None = None,
        X: Any | None = None,
        Z: Any | None = None,
        feature_names_outcome: list[str] | None = None,
        feature_names_selection: list[str] | None = None,
    ) -> "HeckitRegression":
        """Estimate the sample-selection model.

        Two calling styles:

        - **Explicit**: ``model.fit(y, X, Z)`` — pass arrays directly. ``NaN`` in
          ``y`` marks unselected observations.
        - **Formula**: ``HeckitRegression.from_formula(outcome, selection,
          data).fit()`` — both equations specified as patsy formulas.

        Parameters
        ----------
        y : array-like of shape (n,)
            Outcome with ``NaN`` for unselected observations (or zero/any).
        X : array-like of shape (n, k_x)
            Outcome-equation regressors.
        Z : array-like of shape (n, k_z)
            Selection-equation regressors.
        """
        if y is None and X is None and Z is None:
            if self._pending_formula_data is None:
                raise ValueError(
                    "Either pass (y, X, Z) explicitly or use "
                    "HeckitRegression.from_formula(outcome, selection, data) first."
                )
            (
                y, X, Z,
                feature_names_outcome,
                feature_names_selection,
            ) = self._pending_formula_data
            self._pending_formula_data = None
            self.fit_intercept = False
        elif y is None or X is None or Z is None:
            raise ValueError("All of y, X, Z must be supplied")
        if self.method not in ("twostep", "mle"):
            raise ValueError(f"Unknown method: {self.method!r}")

        X_design, x_cols = _prepare_design_matrix(
            X, fit_intercept=self.fit_intercept, feature_names=feature_names_outcome
        )
        Z_design, z_cols = _prepare_design_matrix(
            Z, fit_intercept=self.fit_intercept, feature_names=feature_names_selection
        )
        if X_design.shape[0] != Z_design.shape[0]:
            raise ValueError("X and Z must have the same number of rows")
        n = X_design.shape[0]

        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.shape[0] != n:
            raise ValueError(f"y length ({y_arr.shape[0]}) must equal X rows ({n})")

        s = (~np.isnan(y_arr)).astype(float)
        # replace NaN with 0 so dot products with masked beta don't propagate
        y_safe = np.where(np.isnan(y_arr), 0.0, y_arr)

        self.outcome_feature_names_ = x_cols
        self.selection_feature_names_ = z_cols
        self.n_obs_ = int(n)
        self.n_selected_ = int(s.sum())
        if self.n_selected_ < X_design.shape[1] + 1:
            raise ValueError(
                f"Need more than {X_design.shape[1]} selected observations to fit the outcome equation"
            )

        if self.method == "twostep":
            self._fit_twostep(X_design, y_safe, Z_design, s)
        else:
            self._fit_mle(X_design, y_safe, Z_design, s)

        # Cache for get_margeff (cheap; we already hold these via the optimiser).
        self._X_train_ = X_design
        self._Z_train_ = Z_design
        self._s_train_ = s
        self._fitted = True
        return self

    @classmethod
    def from_formula(
        cls,
        outcome: str,
        selection: str,
        data: Any,
        **init_kwargs: Any,
    ) -> "HeckitRegression":
        """Build a (yet-unfitted) Heckman model from two patsy formulas.

        Example::

            HeckitRegression.from_formula(
                outcome='lwage   ~ 1 + educ + exper + expersq',
                selection='inlf  ~ 1 + educ + exper + age + kidslt6 + kidsge6',
                data=mroz,
            ).fit()

        The left-hand side of the **selection** formula must be a binary
        (0/1) indicator. Wherever the indicator is 0 the outcome is treated as
        unobserved (NaN), regardless of what the outcome column contains. The
        intercept is supplied by the formula in both equations.
        """
        from ._formula import parse_single_formula

        init_kwargs.pop("fit_intercept", None)
        instance = cls(fit_intercept=False, **init_kwargs)

        s_arr, Z_arr, _s_name, z_names = parse_single_formula(selection, data)
        s_arr = np.asarray(s_arr).astype(float)
        if not np.all((s_arr == 0) | (s_arr == 1)):
            raise ValueError(
                f"Left-hand side of the selection formula must be binary (0/1); "
                f"got values: {np.unique(s_arr)[:5]} ..."
            )
        # The outcome y is missing for unselected rows (e.g. wage for non-workers).
        # We must keep those rows so X and Z line up with the selection vector,
        # so disable patsy's automatic NA-drop here.
        y_arr, X_arr, _y_name, x_names = parse_single_formula(
            outcome, data, keep_missing=True,
        )
        if y_arr.shape[0] != s_arr.shape[0]:
            raise ValueError(
                "Outcome and selection formulas resolved to different sample sizes "
                f"({y_arr.shape[0]} vs {s_arr.shape[0]}). Did you drop rows from `data` "
                "before passing it in?"
            )
        # Mark unselected observations as NaN so the fit pipeline handles them.
        y_arr = np.where(s_arr == 1, y_arr, np.nan)
        instance._pending_formula_data = (y_arr, X_arr, Z_arr, x_names, z_names)
        return instance

    # ------------------------------------------------------------------
    # Two-step estimator
    # ------------------------------------------------------------------

    def _fit_twostep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        s: np.ndarray,
    ) -> None:
        gamma_hat, vcov_gamma, _ = _fit_probit(Z, s)
        self.gamma_ = gamma_hat
        self.bse_gamma_ = np.sqrt(np.diag(vcov_gamma))
        self.pvalues_gamma_ = 2 * (1 - norm.cdf(np.abs(gamma_hat / self.bse_gamma_)))

        # Second-stage OLS on selected subsample with augmented regressor lambda.
        sel = s == 1
        Zg = Z[sel] @ gamma_hat
        lam = _inverse_mills(Zg)
        X_aug = np.hstack([X[sel], lam.reshape(-1, 1)])
        y_sel = y[sel]
        # (X_aug' X_aug)^-1 X_aug' y
        coef_aug, *_ = np.linalg.lstsq(X_aug, y_sel, rcond=None)
        self.coef_ = coef_aug[:-1]
        self.sigma_eu_ = float(coef_aug[-1])

        # Residual variance estimate (Heckman-corrected).
        e = y_sel - X_aug @ coef_aug
        # delta_i = lambda_i (lambda_i + z_i'gamma_hat)
        delta = lam * (lam + Zg)
        n_sel = sel.sum()
        sigma_sq = (e @ e) / n_sel + (self.sigma_eu_ ** 2) * delta.mean()
        # Numerical guard
        sigma_sq = max(sigma_sq, 1e-12)
        self.sigma_ = float(np.sqrt(sigma_sq))
        self.rho_ = float(self.sigma_eu_ / self.sigma_)

        # Naive second-stage covariance (uses corrected sigma^2). The proper
        # Heckman-corrected covariance also incorporates the variability of
        # gamma_hat; users wanting that should call .bootstrap().
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
        cov_naive = sigma_sq * XtX_inv
        bse_aug = np.sqrt(np.diag(cov_naive))
        self.bse_ = bse_aug[:-1]
        # se on sigma_eu coefficient
        self._bse_sigma_eu = float(bse_aug[-1])

        self.pvalues_ = 2 * (1 - norm.cdf(np.abs(self.coef_ / self.bse_)))
        self.converged_ = True
        self.llf_ = np.nan  # not defined for two-step

    # ------------------------------------------------------------------
    # Joint MLE
    # ------------------------------------------------------------------

    def _fit_mle(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        s: np.ndarray,
    ) -> None:
        # Warm start from two-step estimates.
        twostep = HeckitRegression(method="twostep", fit_intercept=False)
        twostep.coef_ = None  # noqa
        twostep._fit_twostep(X, y, Z, s)

        k_x = X.shape[1]
        k_z = Z.shape[1]
        x0 = np.concatenate(
            [
                [twostep.sigma_, np.clip(twostep.rho_, -0.95, 0.95)],
                twostep.coef_,
                twostep.gamma_,
            ]
        )
        bounds = (
            [(1e-6, None), (-0.999, 0.999)]
            + [(None, None)] * k_x
            + [(None, None)] * k_z
        )
        res = minimize(
            fun=_neg_loglik_heckit,
            x0=x0,
            args=(X, y, Z, s, k_x, k_z),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        self.converged_ = bool(res.success)
        params = res.x
        self.sigma_ = float(params[0])
        self.rho_ = float(params[1])
        self.sigma_eu_ = self.rho_ * self.sigma_
        self.coef_ = params[2 : 2 + k_x].copy()
        self.gamma_ = params[2 + k_x : 2 + k_x + k_z].copy()
        self.llf_ = -float(res.fun)

        # Standard errors via numerical Hessian of the negative joint log-lik.
        cov_full = self._numeric_cov_params(params, X, y, Z, s, k_x, k_z)
        self.cov_params_ = cov_full
        bse_full = np.sqrt(np.maximum(np.diag(cov_full), 0.0))
        # order: [sigma, rho, beta..., gamma...]
        self.bse_ = bse_full[2 : 2 + k_x]
        self.bse_gamma_ = bse_full[2 + k_x : 2 + k_x + k_z]
        self.pvalues_ = 2 * (1 - norm.cdf(np.abs(self.coef_ / self.bse_)))
        self.pvalues_gamma_ = 2 * (1 - norm.cdf(np.abs(self.gamma_ / self.bse_gamma_)))
        self._bse_sigma = float(bse_full[0])
        self._bse_rho = float(bse_full[1])

    @staticmethod
    def _numeric_cov_params(
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        s: np.ndarray,
        k_x: int,
        k_z: int,
    ) -> np.ndarray:
        """Observed-information covariance: inverse of the numerical Hessian
        of the negative joint log-likelihood at the MLE."""
        step = np.cbrt(np.finfo(float).eps)
        p = params.shape[0]
        h = step * np.maximum(np.abs(params), 1.0)
        H = np.zeros((p, p))
        f0 = _neg_loglik_heckit(params, X, y, Z, s, k_x, k_z)
        for i in range(p):
            x_p = params.copy(); x_p[i] += h[i]
            x_m = params.copy(); x_m[i] -= h[i]
            H[i, i] = (
                _neg_loglik_heckit(x_p, X, y, Z, s, k_x, k_z)
                - 2 * f0
                + _neg_loglik_heckit(x_m, X, y, Z, s, k_x, k_z)
            ) / h[i] ** 2
        for i in range(p):
            for j in range(i + 1, p):
                xpp = params.copy(); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = params.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = params.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = params.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = (
                    _neg_loglik_heckit(xpp, X, y, Z, s, k_x, k_z)
                    - _neg_loglik_heckit(xpm, X, y, Z, s, k_x, k_z)
                    - _neg_loglik_heckit(xmp, X, y, Z, s, k_x, k_z)
                    + _neg_loglik_heckit(xmm, X, y, Z, s, k_x, k_z)
                ) / (4 * h[i] * h[j])
                H[j, i] = H[i, j]
        H = 0.5 * (H + H.T)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:  # pragma: no cover
            cov = np.linalg.pinv(H)
        return cov

    # ------------------------------------------------------------------
    # Bootstrap (proper two-step SE)
    # ------------------------------------------------------------------

    def bootstrap(self, y: Any, X: Any, Z: Any, n_boot: int = 200, seed: int | None = 0) -> dict[str, np.ndarray]:
        """Paired bootstrap of the two-step estimator. Returns SEs for beta and gamma."""
        self._check_fitted()
        rng = np.random.default_rng(seed)
        y_arr = np.asarray(y, dtype=float).ravel()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        n = y_arr.shape[0]
        betas = []
        gammas = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                mr = HeckitRegression(method="twostep", fit_intercept=self.fit_intercept).fit(
                    y_arr[idx], X_arr[idx], Z_arr[idx]
                )
                betas.append(mr.coef_)
                gammas.append(mr.gamma_)
            except Exception:
                continue
        return {
            "beta_se": np.std(np.asarray(betas), axis=0, ddof=1),
            "gamma_se": np.std(np.asarray(gammas), axis=0, ddof=1),
            "beta_samples": np.asarray(betas),
            "gamma_samples": np.asarray(gammas),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: Any | None = None,
        Z: Any | None = None,
        kind: str | None = None,
    ) -> Any:
        """Predict any combination of six Heckman quantities.

        Six quantities are produced; each has a one-letter code that may be
        combined (in any order, repeats ignored) to request several columns at
        once. With ``lam(t) = phi(t)/Phi(t)`` the inverse Mills ratio:

        =====  =============================  ======================================================================
        Code   Quantity                        Formula
        =====  =============================  ======================================================================
        ``s``  selection probability           ``P(S = 1 | Z) = Phi(Z'gamma)``
        ``n``  non-selection probability       ``P(S = 0 | Z) = 1 - Phi(Z'gamma)``
        ``p``  selection propensity (latent)   ``Z'gamma``
        ``o``  E[Y | observed]                 ``E[Y | X, Z, S = 1] = X'beta + rho*sigma*lam(Z'gamma)``
        ``h``  E[Y*] hidden (unconditional)    ``E[Y* | X] = X'beta``
        ``u``  E[Y* | unobserved]              ``E[Y* | X, Z, S = 0] = X'beta - rho*sigma*phi(Z'gamma)/(1 - Phi(Z'gamma))``
        =====  =============================  ======================================================================

        Required inputs per code:

        - ``s``, ``n``, ``p``  -- need only ``Z``;
        - ``h``                -- needs only ``X``;
        - ``o``, ``u``         -- need both ``X`` and ``Z``.

        Parameters
        ----------
        X : array-like or None
            Outcome-equation regressors. Required if any requested code touches
            X (``o``, ``h``, ``u``).
        Z : array-like or None
            Selection-equation regressors. Required if any requested code
            touches Z (``s``, ``n``, ``p``, ``o``, ``u``).
        kind : str or None, default ``None``
            - ``None`` returns all six columns (= ``'snpohu'``) -- requires
              both ``X`` and ``Z``.
            - A letter string (any subset of ``'snpohu'``). A single letter
              returns a 1-D ``ndarray``; multiple letters return a pandas
              ``DataFrame`` whose columns are
              ``['prob_selected', 'prob_not_selected', 'propensity', 'observed',
              'hidden', 'unobserved']`` in the order requested.
            - Backward-compatible long names ``'outcome'``,
              ``'selection_prob'``, ``'conditional'`` continue to return
              1-D ``ndarray``\\ s mapped to ``'h'``, ``'s'``, ``'o'``
              respectively.

        Returns
        -------
        ndarray or pandas.DataFrame
        """
        self._check_fitted()

        # Backward-compat long names -> single-letter codes.
        long_to_letter = {
            "outcome":         "h",
            "selection_prob":  "s",
            "conditional":     "o",
        }
        if isinstance(kind, str) and kind in long_to_letter:
            return self.predict(X=X, Z=Z, kind=long_to_letter[kind])

        if kind is None:
            kind = "snpohu"
        if not isinstance(kind, str):
            raise TypeError(f"kind must be a string or None; got {type(kind).__name__}")
        if not kind:
            raise ValueError("kind is empty")

        valid_letters = set("snpohu")
        unknown = [c for c in kind if c not in valid_letters]
        if unknown:
            raise ValueError(
                f"Unknown kind letter(s): {unknown}; valid letters: {sorted(valid_letters)}"
            )
        # De-duplicate, keep order.
        seen: set[str] = set()
        ordered = [c for c in kind if not (c in seen or seen.add(c))]

        needs_x = any(c in "ohu" for c in ordered)
        needs_z = any(c in "snpou" for c in ordered)
        if needs_x and X is None:
            raise ValueError(f"kind={kind!r} requires X (for letter(s) in 'ohu')")
        if needs_z and Z is None:
            raise ValueError(f"kind={kind!r} requires Z (for letter(s) in 'snpou')")

        # Build the shared shorthand quantities once per call.
        Xb = None
        if needs_x:
            Xd = _prepare_predict_design(
                X, fit_intercept=self.fit_intercept,
                feature_names=self.outcome_feature_names_,
            )
            Xb = Xd @ self.coef_
        Zg = None
        Phi_Zg = None
        phi_Zg = None
        if needs_z:
            Zd = _prepare_predict_design(
                Z, fit_intercept=self.fit_intercept,
                feature_names=self.selection_feature_names_,
            )
            Zg = Zd @ self.gamma_
            Phi_Zg = norm.cdf(Zg)
            phi_Zg = norm.pdf(Zg)

        column_for: dict[str, np.ndarray] = {}
        label_for = {
            "s": "prob_selected",
            "n": "prob_not_selected",
            "p": "propensity",
            "o": "observed",
            "h": "hidden",
            "u": "unobserved",
        }

        for c in ordered:
            if c == "s":
                column_for[c] = Phi_Zg
            elif c == "n":
                column_for[c] = 1.0 - Phi_Zg
            elif c == "p":
                column_for[c] = Zg
            elif c == "h":
                column_for[c] = Xb
            elif c == "o":
                # inverse Mills lam = phi/Phi  -- numerically stable variant.
                lam = _inverse_mills(Zg)
                column_for[c] = Xb + self.sigma_eu_ * lam
            elif c == "u":
                # 'unobserved' uses the lower-tail inverse-Mills phi/(1-Phi).
                # Stabilised by working in -Zg (the upper tail of -Zg is
                # the lower tail of Zg).
                lam_minus = _inverse_mills(-Zg)
                column_for[c] = Xb - self.sigma_eu_ * lam_minus

        if len(ordered) == 1:
            return column_for[ordered[0]]
        cols = {label_for[c]: column_for[c] for c in ordered}
        try:
            import pandas as pd
            return pd.DataFrame(cols)
        except ImportError:  # pragma: no cover
            return np.column_stack(list(cols.values()))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Render a multi-block summary similar in spirit to ``py4etrics``."""
        self._check_fitted()
        lines: list[str] = []
        head = (
            f"{'Heckman Selection Regression':^78}",
            "=" * 78,
            f"{'Method:':<25}{self.method:<25}"
            f"{'No. Observations:':<25}{self.n_obs_}",
            f"{'Selected (S=1):':<25}{self.n_selected_:<25}"
            f"{'Censored (S=0):':<25}{self.n_obs_ - self.n_selected_}",
            f"{'sigma_e:':<25}{self.sigma_:<25.6f}"
            f"{'rho:':<25}{self.rho_:.6f}",
        )
        if self.method == "mle":
            head = head + (f"{'Log-Likelihood:':<25}{self.llf_:.4f}",)
        lines.extend(head)
        lines.append("=" * 78)

        # Outcome equation table
        lines.append(f"{'Outcome equation (Y* = X * beta + e)':^78}")
        lines.append("-" * 78)
        lines.append(
            f"{'':<14}{'coef':>11}{'std err':>12}{'z':>10}{'P>|z|':>10}{'[0.025':>12}{'0.975]':>10}"
        )
        lines.append("-" * 78)
        z_crit = norm.ppf(0.975)
        for nm, b, se, p in zip(
            self.outcome_feature_names_, self.coef_, self.bse_, self.pvalues_
        ):
            t = b / se if se > 0 else np.nan
            lo, hi = b - z_crit * se, b + z_crit * se
            lines.append(
                f"{nm:<14}{b:>11.4f}{se:>12.4f}{t:>10.4f}{p:>10.4f}{lo:>12.4f}{hi:>10.4f}"
            )
        lines.append("-" * 78)

        # Selection equation table
        lines.append(f"{'Selection equation (S* = Z * gamma + u, var(u) = 1)':^78}")
        lines.append("-" * 78)
        lines.append(
            f"{'':<14}{'coef':>11}{'std err':>12}{'z':>10}{'P>|z|':>10}{'[0.025':>12}{'0.975]':>10}"
        )
        lines.append("-" * 78)
        for nm, g, se, p in zip(
            self.selection_feature_names_,
            self.gamma_,
            self.bse_gamma_,
            self.pvalues_gamma_,
        ):
            t = g / se if se > 0 else np.nan
            lo, hi = g - z_crit * se, g + z_crit * se
            lines.append(
                f"{nm:<14}{g:>11.4f}{se:>12.4f}{t:>10.4f}{p:>10.4f}{lo:>12.4f}{hi:>10.4f}"
            )
        lines.append("=" * 78)
        if self.method == "twostep":
            lines.append(
                "Note: two-step second-stage standard errors are naive; call .bootstrap(y, X, Z) "
                "for proper Heckman-corrected SEs that account for the generated regressor."
            )
            lines.append("=" * 78)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary() if self._fitted else f"HeckitRegression(method={self.method!r}) [not fitted]"

    # ------------------------------------------------------------------
    # Marginal effects
    # ------------------------------------------------------------------

    def get_margeff(
        self,
        at: str = "overall",
        kind: str = "conditional",
        atexog: dict | None = None,
    ) -> Any:
        """Marginal effects for the Heckman selection model.

        Parameters
        ----------
        at : {'overall', 'mean', 'median', 'zero'}, default ``'overall'``
            Point at which to evaluate the effect. ``'overall'`` averages the
            per-observation effects (the *AME*); the others evaluate at a
            single representative point (the *MEM* family).
        kind : {'conditional', 'unconditional', 'prob-selected', 'latent'}
            Which quantity to differentiate (see :mod:`_heckit_effects`):

            - ``'latent'``       — ``E[Y* | X]`` (returns the outcome slopes).
            - ``'conditional'``  — ``E[Y | X, Z, S=1]`` (default; includes the
              Mills-ratio correction on Z-side variables).
            - ``'unconditional'``— ``E[Y * S | X, Z]`` (treats unselected
              observations as ``Y = 0``; useful when zero is the natural
              "no-selection" value, e.g. wage for non-workers).
            - ``'prob-selected'``— ``P(S = 1 | Z)``.

        Returns
        -------
        MarginalEffects
            Object with attributes ``margeff``, ``margeff_se``, ``tvalues``,
            ``pvalues``, ``names`` (same dataclass used by the censored /
            truncated models). Standard errors are delta-method on the joint
            MLE covariance; for a two-step fit the standard errors are
            ``NaN`` (a bootstrap is the right tool there -- see
            :meth:`bootstrap`).
        """
        self._check_fitted()
        from ._heckit_effects import get_heckit_margeff

        return get_heckit_margeff(self, at=at, kind=kind, atexog=atexog)

    def ame(self, kind: str = "conditional") -> Any:
        """Shortcut for :meth:`get_margeff` at the **A**verage **M**arginal
        **E**ffect (i.e. ``at='overall'``)."""
        return self.get_margeff(at="overall", kind=kind)

    def mem(self, kind: str = "conditional") -> Any:
        """Shortcut for :meth:`get_margeff` at the sample-**M**ean
        **E**valuation point (``at='mean'``)."""
        return self.get_margeff(at="mean", kind=kind)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This model has not been fitted yet. Call `fit(y, X, Z)` first.")

    def _user_outcome_names(self) -> list[str] | None:
        if not self.outcome_feature_names_:
            return None
        if self.fit_intercept and self.outcome_feature_names_[0] == "const":
            return self.outcome_feature_names_[1:] or None
        return self.outcome_feature_names_

    def _user_selection_names(self) -> list[str] | None:
        if not self.selection_feature_names_:
            return None
        if self.fit_intercept and self.selection_feature_names_[0] == "const":
            return self.selection_feature_names_[1:] or None
        return self.selection_feature_names_
