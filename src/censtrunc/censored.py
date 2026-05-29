"""Censored normal regression (generalised Type-1 Tobit) with arbitrary left/right thresholds.

The model is

    Y* = X'beta + e,   e | X ~ N(0, sigma^2),

    Y =  L              if Y* <= L,
         Y*             if L < Y* < R,
         R              if Y* >= R.

Either ``L`` or ``R`` (or both) may be set, allowing one- or two-sided censoring.
The classical Tobit corresponds to ``L = 0``, ``R = None``.

Estimation is by maximum likelihood. Internally the optimisation is performed in
Olsen's reparameterisation ``(gamma, nu) = (beta/sigma, 1/sigma)``, where the
log-likelihood is globally concave (Olsen 1978), guaranteeing convergence of
gradient-based optimisers to the unique MLE. Standard errors are obtained from
the observed information matrix in the natural ``(beta, sigma)`` parameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from . import _likelihood as _llf
from . import _means
from ._utils import (
    _classify_observations,
    _prepare_design_matrix,
    _prepare_y,
    _resolve_thresholds,
)


@dataclass
class _FitDiagnostics:
    """Diagnostic information from the optimiser."""

    converged: bool
    n_iterations: int
    optimiser_message: str
    final_grad_norm: float


@dataclass
class CensoredRegression:
    """Maximum-likelihood Tobit-type regression with arbitrary censoring thresholds.

    Parameters
    ----------
    left : float or None, default ``None``
        Left censoring threshold ``L``. ``None`` means no left censoring.
    right : float or None, default ``None``
        Right censoring threshold ``R``. ``None`` means no right censoring.
    fit_intercept : bool, default ``True``
        If ``True``, an intercept column of ones is prepended to ``X``.
    method : {'olsen', 'standard'}, default ``'olsen'``
        Parameterisation used for optimisation. ``'olsen'`` (recommended) gives
        a globally concave log-likelihood. ``'standard'`` optimises directly in
        ``(beta, sigma)`` and may fail to converge on harder problems.
    optimizer : str, default ``'L-BFGS-B'``
        Name of the ``scipy.optimize.minimize`` method.
    max_iter : int, default ``200``
        Maximum number of optimiser iterations.
    tol : float, default ``1e-8``
        Convergence tolerance on the gradient norm.

    Attributes
    ----------
    params_ : ndarray of shape (k+1,)
        Estimated ``[sigma, beta_1, ..., beta_k]`` after ``fit``.
    coef_ : ndarray of shape (k,)
        Estimated regression coefficients (``beta``).
    sigma_ : float
        Estimated scale parameter.
    bse_ : ndarray
        Standard errors of ``[sigma, beta_1, ..., beta_k]``.
    tvalues_, pvalues_, conf_int_ :
        Wald-type test statistics, two-sided p-values, and 95% confidence intervals.
    llf_, llnull_ : float
        Log-likelihood of the fitted model and of the intercept-only null model.
    llr_, llr_pvalue_ : float
        Overall likelihood-ratio test statistic and p-value vs. the null model.
    prsquared_ : float
        McFadden's pseudo-R^2.
    aic_, bic_ : float
        Akaike and Bayesian information criteria.
    n_obs_, n_left_censored_, n_right_censored_, n_uncensored_ : int
        Sample size and counts of observations in each region.
    """

    left: float | None = None
    right: float | None = None
    fit_intercept: bool = True
    method: str = "olsen"
    optimizer: str = "L-BFGS-B"
    max_iter: int = 200
    tol: float = 1e-8

    # Fitted attributes — populated by ``fit``. ``init=False`` keeps them out of
    # the constructor signature.
    params_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    coef_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    sigma_: float = field(default=np.nan, init=False, repr=False)
    bse_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    cov_params_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    tvalues_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    pvalues_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    conf_int_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    llf_: float = field(default=np.nan, init=False, repr=False)
    llnull_: float = field(default=np.nan, init=False, repr=False)
    llr_: float = field(default=np.nan, init=False, repr=False)
    llr_pvalue_: float = field(default=np.nan, init=False, repr=False)
    prsquared_: float = field(default=np.nan, init=False, repr=False)
    aic_: float = field(default=np.nan, init=False, repr=False)
    bic_: float = field(default=np.nan, init=False, repr=False)
    n_obs_: int = field(default=0, init=False, repr=False)
    n_left_censored_: int = field(default=0, init=False, repr=False)
    n_right_censored_: int = field(default=0, init=False, repr=False)
    n_uncensored_: int = field(default=0, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    diagnostics_: _FitDiagnostics | None = field(default=None, init=False, repr=False)
    _X_train_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _left: float = field(default=-np.inf, init=False, repr=False)
    _right: float = field(default=np.inf, init=False, repr=False)
    _has_left: bool = field(default=False, init=False, repr=False)
    _has_right: bool = field(default=False, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: list[str] | None = None,
    ) -> "CensoredRegression":
        """Estimate the model parameters by maximum likelihood.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Design matrix of regressors. If ``fit_intercept`` is ``True`` an
            intercept column is prepended automatically.
        y : array-like of shape (n,)
            Observed (possibly censored) dependent variable. Values must lie in
            the closed interval ``[left, right]``; values at the thresholds are
            treated as censored observations.
        feature_names : list of str, optional
            Names of the regressor columns. If ``X`` is a pandas DataFrame the
            column names are used automatically; this argument lets callers
            override them or supply names for plain numpy input.

        Returns
        -------
        self : CensoredRegression
            The fitted estimator (allowing ``model.fit(X, y).predict(X_new)``).
        """
        return self._fit(X, y, feature_names=feature_names, refit_null=True)

    def _fit(
        self,
        X: Any,
        y: Any,
        *,
        feature_names: list[str] | None,
        refit_null: bool,
    ) -> "CensoredRegression":
        # --- input handling ---------------------------------------------------
        if self.left is None and self.right is None:
            raise ValueError(
                "Both `left` and `right` are None: there is no censoring. "
                "Use OLS instead, or supply at least one threshold."
            )

        self._left, self._right, self._has_left, self._has_right = _resolve_thresholds(
            self.left, self.right
        )

        X_design, columns = _prepare_design_matrix(
            X, fit_intercept=self.fit_intercept, feature_names=feature_names
        )
        y_arr = _prepare_y(y, n_expected=X_design.shape[0])
        self.feature_names_ = columns

        mask_left, mask_right, mask_free = _classify_observations(
            y_arr, self._left, self._right, self._has_left, self._has_right
        )
        if not mask_free.any():
            raise ValueError(
                "All observations are at the censoring thresholds; the MLE is undefined."
            )

        # --- starting values from OLS on the uncensored subsample -------------
        beta_init, sigma_init = self._ols_starting_values(
            X_design[mask_free], y_arr[mask_free]
        )

        # --- optimise ---------------------------------------------------------
        if self.method == "olsen":
            params_hat = self._fit_olsen(
                X_design, y_arr, mask_left, mask_right, mask_free, beta_init, sigma_init
            )
        elif self.method == "standard":
            params_hat = self._fit_standard(
                X_design, y_arr, mask_left, mask_right, mask_free, beta_init, sigma_init
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {self.method!r}")

        sigma_hat = float(params_hat[0])
        beta_hat = params_hat[1:].copy()

        # --- standard errors via observed information in (beta, sigma) --------
        cov = self._covariance_matrix(
            params_hat, X_design, y_arr, mask_left, mask_right, mask_free
        )
        bse = np.sqrt(np.diag(cov))

        # --- log-likelihood at MLE -------------------------------------------
        llf = -_llf.neg_loglik_censored(
            params_hat, X_design, y_arr, self._left, self._right,
            mask_left, mask_right, mask_free,
        )

        # --- store ------------------------------------------------------------
        self.params_ = params_hat
        self.coef_ = beta_hat
        self.sigma_ = sigma_hat
        self.cov_params_ = cov
        self.bse_ = bse
        # Guard against zero or near-zero standard errors (degenerate covariance).
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tvalues_ = np.where(bse > 0, params_hat / bse, np.nan)
        self.pvalues_ = 2.0 * (1.0 - norm.cdf(np.abs(self.tvalues_)))
        z_crit = norm.ppf(0.975)
        self.conf_int_ = np.column_stack(
            [params_hat - z_crit * bse, params_hat + z_crit * bse]
        )
        self.llf_ = float(llf)
        k_params = params_hat.shape[0]
        self.n_obs_ = int(X_design.shape[0])
        self.aic_ = -2.0 * llf + 2.0 * k_params
        self.bic_ = -2.0 * llf + k_params * np.log(self.n_obs_)

        self.n_left_censored_ = int(mask_left.sum())
        self.n_right_censored_ = int(mask_right.sum())
        self.n_uncensored_ = int(mask_free.sum())
        # Cache the training design matrix for default AME computation
        self._X_train_ = X_design
        self._fitted = True

        # --- null model for LR test and pseudo R^2 ----------------------------
        if refit_null and (self.fit_intercept or X_design.shape[1] > 1):
            null_llf = self._fit_null_model(X_design, y_arr, mask_left, mask_right, mask_free)
            self.llnull_ = null_llf
            self.llr_ = 2.0 * (self.llf_ - null_llf)
            # degrees of freedom for the LR test = number of slope coefficients
            # (everything beyond the intercept and sigma)
            df_model = X_design.shape[1] - (1 if self.fit_intercept else 0)
            df_model = max(df_model, 0)
            if df_model > 0:
                self.llr_pvalue_ = float(chi2.sf(self.llr_, df_model))
                self.prsquared_ = 1.0 - self.llf_ / null_llf if null_llf != 0 else np.nan

        return self

    # ------------------------------------------------------------------
    # Optimisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_starting_values(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Initial ``(beta, sigma)`` from OLS on the uncensored subsample."""
        # least-squares via QR for stability
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        # use unbiased estimator; fall back to small positive value if degenerate
        dof = max(X.shape[0] - X.shape[1], 1)
        sigma = float(np.sqrt(max(resid @ resid / dof, 1e-6)))
        return beta, sigma

    def _fit_olsen(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask_left: np.ndarray,
        mask_right: np.ndarray,
        mask_free: np.ndarray,
        beta_init: np.ndarray,
        sigma_init: float,
    ) -> np.ndarray:
        """Optimise in Olsen parameterisation; return ``[sigma, beta]`` after back-transform."""
        nu0 = 1.0 / sigma_init
        gamma0 = beta_init / sigma_init
        x0 = np.concatenate([[nu0], gamma0])

        bounds = [(1e-8, None)] + [(None, None)] * gamma0.shape[0]

        res = minimize(
            fun=_llf.neg_loglik_censored_olsen,
            x0=x0,
            jac=_llf.neg_loglik_grad_censored_olsen,
            args=(X, y, self._left, self._right, mask_left, mask_right, mask_free),
            method=self.optimizer,
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        self.diagnostics_ = _FitDiagnostics(
            converged=bool(res.success),
            n_iterations=int(getattr(res, "nit", -1)),
            optimiser_message=str(res.message),
            final_grad_norm=float(np.linalg.norm(res.jac) if res.jac is not None else np.nan),
        )

        if not res.success:
            import warnings
            warnings.warn(
                f"Optimiser did not converge cleanly: {res.message}",
                RuntimeWarning,
                stacklevel=3,
            )

        nu_hat = float(res.x[0])
        gamma_hat = res.x[1:]
        sigma_hat = 1.0 / nu_hat
        beta_hat = gamma_hat * sigma_hat
        return np.concatenate([[sigma_hat], beta_hat])

    def _fit_standard(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask_left: np.ndarray,
        mask_right: np.ndarray,
        mask_free: np.ndarray,
        beta_init: np.ndarray,
        sigma_init: float,
    ) -> np.ndarray:
        x0 = np.concatenate([[sigma_init], beta_init])
        bounds = [(1e-8, None)] + [(None, None)] * beta_init.shape[0]
        res = minimize(
            fun=_llf.neg_loglik_censored,
            x0=x0,
            args=(X, y, self._left, self._right, mask_left, mask_right, mask_free),
            method=self.optimizer,
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        self.diagnostics_ = _FitDiagnostics(
            converged=bool(res.success),
            n_iterations=int(getattr(res, "nit", -1)),
            optimiser_message=str(res.message),
            final_grad_norm=float(np.linalg.norm(res.jac) if res.jac is not None else np.nan),
        )
        return res.x

    # ------------------------------------------------------------------
    # Covariance / standard errors
    # ------------------------------------------------------------------

    @staticmethod
    def _numerical_hessian(
        f: Any, x: np.ndarray, args: tuple, step: float | None = None
    ) -> np.ndarray:
        """Central-difference Hessian of a scalar function ``f(x, *args)``."""
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if step is None:
            step = np.cbrt(np.finfo(float).eps)
        h = step * np.maximum(np.abs(x), 1.0)
        H = np.zeros((n, n))
        # Diagonal: f(x + h e_i) - 2 f(x) + f(x - h e_i)  / h^2
        fx = f(x, *args)
        for i in range(n):
            x_p = x.copy(); x_p[i] += h[i]
            x_m = x.copy(); x_m[i] -= h[i]
            H[i, i] = (f(x_p, *args) - 2 * fx + f(x_m, *args)) / (h[i] ** 2)
        # Off-diagonals: 4-point stencil
        for i in range(n):
            for j in range(i + 1, n):
                xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
                H[i, j] = (f(xpp, *args) - f(xpm, *args) - f(xmp, *args) + f(xmm, *args)) / (
                    4 * h[i] * h[j]
                )
                H[j, i] = H[i, j]
        # Symmetrise to defend against round-off
        return 0.5 * (H + H.T)

    def _covariance_matrix(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        mask_left: np.ndarray,
        mask_right: np.ndarray,
        mask_free: np.ndarray,
    ) -> np.ndarray:
        """Observed-information covariance matrix in (sigma, beta) parameterisation.

        ``params`` are the MLE estimates ``[sigma_hat, beta_hat]`` in the natural
        parameterisation, obtained after back-transforming from Olsen's ``(nu, gamma)``.
        The Hessian is evaluated at this point to obtain standard errors.
        """
        H = self._numerical_hessian(
            _llf.neg_loglik_censored,
            params,
            args=(X, y, self._left, self._right, mask_left, mask_right, mask_free),
        )
        # H is the Hessian of *negative* log-likelihood, which is itself the
        # observed information matrix; covariance = inverse.
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:  # pragma: no cover
            cov = np.linalg.pinv(H)
        return cov

    # ------------------------------------------------------------------
    # Null-model log-likelihood (intercept only) for LR test and pseudo-R^2
    # ------------------------------------------------------------------

    def _fit_null_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask_left: np.ndarray,
        mask_right: np.ndarray,
        mask_free: np.ndarray,
    ) -> float:
        """Fit an intercept-only model and return its log-likelihood."""
        # Build a 1-column intercept design matrix matching the orientation used
        # in the main fit (i.e. the constant column already present in X if
        # ``fit_intercept`` is True; otherwise a fresh column of ones).
        X_null = np.ones((X.shape[0], 1))
        beta_init, sigma_init = self._ols_starting_values(X_null[mask_free], y[mask_free])
        nu0 = 1.0 / sigma_init
        gamma0 = beta_init / sigma_init
        x0 = np.concatenate([[nu0], gamma0])
        bounds = [(1e-8, None), (None, None)]
        res = minimize(
            fun=_llf.neg_loglik_censored_olsen,
            x0=x0,
            jac=_llf.neg_loglik_grad_censored_olsen,
            args=(X_null, y, self._left, self._right, mask_left, mask_right, mask_free),
            method=self.optimizer,
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        nu_hat = float(res.x[0])
        gamma_hat = res.x[1:]
        sigma_hat = 1.0 / nu_hat
        beta_hat = gamma_hat * sigma_hat
        params_null = np.concatenate([[sigma_hat], beta_hat])
        return -_llf.neg_loglik_censored(
            params_null, X_null, y, self._left, self._right, mask_left, mask_right, mask_free,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: Any, kind: str = "censored") -> np.ndarray:
        """Predict conditional expectations.

        Parameters
        ----------
        X : array-like of shape (m, k)
            New design matrix (same convention as in ``fit``).
        kind : {'latent', 'censored', 'truncated'}, default ``'censored'``
            - ``'latent'`` returns ``X'beta`` (uncensored mean).
            - ``'censored'`` returns ``E[Y | X]`` accounting for the threshold mass.
            - ``'truncated'`` returns ``E[Y | X, L < Y < R]`` (mean over the interior).
        """
        self._check_fitted()
        if kind not in _means.CENSORED_KINDS:
            raise ValueError(
                f"Unknown kind: {kind!r}; expected one of {_means.CENSORED_KINDS}."
            )
        X_design, _ = _prepare_design_matrix(
            X, fit_intercept=self.fit_intercept,
            feature_names=self._user_feature_names(),
        )
        return _means.conditional_mean(
            self.coef_, self.sigma_, X_design,
            self._left, self._right, self._has_left, self._has_right, kind,
        )

    # Which conditional means / probabilities support marginal effects
    # (see effects.get_margeff): the three means plus the three region
    # probabilities (prob-left / prob-interior / prob-right).
    _valid_margeff_kinds = _means.MARGEFF_CENSORED_KINDS

    def get_margeff(
        self,
        at: str = "overall",
        method: str = "dydx",
        kind: str = "censored",
        atexog: dict | None = None,
        dummy: bool = False,
        count: bool = False,
    ):
        """Marginal effects, with an API mirroring statsmodels' ``get_margeff``.

        Parameters
        ----------
        at : {'overall', 'mean', 'median', 'zero'}, default ``'overall'``
            Where to evaluate the effect. ``'overall'`` averages the
            per-observation effects (AME); ``'mean'`` evaluates at the mean
            regressor (MEM).
        method : {'dydx', 'eyex', 'dyex', 'eydx'}, default ``'dydx'``
            Derivative (``dydx``) or elasticity / semi-elasticity.
        kind : {'latent', 'censored', 'truncated'}, default ``'censored'``
            Which conditional mean the effect refers to.
        atexog : dict, optional
            ``{design_column_index: value}`` overrides for the evaluation point.
        dummy : bool, default ``False``
            Treat binary regressors with a discrete difference.
        count : bool, default ``False``
            Treat integer-valued regressors with a discrete difference.

        Returns
        -------
        MarginalEffects
            Has ``.summary()``, ``.summary_frame()``, ``.margeff``, ``.margeff_se``,
            ``.pvalues`` (and backward-compatible ``.effects`` / ``.to_dataframe()``).
        """
        from .effects import get_margeff as _get_margeff

        return _get_margeff(
            self, at=at, method=method, kind=kind, atexog=atexog, dummy=dummy, count=count
        )

    def ame(self, kind: str = "censored", method: str = "dydx", dummy: bool = False, count: bool = False):
        """Average Marginal Effects — shorthand for ``get_margeff(at='overall')``."""
        return self.get_margeff(at="overall", method=method, kind=kind, dummy=dummy, count=count)

    def mem(self, kind: str = "censored", method: str = "dydx", dummy: bool = False, count: bool = False):
        """Marginal Effects at the Mean — shorthand for ``get_margeff(at='mean')``."""
        return self.get_margeff(at="mean", method=method, kind=kind, dummy=dummy, count=count)

    def summary(self) -> str:
        """Return a multi-line text summary of the fitted model.

        The format mirrors that of :mod:`statsmodels`: a header block with model
        metadata and overall fit statistics (Log-Likelihood, LL-Null, LLR
        p-value, McFadden's pseudo-R^2, AIC, BIC), a coefficient table with
        Wald z-statistics and 95% confidence intervals, and a footer with
        per-region censoring counts.
        """
        from ._summary import format_summary_censored

        self._check_fitted()
        return format_summary_censored(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self._fitted:
            return self.summary()
        return f"CensoredRegression(left={self.left!r}, right={self.right!r}, fit_intercept={self.fit_intercept!r}) [not fitted]"

    def predict_proba(self, X: Any) -> dict[str, np.ndarray]:
        """Return probabilities of the three regions (left, interior, right).

        Output is a dict with keys ``'left'``, ``'interior'``, ``'right'`` mapping
        to arrays of length ``m``.
        """
        self._check_fitted()
        X_design, _ = _prepare_design_matrix(
            X, fit_intercept=self.fit_intercept,
            feature_names=self._user_feature_names(),
        )
        return _means.region_probabilities(
            self.coef_, self.sigma_, X_design,
            self._left, self._right, self._has_left, self._has_right,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _user_feature_names(self) -> list[str] | None:
        """Feature names excluding the auto-added intercept (or None if defaults)."""
        if not self.feature_names_:
            return None
        if self.fit_intercept and self.feature_names_[0] == "const":
            return self.feature_names_[1:] or None
        return self.feature_names_

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This model has not been fitted yet. Call `fit(X, y)` first.")
