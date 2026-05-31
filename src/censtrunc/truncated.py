"""Truncated normal regression with arbitrary left/right truncation thresholds.

The model is

    Y* = X'beta + e,   e | X ~ N(0, sigma^2),

with the observation rule that only ``Y*`` falling strictly inside the interval
``(L, R)`` is observed. Observations outside this interval are absent from the
sample entirely (unlike censoring, where they would pile up at the threshold).

The log-likelihood divides each normal density by the truncation probability:

    log f(y_i | L < y* < R) = log[(1/sigma) phi((y - X'beta)/sigma)]
                            - log[Phi((R - X'beta)/sigma) - Phi((L - X'beta)/sigma)]

For one-sided truncation either the left or right Phi-term drops out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from . import _likelihood as _llf
from . import _means
from ._utils import _prepare_design_matrix, _prepare_y, _resolve_thresholds


@dataclass
class _FitDiagnostics:
    converged: bool
    n_iterations: int
    optimiser_message: str
    final_grad_norm: float


@dataclass
class TruncatedRegression:
    """Maximum-likelihood estimation of a truncated normal regression.

    Parameters
    ----------
    left, right : float or None
        Truncation thresholds. At least one must be supplied. All ``y`` values
        passed to ``fit`` must lie strictly inside ``(left, right)``.
    fit_intercept : bool
        Whether to prepend an intercept column.
    optimizer, max_iter, tol :
        Forwarded to :func:`scipy.optimize.minimize`.

    Notes
    -----
    Unlike the censored case, the truncated log-likelihood is *not* generally
    concave under Olsen's reparameterisation, so optimisation is performed in
    the natural ``(sigma, beta)`` parameterisation. OLS on the truncated sample
    provides good starting values for the search.
    """

    left: float | None = None
    right: float | None = None
    fit_intercept: bool = True
    optimizer: str = "L-BFGS-B"
    max_iter: int = 200
    tol: float = 1e-8

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
    feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    diagnostics_: _FitDiagnostics | None = field(default=None, init=False, repr=False)
    _left: float = field(default=-np.inf, init=False, repr=False)
    _right: float = field(default=np.inf, init=False, repr=False)
    _has_left: bool = field(default=False, init=False, repr=False)
    _has_right: bool = field(default=False, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _X_train_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _y_train_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    # Truncated models support only latent and truncated conditional means.
    _valid_margeff_kinds = _means.TRUNCATED_KINDS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: list[str] | None = None,
    ) -> "TruncatedRegression":
        """Estimate the truncated regression by maximum likelihood."""
        if self.left is None and self.right is None:
            raise ValueError(
                "Truncated regression requires at least one of `left` or `right` to be set."
            )
        self._left, self._right, self._has_left, self._has_right = _resolve_thresholds(
            self.left, self.right
        )

        X_design, columns = _prepare_design_matrix(
            X, fit_intercept=self.fit_intercept, feature_names=feature_names
        )
        y_arr = _prepare_y(y, n_expected=X_design.shape[0])
        self.feature_names_ = columns
        self._X_train_ = X_design
        self._y_train_ = y_arr

        # Validate that y is strictly inside the truncation interval
        if self._has_left and np.any(y_arr <= self._left):
            raise ValueError(
                f"All y values must satisfy y > left ({self._left}); found {(y_arr <= self._left).sum()} violations."
            )
        if self._has_right and np.any(y_arr >= self._right):
            raise ValueError(
                f"All y values must satisfy y < right ({self._right}); found {(y_arr >= self._right).sum()} violations."
            )

        params_hat = self._fit_optimize(X_design, y_arr)
        sigma_hat = float(params_hat[0])
        beta_hat = params_hat[1:].copy()

        cov = self._covariance_matrix(params_hat, X_design, y_arr)
        bse = np.sqrt(np.diag(cov))

        llf = -_llf.neg_loglik_truncated(
            params_hat, X_design, y_arr, self._left, self._right,
            self._has_left, self._has_right,
        )

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
        self.n_obs_ = int(X_design.shape[0])
        k = params_hat.shape[0]
        self.aic_ = -2 * llf + 2 * k
        self.bic_ = -2 * llf + k * np.log(self.n_obs_)
        self._fitted = True

        if self.fit_intercept or X_design.shape[1] > 1:
            null_llf = self._fit_null_model(X_design, y_arr)
            self.llnull_ = null_llf
            self.llr_ = 2.0 * (self.llf_ - null_llf)
            df_model = X_design.shape[1] - (1 if self.fit_intercept else 0)
            if df_model > 0:
                self.llr_pvalue_ = float(chi2.sf(self.llr_, df_model))
                self.prsquared_ = 1.0 - self.llf_ / null_llf if null_llf != 0 else np.nan

        return self

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _fit_optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # OLS starting values
        beta_init, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta_init
        dof = max(X.shape[0] - X.shape[1], 1)
        sigma_init = float(np.sqrt(max(resid @ resid / dof, 1e-6)))
        x0 = np.concatenate([[sigma_init], beta_init])
        bounds = [(1e-8, None)] + [(None, None)] * beta_init.shape[0]

        res = minimize(
            fun=_llf.neg_loglik_truncated,
            x0=x0,
            args=(X, y, self._left, self._right, self._has_left, self._has_right),
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
        return res.x

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    @staticmethod
    def _numerical_hessian(f: Any, x: np.ndarray, args: tuple) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        step = np.cbrt(np.finfo(float).eps)
        h = step * np.maximum(np.abs(x), 1.0)
        H = np.zeros((n, n))
        fx = f(x, *args)
        for i in range(n):
            xp = x.copy(); xp[i] += h[i]
            xm = x.copy(); xm[i] -= h[i]
            H[i, i] = (f(xp, *args) - 2 * fx + f(xm, *args)) / (h[i] ** 2)
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
        return 0.5 * (H + H.T)

    def _covariance_matrix(
        self, params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        H = self._numerical_hessian(
            _llf.neg_loglik_truncated,
            params,
            args=(X, y, self._left, self._right, self._has_left, self._has_right),
        )
        try:
            return np.linalg.inv(H)
        except np.linalg.LinAlgError:  # pragma: no cover
            return np.linalg.pinv(H)

    # ------------------------------------------------------------------
    # Null model
    # ------------------------------------------------------------------

    def _fit_null_model(self, X: np.ndarray, y: np.ndarray) -> float:
        X_null = np.ones((X.shape[0], 1))
        beta_init = np.array([float(y.mean())])
        resid = y - X_null @ beta_init
        sigma_init = float(np.sqrt(max(resid @ resid / max(X_null.shape[0] - 1, 1), 1e-6)))
        x0 = np.concatenate([[sigma_init], beta_init])
        bounds = [(1e-8, None), (None, None)]
        res = minimize(
            fun=_llf.neg_loglik_truncated,
            x0=x0,
            args=(X_null, y, self._left, self._right, self._has_left, self._has_right),
            method=self.optimizer,
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        return -float(res.fun)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def get_margeff(
        self,
        at: str = "overall",
        method: str = "dydx",
        kind: str = "truncated",
        atexog: dict | None = None,
        dummy: bool = False,
        count: bool = False,
    ):
        """Marginal effects (statsmodels-style ``get_margeff``).

        For the truncated model ``kind`` may be ``'latent'`` (effect on the
        population mean X'beta) or ``'truncated'`` (effect on E[Y | L<Y<R]).
        See :func:`censtrunc.effects.get_margeff` for the full option reference.
        """
        from .effects import get_margeff as _get_margeff

        return _get_margeff(
            self, at=at, method=method, kind=kind, atexog=atexog, dummy=dummy, count=count
        )

    def ame(self, kind: str = "truncated", method: str = "dydx", dummy: bool = False, count: bool = False):
        """Average Marginal Effects — shorthand for ``get_margeff(at='overall')``."""
        return self.get_margeff(at="overall", method=method, kind=kind, dummy=dummy, count=count)

    def mem(self, kind: str = "truncated", method: str = "dydx", dummy: bool = False, count: bool = False):
        """Marginal Effects at the Mean — shorthand for ``get_margeff(at='mean')``."""
        return self.get_margeff(at="mean", method=method, kind=kind, dummy=dummy, count=count)

    # ------------------------------------------------------------------
    # LR test from linear restrictions (statsmodels-style hypothesis strings)
    # ------------------------------------------------------------------

    def lr_test(self, hypotheses):
        """Likelihood-ratio test for linear restrictions on this truncated model.

        Same API as :meth:`CensoredRegression.lr_test`. See its docstring for
        accepted hypothesis formats.
        """
        from ._restrictions import parse_hypotheses
        from .inference import LRTestResult

        self._check_fitted()
        param_names = ["sigma"] + list(self.feature_names_)
        R, r = parse_hypotheses(hypotheses, param_names)
        if R.shape[0] >= self.params_.shape[0]:
            raise ValueError(
                f"Too many restrictions ({R.shape[0]}) for a {self.params_.shape[0]}-parameter model"
            )
        restricted_llf = self._fit_restricted_loglik(R, r)
        stat = 2.0 * (float(self.llf_) - float(restricted_llf))
        if stat < 0:
            import warnings

            warnings.warn(
                f"LR statistic is negative ({stat:.3g}); restricted MLE likely did not converge "
                "to a true constrained optimum.",
                RuntimeWarning,
                stacklevel=2,
            )
        df = int(R.shape[0])
        p_value = float(chi2.sf(max(stat, 0.0), df))
        return LRTestResult(
            statistic=stat,
            df=df,
            p_value=p_value,
            ll_full=float(self.llf_),
            ll_restricted=float(restricted_llf),
            n_obs=int(self.n_obs_),
        )

    def _fit_restricted_loglik(self, R: np.ndarray, r: np.ndarray) -> float:
        """Maximise the truncated log-likelihood subject to ``R @ params == r``."""
        from scipy.optimize import LinearConstraint, minimize

        theta0 = self.params_.copy()
        residual = r - R @ theta0
        theta_start = theta0 + np.linalg.pinv(R) @ residual
        if theta_start[0] <= 1e-6:
            from scipy.linalg import null_space

            N = null_space(R)
            if N.shape[1] > 0:
                k = int(np.argmax(np.abs(N[0])))
                if abs(N[0, k]) > 1e-10:
                    step = (max(self.sigma_, 1e-3) - theta_start[0]) / N[0, k]
                    theta_start = theta_start + step * N[:, k]
            if theta_start[0] <= 1e-6:
                theta_start[0] = max(self.sigma_, 1e-3)

        bounds = [(1e-8, None)] + [(None, None)] * (theta_start.shape[0] - 1)
        constraint = LinearConstraint(R, r, r)
        args = (
            self._X_train_,
            self._y_train_,
            self._left,
            self._right,
            self._has_left,
            self._has_right,
        )
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore",
                message="delta_grad == 0.0",
                category=UserWarning,
                module=r"scipy\.optimize\._differentiable_functions",
            )
            res = minimize(
                fun=_llf.neg_loglik_truncated,
                x0=theta_start,
                args=args,
                method="trust-constr",
                bounds=bounds,
                constraints=[constraint],
                options={"maxiter": max(self.max_iter, 500), "xtol": self.tol, "verbose": 0},
            )
        if not res.success:
            import warnings

            warnings.warn(
                f"Restricted MLE did not converge cleanly: {res.message}",
                RuntimeWarning,
                stacklevel=3,
            )
        return -float(res.fun)

    def summary(self) -> str:
        """Return a multi-line text summary of the fitted model."""
        from ._summary import format_summary_truncated

        self._check_fitted()
        return format_summary_truncated(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self._fitted:
            return self.summary()
        return f"TruncatedRegression(left={self.left!r}, right={self.right!r}, fit_intercept={self.fit_intercept!r}) [not fitted]"

    def predict(self, X: Any, kind: str = "truncated") -> np.ndarray:
        """Predict conditional expectations for the truncated model.

        Parameters
        ----------
        X : array-like
            New design matrix.
        kind : {'latent', 'truncated'}, default ``'truncated'``
            ``'latent'`` returns ``X'beta`` (the untruncated population mean).
            ``'truncated'`` returns ``E[Y | X, L < Y < R]``.
        """
        self._check_fitted()
        if kind not in _means.TRUNCATED_KINDS:
            raise ValueError(
                f"Unknown kind: {kind!r}; expected one of {_means.TRUNCATED_KINDS}."
            )
        X_design, _ = _prepare_design_matrix(
            X, fit_intercept=self.fit_intercept,
            feature_names=self._user_feature_names(),
        )
        return _means.conditional_mean(
            self.coef_, self.sigma_, X_design,
            self._left, self._right, self._has_left, self._has_right, kind,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _user_feature_names(self) -> list[str] | None:
        if not self.feature_names_:
            return None
        if self.fit_intercept and self.feature_names_[0] == "const":
            return self.feature_names_[1:] or None
        return self.feature_names_

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This model has not been fitted yet. Call `fit(X, y)` first.")
