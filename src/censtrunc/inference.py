"""Likelihood-ratio test for nested censored/truncated regression models.

The two models compared must be **nested** in the strict statistical sense:
the restricted model is obtained from the unrestricted model by setting a subset
of its parameters to fixed values (typically zero). Equivalent number of
observations is assumed; the test statistic is

    LR = 2 * (loglik_unrestricted - loglik_restricted)

which under the null (the restrictions hold) is asymptotically distributed as
``chi^2_q``, with ``q`` equal to the number of restrictions imposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass
class LRTestResult:
    """Result of a likelihood-ratio test.

    Attributes
    ----------
    statistic : float
        The LR statistic, ``2 * (ll_full - ll_restricted)``.
    df : int
        Degrees of freedom (number of restrictions).
    p_value : float
        Right-tail p-value under ``chi^2_df``.
    ll_full : float
        Log-likelihood of the unrestricted (full) model.
    ll_restricted : float
        Log-likelihood of the restricted model.
    n_obs : int
        Sample size shared by both models.
    """

    statistic: float
    df: int
    p_value: float
    ll_full: float
    ll_restricted: float
    n_obs: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Likelihood-ratio test\n"
            f"  LR statistic : {self.statistic:.4f}\n"
            f"  df           : {self.df}\n"
            f"  p-value      : {self.p_value:.4g}\n"
            f"  log-lik full : {self.ll_full:.4f}\n"
            f"  log-lik rest.: {self.ll_restricted:.4f}\n"
            f"  n            : {self.n_obs}"
        )


def lr_test(model_full, model_restricted) -> LRTestResult:
    """Likelihood-ratio test of two nested censored/truncated regression models.

    Both arguments must be fitted estimators exposing the attributes
    ``llf_``, ``params_``, and ``n_obs_`` (i.e., :class:`CensoredRegression`
    or :class:`TruncatedRegression`).

    Parameters
    ----------
    model_full : fitted estimator
        The larger (unrestricted) model. Must have at least as many parameters
        as ``model_restricted``.
    model_restricted : fitted estimator
        The smaller (restricted, nested) model.

    Returns
    -------
    LRTestResult

    Raises
    ------
    ValueError
        If the models are not fitted, were fitted on different sample sizes,
        or the "restricted" model has at least as many parameters as the "full".
    RuntimeWarning (via warnings)
        If the LR statistic is negative — typically a sign of poor convergence
        or a misspecified nesting relation.
    """
    for name, m in (("model_full", model_full), ("model_restricted", model_restricted)):
        if not getattr(m, "_fitted", False):
            raise ValueError(f"{name} has not been fitted; call .fit(X, y) first.")
    if model_full.n_obs_ != model_restricted.n_obs_:
        raise ValueError(
            f"Sample sizes differ: full has {model_full.n_obs_}, "
            f"restricted has {model_restricted.n_obs_}. LR test requires identical samples."
        )
    df = int(model_full.params_.shape[0] - model_restricted.params_.shape[0])
    if df <= 0:
        raise ValueError(
            f"`model_full` has {model_full.params_.shape[0]} parameters and "
            f"`model_restricted` has {model_restricted.params_.shape[0]}; the full "
            "model must have strictly more parameters for a proper LR test."
        )

    stat = 2.0 * (float(model_full.llf_) - float(model_restricted.llf_))
    if stat < 0:
        import warnings

        warnings.warn(
            f"LR statistic is negative ({stat:.4g}). This typically indicates "
            "poor convergence in one of the models, or that the models are not "
            "properly nested. Results below are still reported.",
            RuntimeWarning,
            stacklevel=2,
        )
    p = float(chi2.sf(max(stat, 0.0), df))
    return LRTestResult(
        statistic=stat,
        df=df,
        p_value=p,
        ll_full=float(model_full.llf_),
        ll_restricted=float(model_restricted.llf_),
        n_obs=int(model_full.n_obs_),
    )
