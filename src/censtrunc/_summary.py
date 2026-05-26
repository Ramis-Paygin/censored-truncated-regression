"""Formatted text summary for fitted censored / truncated regression models.

Produces a `statsmodels`-style table with three blocks:

1. Header — model name, sample size, log-likelihood, LL-Null, LLR p-value,
   pseudo-R^2, AIC, BIC, censoring thresholds.
2. Coefficient table — beta_hat, std err, z, P>|z|, 95% confidence interval.
3. Footer — counts of observations in the censored / interior regions plus
   optimiser diagnostics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _fmt_threshold(x: float | None) -> str:
    if x is None:
        return "none"
    return f"{x:g}"


def _fmt_number(x: float, width: int = 10) -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN check
        return "—".rjust(width)
    if abs(x) < 1e-4 and x != 0:
        return f"{x:>{width}.3e}"
    return f"{x:>{width}.4f}"


def _coef_table(
    names: list[str],
    coef: list[float],
    bse: list[float],
    tvalues: list[float],
    pvalues: list[float],
    conf_int: list[tuple[float, float]],
) -> str:
    header = f"{'':<14}{'coef':>12}{'std err':>12}{'z':>10}{'P>|z|':>10}{'[0.025':>12}{'0.975]':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for nm, c, s, t, p, (lo, hi) in zip(names, coef, bse, tvalues, pvalues, conf_int):
        lines.append(
            f"{nm:<14}{c:>12.4f}{s:>12.4f}{t:>10.4f}{p:>10.4f}{lo:>12.4f}{hi:>12.4f}"
        )
    return "\n".join(lines)


def _two_column(left: tuple[str, Any], right: tuple[str, Any]) -> str:
    """Render two ``(label, value)`` pairs side-by-side."""
    l_label, l_val = left
    r_label, r_val = right
    if isinstance(l_val, float):
        l_val = _fmt_number(l_val).strip()
    if isinstance(r_val, float):
        r_val = _fmt_number(r_val).strip()
    return f"{l_label:<25}{str(l_val):<20}{r_label:<25}{str(r_val)}"


def format_summary_censored(model) -> str:
    """Render a ``CensoredRegression`` fit as a multi-line text summary."""
    thresholds_line = _two_column(
        ("Left threshold:", _fmt_threshold(model.left)),
        ("Right threshold:", _fmt_threshold(model.right)),
    )

    header_lines = [
        f"{'=' * 88}",
        f"{'Censored Regression Results':^88}",
        f"{'=' * 88}",
        _two_column(
            ("Dep. Variable:", "y"),
            ("No. Observations:", model.n_obs_),
        ),
        _two_column(
            ("Model:", "CensoredRegression"),
            ("Df Model:", model.params_.shape[0] - 1),  # minus sigma
        ),
        _two_column(
            ("Method:", "MLE"),
            ("Df Residuals:", model.n_obs_ - model.params_.shape[0]),
        ),
        _two_column(
            ("Date:", datetime.now().strftime("%a, %d %b %Y")),
            ("Log-Likelihood:", model.llf_),
        ),
        _two_column(
            ("Time:", datetime.now().strftime("%H:%M:%S")),
            ("LL-Null:", model.llnull_),
        ),
        _two_column(
            ("AIC:", model.aic_),
            ("LLR p-value:", model.llr_pvalue_),
        ),
        _two_column(
            ("BIC:", model.bic_),
            ("Pseudo R-squ.:", model.prsquared_),
        ),
        thresholds_line,
        f"{'=' * 88}",
    ]
    names = ["sigma"] + list(model.feature_names_)
    coef_table = _coef_table(
        names,
        list(model.params_),
        list(model.bse_),
        list(model.tvalues_),
        list(model.pvalues_),
        [(lo, hi) for lo, hi in model.conf_int_],
    )
    footer = [
        f"{'=' * 88}",
        f"Left-censored observations:    {model.n_left_censored_:>6} "
        f"({100 * model.n_left_censored_ / model.n_obs_:5.2f}%)",
        f"Right-censored observations:   {model.n_right_censored_:>6} "
        f"({100 * model.n_right_censored_ / model.n_obs_:5.2f}%)",
        f"Uncensored observations:       {model.n_uncensored_:>6} "
        f"({100 * model.n_uncensored_ / model.n_obs_:5.2f}%)",
        f"Optimiser converged:           {model.diagnostics_.converged}",
        f"{'=' * 88}",
    ]
    return "\n".join(header_lines) + "\n" + coef_table + "\n" + "\n".join(footer)


def format_summary_truncated(model) -> str:
    """Render a ``TruncatedRegression`` fit as a multi-line text summary."""
    header_lines = [
        f"{'=' * 88}",
        f"{'Truncated Regression Results':^88}",
        f"{'=' * 88}",
        _two_column(
            ("Dep. Variable:", "y"),
            ("No. Observations:", model.n_obs_),
        ),
        _two_column(
            ("Model:", "TruncatedRegression"),
            ("Df Model:", model.params_.shape[0] - 1),
        ),
        _two_column(
            ("Method:", "MLE"),
            ("Df Residuals:", model.n_obs_ - model.params_.shape[0]),
        ),
        _two_column(
            ("Date:", datetime.now().strftime("%a, %d %b %Y")),
            ("Log-Likelihood:", model.llf_),
        ),
        _two_column(
            ("Time:", datetime.now().strftime("%H:%M:%S")),
            ("LL-Null:", model.llnull_),
        ),
        _two_column(
            ("AIC:", model.aic_),
            ("LLR p-value:", model.llr_pvalue_),
        ),
        _two_column(
            ("BIC:", model.bic_),
            ("Pseudo R-squ.:", model.prsquared_),
        ),
        _two_column(
            ("Left truncation:", _fmt_threshold(model.left)),
            ("Right truncation:", _fmt_threshold(model.right)),
        ),
        f"{'=' * 88}",
    ]
    names = ["sigma"] + list(model.feature_names_)
    coef_table = _coef_table(
        names,
        list(model.params_),
        list(model.bse_),
        list(model.tvalues_),
        list(model.pvalues_),
        [(lo, hi) for lo, hi in model.conf_int_],
    )
    footer = [
        f"{'=' * 88}",
        f"Optimiser converged:           {model.diagnostics_.converged}",
        f"{'=' * 88}",
    ]
    return "\n".join(header_lines) + "\n" + coef_table + "\n" + "\n".join(footer)
