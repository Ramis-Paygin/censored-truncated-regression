# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-26

### Added
- `CensoredRegression`: maximum-likelihood estimator for censored normal regression
  with arbitrary `left` and `right` thresholds (generalised Type-1 Tobit).
- `TruncatedRegression`: maximum-likelihood estimator for truncated normal
  regression with arbitrary `left` and `right` thresholds.
- Three prediction modes — `latent`, `censored`, `truncated` — plus
  `predict_proba` returning region probabilities.
- Marginal effects through a single `get_margeff` method with an API mirroring
  statsmodels (`at` ∈ {overall, mean, median, zero}; `method` ∈ {dydx, eyex,
  dyex, eydx}; `dummy`/`count` for discrete regressors), with delta-method
  standard errors, a statsmodels-style `summary()`, and a `summary_frame()`.
  `ame()` / `mem()` remain as shorthands for `at='overall'` / `at='mean'`.
- Validation tests: in the no-censoring limit the estimator reproduces OLS
  (coefficients, MLE scale, and log-likelihood) to optimiser tolerance, checked
  against `statsmodels.OLS`; an optional comparison against R's `AER::tobit`
  and `truncreg` runs when R is available (`tests/test_r_reference.py`).
- Shared conditional-mean module (`_means.py`) used by both `predict` and the
  marginal-effects machinery, keeping predictions and effects consistent.
- Likelihood-ratio test (`lr_test`) for arbitrary nested models, plus an
  overall LR test against the intercept-only null model included in every
  fitted-model summary.
- statsmodels-style `summary()` with coefficient table, Wald inference,
  AIC/BIC, McFadden's pseudo-R², and per-region censoring counts.
- Three executed example notebooks in `examples/`:
  two-sided censoring, classical Tobit on Fair (1978), and truncated
  regression.
- Test suite (29 tests, including Monte Carlo consistency checks) and a CI
  workflow that runs them on Python 3.10-3.13 across Linux and macOS.
