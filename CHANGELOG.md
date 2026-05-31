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
- `HeckitRegression`: Heckman (1979) sample-selection regression with both the
  two-step estimator (probit on `Z` + OLS of `Y` on `(X, lambda_hat)` on the
  selected subsample) and the joint maximum-likelihood estimator. Includes a
  `bootstrap()` helper for proper two-step standard errors and a `predict()`
  with three kinds: `'outcome'` (`X'beta`), `'selection_prob'` (`Phi(Z'gamma)`),
  and `'conditional'` (`E[Y | X, Z, S=1] = X'beta + rho*sigma*lambda(Z'gamma)`).
- A unified `predict()` that returns any combination of **six** quantities
  via a compact one-letter ``kind`` argument:

      h = hidden (latent) mean,        l = P(Y = L | X),
      c = censored conditional mean,   m = P(L < Y < R | X),
      t = truncated conditional mean,  r = P(Y = R | X).

  ``predict(X)`` (no ``kind``) returns a ``pandas.DataFrame`` with all six
  columns; ``predict(X, kind='lmr')`` returns only the three probabilities;
  ``predict(X, kind='h')`` returns a 1-D ``ndarray``. Long names
  (``'latent'``, ``'censored'``, ``'truncated'``) remain valid and continue to
  return 1-D arrays. The truncated model exposes only the ``h`` and ``t``
  quantities (default ``'ht'``). ``predict_proba`` is still available for the
  legacy dict-of-arrays interface.
- Marginal effects through a single `get_margeff` method with an API mirroring
  statsmodels (`at` ∈ {overall, mean, median, zero}; `method` ∈ {dydx, eyex,
  dyex, eydx}; `dummy`/`count` for discrete regressors), with delta-method
  standard errors, a statsmodels-style `summary()`, and a `summary_frame()`.
  `ame()` / `mem()` remain as shorthands for `at='overall'` / `at='mean'`.
  Beyond the conditional means, `kind` also accepts `prob-left`,
  `prob-interior`, and `prob-right` to report marginal effects on the
  probability of each region (P(Y=L), P(L<Y<R), P(Y=R)); these three effects
  sum to zero by construction.
- Validation tests: in the no-censoring limit the estimator reproduces OLS
  (coefficients, MLE scale, and log-likelihood) to optimiser tolerance, checked
  against `statsmodels.OLS`; an optional comparison against R's `AER::tobit`
  and `truncreg` runs when R is available (`tests/test_r_reference.py`).
- Shared conditional-mean module (`_means.py`) used by both `predict` and the
  marginal-effects machinery, keeping predictions and effects consistent.
- Likelihood-ratio test in two equivalent forms:
  - `lr_test(full, restricted)` for any pair of fitted nested models;
  - `model.lr_test(hypotheses)` for linear restrictions specified as a
    statsmodels-style hypothesis string
    (e.g. `'(x1 = 0), (x2 - x3 = 0.5)'`) or as `(R, r)` arrays.
  Plus an overall LR test against the intercept-only null model included in
  every fitted-model summary.
- statsmodels-style `summary()` with coefficient table, Wald inference,
  AIC/BIC, McFadden's pseudo-R², and per-region censoring counts.
- Five executed example notebooks in `examples/`:
  two-sided censoring, classical Tobit on Fair (1978), truncated regression,
  validation against OLS and R, and a side-by-side visual comparison
  of truncated vs censored fits with an asymptotic prediction band.
- Test suite (29 tests, including Monte Carlo consistency checks) and a CI
  workflow that runs them on Python 3.10-3.13 across Linux and macOS.
