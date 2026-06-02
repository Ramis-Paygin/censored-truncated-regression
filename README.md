# censtrunc

[![Tests](https://github.com/<your-username>/censtrunc/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-username>/censtrunc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**censtrunc** is a Python package for maximum-likelihood estimation of
**censored and truncated normal regression models** with arbitrary left and
right thresholds. It generalises the classical Type-1 Tobit (Tobin 1958) and
the truncated normal regression to two-sided censoring/truncation while
exposing a familiar scikit-learn-style API and a statsmodels-style summary.

```python
from censtrunc import CensoredRegression, lr_test

model = CensoredRegression(left=0.0, right=2.5).fit(X, y)
print(model.summary())                       # full coefficient table
ame  = model.ame(kind="censored").to_dataframe()
mem  = model.mem(kind="censored").to_dataframe()
preds = model.predict(X_new)                   # DataFrame of all 6 quantities
restricted = CensoredRegression(left=0, right=2.5).fit(X[:, :2], y)
lr_test(model, restricted)
```

## What problem does it solve?

When a continuously distributed outcome is **observed only inside an interval
`[L, R]`**, ordinary least squares is biased: slope coefficients are shrunk
toward zero in proportion to the censoring probability (Greene 1981). Two
common observation rules apply:

| Type        | Observation rule                                      | Use this package as          |
|-------------|--------------------------------------------------------|------------------------------|
| Censored    | values outside `[L, R]` are *clipped* to the threshold | `CensoredRegression`         |
| Truncated   | values outside `[L, R]` are *dropped* from the sample  | `TruncatedRegression`        |

The classical Tobit model is the censored case with `L = 0`, `R = ∞`. This
package handles **arbitrary** finite or infinite thresholds, optionally on
both sides.

## Mathematical specification

The latent variable follows a linear regression with normal errors:

$$Y^* = X'\beta + e, \quad e \mid X \sim \mathcal N(0, \sigma^2).$$

**Censored model.** The observed variable is

$$Y = \min(R, \max(L, Y^*)),$$

so that values below `L` pile up at `L` and values above `R` pile up at `R`.
The log-likelihood is a mixture of CDFs (for the censored regions) and a
truncated normal density (for the interior):

$$\ell(\beta, \sigma) = \sum_{Y_i = L} \log\Phi\!\left(\tfrac{L - X_i'\beta}{\sigma}\right)
+ \sum_{Y_i = R} \log\Phi\!\left(\tfrac{X_i'\beta - R}{\sigma}\right)
+ \sum_{L < Y_i < R} \log\!\left[\tfrac{1}{\sigma}\phi\!\left(\tfrac{Y_i - X_i'\beta}{\sigma}\right)\right].$$

Optimisation is performed in **Olsen's reparameterisation** `(γ, ν) = (β/σ, 1/σ)`,
where the log-likelihood is globally concave (Olsen 1978) and any
gradient-based optimiser (L-BFGS-B by default) is guaranteed to converge.

**Truncated model.** Observations outside `(L, R)` are absent from the sample
entirely. The conditional density divides the normal density by the
probability of being in the interval:

$$f(y_i \mid L < Y_i^* < R) = \frac{\sigma^{-1}\,\phi((y_i - X_i'\beta)/\sigma)}
                                {\Phi((R - X_i'\beta)/\sigma) - \Phi((L - X_i'\beta)/\sigma)}.$$

**Six predicted quantities.** With $\mu = X'\beta$, $\alpha_L = (L - \mu)/\sigma$,
and $\alpha_R = (R - \mu)/\sigma$:

| code | quantity | formula |
|------|----------|---------|
| `h`  | hidden / latent mean      | $\mathbb E[Y^* \mid X] = \mu$ |
| `c`  | censored conditional mean | $\mathbb E[Y \mid X] = L\,\Phi(\alpha_L) + R\,[1{-}\Phi(\alpha_R)] + \mu\,[\Phi(\alpha_R){-}\Phi(\alpha_L)] + \sigma\,[\phi(\alpha_L){-}\phi(\alpha_R)]$ |
| `t`  | truncated conditional mean | $\mathbb E[Y \mid X, L{<}Y{<}R] = \mu + \sigma\,\dfrac{\phi(\alpha_L) - \phi(\alpha_R)}{\Phi(\alpha_R) - \Phi(\alpha_L)}$ |
| `l`  | left-region probability    | $P(Y = L \mid X) = \Phi(\alpha_L)$ |
| `m`  | middle-region probability  | $P(L < Y < R \mid X) = \Phi(\alpha_R) - \Phi(\alpha_L)$ |
| `r`  | right-region probability   | $P(Y = R \mid X) = 1 - \Phi(\alpha_R)$ |

For the truncated model only `h` and `t` are defined (no mass piles up at the
thresholds and no observations are reported there).

## Features

- **Censored regression** with arbitrary `left` / `right` thresholds (either
  may be `None` for one-sided censoring).
- **Truncated regression** with arbitrary thresholds.
- **Heckman sample-selection regression** (Heckit) with both the classical
  two-step estimator and joint maximum likelihood; corrects for selection bias
  when the outcome is observed only for a non-random subsample.
- A unified `predict()` that returns any combination of **six** quantities —
  three conditional means and three region probabilities — selected via a
  compact one-letter `kind` argument (e.g. `kind='hctlmr'` for all six,
  `kind='lmr'` for just the probabilities, `kind='h'` for the latent mean).
  Long names (`'latent'`, `'censored'`, `'truncated'`) remain valid.
- **Marginal effects** via a single `get_margeff` method whose API mirrors
  statsmodels' `get_margeff`: choose *where* to evaluate (`at='overall'` = AME,
  `at='mean'` = MEM, plus `'median'`/`'zero'`) and *what* to report
  (`method='dydx'` derivative, or `'eyex'`/`'dyex'`/`'eydx'` elasticities), with
  discrete-variable handling (`dummy=True`) and **delta-method standard errors**.
  `ame()` and `mem()` are provided as convenient shorthands. `HeckitRegression`
  also exposes `get_margeff` with the four Heckman quantities (`latent`,
  `conditional`, `unconditional`, `prob-selected`); variables that appear in
  both the outcome and selection equations (matched by feature name) have their
  X-side and Z-side derivatives combined automatically.
- **Likelihood-ratio test** (`lr_test`) for any pair of nested models, plus
  `model.lr_test(hypotheses)` for linear restrictions specified as a
  statsmodels-style hypothesis string (`'(x1 = 0), (x2 = x3)'`,
  `'2*x1 + x4 = 1'`, ...) or as `(R, r)` arrays.
- **statsmodels-style `summary()`** with coefficient table, Wald confidence
  intervals, AIC/BIC, McFadden's pseudo-$R^2$, overall LR test, and per-region
  censoring counts. `HeckitRegression.summary()` prints separate blocks for the
  outcome and selection equations and reports the joint $\rho$ and $\sigma$.
- **Cross-validation against external references** in the test suite:
  OLS-equivalence in the no-censoring limit; **probit-equivalence** when
  $L \approx R$ (the censored MLE reduces to a probit on $\mathbf 1\{Y^* > c\}$);
  R's `survival::survreg`, `truncreg::truncreg`, and
  `sampleSelection::selection` (two-step + joint MLE) for Heckit;
  Python's `py4etrics.Heckit` for the two-step.

## What's already available in Python (and what isn't)

Before writing `censtrunc` we surveyed the existing Python options for the
three model families it covers. The short version: the Python landscape for
censored / truncated / Heckman models is patchy compared to R, and several
common needs have no production-quality estimator behind them.

| Package | Censored / Tobit | Truncated | Heckit | Notes |
|---------|------------------|-----------|--------|-------|
| **statsmodels** (`0.14`) | no public Tobit class | no public truncated regression | not implemented | Has Probit, Logit, OLS, GLM, robust regression; Tobit-family models are the documented gap. |
| **scikit-learn** | — | — | — | No limited-dependent-variable models. |
| **linearmodels** | — | — | — | Panel / IV / GMM specialist. |
| **lifelines** | left-censored Weibull / log-normal AFT regressions | — | — | A survival framework: coefficients have hazard-ratio interpretations, and a non-zero left threshold is awkward to express. |
| **scikit-survival** (`sksurv`) | right-censored only, AFT-style | — | — | Same survival framing as `lifelines`. |
| **py4etrics** (`0.1.9`) | `Tobit` with arbitrary `left` / `right` | `Truncreg` with arbitrary `left` / `right` | `Heckit` two-step only (`method='mle'` is silently ignored) | An educational package by Hasebe et al.; no marginal effects, no LR test, no formula support, no Heckit MLE. |
| **PyMC / bambi** | Bayesian censored regression via `Censored` | Bayesian truncated regression via `Truncated` | possible to hand-code | Bayesian / MCMC route. |
| **marginaleffects** (Python) | — | — | — | A general-purpose slopes / contrasts toolkit, not a model-fitting library. |

R has all of these covered out of the box: `survival::survreg` (wrapped by
`AER::tobit`) for the censored case, `truncreg::truncreg` for the truncated
case, and `sampleSelection::selection` for both Heckit estimators.

### What `censtrunc` adds

- A single censored MLE that handles **arbitrary $L$ and $R$ thresholds**
  (left-only, right-only, or two-sided), with Olsen's globally concave
  reparameterisation so the optimiser is guaranteed to find the unique
  maximum.
- A matching truncated MLE for **two-sided truncation**, which is what you
  need when survey or administrative data only records observations inside
  a window.
- A Heckman selection estimator with **both** the two-step and the joint
  MLE (matching `sampleSelection::selection`), bootstrap SEs for the
  two-step, and a `summary()` that prints both equation blocks plus the
  estimated $\rho$ and $\sigma$.
- A unified `predict()` returning any combination of six quantities (three
  conditional means + three region probabilities) via a one-letter `kind`
  string.
- A `get_margeff()` mirroring statsmodels' API for all three estimators,
  including the four Heckman-specific kinds (`latent`, `conditional`,
  `unconditional`, `prob-selected`) and the three region-probability kinds
  for the censored model.
- A `lr_test()` that accepts statsmodels-style hypothesis strings
  (`'(x1 = 0), (x2 = x3)'`, `'x1 - 2*x2 = 0'`, ...) as well as two-model
  comparisons.
- Cross-validated in the test suite against `statsmodels.OLS` (no-censoring
  and probit limits), `survival::survreg`, `truncreg::truncreg`,
  `sampleSelection::selection`, and `py4etrics.Heckit`.

If your problem is the classical left-censored-at-zero Tobit, `py4etrics.Tobit`
will work too; `censtrunc` is aimed at what's missing elsewhere — two-sided
thresholds, joint-MLE Heckit, marginal effects, LR tests, and patsy formulas
on all three model classes.

## Installation

From PyPI (after publication):

```bash
pip install censtrunc
```

From source:

```bash
git clone https://github.com/<your-username>/censtrunc.git
cd censtrunc
pip install -e .[examples]   # examples extra installs pandas, matplotlib, jupyter
```

`censtrunc` supports Python 3.10+ and depends only on **numpy** and **scipy**
at runtime.

## Quick start

### Two-sided censoring

```python
import numpy as np
from censtrunc import CensoredRegression

rng = np.random.default_rng(0)
n = 2000
X = rng.normal(size=(n, 2))
y_star = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(size=n)
y = np.clip(y_star, 0.0, 2.5)   # two-sided clipping

model = CensoredRegression(left=0.0, right=2.5).fit(X, y)
print(model.summary())
print("β̂ =", model.coef_, "σ̂ =", model.sigma_)

# All six predicted quantities at once
preds = model.predict(X[:5])
#        latent  censored  truncated  prob_left  prob_interior  prob_right
print(preds.round(3))

# Just the censored mean (1-D ndarray, backward-compatible)
y_hat = model.predict(X[:5], kind="censored")     # same as kind="c"
# Just the three region probabilities (DataFrame, sums to 1 per row)
probs = model.predict(X[:5], kind="lmr")
```

### Classical Tobit (left-only)

```python
import statsmodels.api as sm
from censtrunc import CensoredRegression

data = sm.datasets.fair.load_pandas().data
X = data[["rate_marriage", "age", "yrs_married", "religious", "educ"]]
y = data["affairs"]

model = CensoredRegression(left=0.0).fit(X, y)
print(model.summary())
```

### Truncated regression

```python
from censtrunc import TruncatedRegression
# All `y` values must satisfy L < y < R (otherwise an error is raised).
model = TruncatedRegression(left=0.0, right=2.5).fit(X, y)
```

### Heckman selection (Heckit)

```python
from censtrunc import HeckitRegression
# y has NaN where the observation is *not selected* (e.g. wages for non-workers).
# X = outcome-equation regressors; Z = selection-equation regressors (often
# wider than X — it must include an "exclusion" variable that affects selection
# but not the outcome).
twostep = HeckitRegression(method="twostep").fit(y, X, Z)
mle     = HeckitRegression(method="mle").fit(y, X, Z)
print(twostep.summary())                # outcome + selection blocks + rho/sigma
yhat = mle.predict(X=X_new, Z=Z_new, kind="conditional")  # E[Y | X, Z, S=1]
```

### Patsy-style formulas

Every model class accepts a `from_formula(formula, data, ...)` constructor,
mirroring statsmodels:

```python
# Censored / truncated: one formula
CensoredRegression.from_formula(
    'lwage ~ 1 + educ + exper + expersq', data=mroz, left=0.0,
).fit()

TruncatedRegression.from_formula(
    'lwage ~ 1 + educ + exper + expersq',
    data=mroz.dropna(subset=['lwage']),
    left=thresh,
).fit()

# Heckit: two formulas (outcome + selection); selection LHS must be binary
HeckitRegression.from_formula(
    outcome   = 'lwage ~ 1 + educ + exper + expersq',
    selection = 'inlf  ~ 1 + educ + exper + age + kidslt6 + kidsge6',
    data=mroz,
    method='twostep',
).fit()
```

Patsy transforms (`np.log(x)`, `I(x**2)`, `C(z)` for categorical contrasts,
`x1:x2` interactions) all work because the formula is parsed by `patsy` before
the model sees it. The intercept is handled by the formula (`+ 1` is implicit,
`- 1` suppresses it).

### Marginal effects (`get_margeff`)

```python
# statsmodels-style: one method, options select where and what
me = model.get_margeff(at="overall", method="dydx", kind="censored")
print(me.summary())            # statsmodels-style table
me.summary_frame()             # pandas DataFrame

# elasticity at the sample mean
model.get_margeff(at="mean", method="eyex", kind="censored")

# discrete-difference for binary regressors
model.get_margeff(method="dydx", dummy=True)

# effect on the probability of each region (these three sum to zero)
model.get_margeff(kind="prob-left")      # d P(Y = L) / dx
model.get_margeff(kind="prob-interior")  # d P(L < Y < R) / dx
model.get_margeff(kind="prob-right")     # d P(Y = R) / dx

# convenient shorthands
model.ame(kind="censored")     # == get_margeff(at="overall")
model.mem(kind="censored")     # == get_margeff(at="mean")
```

`kind` selects the quantity whose marginal effect is reported: a conditional
mean (`latent`, `censored`, `truncated`) or a **region probability**
(`prob-left`, `prob-interior`, `prob-right`).

| `at` | meaning | | `method` | meaning |
|------|---------|---|----------|---------|
| `overall` | average over sample (AME) | | `dydx` | derivative dy/dx |
| `mean` | at the mean (MEM) | | `eyex` | elasticity |
| `median` | at the median | | `dyex` | semi-elasticity (dy/d ln x) |
| `zero` | at zero | | `eydx` | semi-elasticity (d ln y/dx) |

### Likelihood-ratio test

Two equivalent routes:

```python
# 1) Two fitted models (any nested pair)
from censtrunc import lr_test
full = CensoredRegression(left=0.0, right=2.5).fit(X, y)
restricted = CensoredRegression(left=0.0, right=2.5).fit(X[:, :2], y)
print(lr_test(full, restricted))

# 2) One model + a statsmodels-style hypothesis string (`f_test` syntax)
print(full.lr_test('x3 = 0'))                       # single restriction
print(full.lr_test('(x2 = 0), (x3 = 0)'))           # joint restriction
print(full.lr_test('x1 - 2*x2 = 0'))                # composite / arithmetic
print(full.lr_test((R, r)))                         # raw (R, r) arrays
```

Route 2 refits the model under the linear constraint $R\hat\theta = r$ via
`scipy.optimize` with `LinearConstraint`, then reports
$LR = 2(\ell_{\text{full}} - \ell_{\text{rest.}})\sim \chi^2_q$ where $q$ is
the number of restrictions.

## API at a glance

| Object                       | Purpose                                              |
|------------------------------|------------------------------------------------------|
| `CensoredRegression`         | Tobit-style estimator with arbitrary `left, right`   |
| `TruncatedRegression`        | Truncated normal regression                          |
| `HeckitRegression`           | Heckman sample-selection (two-step or joint MLE)     |
| `lr_test(full, restricted)`  | Likelihood-ratio test between two nested models      |
| `model.lr_test(hypotheses)`  | LR test for linear restrictions (`'x1 = 0, x2 = x3'`) |
| `MarginalEffects`            | Returned by `.ame()` / `.mem()`; has `.to_dataframe()` |
| `LRTestResult`               | Returned by `lr_test`                                 |

Fitted-model attributes (censored and truncated):

| Attribute            | Description                                              |
|----------------------|----------------------------------------------------------|
| `params_`            | `[sigma, beta_0, beta_1, ...]`                            |
| `coef_`              | Regression coefficients (`beta`)                          |
| `sigma_`             | Scale parameter                                           |
| `bse_`, `tvalues_`, `pvalues_`, `conf_int_` | Standard errors and Wald inference |
| `llf_`, `llnull_`    | Log-likelihood of fitted and null (intercept-only) models |
| `llr_`, `llr_pvalue_`| Overall LR statistic and p-value                          |
| `prsquared_`         | McFadden's pseudo-$R^2$                                   |
| `aic_`, `bic_`       | Information criteria                                      |
| `n_left_censored_`, `n_right_censored_`, `n_uncensored_` | Region counts (censored only) |
| `diagnostics_`       | Optimiser convergence message and iteration count         |

`HeckitRegression` has the same `coef_` / `sigma_` / `bse_` / `pvalues_` (for
the outcome equation) plus selection-equation analogues `gamma_`, `bse_gamma_`,
`pvalues_gamma_`, the joint correlation `rho_`, the Heckman covariance term
`sigma_eu_` (= $\rho\sigma$), and `n_obs_` / `n_selected_`. After a
`method='mle'` fit the joint covariance `cov_params_` is also exposed (used by
`get_margeff` for delta-method SEs).

## Examples

The `examples/` directory contains **six** executed Jupyter notebooks:

1. **`01_two_sided_censoring.ipynb`** — the headline use case: two-sided
   censoring on simulated data, with histograms, predictions, AME/MEM, and
   an LR test.
2. **`02_classical_tobit_affairs.ipynb`** — classical left-censored Tobit on
   the Fair (1978) extramarital-affairs dataset, with marginal effects and a
   comparison against OLS.
3. **`03_truncated_regression.ipynb`** — truncated regression on simulated
   data, showing the OLS bias.
4. **`04_validation.ipynb`** — equivalence with OLS in the no-censoring limit,
   the probit limit at tight thresholds ($L \approx R$), plus a live
   comparison against R's `survreg` / `truncreg`.
5. **`05_truncated_vs_censored_visual.ipynb`** — side-by-side plot of fits
   under truncation and censoring, with an asymptotic prediction band
   (frequentist analogue of the pymc-devs GLM-truncated-censored figure).
6. **`06_heckit.ipynb`** — Heckman sample-selection (Heckit) on a simulated
   DGP and on the canonical Mroz (1987) wage data, including cross-validation
   against `py4etrics` and a base-R reference, marginal effects for all four
   Heckit `kind`s, and a finite-difference check of the derivative formulas.

Rebuild the notebooks from source with

```bash
python examples/_build_notebooks.py
```

### Single combined PDF report

For a self-contained document (title page, model description with formulas, and
all three examples as sections), build and render the combined report:

```bash
pip install -e .[docs]              # adds nbconvert[webpdf]
python examples/_build_report.py    # assembles examples/censtrunc_report.ipynb
playwright install chromium         # one-time: headless browser for PDF export
jupyter nbconvert --to webpdf examples/censtrunc_report.ipynb
# -> examples/censtrunc_report.pdf
```

(The `webpdf` route needs no LaTeX. If you have a LaTeX toolchain installed,
`--to pdf` works too.)

## Testing

```bash
pip install -e .[test]
pytest                                 # full suite
pytest -m "not slow"                   # skip Monte Carlo tests
```

The test suite (~120 tests, run in well under a minute on a laptop) covers:

- Smoke tests for all four supported configurations (left-only, right-only,
  both, truncated) on all three estimators.
- Recovery of true parameters on simulated data within a few standard errors.
- Analytical gradient matches finite-difference gradient (catches sign errors
  in the closed-form derivative expressions).
- LR test correctly rejects irrelevant restrictions and accepts true ones.
- Predictions and marginal effects in the right shape and direction, including
  a finite-difference cross-check between `get_margeff` and `predict` for the
  Heckman model.
- **Cross-validation against external implementations**: OLS equivalence in
  the no-censoring limit; probit equivalence in the narrow-band limit; R's
  `survival::survreg` and `truncreg::truncreg` (gated on R availability);
  `py4etrics.Heckit` two-step; a base-R Heckit MLE (`glm(probit)` + `optim()`
  on the joint log-likelihood).
- Input validation (NaN handling, threshold ordering, etc.).
- Monte Carlo (marked `slow`) confirming approximate unbiasedness over 60
  repetitions.

## Notes on `marginaleffects`

The Python [`marginaleffects`](https://marginaleffects.com/) package is a
general-purpose toolkit for predictions, comparisons, slopes, and hypothesis
tests across many model backends. We considered providing a direct adapter
but decided **not** to ship one. The reasoning:

- `marginaleffects` requires a backend-specific subclass of an internal
  `ModelAbstract` that exposes a `vault` (named pandas coefficients, a
  formula, polars data, formula-engine metadata) and overrides `get_predict`,
  `get_exog`, `get_vcov`. A working adapter would couple `censtrunc` to those
  private internals and would need ongoing maintenance.
- Our `get_margeff` already mirrors the design (one method, options) and
  covers the standard slopes use cases (`dydx`, elasticities, AME/MEM,
  delta-method SEs, hypothesis tests for linear restrictions). It additionally
  exposes Tobit-specific quantities — `kind` ∈ {`latent`, `censored`,
  `truncated`, `prob-left`, `prob-interior`, `prob-right`} — that
  `marginaleffects` does not natively understand (it would only see one
  default prediction).
- For users who want `marginaleffects`-style diagnostic plots, the recommended
  workflow is to fit a parallel OLS on the same data via `statsmodels` and use
  `marginaleffects.slopes` on it for visualisation, while reporting the
  unbiased Tobit marginal effects from `censtrunc.get_margeff`.

## References

- Tobin, J. (1958). "Estimation of Relationships for Limited Dependent
  Variables." *Econometrica*, 26(1), 24-36.
- Olsen, R. J. (1978). "Note on the Uniqueness of the Maximum Likelihood
  Estimator for the Tobit Model." *Econometrica*, 46(5), 1211-1215.
- Greene, W. H. (1981). "On the Asymptotic Bias of the Ordinary Least Squares
  Estimator of the Tobit Model." *Econometrica*, 49(2), 505-513.
- Hansen, B. E. (2022). *Econometrics*. Princeton University Press. Chapter 27,
  "Censoring and Selection."

## License

[MIT](LICENSE).
