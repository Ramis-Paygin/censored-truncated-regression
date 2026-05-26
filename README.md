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
preds = model.predict(X_new, kind="censored")  # E[Y | X]
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

## Features

- **Censored regression** with arbitrary `left` / `right` thresholds (either
  may be `None` for one-sided censoring).
- **Truncated regression** with arbitrary thresholds.
- Three flavours of prediction: `latent` ($X'\beta$), `censored` ($\mathbb E[Y\mid X]$),
  and `truncated` ($\mathbb E[Y\mid X, L<Y<R]$); plus `predict_proba` returning
  region probabilities.
- **AME** (Average Marginal Effect) and **MEM** (Marginal Effect at the Mean)
  for all three notions of conditional mean, with **delta-method standard
  errors**.
- **Likelihood-ratio test** (`lr_test`) for any pair of nested models.
- **statsmodels-style `summary()`** with coefficient table, Wald confidence
  intervals, AIC/BIC, McFadden's pseudo-$R^2$, overall LR test, and per-region
  censoring counts.

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

### Marginal effects

```python
ame  = model.ame(kind="censored")           # MarginalEffects object
print(ame.to_dataframe())                    # pandas DataFrame view
mem  = model.mem(kind="censored")            # at the sample mean
```

### Likelihood-ratio test

```python
from censtrunc import lr_test
full = CensoredRegression(left=0.0, right=2.5).fit(X, y)
restricted = CensoredRegression(left=0.0, right=2.5).fit(X[:, :2], y)
result = lr_test(full, restricted)
print(result)        # statistic, df, p-value, log-likelihoods
```

## API at a glance

| Object                       | Purpose                                              |
|------------------------------|------------------------------------------------------|
| `CensoredRegression`         | Tobit-style estimator with arbitrary `left, right`   |
| `TruncatedRegression`        | Truncated normal regression                          |
| `lr_test(full, restricted)`  | Likelihood-ratio test between two nested models      |
| `MarginalEffects`            | Returned by `.ame()` / `.mem()`; has `.to_dataframe()` |
| `LRTestResult`               | Returned by `lr_test`                                 |

Fitted-model attributes (both estimators):

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
| `n_left_censored_`, `n_right_censored_`, `n_uncensored_` | Region counts |
| `diagnostics_`       | Optimiser convergence message and iteration count         |

## Examples

The `examples/` directory contains three executed Jupyter notebooks:

1. **`01_two_sided_censoring.ipynb`** — the headline use case: two-sided
   censoring on simulated data, with histograms, predictions, AME/MEM, and
   an LR test.
2. **`02_classical_tobit_affairs.ipynb`** — classical left-censored Tobit on
   the Fair (1978) extramarital-affairs dataset, with marginal effects and a
   comparison against OLS.
3. **`03_truncated_regression.ipynb`** — truncated regression on simulated
   data, showing the OLS bias.

Rebuild the notebooks from source with

```bash
python examples/_build_notebooks.py
```

## Testing

```bash
pip install -e .[test]
pytest                                 # full suite
pytest -m "not slow"                   # skip Monte Carlo tests
```

The test suite covers:

- Smoke tests for all four supported configurations (left-only, right-only,
  both, truncated).
- Recovery of true parameters on simulated data within a few standard errors.
- Analytical gradient matches finite-difference gradient (catches sign errors
  in the closed-form derivative expressions).
- LR test correctly rejects irrelevant restrictions and accepts true ones.
- Predictions and marginal effects in the right shape and direction.
- Input validation (NaN handling, threshold ordering, etc.).
- Monte Carlo (marked `slow`) confirming approximate unbiasedness over 60
  repetitions.

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
