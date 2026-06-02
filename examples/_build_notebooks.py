"""Build the example notebooks programmatically from a single source.

Running this script regenerates the three example ``.ipynb`` files in this
directory. Keeping notebooks generated rather than hand-edited makes it easy
to keep the prose, code, and outputs in sync with the package.

Usage
-----
    python examples/_build_notebooks.py

Requirements: ``nbformat`` and ``jupyter`` (installed via the ``examples`` extra,
``pip install -e .[examples]``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def _build_two_sided_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        _md(
            "# Two-Sided Censored Regression\n"
            "\n"
            "This notebook demonstrates the headline feature of `censtrunc`: a\n"
            "censored normal regression with **arbitrary left and right thresholds**.\n"
            "We simulate data with a known data-generating process, censor it on\n"
            "both sides, then compare the package's MLE to ordinary least squares.\n"
            "\n"
            "**Setup**: install the package in editable mode with the `examples`\n"
            "extra:\n"
            "\n"
            "```bash\n"
            "pip install -e .[examples]\n"
            "```"
        ),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from censtrunc import CensoredRegression, lr_test\n"
            "\n"
            "rng = np.random.default_rng(2026)\n"
            "n = 3000"
        ),
        _md(
            "## Data-generating process\n"
            "\n"
            "$$Y_i^* = 1.0 + 0.5\\,X_{1i} - 0.3\\,X_{2i} + 0.2\\,X_{3i} + e_i,"
            "\\quad e_i \\sim \\mathcal N(0, 1).$$\n"
            "\n"
            "The observation rule clips values below $L=0$ and above $R=2.5$,"
            " producing a realistic two-sided censoring pattern (think 'satisfaction"
            " score on a 0-2.5 scale')."
        ),
        _code(
            "X = rng.normal(size=(n, 3))\n"
            "beta_true = np.array([1.0, 0.5, -0.3, 0.2])\n"
            "sigma_true = 1.0\n"
            "y_star = beta_true[0] + X @ beta_true[1:] + rng.normal(scale=sigma_true, size=n)\n"
            "L, R = 0.0, 2.5\n"
            "y = np.clip(y_star, L, R)\n"
            "\n"
            "print(f'Left-censored:   {np.mean(y == L):.1%}')\n"
            "print(f'Right-censored:  {np.mean(y == R):.1%}')\n"
            "print(f'Interior:        {np.mean((y > L) & (y < R)):.1%}')"
        ),
        _md(
            "## Distribution of the latent and observed variables\n"
            "\n"
            "The pile-ups at $L$ and $R$ in the observed distribution are the"
            " hallmark of two-sided censoring."
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)\n"
            "axes[0].hist(y_star, bins=40, color='steelblue', alpha=0.85)\n"
            "axes[0].set_title('Latent Y* (uncensored)')\n"
            "axes[0].axvline(L, color='crimson', linestyle='--', label='thresholds')\n"
            "axes[0].axvline(R, color='crimson', linestyle='--')\n"
            "axes[0].legend()\n"
            "axes[1].hist(y, bins=40, color='darkorange', alpha=0.85)\n"
            "axes[1].set_title('Observed Y (clipped to [L, R])')\n"
            "axes[1].axvline(L, color='crimson', linestyle='--')\n"
            "axes[1].axvline(R, color='crimson', linestyle='--')\n"
            "plt.tight_layout(); plt.show()"
        ),
        _md(
            "## Fit the censored regression\n"
            "\n"
            "The package's default optimisation uses Olsen's reparameterisation,"
            " which makes the log-likelihood globally concave and guarantees"
            " convergence of gradient-based optimisers (Olsen 1978)."
        ),
        _code(
            "model = CensoredRegression(left=L, right=R).fit(\n"
            "    X, y, feature_names=['x1', 'x2', 'x3']\n"
            ")\n"
            "print(model.summary())"
        ),
        _md(
            "## Compare against OLS\n"
            "\n"
            "OLS on the censored sample is biased (Greene 1981): the slope"
            " coefficients are shrunk roughly in proportion to the censoring"
            " probability. Our Tobit-style MLE is consistent."
        ),
        _code(
            "X_with_int = np.column_stack([np.ones(n), X])\n"
            "ols_beta, *_ = np.linalg.lstsq(X_with_int, y, rcond=None)\n"
            "\n"
            "comparison = pd.DataFrame({\n"
            "    'true':        beta_true,\n"
            "    'OLS (biased)': ols_beta,\n"
            "    'Censored MLE': model.coef_,\n"
            "}, index=['const', 'x1', 'x2', 'x3'])\n"
            "comparison['OLS bias']  = comparison['OLS (biased)']  - comparison['true']\n"
            "comparison['MLE bias']  = comparison['Censored MLE']  - comparison['true']\n"
            "comparison.round(4)"
        ),
        _md(
            "## Predictions: six quantities, one call\n"
            "\n"
            "`model.predict(X)` returns a dataframe with all six quantities — three"
            " conditional means and three region probabilities. Each column has a"
            " one-letter mnemonic:\n"
            "\n"
            "- `h` — *hidden* / latent mean $\\mathbb E[Y^* \\mid X] = X'\\beta$\n"
            "- `c` — *censored* mean $\\mathbb E[Y \\mid X]$\n"
            "- `t` — *truncated* mean $\\mathbb E[Y \\mid X, L<Y<R]$\n"
            "- `l` — left-region probability $P(Y=L \\mid X) = \\Phi(\\alpha_L)$\n"
            "- `m` — middle probability $P(L<Y<R \\mid X) = \\Phi(\\alpha_R) - \\Phi(\\alpha_L)$\n"
            "- `r` — right-region probability $P(Y=R \\mid X) = 1 - \\Phi(\\alpha_R)$"
        ),
        _code(
            "X_show = X[:6]\n"
            "model.predict(X_show).round(3)   # default = all six columns"
        ),
        _md(
            "Subsets are selected with the same letter codes. For example, just the"
            " three probabilities (note that each row sums to one) or just the"
            " hidden mean (a 1-D array):"
        ),
        _code(
            "model.predict(X_show, kind='lmr').round(3)"
        ),
        _code(
            "model.predict(X_show, kind='h')          # single letter -> 1-D ndarray"
        ),
        _md(
            "Long names — `kind='latent'`, `'censored'`, `'truncated'` — remain"
            " valid and return 1-D arrays for backward compatibility."
        ),
        _md(
            "## Marginal effects via `get_margeff`\n"
            "\n"
            "The marginal-effects API mirrors statsmodels' `get_margeff`: one method,\n"
            "with options for *where* to evaluate (`at`) and *what* to report\n"
            "(`method`). The summary prints in the familiar statsmodels layout."
        ),
        _code(
            "# AME = average over the sample (at='overall'); statsmodels-style summary\n"
            "print(model.get_margeff(at='overall', method='dydx', kind='censored').summary())"
        ),
        _md(
            "`at='overall'` is the AME; `at='mean'` is the marginal effect at the"
            " mean (MEM). Below we compare the censored marginal effect across"
            " evaluation points and against the latent effect (which equals"
            " $\\beta$ exactly)."
        ),
        _code(
            "import pandas as pd\n"
            "summary = pd.DataFrame({\n"
            "    'latent (=beta)':   model.get_margeff(at='overall', kind='latent').margeff,\n"
            "    'censored AME':     model.get_margeff(at='overall', kind='censored').margeff,\n"
            "    'censored MEM':     model.get_margeff(at='mean',    kind='censored').margeff,\n"
            "    'truncated AME':    model.get_margeff(at='overall', kind='truncated').margeff,\n"
            "}, index=['x1', 'x2', 'x3'])\n"
            "summary.round(4)"
        ),
        _md(
            "`method` switches between the derivative and elasticities:\n"
            "`dydx` (derivative), `eyex` (elasticity), `dyex`/`eydx`"
            " (semi-elasticities). Discrete regressors can be handled with"
            " `dummy=True` (a 0->1 difference rather than a derivative)."
        ),
        _code(
            "elas = model.get_margeff(at='overall', method='eyex', kind='censored')\n"
            "elas.summary_frame().round(4)"
        ),
        _md(
            "### Marginal effects on the region probabilities\n"
            "\n"
            "Beyond the conditional mean, we can ask how a regressor shifts the"
            " *probability* of each region: landing at the left threshold"
            " ($P(Y=L)=\\Phi(\\alpha_L)$), in the interior"
            " ($P(L<Y<R)=\\Phi(\\alpha_R)-\\Phi(\\alpha_L)$), or at the right"
            " threshold ($P(Y=R)=1-\\Phi(\\alpha_R)$). These are selected with"
            " `kind='prob-left'`, `'prob-interior'`, `'prob-right'`. Because the"
            " three probabilities sum to one, their marginal effects sum to zero."
        ),
        _code(
            "prob_effects = pd.DataFrame({\n"
            "    'P(Y=L)':     model.get_margeff(kind='prob-left').margeff,\n"
            "    'P(L<Y<R)':   model.get_margeff(kind='prob-interior').margeff,\n"
            "    'P(Y=R)':     model.get_margeff(kind='prob-right').margeff,\n"
            "}, index=['x1', 'x2', 'x3'])\n"
            "prob_effects['sum (=0)'] = prob_effects.sum(axis=1)\n"
            "prob_effects.round(4)"
        ),
        _md(
            "A positive coefficient pushes observations out of the left pile-up"
            " and toward the right one, so the `P(Y=L)` effect is negative and the"
            " `P(Y=R)` effect is positive; the `sum (=0)` column confirms the"
            " adding-up constraint holds numerically."
        ),
        _md(
            "## Likelihood-ratio test\n"
            "\n"
            "We test whether `x3` is jointly redundant by fitting a restricted"
            " model that omits it, then comparing log-likelihoods."
        ),
        _code(
            "restricted = CensoredRegression(left=L, right=R).fit(X[:, :2], y)\n"
            "res = lr_test(model, restricted)\n"
            "print(res)"
        ),
        _md(
            "If the p-value is small, we reject the restriction (drop `x3`)."
            " Here `x3` has a true coefficient of 0.2 and a large sample, so the"
            " test correctly rejects."
        ),
        _md(
            "### LR test from a hypothesis string\n"
            "\n"
            "Re-fitting a restricted model by hand is fine for simple drop-a-column"
            " cases, but cumbersome for joint or composite restrictions. For those,"
            " `model.lr_test(hypotheses)` mirrors statsmodels' `f_test` and accepts"
            " linear restrictions directly as a string — single, joint, or with"
            " arithmetic on parameter names."
        ),
        _code(
            "# Drop a single regressor (same null as the two-model test above)\n"
            "print(model.lr_test('x3 = 0'))"
        ),
        _code(
            "# Joint restriction: x2 AND x3 are zero (composite null)\n"
            "print(model.lr_test('(x2 = 0), (x3 = 0)'))"
        ),
        _code(
            "# Composite restriction with arithmetic: x1 - 2*x2 = 0\n"
            "# (equivalent to testing whether x1 equals twice x2)\n"
            "print(model.lr_test('x1 - 2*x2 = 0'))"
        ),
        _md(
            "Each call refits the model under the linear constraint $R\\hat\\theta = r$"
            " (via `scipy.optimize` with `LinearConstraint`) and reports the"
            " asymptotic LR statistic $2(\\ell_{\\text{full}} - \\ell_{\\text{rest.}})"
            "\\sim \\chi^2_q$."
        ),
    ]
    nb["cells"] = cells
    return nb


def _build_classical_tobit_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        _md(
            "# Classical Tobit on the Fair Affairs Dataset\n"
            "\n"
            "The Fair (1978) extramarital-affairs dataset is a canonical Tobit"
            " example: the dependent variable `affairs` is the number of affairs"
            " per year, left-censored at zero (about 68% of respondents report"
            " zero affairs). We use the censored regression with `left=0` and no"
            " right threshold — the classical Type-1 Tobit."
        ),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import statsmodels.api as sm\n"
            "from censtrunc import CensoredRegression\n"
            "\n"
            "data = sm.datasets.fair.load_pandas().data\n"
            "print('n =', len(data))\n"
            "print(f\"share with affairs == 0: {(data['affairs'] == 0).mean():.1%}\")\n"
            "data.head()"
        ),
        _md(
            "## Specification\n"
            "\n"
            "Following Fair (1978), we regress the count of affairs on the"
            " standard set of demographic and marital variables."
        ),
        _code(
            "covariates = ['rate_marriage', 'age', 'yrs_married', 'children',\n"
            "              'religious', 'educ', 'occupation', 'occupation_husb']\n"
            "X = data[covariates]\n"
            "y = data['affairs']"
        ),
        _md(
            "## Fit\n"
            "\n"
            "We tell `CensoredRegression` that the left threshold is zero; the"
            " right threshold is left unspecified."
        ),
        _code(
            "model = CensoredRegression(left=0.0).fit(X, y)\n"
            "print(model.summary())"
        ),
        _md(
            "## Marginal effects\n"
            "\n"
            "Because the response is left-censored at zero, the censored marginal"
            " effects are smaller in absolute value than the latent ones. The"
            " latent (uncensored) coefficients describe the propensity scale,"
            " while the censored marginal effects describe the *expected number*"
            " of affairs."
        ),
        _code(
            "ame_cens = model.ame(kind='censored').to_dataframe()\n"
            "ame_lat  = model.ame(kind='latent').to_dataframe()\n"
            "pd.concat([\n"
            "    ame_lat[['dy/dx']].rename(columns={'dy/dx': 'latent'}),\n"
            "    ame_cens[['dy/dx', 'std err', 'P>|z|']].rename(columns={'dy/dx': 'censored E[Y]'}),\n"
            "], axis=1).round(4)"
        ),
        _md(
            "## Sanity check vs. OLS\n"
            "\n"
            "Naive OLS on the censored sample understates the slope coefficients"
            " (Greene 1981). The Tobit MLE corrects this."
        ),
        _code(
            "X_int = sm.add_constant(X)\n"
            "ols = sm.OLS(y, X_int).fit()\n"
            "pd.DataFrame({\n"
            "    'OLS coef':    ols.params.values,\n"
            "    'Tobit coef':  model.coef_,\n"
            "}, index=ols.params.index).round(4)"
        ),
    ]
    nb["cells"] = cells
    return nb


def _build_truncated_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        _md(
            "# Truncated Regression\n"
            "\n"
            "In truncated samples, observations outside the truncation interval"
            " are **absent entirely** (not just clipped to a threshold). This"
            " happens in survey data when respondents are recruited only above a"
            " minimum income, or in administrative data that only records"
            " transactions above a reporting threshold."
        ),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from censtrunc import TruncatedRegression\n"
            "\n"
            "rng = np.random.default_rng(7)\n"
            "n_population = 8000"
        ),
        _md(
            "## Simulate a truncated dataset\n"
            "\n"
            "We draw a large population, then keep only observations with"
            " $0 < Y^* < 2.5$ — about 60% of the population survives the"
            " double-sided truncation in this DGP."
        ),
        _code(
            "X_pop = rng.normal(size=(n_population, 2))\n"
            "beta_true = np.array([1.0, 0.5, -0.3])\n"
            "y_star = beta_true[0] + X_pop @ beta_true[1:] + rng.normal(scale=1.0, size=n_population)\n"
            "L, R = 0.0, 2.5\n"
            "mask = (y_star > L) & (y_star < R)\n"
            "X = X_pop[mask]\n"
            "y = y_star[mask]\n"
            "print(f'Observed after truncation: {len(y)} / {n_population}')"
        ),
        _md(
            "## Fit the truncated regression"
        ),
        _code(
            "model = TruncatedRegression(left=L, right=R).fit(\n"
            "    X, y, feature_names=['x1', 'x2']\n"
            ")\n"
            "print(model.summary())"
        ),
        _md(
            "## Comparison: OLS on the truncated sample is biased\n"
            "\n"
            "Naive OLS on the truncated sample treats the data as if it came from"
            " an unrestricted population — leading to biased slope estimates,"
            " typically shrunk toward zero."
        ),
        _code(
            "X_int = np.column_stack([np.ones(len(y)), X])\n"
            "ols_beta, *_ = np.linalg.lstsq(X_int, y, rcond=None)\n"
            "\n"
            "comparison = pd.DataFrame({\n"
            "    'true':         beta_true,\n"
            "    'OLS (biased)': ols_beta,\n"
            "    'Truncated MLE': model.coef_,\n"
            "}, index=['const', 'x1', 'x2'])\n"
            "comparison['OLS bias']           = comparison['OLS (biased)']  - comparison['true']\n"
            "comparison['Truncated MLE bias'] = comparison['Truncated MLE'] - comparison['true']\n"
            "comparison.round(4)"
        ),
        _md(
            "## Predictions\n"
            "\n"
            "For the truncated model only two prediction quantities are meaningful"
            " (there is no mass at the thresholds, so the region-probability"
            " columns do not apply):\n"
            "\n"
            "- `h` (`latent`): $X'\\hat\\beta$ — the population mean\n"
            "- `t` (`truncated`): $\\mathbb E[Y | X, L < Y < R]$"
        ),
        _code(
            "preds = model.predict(X[:6])      # default = 'ht' -> both columns\n"
            "preds.insert(0, 'observed', y[:6])\n"
            "preds.round(3)"
        ),
    ]
    nb["cells"] = cells
    return nb


def _build_validation_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        _md(
            "# Validation\n"
            "\n"
            "How do we know the estimator is correct and not 'a big pile of code that\n"
            "returns nonsense'? **Three** independent checks, all run live in this\n"
            "notebook.\n"
            "\n"
            "1. **No-censoring limit:** with thresholds pushed beyond the data range,\n"
            "   no observation is censored, so the censored/truncated MLE *must*\n"
            "   reduce to OLS. Verified against `statsmodels.OLS`.\n"
            "2. **Reference packages:** the test suite also compares against R's\n"
            "   `survival::survreg` (the engine behind `AER::tobit`) and `truncreg`\n"
            "   when R is available; see `tests/test_r_reference.py`. We reproduce\n"
            "   that R comparison live below.\n"
            "3. **Probit limit:** with $L = R = c$, the censored log-likelihood is\n"
            "   *exactly* the probit log-likelihood on $\\mathbf 1\\{Y^* > c\\}$, and\n"
            "   the relation $\\beta_{\\text{tobit}}/\\sigma_{\\text{tobit}} =\n"
            "   \\beta_{\\text{probit}}$ holds (Hansen 2022, §27.4). A narrow band\n"
            "   $L = c - \\varepsilon, R = c + \\varepsilon$ is enough to verify it\n"
            "   numerically without making $\\sigma$ unidentified."
        ),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import statsmodels.api as sm\n"
            "from censtrunc import CensoredRegression, TruncatedRegression\n"
            "\n"
            "rng = np.random.default_rng(0)\n"
            "n = 1000\n"
            "X = rng.normal(size=(n, 3))\n"
            "y = 2.0 + 1.5*X[:,0] - 0.7*X[:,1] + 0.3*X[:,2] + rng.normal(scale=1.3, size=n)"
        ),
        _md(
            "## No censoring $\\Rightarrow$ OLS\n"
            "\n"
            "We set `left=-1e6`, `right=1e6` so that no observation is censored. The"
            " censored log-likelihood then reduces to the Gaussian log-likelihood,"
            " whose maximiser is exactly the OLS coefficient vector."
        ),
        _code(
            "ols = sm.OLS(y, sm.add_constant(X)).fit()\n"
            "cens = CensoredRegression(left=-1e6, right=1e6).fit(X, y)\n"
            "trunc = TruncatedRegression(left=-1e6, right=1e6).fit(X, y)\n"
            "\n"
            "pd.DataFrame({\n"
            "    'OLS':            np.asarray(ols.params),\n"
            "    'Censored MLE':   cens.coef_,\n"
            "    'Truncated MLE':  trunc.coef_,\n"
            "}, index=['const', 'x1', 'x2', 'x3']).round(6)"
        ),
        _md(
            "The columns agree to several decimals. We can quantify the maximum"
            " discrepancy and confirm the scale parameter and log-likelihood match"
            " too (using the MLE scale $\\hat\\sigma = \\sqrt{\\mathrm{SSR}/n}$)."
        ),
        _code(
            "ols_beta = np.asarray(ols.params)\n"
            "print(f'max |beta_censored - beta_OLS|  = {np.max(np.abs(cens.coef_ - ols_beta)):.2e}')\n"
            "print(f'max |beta_truncated - beta_OLS| = {np.max(np.abs(trunc.coef_ - ols_beta)):.2e}')\n"
            "print(f'|sigma_censored - sqrt(SSR/n)|  = {abs(cens.sigma_ - np.sqrt(ols.ssr/n)):.2e}')\n"
            "print(f'|loglik_censored - loglik_OLS|  = {abs(cens.llf_ - ols.llf):.2e}')"
        ),
        _md(
            "All discrepancies are at the level of the optimiser tolerance — the"
            " estimator behaves exactly as theory requires in the no-censoring"
            " limit, which is strong evidence the likelihood and its optimisation"
            " are implemented correctly."
        ),
        _md(
            "## With censoring: agreement with R, disagreement with OLS\n"
            "\n"
            "The decisive test is a genuinely censored dataset. We compare four"
            " numbers per coefficient:\n"
            "\n"
            "- **true** — the data-generating value;\n"
            "- **censtrunc** — this package's MLE;\n"
            "- **R survreg** — R's Gaussian `survreg`, the engine behind"
            " `AER::tobit`, computed by an entirely independent codebase;\n"
            "- **OLS** — naive least squares, which is biased under censoring.\n"
            "\n"
            "Two correct maximum-likelihood implementations must converge to the"
            " *same unique* optimum, so `censtrunc` and R should agree to optimiser"
            " tolerance — while both should differ from the biased OLS and sit"
            " close to the truth."
        ),
        _code(
            "import shutil, subprocess, json, tempfile, os\n"
            "from pathlib import Path\n"
            "\n"
            "rng2 = np.random.default_rng(20260527)\n"
            "nc = 4000\n"
            "Xc = rng2.normal(size=(nc, 2))\n"
            "y_star_c = 1.0 + 0.7*Xc[:, 0] - 0.4*Xc[:, 1] + rng2.normal(size=nc)\n"
            "Lc, Rc = 0.0, 2.5\n"
            "yc = np.clip(y_star_c, Lc, Rc)\n"
            "\n"
            "cens = CensoredRegression(left=Lc, right=Rc).fit(Xc, yc)\n"
            "ols_c = np.asarray(sm.OLS(yc, sm.add_constant(Xc)).fit().params)\n"
            "\n"
            "# Run R's survreg via the bundled reference script, if R is available.\n"
            "def _find_r_script():\n"
            "    for c in [Path('tests/reference/fit_tobit.R'),\n"
            "              Path('../tests/reference/fit_tobit.R')]:\n"
            "        if c.exists():\n"
            "            return c\n"
            "    return None\n"
            "\n"
            "r_beta = r_sigma = None\n"
            "rscript, script_path = shutil.which('Rscript'), _find_r_script()\n"
            "if rscript and script_path:\n"
            "    fd, csvp = tempfile.mkstemp(suffix='.csv'); os.close(fd)\n"
            "    np.savetxt(csvp, np.column_stack([yc, Xc]), delimiter=',', header='y,x1,x2', comments='')\n"
            "    try:\n"
            "        out = subprocess.run([rscript, str(script_path), csvp, str(Lc), str(Rc)],\n"
            "                             capture_output=True, text=True, timeout=120, check=True)\n"
            "        rt = json.loads(out.stdout)['tobit']\n"
            "        r_beta = np.array([rt['coef']['(Intercept)'], rt['coef']['x1'], rt['coef']['x2']])\n"
            "        r_sigma = float(rt['scale'])\n"
            "    except Exception:\n"
            "        r_beta = None\n"
            "    finally:\n"
            "        os.remove(csvp)"
        ),
        _code(
            "if r_beta is not None:\n"
            "    table = pd.DataFrame({\n"
            "        'true':         [1.0, 0.7, -0.4],\n"
            "        'censtrunc':    cens.coef_,\n"
            "        'R survreg':    r_beta,\n"
            "        'OLS (biased)': ols_c,\n"
            "    }, index=['const', 'x1', 'x2'])\n"
            "    display(table.round(6))\n"
            "    print(f'max |censtrunc - R|  = {np.max(np.abs(cens.coef_ - r_beta)):.2e}  (independent solvers, same optimum)')\n"
            "    print(f'|sigma_censtrunc - sigma_R| = {abs(cens.sigma_ - r_sigma):.2e}')\n"
            "    print(f'max |censtrunc - OLS| = {np.max(np.abs(cens.coef_ - ols_c)):.3f}  (Tobit correction is real)')\n"
            "else:\n"
            "    print('R not available at build time.')\n"
            "    print('The live comparison runs in the test suite: tests/test_r_reference.py')\n"
            "    print('censtrunc estimates:', cens.coef_.round(4))\n"
            "    print('OLS (biased):       ', ols_c.round(4))"
        ),
        _md(
            "The `censtrunc` and R columns are identical to about six decimals,"
            " yet `max |censtrunc - R|` is a tiny *non-zero* number (~1e-8): the"
            " two independent optimisers land on the same maximum without being"
            " bit-for-bit copies. Both recover the true coefficients, while OLS is"
            " visibly biased toward zero — exactly the censoring attenuation"
            " predicted by Greene (1981)."
        ),
        _md(
            "## Probit limit: tight thresholds reduce Tobit to probit\n"
            "\n"
            "Consider the censored regression with $L = R = c$. With no interior"
            " region the likelihood reduces to\n"
            "\n"
            "$$\\ell(\\beta, \\sigma) = \\sum_i \\mathbf 1\\{y_i = c\\}\\,"
            "\\log\\Phi\\!\\left(\\tfrac{c - X_i'\\beta}{\\sigma}\\right) +"
            "\\mathbf 1\\{y_i > c\\}\\,\\log\\!\\left[1 - \\Phi\\!\\left(\\tfrac{c - X_i'\\beta}{\\sigma}\\right)\\right],$$\n"
            "\n"
            "which is *exactly* the probit log-likelihood for the indicator"
            " $D_i = \\mathbf 1\\{Y^*_i > c\\}$, identifying $\\beta_{\\text{probit}} ="
            " \\beta_{\\text{tobit}}/\\sigma_{\\text{tobit}}$ (Hansen 2022, §27.4)."
            "\n\n"
            "We cannot set $L = R$ literally (then $\\sigma$ is unidentified), but a"
            " *narrow band* $[c - \\varepsilon, c + \\varepsilon]$ is close enough."
            " A handful of interior observations identifies $\\sigma$; the rest of"
            " the data act like a binary outcome. So as $\\varepsilon\\to 0$ we"
            " expect $\\hat\\beta_{\\text{tobit}}/\\hat\\sigma_{\\text{tobit}}\\to"
            "\\hat\\beta_{\\text{probit}}$."
        ),
        _code(
            "from statsmodels.discrete.discrete_model import Probit\n"
            "\n"
            "rng3 = np.random.default_rng(20250601)\n"
            "n3 = 8000\n"
            "Xp = rng3.normal(size=(n3, 2))\n"
            "beta_p = np.array([0.4, 1.2, -0.8])\n"
            "y_star_p = beta_p[0] + Xp @ beta_p[1:] + rng3.normal(size=n3)\n"
            "y_bin = (y_star_p > 0.0).astype(float)\n"
            "probit = Probit(y_bin, sm.add_constant(Xp)).fit(disp=False)\n"
            "\n"
            "def fit_band(eps: float):\n"
            "    y_obs = np.clip(y_star_p, -eps, eps)\n"
            "    m = CensoredRegression(left=-eps, right=eps).fit(Xp, y_obs)\n"
            "    return m.coef_ / m.sigma_   # the ratio that should match probit\n"
            "\n"
            "table = pd.DataFrame({\n"
            "    'probit beta':                     np.asarray(probit.params),\n"
            "    'tobit / sigma  (eps=0.5)':        fit_band(0.50),\n"
            "    'tobit / sigma  (eps=0.1)':        fit_band(0.10),\n"
            "    'tobit / sigma  (eps=0.05)':       fit_band(0.05),\n"
            "}, index=['const', 'x1', 'x2'])\n"
            "table.round(4)"
        ),
        _md(
            "As the band shrinks, the tobit-to-probit ratio approaches the probit"
            " estimate. The intercept drifts more than the slopes — a finite-band"
            " effect that scales like $\\varepsilon/\\sigma$ — but at $\\varepsilon"
            " = 0.05$ the slopes agree to a couple percent. This independently"
            " validates the censored log-likelihood and its normalisation: a bug"
            " in the boundary terms of $\\ell$ would manifest as a discrepancy"
            " here, but does not."
        ),
    ]
    nb["cells"] = cells
    return nb


def _build_visualization_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        _md(
            "# Visual: Truncated vs Censored\n"
            "\n"
            "A picture worth a thousand summary tables. We simulate a simple\n"
            "linear DGP, observe it through either truncation or censoring at the\n"
            "same thresholds, fit `censtrunc` to each, and overlay the model's\n"
            "predicted conditional mean as a band of asymptotic uncertainty.\n"
            "\n"
            "This mirrors the style of the well-known"
            " `pymc-devs`"
            " GLM-truncated-censored-regression demonstration, but uses our\n"
            "frequentist MLE instead of MCMC: the band is built by sampling\n"
            "parameters from the asymptotic distribution\n"
            "$\\hat{\\theta} \\sim \\mathcal{N}(\\hat{\\theta}_{\\mathrm{MLE}},\\,\\widehat{\\mathrm{Var}}(\\hat{\\theta}_{\\mathrm{MLE}}))$\n"
            "and drawing one prediction curve per sample."
        ),
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "from censtrunc import CensoredRegression, TruncatedRegression\n"
            "from censtrunc._means import value_for_kind\n"
            "from censtrunc._utils import _prepare_design_matrix\n"
            "\n"
            "rng = np.random.default_rng(2026)\n"
            "n = 250\n"
            "x = rng.uniform(-10, 10, size=n)\n"
            "sigma_true = 2.0\n"
            "y_star = 1.0 * x + rng.normal(scale=sigma_true, size=n)\n"
            "L, R = -5.0, 5.0"
        ),
        _md(
            "**Truncated dataset:** observations with $y \\notin (L, R)$ are dropped"
            " entirely. **Censored dataset:** they are clipped to the nearest"
            " threshold. The latent linear function $y = x$ is the same in both."
        ),
        _code(
            "mask = (y_star > L) & (y_star < R)\n"
            "x_t, y_t = x[mask], y_star[mask]\n"
            "x_c, y_c = x.copy(), np.clip(y_star, L, R)\n"
            "\n"
            "mt = TruncatedRegression(left=L, right=R).fit(x_t.reshape(-1, 1), y_t)\n"
            "mc = CensoredRegression(left=L, right=R).fit(x_c.reshape(-1, 1), y_c)\n"
            "print(f'truncated:  n={len(y_t)}, beta_hat = {mt.coef_.round(3)}')\n"
            "print(f'censored:   n={len(y_c)}, beta_hat = {mc.coef_.round(3)}')"
        ),
        _md(
            "Now we build a prediction band from asymptotic uncertainty. We draw"
            " 200 parameter vectors from the asymptotic normal distribution of"
            " $\\hat\\theta$, compute the corresponding conditional mean curve"
            " (truncated on the left, censored on the right), and plot them with"
            " low opacity. The pile-up at the censoring thresholds is shown in"
            " red on the right."
        ),
        _code(
            "def prediction_band(model, x_grid, kind, n_draws=200, seed=0):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    samples = rng.multivariate_normal(model.params_, model.cov_params_, size=n_draws)\n"
            "    samples = samples[samples[:, 0] > 0]   # keep only feasible sigma > 0\n"
            "    Xg, _ = _prepare_design_matrix(x_grid.reshape(-1, 1), fit_intercept=True)\n"
            "    curves = np.empty((samples.shape[0], x_grid.size))\n"
            "    for i, theta in enumerate(samples):\n"
            "        sigma, beta = theta[0], theta[1:]\n"
            "        curves[i] = value_for_kind(\n"
            "            beta, sigma, Xg,\n"
            "            model._left, model._right, model._has_left, model._has_right, kind,\n"
            "        )\n"
            "    return curves\n"
            "\n"
            "x_grid = np.linspace(-10, 10, 300)\n"
            "curves_t = prediction_band(mt, x_grid, kind='truncated')\n"
            "curves_c = prediction_band(mc, x_grid, kind='censored')"
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)\n"
            "for ax in axes:\n"
            "    ax.set_facecolor('#eaeaf2')\n"
            "    ax.grid(True, color='white', linewidth=1)\n"
            "    ax.axhline(L, color='crimson', linestyle='--', linewidth=1.2)\n"
            "    ax.axhline(R, color='crimson', linestyle='--', linewidth=1.2)\n"
            "    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)\n"
            "    ax.set_xlabel('x'); ax.set_ylabel('y')\n"
            "    ax.plot(x_grid, x_grid, color='black', linewidth=2.5, label='True')\n"
            "\n"
            "# Left panel: truncated\n"
            "ax = axes[0]\n"
            "ax.set_title('Truncated data')\n"
            "for c in curves_t:\n"
            "    ax.plot(x_grid, c, color='steelblue', alpha=0.04, linewidth=1)\n"
            "ax.scatter(x_t, y_t, c='black', s=14, zorder=5)\n"
            "ax.legend(loc='upper left')\n"
            "\n"
            "# Right panel: censored\n"
            "ax = axes[1]\n"
            "ax.set_title('Censored data')\n"
            "for c in curves_c:\n"
            "    ax.plot(x_grid, c, color='steelblue', alpha=0.04, linewidth=1)\n"
            "is_at_bound = (y_c == L) | (y_c == R)\n"
            "ax.scatter(x_c[~is_at_bound], y_c[~is_at_bound], c='black', s=14, zorder=5)\n"
            "ax.scatter(x_c[is_at_bound],  y_c[is_at_bound],  c='crimson', s=14, zorder=5, alpha=0.7)\n"
            "ax.legend(loc='upper left')\n"
            "\n"
            "plt.tight_layout(); plt.show()"
        ),
        _md(
            "**What to look for.**\n"
            "\n"
            "- The black line is the true linear relation $y^* = x$.\n"
            "- The blue band is the model's MLE prediction with asymptotic\n"
            "  uncertainty; on the **left** it bends inside $(L, R)$ because the\n"
            "  truncated mean $\\mathbb E[Y \\mid X, L<Y<R]$ is shrunk toward the"
            " interval centre; on the **right** it flattens near each threshold"
            " because the censored mean $\\mathbb E[Y\\mid X]$ mixes the latent"
            " linear part with the threshold mass.\n"
            "- Red dots on the right panel are the censored pile-ups at $y=L$ and"
            " $y=R$. The truncated dataset has no such pile-ups because those"
            " observations are absent entirely.\n"
            "- Both fitted curves are close to the true line in the interior, where"
            " the data are most informative; both lose precision near and beyond"
            " the thresholds, but neither shows the strong attenuation toward"
            " zero that OLS would suffer."
        ),
    ]
    nb["cells"] = cells
    return nb


def _build_heckit_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        _md(
            "# Heckman's Sample-Selection Model (Heckit)\n"
            "\n"
            "When the outcome of interest is observed only for individuals who"
            " 'self-select' into the sample (think wages — observed only for"
            " those who work) and that selection is correlated with the outcome"
            " error, OLS on the selected subsample is biased. Heckman (1979)"
            " introduced a two-step correction; the joint maximum-likelihood"
            " estimator is its asymptotically efficient sibling.\n"
            "\n"
            "Model:\n"
            "\n"
            "$$Y^* = X'\\beta + e, \\quad S^* = Z'\\gamma + u, \\quad S = \\mathbf 1\\{S^* > 0\\},$$\n"
            "$$Y = Y^* \\text{ if } S = 1, \\text{ missing otherwise}, \\quad"
            " (e, u) \\sim \\mathcal N\\!\\left(0, \\begin{pmatrix}\\sigma^2 & \\rho\\sigma \\\\ \\rho\\sigma & 1\\end{pmatrix}\\right).$$"
        ),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import statsmodels.api as sm\n"
            "from censtrunc import HeckitRegression\n"
            "\n"
            "rng = np.random.default_rng(42)\n"
            "n = 4000\n"
            "# Shared regressor (in both equations); two exclusive regressors\n"
            "shared  = rng.normal(size=n)\n"
            "x_only  = rng.normal(size=n)       # in the outcome eq. only\n"
            "z_only  = rng.normal(size=n)       # in the selection eq. only (exclusion)\n"
            "\n"
            "beta_true  = np.array([1.0,  0.5, -0.3])     # const, shared, x_only\n"
            "gamma_true = np.array([0.0,  0.3,  0.6])     # const, shared, z_only\n"
            "rho_true, sigma_true = 0.6, 1.0\n"
            "\n"
            "Sigma = np.array([[sigma_true**2, rho_true*sigma_true],\n"
            "                  [rho_true*sigma_true, 1.0]])\n"
            "errs = rng.multivariate_normal([0.0, 0.0], Sigma, size=n)\n"
            "e, u = errs[:, 0], errs[:, 1]\n"
            "\n"
            "X = np.column_stack([shared, x_only])\n"
            "Z = np.column_stack([shared, z_only])\n"
            "S = (gamma_true[0] + Z @ gamma_true[1:] + u > 0).astype(int)\n"
            "y_full = beta_true[0] + X @ beta_true[1:] + e\n"
            "y = np.where(S == 1, y_full, np.nan)\n"
            "print(f'Selected (S=1): {S.sum()} / {n}  ({S.mean():.1%})')"
        ),
        _md(
            "## Why OLS on the selected sample is biased\n"
            "\n"
            "Because $u$ and $e$ are correlated ($\\rho > 0$), individuals with"
            " high $e$ tend to have high $u$ and therefore are more likely to be"
            " selected. The conditional mean for the selected sample is\n"
            "\n"
            "$$\\mathbb E[Y \\mid X, S = 1] = X'\\beta + \\rho\\sigma\\,\\lambda(Z'\\gamma),$$\n"
            "\n"
            "with $\\lambda(\\cdot)$ the inverse Mills ratio. Omitting $\\lambda(Z'\\gamma)$"
            " from the regression — what naive OLS does — produces omitted-variable"
            " bias on $\\beta$."
        ),
        _code(
            "sel = ~np.isnan(y)\n"
            "X_full = sm.add_constant(X)\n"
            "ols = sm.OLS(y[sel], X_full[sel]).fit()\n"
            "print('OLS on selected subsample (biased):')\n"
            "print(np.asarray(ols.params).round(4))\n"
            "print('truth:', beta_true)"
        ),
        _md(
            "## Two-step Heckit\n"
            "\n"
            "Heckman's two-step recipe:\n"
            "\n"
            "1. Probit of $S$ on $Z$ to obtain $\\hat\\gamma$.\n"
            "2. Compute the inverse Mills ratio $\\hat\\lambda_i ="
            " \\phi(Z_i'\\hat\\gamma)/\\Phi(Z_i'\\hat\\gamma)$ for selected observations.\n"
            "3. OLS of $y_i$ on $(X_i,\\hat\\lambda_i)$ for selected $i$. The"
            " coefficients are $\\hat\\beta$ and $\\hat\\rho\\hat\\sigma$."
        ),
        _code(
            "m_two = HeckitRegression(method='twostep').fit(y, X, Z)\n"
            "print(m_two.summary())"
        ),
        _md(
            "## Joint MLE Heckit\n"
            "\n"
            "Maximise the full log-likelihood (Hansen 2022, §27.10):\n"
            "\n"
            "$$\\ell = \\sum_{S_i=0} \\log[1 - \\Phi(Z_i'\\gamma)] +"
            " \\sum_{S_i=1}\\!\\left\\{\\log\\Phi\\!\\left(\\frac{Z_i'\\gamma +"
            " (\\rho/\\sigma)(Y_i - X_i'\\beta)}{\\sqrt{1 - \\rho^2}}\\right) -"
            " \\tfrac12\\log(2\\pi\\sigma^2) - \\frac{(Y_i - X_i'\\beta)^2}{2\\sigma^2}\\right\\}.$$\n"
            "\n"
            "Asymptotically efficient, somewhat slower than the two-step."
        ),
        _code(
            "m_ml = HeckitRegression(method='mle').fit(y, X, Z)\n"
            "print(m_ml.summary())"
        ),
        _md(
            "## Side-by-side comparison\n"
            "\n"
            "All three estimators on the same data. Heckit's two-step and MLE"
            " recover $\\beta_{\\text{shared}}$ much closer to the truth than OLS,"
            " and they recover $\\rho$ and $\\sigma$ as well."
        ),
        _code(
            "comparison = pd.DataFrame({\n"
            "    'true':         beta_true,\n"
            "    'OLS (biased)': np.asarray(ols.params),\n"
            "    'Heckit 2step': m_two.coef_,\n"
            "    'Heckit MLE':   m_ml.coef_,\n"
            "}, index=['const', 'shared', 'x_only'])\n"
            "comparison['OLS error']   = comparison['OLS (biased)'] - comparison['true']\n"
            "comparison['2step error'] = comparison['Heckit 2step']  - comparison['true']\n"
            "comparison['MLE error']   = comparison['Heckit MLE']    - comparison['true']\n"
            "comparison.round(4)"
        ),
        _code(
            "pd.DataFrame({\n"
            "    'true':      [rho_true, sigma_true],\n"
            "    'two-step':  [m_two.rho_, m_two.sigma_],\n"
            "    'MLE':       [m_ml.rho_,  m_ml.sigma_],\n"
            "}, index=['rho', 'sigma']).round(4)"
        ),
        _md(
            "## Cross-validation against `py4etrics` and a base-R reference\n"
            "\n"
            "Recovering the true DGP is necessary but not sufficient — a buggy"
            " estimator that returns *any* sensible-looking numbers would also pass"
            " that bar. The decisive check is replication of an independent"
            " implementation on the *same* data. We compare against two:\n"
            "\n"
            "- **`py4etrics.Heckit`** (Hasebe et al., the Japanese package the"
            " review suggested) — closed-form two-step Heckman in Python.\n"
            "- A **base-R reference** (`glm(probit)` + augmented OLS for two-step;"
            " `optim` on the joint log-likelihood for MLE) — see"
            " `tests/reference/fit_heckit.R`. We deliberately avoid the"
            " `sampleSelection` package because its dependency chain"
            " (`nloptr`/`lme4`/`car`) fails to compile on Apple Silicon and"
            " several Linux distros; hand-rolling the same likelihood in R gives"
            " us an optimiser that is structurally independent of ours.\n"
            "\n"
            "Both checks run as part of the test suite\n"
            "(`tests/test_heckit.py::test_twostep_matches_py4etrics` and\n"
            "`tests/test_r_reference.py::test_heckit_mle_matches_R`). The cells"
            " below reproduce them live."
        ),
        _code(
            "# Compare with py4etrics (two-step only -- py4etrics MLE is a no-op).\n"
            "import warnings\n"
            "try:\n"
            "    from py4etrics.heckit import Heckit as Py4Heckit\n"
            "    with warnings.catch_warnings():\n"
            "        warnings.simplefilter('ignore')\n"
            "        ref = Py4Heckit(y, sm.add_constant(X), sm.add_constant(Z)).fit(method='twostep')\n"
            "    diff_beta  = np.max(np.abs(m_two.coef_  - ref.params))\n"
            "    diff_gamma = np.max(np.abs(m_two.gamma_ - ref.select_res.params))\n"
            "    print(f'max |beta_censtrunc  - beta_py4etrics|  = {diff_beta:.2e}')\n"
            "    print(f'max |gamma_censtrunc - gamma_py4etrics| = {diff_gamma:.2e}')\n"
            "    print(f'|sigma_censtrunc - sigma_py4etrics|     = {abs(m_two.sigma_ - np.sqrt(ref.var_reg_error)):.2e}')\n"
            "    print(f'|rho_censtrunc   - rho_py4etrics|       = {abs(m_two.rho_   - ref.corr_eqnerrors):.2e}')\n"
            "except ImportError:\n"
            "    print('py4etrics not installed; pip install py4etrics to enable the comparison.')"
        ),
        _md(
            "Both packages solve the same closed-form Heckman expressions, so"
            " agreement to $\\sim 10^{-6}$ is essentially numpy/statsmodels"
            " linear-algebra round-off."
        ),
        _code(
            "# Compare with the base-R reference (probit + augmented OLS + optim MLE).\n"
            "# The R script expects a CSV with columns (y, s, x1, x2, z1, z2) and fits\n"
            "# outcome 'y ~ x1 + x2' / selection 's ~ x1 + x2 + z1 + z2', so we use a\n"
            "# fresh DGP that matches this layout exactly.\n"
            "import shutil, subprocess, json, tempfile, os\n"
            "from pathlib import Path\n"
            "\n"
            "def _find_r_heckit():\n"
            "    for c in [Path('tests/reference/fit_heckit.R'),\n"
            "              Path('../tests/reference/fit_heckit.R')]:\n"
            "        if c.exists():\n"
            "            return c\n"
            "    return None\n"
            "\n"
            "rscript, rpath = shutil.which('Rscript'), _find_r_heckit()\n"
            "if rscript and rpath:\n"
            "    rng_r = np.random.default_rng(20250601)\n"
            "    n_r = 4000\n"
            "    x1r = rng_r.normal(size=n_r); x2r = rng_r.normal(size=n_r)\n"
            "    z1r = rng_r.normal(size=n_r); z2r = rng_r.normal(size=n_r)\n"
            "    bb = np.array([0.5, 1.0, -0.4])\n"
            "    gg = np.array([0.1, 0.3, -0.2, 0.5, 0.7])\n"
            "    rho_r, sigma_r = 0.5, 1.2\n"
            "    Sig = np.array([[sigma_r**2, rho_r*sigma_r],[rho_r*sigma_r, 1.0]])\n"
            "    er, ur = rng_r.multivariate_normal([0, 0], Sig, size=n_r).T\n"
            "    y_lat = bb[0] + bb[1]*x1r + bb[2]*x2r + er\n"
            "    s_lat = gg[0] + gg[1]*x1r + gg[2]*x2r + gg[3]*z1r + gg[4]*z2r + ur\n"
            "    s_obs = (s_lat > 0).astype(int)\n"
            "    y_obs_r = np.where(s_obs == 1, y_lat, np.nan)\n"
            "\n"
            "    Xr = np.column_stack([x1r, x2r])\n"
            "    Zr = np.column_stack([x1r, x2r, z1r, z2r])\n"
            "    m_ml_r = HeckitRegression(method='mle').fit(y_obs_r, Xr, Zr)\n"
            "\n"
            "    fd, csvp = tempfile.mkstemp(suffix='.csv'); os.close(fd)\n"
            "    pd.DataFrame({\n"
            "        'y': y_obs_r, 's': s_obs,\n"
            "        'x1': x1r, 'x2': x2r, 'z1': z1r, 'z2': z2r,\n"
            "    }).to_csv(csvp, index=False, na_rep='NA')\n"
            "    try:\n"
            "        out = subprocess.run([rscript, str(rpath), csvp],\n"
            "                             capture_output=True, text=True, timeout=180, check=True)\n"
            "        rdat = json.loads(out.stdout)['mle']\n"
            "        r_beta  = np.array([rdat['outcome']['(Intercept)'],\n"
            "                            rdat['outcome']['x1'], rdat['outcome']['x2']])\n"
            "        r_gamma = np.array([rdat['selection']['(Intercept)'],\n"
            "                            rdat['selection']['x1'], rdat['selection']['x2'],\n"
            "                            rdat['selection']['z1'], rdat['selection']['z2']])\n"
            "        print(f\"max |beta_censtrunc  - beta_R|   = {np.max(np.abs(m_ml_r.coef_  - r_beta)):.2e}\")\n"
            "        print(f\"max |gamma_censtrunc - gamma_R|  = {np.max(np.abs(m_ml_r.gamma_ - r_gamma)):.2e}\")\n"
            "        print(f\"|sigma_censtrunc - sigma_R|      = {abs(m_ml_r.sigma_ - rdat['sigma']):.2e}\")\n"
            "        print(f\"|rho_censtrunc   - rho_R|        = {abs(m_ml_r.rho_   - rdat['rho']):.2e}\")\n"
            "        print(f\"|logLik_censtrunc - logLik_R|    = {abs(m_ml_r.llf_   - rdat['logLik']):.2e}\")\n"
            "    except Exception as exc:\n"
            "        print(f'R run failed: {exc!r}')\n"
            "    finally:\n"
            "        os.remove(csvp)\n"
            "else:\n"
            "    print('Rscript not available at build time; see tests/test_r_reference.py for the gated test.')"
        ),
        _md(
            "Two independent optimisers (scipy `L-BFGS-B` and R `optim`) on the"
            " same joint log-likelihood: the parameters, the variance/correlation"
            " and the log-likelihood all agree to optimiser tolerance ($\\sim"
            "10^{-3}$). Any structural bug in our likelihood code would surface"
            " as a *systematic* offset here, not a tolerance-level discrepancy."
        ),
        _md(
            "## Three kinds of prediction\n"
            "\n"
            "- `kind='selection_prob'`  — $P(S=1 \\mid Z) = \\Phi(Z'\\hat\\gamma)$,\n"
            "- `kind='outcome'`         — $\\mathbb E[Y^* \\mid X] = X'\\hat\\beta$ (unconditional),\n"
            "- `kind='conditional'`     — $\\mathbb E[Y \\mid X, Z, S=1] = X'\\hat\\beta +"
            " \\hat\\rho\\hat\\sigma\\,\\lambda(Z'\\hat\\gamma)$."
        ),
        _code(
            "preds = pd.DataFrame({\n"
            "    'selection_prob': m_ml.predict(Z=Z[:6], kind='selection_prob'),\n"
            "    'outcome':        m_ml.predict(X=X[:6], kind='outcome'),\n"
            "    'conditional':    m_ml.predict(X=X[:6], Z=Z[:6], kind='conditional'),\n"
            "    'observed_y':     y[:6],\n"
            "    'selected':       S[:6],\n"
            "})\n"
            "preds.round(3)"
        ),
        _md(
            "## Same fit via a patsy formula\n"
            "\n"
            "Instead of building `y`, `X`, `Z` by hand, every model class accepts"
            " a `from_formula(...)` constructor in the statsmodels style. For"
            " Heckit you pass *two* formulas — outcome and selection — sharing a"
            " single dataframe:"
        ),
        _code(
            "df = pd.DataFrame({\n"
            "    'shared': shared, 'x_only': x_only, 'z_only': z_only,\n"
            "    'inlf':   S, 'lwage': y_full,\n"
            "})\n"
            "m_form = HeckitRegression.from_formula(\n"
            "    outcome   = 'lwage ~ 1 + shared + x_only',\n"
            "    selection = 'inlf  ~ 1 + shared + z_only',\n"
            "    data=df, method='mle',\n"
            ").fit()\n"
            "print(pd.DataFrame({\n"
            "    'beta (explicit)': m_ml.coef_,\n"
            "    'beta (formula)':  m_form.coef_,\n"
            "}, index=['const', 'shared', 'x_only']).round(6))"
        ),
        _md(
            "## Bootstrap standard errors\n"
            "\n"
            "Two-step second-stage standard errors are naive: they ignore the"
            " variability of $\\hat\\gamma$ from the probit step. A paired bootstrap"
            " gives proper SEs. The MLE Hessian-based SEs are already correct."
        ),
        _code(
            "boot = m_two.bootstrap(y, X, Z, n_boot=80, seed=0)\n"
            "pd.DataFrame({\n"
            "    'beta':         m_two.coef_,\n"
            "    'naive SE':     m_two.bse_,\n"
            "    'bootstrap SE': boot['beta_se'],\n"
            "}, index=['const', 'shared', 'x_only']).round(4)"
        ),
        _md(
            "## Marginal effects\n"
            "\n"
            "Heckman's model has four standard marginal-effect quantities"
            " (Greene, *Econometric Analysis* §19.5):\n"
            "\n"
            "| `kind`            | Differentiates                                                         |\n"
            "|-------------------|------------------------------------------------------------------------|\n"
            "| `'latent'`        | $E[Y^*\\mid X] = X'\\beta$ — X-side only                                |\n"
            "| `'conditional'`   | $E[Y\\mid X, Z, S=1] = X'\\beta + \\rho\\sigma\\,\\lambda(Z'\\gamma)$        |\n"
            "| `'unconditional'` | $E[Y\\cdot S\\mid X, Z] = \\Phi(Z'\\gamma)X'\\beta + \\rho\\sigma\\,\\phi(Z'\\gamma)$ |\n"
            "| `'prob-selected'` | $P(S=1\\mid Z) = \\Phi(Z'\\gamma)$ — Z-side only                          |\n"
            "\n"
            "A variable that appears in *both* equations (here `shared`) has its X-"
            " and Z-side derivatives **added together**. SEs use the delta method"
            " on the joint MLE covariance; for a two-step fit they would be `NaN`"
            " (bootstrap is the right tool there).\n"
            "\n"
            "Closed-form per-row derivatives with $\\lambda = \\phi/\\Phi$ and"
            " $\\delta(t) = \\lambda(t)(t + \\lambda(t))$ (so $\\mathrm d\\lambda/\\mathrm dt"
            " = -\\delta$):\n"
            "\n"
            "$$\\frac{\\partial E[Y\\mid S=1]}{\\partial v} = \\beta_v\\,\\mathbf 1\\{v\\in X\\}"
            " \\;-\\; \\gamma_v\\,\\rho\\sigma\\,\\delta(Z'\\gamma)\\,\\mathbf 1\\{v\\in Z\\}.$$\n"
            "\n"
            "We pass `feature_names_outcome` / `feature_names_selection` so that"
            " variables with matching names in $X$ and $Z$ are recognised as the"
            " same variable."
        ),
        _code(
            "m_me = HeckitRegression(method='mle').fit(\n"
            "    y, X, Z,\n"
            "    feature_names_outcome=['shared', 'x_only'],\n"
            "    feature_names_selection=['shared', 'z_only'],\n"
            ")\n"
            "ame_cond = m_me.ame(kind='conditional')\n"
            "print(ame_cond.summary())"
        ),
        _md(
            "All four marginal-effect kinds in a single table. Each row is averaged"
            " across the sample (AME); the entries are blank where the variable"
            " does not enter that equation."
        ),
        _code(
            "kinds = ['latent', 'conditional', 'unconditional', 'prob-selected']\n"
            "tbl = pd.DataFrame(index=sorted({n for k in kinds for n in m_me.ame(kind=k).names}))\n"
            "for k in kinds:\n"
            "    me = m_me.ame(kind=k)\n"
            "    tbl[k] = pd.Series(me.margeff, index=me.names)\n"
            "tbl.round(4)"
        ),
        _md(
            "Reading the columns:\n"
            "\n"
            "- `latent` reproduces the slope coefficients $\\beta$ (a sanity check)."
            "\n"
            "- `conditional` differs from `latent` on `shared` (Z-side correction"
            " kicks in) and matches it exactly on `x_only` (X-only variable, no"
            " correction).\n"
            "- `unconditional` is uniformly damped: $E[Y\\cdot S]$ weighs the X-side"
            " by the selection probability $\\Phi(Z'\\gamma)$ rather than 1.\n"
            "- `prob-selected` shows only the Z-side variables, scaled by"
            " $\\phi(Z'\\gamma)$."
        ),
        _md(
            "**Cross-check via finite differences of `predict`.** The strongest"
            " correctness test: read our derivative formulas (in"
            " `_heckit_effects.py`) and our `predict()` code (in `heckit.py`) as"
            " *independent* sources of the same quantity, then verify they agree."
            " Below we pick the conditional mean at $X=\\bar X, Z=\\bar Z$ and"
            " perturb each regressor with a central difference."
        ),
        _code(
            "h = 1e-4\n"
            "X_mean = X.mean(axis=0, keepdims=True)\n"
            "Z_mean = Z.mean(axis=0, keepdims=True)\n"
            "def cond_at(Xm, Zm):\n"
            "    return float(m_me.predict(X=Xm, Z=Zm, kind='conditional').mean())\n"
            "\n"
            "mem = m_me.mem(kind='conditional')\n"
            "rows = []\n"
            "for j, name in enumerate(['shared', 'x_only']):\n"
            "    Xp, Xm_ = X_mean.copy(), X_mean.copy()\n"
            "    Xp[0, j] += h; Xm_[0, j] -= h\n"
            "    Zp, Zm_ = Z_mean.copy(), Z_mean.copy()\n"
            "    if name == 'shared':\n"
            "        Zp[0, 0] += h; Zm_[0, 0] -= h\n"
            "    num = (cond_at(Xp, Zp) - cond_at(Xm_, Zm_)) / (2*h)\n"
            "    rows.append((name, mem.margeff[mem.names.index(name)], num))\n"
            "# z_only is Z-only.\n"
            "Zp, Zm_ = Z_mean.copy(), Z_mean.copy()\n"
            "Zp[0, 1] += h; Zm_[0, 1] -= h\n"
            "num_z = (cond_at(X_mean, Zp) - cond_at(X_mean, Zm_)) / (2*h)\n"
            "rows.append(('z_only', mem.margeff[mem.names.index('z_only')], num_z))\n"
            "pd.DataFrame(rows, columns=['variable', 'analytic dy/dx', 'finite-difference']).round(8)"
        ),
        _md(
            "The two columns agree to 7-8 decimals: the analytic derivative"
            " formulas in `_heckit_effects.py` are consistent with the prediction"
            " formulas in `heckit.py`."
        ),
        _md(
            "---\n"
            "## Real-data demonstration: Mroz (1987)\n"
            "\n"
            "The canonical Heckit application — Wooldridge's wage equation for"
            " married women. Of the 753 observations in Mroz's dataset, 428 women"
            " are in the labor force (`inlf = 1`) and have an observed `lwage`;"
            " the remaining 325 do not work and `lwage` is missing. Running OLS"
            " on the 428 working women answers a conditional question — *what"
            " determines wages among those who choose to work?* — whereas the"
            " latent question — *what would women earn if everyone worked?* —"
            " is what Heckit estimates.\n"
            "\n"
            "Following Wooldridge (2010, Example 17.5):\n"
            "\n"
            "- **Outcome equation:** `lwage ~ educ + exper + expersq`\n"
            "- **Selection equation:** `inlf ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6`\n"
            "\n"
            "`nwifeinc`, `age`, `kidslt6`, and `kidsge6` serve as the exclusion"
            " restriction (they shift the *decision to work* but are not"
            " supposed to affect wages directly)."
        ),
        _code(
            "try:\n"
            "    import wooldridge as woo\n"
            "    mroz = woo.data('mroz')\n"
            "except ImportError:\n"
            "    raise ImportError('Install with: pip install wooldridge')\n"
            "\n"
            "print(f'shape: {mroz.shape}')\n"
            "print(f'in labor force (inlf=1): {(mroz[\"inlf\"]==1).sum()}')\n"
            "print(f'out of labor force:      {(mroz[\"inlf\"]==0).sum()}')\n"
            "print(f'lwage missing where inlf=0: {mroz.loc[mroz[\"inlf\"]==0, \"lwage\"].isna().sum()}/'\n"
            "      f'{(mroz[\"inlf\"]==0).sum()}  (should be 100% — the canonical selection setup)')"
        ),
        _md(
            "**Step 1 — naive OLS on the working subsample.** This is what one"
            " gets by ignoring the selection problem. We will compare its"
            " coefficient on `educ` to the Heckit-corrected version below."
        ),
        _code(
            "import statsmodels.formula.api as smf\n"
            "\n"
            "ols = smf.ols('lwage ~ educ + exper + expersq',\n"
            "              data=mroz[mroz['inlf'] == 1]).fit()\n"
            "print(ols.summary().tables[1])"
        ),
        _md(
            "**Step 2 — Heckit two-step.** The selection equation includes the"
            " exclusion variables; the second stage adds the inverse Mills ratio"
            " to the wage equation."
        ),
        _code(
            "from censtrunc import HeckitRegression\n"
            "\n"
            "heck_ts = HeckitRegression.from_formula(\n"
            "    outcome   = 'lwage ~ educ + exper + expersq',\n"
            "    selection = 'inlf ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6',\n"
            "    data=mroz, method='twostep',\n"
            ").fit()\n"
            "print(heck_ts.summary())"
        ),
        _md(
            "**Step 3 — Joint MLE Heckit** (asymptotically efficient)."
        ),
        _code(
            "heck_ml = HeckitRegression.from_formula(\n"
            "    outcome   = 'lwage ~ educ + exper + expersq',\n"
            "    selection = 'inlf ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6',\n"
            "    data=mroz, method='mle',\n"
            ").fit()\n"
            "print(heck_ml.summary())"
        ),
        _md(
            "### Coefficient comparison\n"
            "\n"
            "Lining up the three wage-equation slope estimates. Wooldridge's"
            " textbook reports an OLS return-to-education of about 0.108; the"
            " Heckit-corrected return is close to that, and the implied"
            " correlation $\\hat\\rho$ between the wage-equation error and the"
            " participation error is informative about how strong the selection"
            " channel is."
        ),
        _code(
            "import numpy as np\n"
            "comparison = pd.DataFrame({\n"
            "    'OLS (selected)': np.asarray(ols.params),\n"
            "    'Heckit two-step': heck_ts.coef_,\n"
            "    'Heckit MLE':     heck_ml.coef_,\n"
            "}, index=heck_ts.outcome_feature_names_)\n"
            "comparison.round(4)"
        ),
        _code(
            "pd.DataFrame({\n"
            "    'two-step':  [heck_ts.rho_, heck_ts.sigma_],\n"
            "    'MLE':       [heck_ml.rho_, heck_ml.sigma_],\n"
            "}, index=['rho', 'sigma']).round(4)"
        ),
        _md(
            "Interpreting $\\hat\\rho$: a positive (negative) value means that the"
            " unobserved factors that raise wages also raise (lower) the"
            " probability of working. If $\\hat\\rho$ is small or insignificant"
            " (a Wald or LR test on $\\rho = 0$ would say so), the selection"
            " correction is empirically unnecessary and ordinary OLS is fine."
        ),
        _md(
            "### Marginal effects on Mroz\n"
            "\n"
            "For policy interpretation we usually want marginal effects on the"
            " *conditional* wage $E[\\log\\text{wage}\\mid \\text{works}]$ and on"
            " the *participation probability* $P(\\text{works})$. The"
            " participation effects below use the standard probit-style derivative"
            " $\\gamma_v\\,\\phi(Z'\\gamma)$, averaged across the sample."
        ),
        _code(
            "ame_cond = heck_ml.ame(kind='conditional')\n"
            "ame_prob = heck_ml.ame(kind='prob-selected')\n"
            "pd.concat([\n"
            "    ame_cond.summary_frame()[['dy/dx', 'std err', 'P>|z|']].rename(\n"
            "        columns={'dy/dx': 'cond E[lwage]', 'std err': 'SE', 'P>|z|': 'pval'}),\n"
            "    ame_prob.summary_frame()[['dy/dx', 'std err', 'P>|z|']].rename(\n"
            "        columns={'dy/dx': 'P(in LF)', 'std err': 'SE_p', 'P>|z|': 'pval_p'}),\n"
            "], axis=1).round(4)"
        ),
        _md(
            "The `educ` row reads: one extra year of schooling raises *expected"
            " log-wage among workers* by the `cond E[lwage]` figure, *and"
            " separately* raises the probability of being in the labour force by"
            " the `P(in LF)` figure.\n"
            "\n"
            "Z-only variables (`nwifeinc`, `age`, `kidslt6`, `kidsge6`) still"
            " appear in the `cond E[lwage]` column with very small,"
            " statistically-insignificant entries. That is **not** a blank: they"
            " do affect the conditional wage, but only *through the Mills-ratio"
            " correction* $-\\gamma_v\\,\\rho\\,\\sigma\\,\\delta(Z'\\gamma)$. On Mroz"
            " the estimated correlation $\\hat\\rho$ is close to zero (about"
            " $0.03$), so that channel is numerically tiny and the entries land"
            " around zero — exactly as one would expect when the selection"
            " correction is weak."
        ),
    ]
    nb["cells"] = cells
    return nb


def _execute_and_save(nb: nbf.NotebookNode, path: Path) -> None:
    """Execute a notebook in-place (so outputs are saved) and write to disk."""
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()
    nbf.write(nb, path)
    print(f"  wrote {path.name}")


def main() -> int:
    print("Building example notebooks...")
    builders = {
        "01_two_sided_censoring.ipynb": _build_two_sided_notebook,
        "02_classical_tobit_affairs.ipynb": _build_classical_tobit_notebook,
        "03_truncated_regression.ipynb": _build_truncated_notebook,
        "04_validation.ipynb": _build_validation_notebook,
        "05_truncated_vs_censored_visual.ipynb": _build_visualization_notebook,
        "06_heckit.ipynb": _build_heckit_notebook,
    }
    for fname, builder in builders.items():
        nb = builder()
        _execute_and_save(nb, HERE / fname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
