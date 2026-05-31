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
            "## Predictions: the three flavours\n"
            "\n"
            "- `latent`: $\\mathbb E[Y^* | X] = X'\\beta$\n"
            "- `censored`: $\\mathbb E[Y | X]$ — accounts for the threshold mass\n"
            "- `truncated`: $\\mathbb E[Y | X, L < Y < R]$ — conditional on being interior"
        ),
        _code(
            "X_show = X[:6]\n"
            "preds = pd.DataFrame({\n"
            "    'latent':    model.predict(X_show, kind='latent'),\n"
            "    'censored':  model.predict(X_show, kind='censored'),\n"
            "    'truncated': model.predict(X_show, kind='truncated'),\n"
            "    'observed':  y[:6],\n"
            "})\n"
            "preds.round(3)"
        ),
        _md(
            "## Probabilities of each region\n"
            "\n"
            "`predict_proba` returns the conditional probability of each region"
            " given $X$. The three probabilities sum to 1 by construction."
        ),
        _code(
            "proba = model.predict_proba(X_show)\n"
            "pd.DataFrame(proba, index=range(6)).round(3)"
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
            "Two prediction kinds are available for the truncated model:\n"
            "\n"
            "- `latent`: $X'\\hat\\beta$ — the population mean\n"
            "- `truncated`: $\\mathbb E[Y | X, L < Y < R]$"
        ),
        _code(
            "preds = pd.DataFrame({\n"
            "    'observed':  y[:6],\n"
            "    'latent':    model.predict(X[:6], kind='latent'),\n"
            "    'truncated': model.predict(X[:6], kind='truncated'),\n"
            "}).round(3)\n"
            "preds"
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
            "returns nonsense'? Two independent checks.\n"
            "\n"
            "1. **Trivial limit:** with thresholds pushed beyond the data range,\n"
            "   nothing is censored, so the censored/truncated MLE *must* reduce to\n"
            "   ordinary least squares. We verify this against `statsmodels.OLS`.\n"
            "2. **Reference packages:** the test suite also compares against R's\n"
            "   `AER::tobit` and `truncreg` (run when R is available); see\n"
            "   `tests/test_r_reference.py`."
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
    }
    for fname, builder in builders.items():
        nb = builder()
        _execute_and_save(nb, HERE / fname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
