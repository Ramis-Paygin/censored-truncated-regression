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
            "## Average marginal effects (AME)\n"
            "\n"
            "Latent AME equals $\\beta$ exactly. The censored AME is $\\beta$"
            " shrunk by the probability of being interior. The truncated AME"
            " shrinks further. Standard errors are computed via the delta method."
        ),
        _code(
            "ame_latent    = model.ame(kind='latent').to_dataframe()\n"
            "ame_censored  = model.ame(kind='censored').to_dataframe()\n"
            "ame_truncated = model.ame(kind='truncated').to_dataframe()\n"
            "\n"
            "print('AME — latent\\n', ame_latent.round(4))\n"
            "print('\\nAME — censored\\n', ame_censored.round(4))\n"
            "print('\\nAME — truncated\\n', ame_truncated.round(4))"
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
    }
    for fname, builder in builders.items():
        nb = builder()
        _execute_and_save(nb, HERE / fname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
