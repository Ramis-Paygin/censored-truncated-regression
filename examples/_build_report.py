"""Assemble a single combined report notebook from the overview text and the
three example notebooks, then (optionally) execute it so outputs are embedded.

The resulting ``censtrunc_report.ipynb`` is meant to be converted to PDF with

    jupyter nbconvert --to pdf examples/censtrunc_report.ipynb

(LaTeX route) or ``--to webpdf`` (headless-Chromium route). Building the report
separately from the per-topic notebooks keeps each example self-contained while
still producing one readable document for review.

Usage
-----
    python examples/_build_report.py            # build + execute
    python examples/_build_report.py --no-exec  # build without executing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SOURCE_NOTEBOOKS = [
    ("Example 1 — Two-Sided Censored Regression", "01_two_sided_censoring.ipynb"),
    ("Example 2 — Classical Tobit (Fair Affairs Data)", "02_classical_tobit_affairs.ipynb"),
    ("Example 3 — Truncated Regression", "03_truncated_regression.ipynb"),
    ("Example 4 — Validation against OLS and R", "04_validation.ipynb"),
    ("Example 5 — Visual: Truncated vs Censored", "05_truncated_vs_censored_visual.ipynb"),
    ("Example 6 — Heckman Selection (Heckit)", "06_heckit.ipynb"),
]


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


_PRINT_CSS = """<style>
/* --- print/PDF tuning ---------------------------------------------------
   The statsmodels-style summary() output is 88 characters wide. At the
   template's default font size those lines wrap awkwardly in the PDF, so we
   shrink monospaced output (stdout text and code) just enough to fit one line
   on the page. Screen rendering of the notebook is unaffected when viewed in
   Jupyter, since this style only matters for the static HTML/PDF export. */
.jp-OutputArea-output pre,
.jp-RenderedText pre,
div.output_text pre,
pre {
    font-size: 9.5px !important;
    line-height: 1.25 !important;
    white-space: pre !important;
}
.jp-InputArea-editor,
.highlight pre,
.jp-CodeMirrorEditor {
    font-size: 10.5px !important;
}
@page { margin: 1.3cm; }
</style>"""


def _title_cells() -> list[nbf.NotebookNode]:
    return [
        _md(_PRINT_CSS),
        _md(
            "# censtrunc\n"
            "## Censored and Truncated Normal Regression with Arbitrary Thresholds\n"
            "\n"
            "**Author:** &lt;Your Name&gt;  \n"
            "**Package:** `censtrunc` v0.1.0  \n"
            "\n"
            "---\n"
            "\n"
            "### Abstract\n"
            "\n"
            "`censtrunc` is a Python package for maximum-likelihood estimation of"
            " censored and truncated normal regression models with **arbitrary left"
            " and right thresholds**, plus the Heckman (1979) sample-selection"
            " model. It generalises the classical Type-1 Tobit (Tobin, 1958) and the"
            " truncated normal regression to two-sided limits, exposes a"
            " scikit-learn-style API, a statsmodels-style results summary, marginal"
            " effects, likelihood-ratio tests, and patsy-style formulas. Estimation"
            " uses Olsen's (1978) globally concave reparameterisation for the"
            " censored case. The report opens with a survey of what is and is not"
            " available in Python for these models today, then states the"
            " statistical model and presents six worked examples."
        ),
        _md(
            "## 1. Python landscape, and what `censtrunc` adds\n"
            "\n"
            "Compared to R -- where `survival::survreg` (wrapped by `AER::tobit`)"
            " covers the censored case, `truncreg::truncreg` covers truncation, and"
            " `sampleSelection::selection` covers Heckman selection with both"
            " estimators -- the Python ecosystem for these models is fragmented.\n"
            "\n"
            "| Package | Censored / Tobit | Truncated | Heckit | Notes |\n"
            "|---------|------------------|-----------|--------|-------|\n"
            "| **statsmodels** | no public Tobit class | not in the public API | not implemented | The obvious place to look first; the documented gap. |\n"
            "| **scikit-learn** | — | — | — | No limited-dependent-variable models. |\n"
            "| **linearmodels** | — | — | — | Panel / IV / GMM specialist. |\n"
            "| **lifelines** | left-censored Weibull / log-normal AFT | — | — | Survival framing: hazard-ratio coefficients, awkward for non-zero $L$. |\n"
            "| **scikit-survival** | right-censored, AFT-style | — | — | Same survival framing. |\n"
            "| **py4etrics** | `Tobit` with arbitrary $L, R$ | `Truncreg` with arbitrary $L, R$ | two-step `Heckit` (`method='mle'` silently ignored) | Educational package; no marginal effects, no LR test, no formula API. |\n"
            "| **stnwanekezie/TobitRegression** | a single `Tobit` class on top of `statsmodels.OLS` with Olsen's reparameterisation | — | — | 425-line script; `summary()` only. |\n"
            "| **PyMC / bambi** | Bayesian via `Censored` | Bayesian via `Truncated` | possible to hand-code | Bayesian / MCMC route. |\n"
            "| **marginaleffects** | — | — | — | Slopes/contrasts toolkit, not a fitting library. |\n"
            "\n"
            "### What `censtrunc` adds on top\n"
            "\n"
            "- **Two-sided censoring and truncation** with arbitrary $L$, $R$ in"
            " every estimator (one-sided becomes a special case via `left=None` or"
            " `right=None`).\n"
            "- **Heckman selection with joint MLE** (matching"
            " `sampleSelection::selection`), not just the two-step. The two-step"
            " ships with a paired bootstrap for the proper Heckman-corrected SEs.\n"
            "- **Marginal effects** through a single `get_margeff` method mirroring"
            " statsmodels' API, for all three estimators. Censored models get the"
            " three region-probability kinds (`prob-left`, `prob-interior`,"
            " `prob-right`) in addition to `latent` / `censored` / `truncated`;"
            " Heckit gets the four Heckman quantities (`latent`, `conditional`,"
            " `unconditional`, `prob-selected`), with variables matched by name"
            " across the outcome and selection equations.\n"
            "- **Likelihood-ratio test** in two forms: between any pair of fitted"
            " models, or `model.lr_test(hypotheses)` with statsmodels-style"
            " hypothesis strings (`'(x1 = 0), (x2 = x3)'`, `'2*x1 + x4 = 1'`).\n"
            "- **Patsy formulas** on all three classes; the same model can be fit"
            " either by passing arrays `(X, y[, Z])` or via"
            " `Model.from_formula(formula, data=df, ...)`.\n"
            "- A unified `predict()` returning any combination of **six** quantities"
            " (three conditional means + three region probabilities) via a"
            " one-letter `kind` string -- e.g. `'hctlmr'` for all of them, `'lmr'`"
            " for just the probabilities.\n"
            "- **Cross-validation against external implementations** in the test"
            " suite: OLS-equivalence and probit-equivalence limits, R's"
            " `survival::survreg`, `truncreg::truncreg`,"
            " `sampleSelection::selection`, and `py4etrics.Heckit`."
        ),
        _md(
            "## 2. Statistical Model\n"
            "\n"
            "### 2.1 Latent regression\n"
            "\n"
            "All models share the latent linear specification with normal errors:\n"
            "\n"
            "$$Y^* = X'\\beta + e, \\qquad e \\mid X \\sim \\mathcal{N}(0, \\sigma^2).$$\n"
            "\n"
            "What differs is the rule linking the latent $Y^*$ to the observed data.\n"
            "\n"
            "### 2.2 Censored model\n"
            "\n"
            "Values outside $[L, R]$ are **clipped** to the nearest threshold:\n"
            "\n"
            "$$Y = \\min\\bigl(R,\\; \\max(L,\\; Y^*)\\bigr).$$\n"
            "\n"
            "The log-likelihood combines the probability mass at each threshold with\n"
            "the normal density on the interior. With\n"
            "$\\alpha_L = (L - X'\\beta)/\\sigma$ and $\\alpha_R = (R - X'\\beta)/\\sigma$,\n"
            "\n"
            "$$\\ell(\\beta, \\sigma) = \\sum_{Y_i = L} \\log\\Phi(\\alpha_{L,i})\n"
            "+ \\sum_{Y_i = R} \\log\\bigl[1 - \\Phi(\\alpha_{R,i})\\bigr]\n"
            "+ \\sum_{L < Y_i < R} \\Bigl[\\log\\phi(\\alpha_i) - \\log\\sigma\\Bigr],$$\n"
            "\n"
            "where $\\phi, \\Phi$ are the standard normal pdf and cdf. The classical\n"
            "Tobit is the special case $L = 0$, $R = +\\infty$.\n"
            "\n"
            "### 2.3 Olsen's reparameterisation\n"
            "\n"
            "Optimisation is carried out in $(\\gamma, \\nu) = (\\beta/\\sigma,\\, 1/\\sigma)$.\n"
            "Olsen (1978) showed the resulting log-likelihood is **globally concave**,\n"
            "so any gradient-based optimiser converges to the unique maximum. After\n"
            "convergence the package back-transforms to $(\\beta, \\sigma)$ and computes\n"
            "standard errors from the observed information matrix.\n"
            "\n"
            "### 2.4 Truncated model\n"
            "\n"
            "In a truncated sample, observations outside $(L, R)$ are **absent**\n"
            "entirely. The density is the normal density renormalised by the\n"
            "probability of falling inside the interval:\n"
            "\n"
            "$$f(y_i \\mid L < Y_i^* < R) =\n"
            "\\frac{\\sigma^{-1}\\,\\phi\\!\\bigl((y_i - X_i'\\beta)/\\sigma\\bigr)}\n"
            "     {\\Phi(\\alpha_{R,i}) - \\Phi(\\alpha_{L,i})}.$$\n"
            "\n"
            "### 2.5 Conditional means and marginal effects\n"
            "\n"
            "Three conditional means are reported (Hansen, 2022, ch. 27):\n"
            "\n"
            "- **Latent:** $\\mathbb{E}[Y^* \\mid X] = X'\\beta$\n"
            "- **Censored:** $\\mathbb{E}[Y \\mid X]$, accounting for the threshold mass\n"
            "- **Truncated:** $\\mathbb{E}[Y \\mid X,\\, L < Y < R]$\n"
            "\n"
            "Marginal effects are available as **AME** (average over the sample) and\n"
            "**MEM** (evaluated at the mean regressor), each with delta-method standard\n"
            "errors. The censored marginal effect on $\\mathbb{E}[Y\\mid X]$ equals\n"
            "$\\beta_j \\cdot [\\Phi(\\alpha_R) - \\Phi(\\alpha_L)]$ — the latent slope\n"
            "shrunk by the probability of being interior.\n"
            "\n"
            "### 2.6 Inference\n"
            "\n"
            "Every fitted model carries an overall likelihood-ratio test against the\n"
            "intercept-only null model and McFadden's pseudo-$R^2$. Arbitrary nested\n"
            "models can be compared with `lr_test`, which returns\n"
            "$LR = 2(\\ell_{\\text{full}} - \\ell_{\\text{restricted}}) \\sim \\chi^2_q$."
        ),
        _md(
            "## 3. Worked Examples\n"
            "\n"
            "The remainder of this report reproduces the six example notebooks."
            " Each is fully executed; the code and its output appear inline."
        ),
    ]


def build_report(execute: bool = True) -> Path:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = list(_title_cells())

    for idx, (heading, fname) in enumerate(SOURCE_NOTEBOOKS, start=1):
        cells.append(_md(f"---\n\n## 3.{idx} {heading}"))
        src = nbf.read(HERE / fname, as_version=4)
        # Skip the first cell of each source notebook (its own H1 title) to avoid
        # duplicate top-level headings; keep everything else.
        src_cells = src.cells[1:] if src.cells and src.cells[0].cell_type == "markdown" else src.cells
        cells.extend(src_cells)

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "censtrunc — Censored and Truncated Regression",
    }

    out_path = HERE / "censtrunc_report.ipynb"
    if execute:
        print("Executing combined report (this runs all example code)...")
        client = NotebookClient(nb, timeout=180, kernel_name="python3")
        client.execute()
    nbf.write(nb, out_path)
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-exec", action="store_true", help="build without executing")
    args = parser.parse_args()
    build_report(execute=not args.no_exec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
