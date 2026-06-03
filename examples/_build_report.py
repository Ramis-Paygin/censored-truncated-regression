"""Assemble the combined PDF report.

The report is structured around the two model families that the package
addresses, exactly as the supervisor recommended:

  Part A (Censored / Truncated regression)
    A.1  Theory
    A.2  Worked applications (examples 1, 2, 3, 5)
    A.3  Empirical correctness checks (example 4)

  Part B (Heckman sample-selection)
    B.1  Theory
    B.2  Worked applications (extracted from example 6)
    B.3  Empirical correctness checks (extracted from example 6)

Source notebooks are kept intact; this script slices Example 6 into
its "application" and "validation" halves on the fly so the assembled
report follows the theory → application → validation rhythm twice.

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
@page {
    margin: 1.3cm 1.3cm 1.6cm 1.3cm;   /* extra bottom margin for the page number */
    @bottom-right {
        content: counter(page);
        font-family: sans-serif;
        font-size: 9px;
        color: #666;
    }
}
</style>"""


# ----------------------------------------------------------------------
# Title + introduction + Python-landscape cells
# ----------------------------------------------------------------------


def _intro_cells() -> list[nbf.NotebookNode]:
    return [
        _md(_PRINT_CSS),
        _md(
            "# censtrunc\n"
            "## Censored and Truncated Normal Regression with Arbitrary Thresholds\n"
            "\n"
            "**Author:** Ramis Paygin  \n"
            "**Package:** `censtrunc` v0.1.0  \n"
            "**Repository:** [github.com/Ramis-Paygin/censored-truncated-regression]"
            "(https://github.com/Ramis-Paygin/censored-truncated-regression)  \n"
            "\n"
            "---\n"
            "\n"
            "### Abstract\n"
            "\n"
            "`censtrunc` is a Python package for maximum-likelihood estimation of"
            " two families of regression models with limited dependent variables:"
            " (i) **censored and truncated normal regression** with arbitrary left"
            " and right thresholds — generalising the classical Tobit (Tobin, 1958)"
            " — and (ii) **Heckman's sample-selection model** (Heckman, 1979),"
            " with both the two-step estimator and joint MLE. The package exposes"
            " a scikit-learn-style fitting API, a statsmodels-style summary,"
            " marginal effects via `get_margeff`, the likelihood-ratio test in"
            " hypothesis-string form, and patsy formulas. Estimation uses Olsen's"
            " (1978) globally concave reparameterisation in the censored case.\n"
            "\n"
            "The report is organised in two parts following the two model families."
            " Each part contains three sections: **theory**, **applications**"
            " demonstrating the API on simulated and real data, and **empirical"
            " correctness checks** — analytical limits and cross-validation"
            " against independent implementations."
        ),
        _md(
            "## 1. Introduction\n"
            "\n"
            "Linear regression assumes the dependent variable is observed on the"
            " whole real line. In practice it is often observed on a restricted"
            " range:\n"
            "\n"
            "- **Censored**: values outside an interval $[L, R]$ are pushed to"
            " the nearest threshold. A satisfaction score on a 0–10 scale or a"
            " salary capped at an institutional ceiling are typical examples.\n"
            "- **Truncated**: values outside $[L, R]$ are absent entirely. Survey"
            " data that only records respondents above an income threshold, or"
            " administrative data on transactions above a reporting limit, behave"
            " this way.\n"
            "- **Selected**: the variable is observed only when a separate"
            " *selection equation* fires. Wages are observed only for those in"
            " the labour force; consumption is observed only for those who"
            " chose to participate in the survey.\n"
            "\n"
            "Ordinary least squares is biased in all three cases. The classical"
            " solutions are: **Tobit** (Tobin, 1958) for one-sided censoring at"
            " zero, **truncated normal regression** for one-sided truncation,"
            " and **Heckman's selection model** (Heckman, 1979) for the third"
            " case.\n"
            "\n"
            "`censtrunc` packages all three in a single Python library with a"
            " consistent API and a common evaluation surface (`predict`,"
            " `get_margeff`, `lr_test`, `from_formula`). Sections 3 and 4 below"
            " cover the two model families in detail."
        ),
        _md(
            "## 2. What is already available in Python\n"
            "\n"
            "Compared to R — where `survival::survreg` (used by `AER::tobit`)"
            " covers censoring, `truncreg::truncreg` covers truncation, and"
            " `sampleSelection::selection` covers Heckman selection — the"
            " Python ecosystem for these models is fragmented.\n"
            "\n"
            "| Package | Censored / Tobit | Truncated | Heckit | Notes |\n"
            "|---|---|---|---|---|\n"
            "| **statsmodels** | no public Tobit class | not in the public API | not implemented | The natural first place to look; this is the documented gap. |\n"
            "| **scikit-learn** | — | — | — | No limited-dependent-variable models. |\n"
            "| **linearmodels** | — | — | — | Panel / IV / GMM specialist. |\n"
            "| **lifelines** | left-censored Weibull / log-normal AFT | — | — | Survival framing: hazard-ratio coefficients; non-zero $L$ is awkward to express. |\n"
            "| **scikit-survival** | right-censored, AFT-style | — | — | Same survival framing. |\n"
            "| **py4etrics** | `Tobit` with arbitrary $L, R$ | `Truncreg` with arbitrary $L, R$ | two-step `Heckit` (`method='mle'` silently ignored) | Educational package; no marginal effects, no LR test, no formula API. |\n"
            "| **stnwanekezie/TobitRegression** | a single `Tobit` class on top of `statsmodels.OLS` | — | — | 425-line script; `summary()` only. |\n"
            "| **PyMC / bambi** | Bayesian via `Censored` | Bayesian via `Truncated` | possible to hand-code | Bayesian / MCMC route. |\n"
            "| **marginaleffects** | — | — | — | Slopes/contrasts toolkit, not a fitting library. |\n"
            "\n"
            "### What `censtrunc` adds on top\n"
            "\n"
            "- **Two-sided censoring and truncation** with arbitrary $L$, $R$"
            " in every estimator (one-sided is a special case).\n"
            "- **Heckman selection with joint MLE**, not just two-step, plus a"
            " paired bootstrap for two-step standard errors.\n"
            "- **Marginal effects** through a single `get_margeff` method with"
            " statsmodels-style options, for all three model classes. Censored"
            " models also expose three region-probability kinds; Heckit exposes"
            " four (`latent`, `conditional`, `unconditional`, `prob-selected`).\n"
            "- **Likelihood-ratio test** in two forms: between any pair of"
            " fitted models, or `model.lr_test(hypotheses)` with hypothesis-"
            "string syntax (`'(x1 = 0), (x2 = x3)'`, `'2*x1 + x4 = 1'`).\n"
            "- **Patsy formulas** on all three classes.\n"
            "- A unified `predict()` returning any combination of six quantities"
            " via a one-letter `kind` string — for the censored model `'hctlmr'`,"
            " for the Heckman model `'snpohu'`.\n"
            "- **Cross-validation against external implementations** in the"
            " test suite — see §3.3 and §4.3."
        ),
    ]


# ----------------------------------------------------------------------
# Part A: Censored / Truncated
# ----------------------------------------------------------------------


def _part_a_cells() -> list[nbf.NotebookNode]:
    """Headings and theory text for Part A. Application and validation
    notebooks are embedded by build_report() after these."""
    return [
        _md(
            "---\n"
            "\n"
            "# Part A. Censored and Truncated Normal Regression"
        ),
        _md(
            "## 3.1 Theory\n"
            "\n"
            "### Latent regression\n"
            "\n"
            "Both models share the latent linear specification with normal errors:\n"
            "\n"
            "$$Y^* = X'\\beta + e, \\qquad e \\mid X \\sim \\mathcal{N}(0, \\sigma^2).$$\n"
            "\n"
            "What differs is the rule linking the latent $Y^*$ to the observed"
            " data.\n"
            "\n"
            "### Censored model\n"
            "\n"
            "Values outside $[L, R]$ are **clipped** to the nearest threshold:\n"
            "\n"
            "$$Y = \\min\\bigl(R,\\; \\max(L,\\; Y^*)\\bigr).$$\n"
            "\n"
            "With $\\alpha_L = (L - X'\\beta)/\\sigma$ and"
            " $\\alpha_R = (R - X'\\beta)/\\sigma$, the log-likelihood combines"
            " the probability mass at each threshold with the normal density on"
            " the interior:\n"
            "\n"
            "$$\\ell(\\beta, \\sigma) = \\sum_{Y_i = L} \\log\\Phi(\\alpha_{L,i})\n"
            "+ \\sum_{Y_i = R} \\log\\bigl[1 - \\Phi(\\alpha_{R,i})\\bigr]\n"
            "+ \\sum_{L < Y_i < R} \\Bigl[\\log\\phi(\\alpha_i) - \\log\\sigma\\Bigr],$$\n"
            "\n"
            "where $\\phi, \\Phi$ are the standard normal pdf and cdf. The"
            " classical Tobit is the special case $L = 0$, $R = +\\infty$.\n"
            "\n"
            "### Olsen's reparameterisation\n"
            "\n"
            "Optimisation is carried out in $(\\gamma, \\nu) = (\\beta/\\sigma,\\, 1/\\sigma)$."
            " Olsen (1978) showed the resulting log-likelihood is **globally"
            " concave**, so any gradient-based optimiser converges to the unique"
            " maximum. After convergence the package back-transforms to"
            " $(\\beta, \\sigma)$ and computes standard errors from the observed"
            " information matrix.\n"
            "\n"
            "### Truncated model\n"
            "\n"
            "In a truncated sample, observations outside $(L, R)$ are **absent**"
            " entirely. The density is the normal density renormalised by the"
            " probability of falling inside the interval:\n"
            "\n"
            "$$f(y_i \\mid L < Y_i^* < R) =\n"
            "\\frac{\\sigma^{-1}\\,\\phi\\!\\bigl((y_i - X_i'\\beta)/\\sigma\\bigr)}\n"
            "     {\\Phi(\\alpha_{R,i}) - \\Phi(\\alpha_{L,i})}.$$\n"
            "\n"
            "### Conditional means and marginal effects\n"
            "\n"
            "Three conditional means are reported (Hansen, 2022, ch. 27):\n"
            "\n"
            "- **Latent:** $\\mathbb{E}[Y^* \\mid X] = X'\\beta$\n"
            "- **Censored:** $\\mathbb{E}[Y \\mid X]$, with the threshold mass\n"
            "- **Truncated:** $\\mathbb{E}[Y \\mid X,\\, L < Y < R]$\n"
            "\n"
            "Marginal effects are available as **AME** (average over the sample)"
            " and **MEM** (at the mean regressor), each with delta-method"
            " standard errors. The censored marginal effect on $\\mathbb{E}[Y\\mid X]$"
            " equals $\\beta_j \\cdot [\\Phi(\\alpha_R) - \\Phi(\\alpha_L)]$ — the"
            " latent slope multiplied by the probability of being interior.\n"
            "\n"
            "### Inference\n"
            "\n"
            "Every fitted model carries an overall likelihood-ratio test against"
            " the intercept-only null and McFadden's pseudo-$R^2$. Arbitrary"
            " nested models can be compared with `lr_test`, which returns"
            " $LR = 2(\\ell_{\\text{full}} - \\ell_{\\text{restricted}}) \\sim \\chi^2_q$."
        ),
        _md(
            "## 3.2 Applications\n"
            "\n"
            "Four worked examples demonstrate the API on increasingly involved"
            " data:\n"
            "\n"
            "- 3.2.1 two-sided censoring on simulated data,\n"
            "- 3.2.2 classical left-censored Tobit on the Fair (1978) affairs"
            " dataset,\n"
            "- 3.2.3 truncated regression on simulated data,\n"
            "- 3.2.4 side-by-side visual of truncation vs censoring."
        ),
    ]


# Examples that go into §3.2 (application) and §3.3 (validation).
PART_A_APPLICATION_NOTEBOOKS = [
    ("3.2.1 Two-sided censored regression", "01_two_sided_censoring.ipynb"),
    ("3.2.2 Classical Tobit on the Fair (1978) affairs data", "02_classical_tobit_affairs.ipynb"),
    ("3.2.3 Truncated regression", "03_truncated_regression.ipynb"),
    ("3.2.4 Visual: truncated vs censored", "05_truncated_vs_censored_visual.ipynb"),
]

PART_A_VALIDATION_HEADING = (
    "## 3.3 Empirical correctness checks\n"
    "\n"
    "Three independent analytical / numerical checks on the censored MLE:\n"
    "\n"
    "1. **No-censoring limit** — with thresholds pushed beyond the data range,"
    " the MLE must coincide with OLS;\n"
    "2. **Probit limit** — with $L \\approx R$, the censored log-likelihood"
    " reduces to a probit log-likelihood on $\\mathbf 1\\{Y^* > c\\}$ with"
    " $\\beta_{\\text{tobit}}/\\sigma_{\\text{tobit}} = \\beta_{\\text{probit}}$"
    " (Hansen 2022, §27.4);\n"
    "3. **Cross-validation with R** — `survival::survreg` (the engine behind"
    " `AER::tobit`) and `truncreg::truncreg` solve the same MLE; the two"
    " implementations must converge to the same point."
)

PART_A_VALIDATION_NOTEBOOK = ("3.3 Validation", "04_validation.ipynb")


# ----------------------------------------------------------------------
# Part B: Heckman
# ----------------------------------------------------------------------


def _part_b_cells() -> list[nbf.NotebookNode]:
    return [
        _md(
            "---\n"
            "\n"
            "# Part B. Heckman's Sample-Selection Model"
        ),
        _md(
            "## 4.1 Theory\n"
            "\n"
            "### The selection problem\n"
            "\n"
            "Suppose the outcome of interest $Y$ is observed only when an"
            " individual is *selected* into the sample by an unobserved"
            " mechanism. The canonical example is the wage equation: $Y ="
            " \\log W$ is observed only for those in the labour force —"
            " the standard Mincer specification [11]. The model is (with"
            " $Y^*$ denoting latent log-wage)\n"
            "\n"
            "$$\n"
            "Y^* = X'\\beta + e, \\qquad S^* = Z'\\gamma + u, \\qquad"
            " S = \\mathbf 1\\{S^* > 0\\},\n"
            "$$\n"
            "$$Y = Y^* \\text{ if } S = 1, \\text{ missing otherwise}, \\qquad\n"
            "(e, u) \\sim \\mathcal N\\!\\left(0,\n"
            " \\begin{pmatrix}\\sigma^2 & \\rho\\sigma \\\\ \\rho\\sigma & 1\\end{pmatrix}\\right).$$\n"
            "\n"
            "The variance of $u$ is normalised to one for identification (the"
            " probit convention). When $\\rho \\ne 0$, OLS on the selected"
            " subsample is biased because\n"
            "\n"
            "$$\\mathbb E[Y \\mid X, Z, S=1] = X'\\beta + \\rho\\sigma\\,\\lambda(Z'\\gamma),$$\n"
            "\n"
            "with $\\lambda(t) = \\phi(t)/\\Phi(t)$ the inverse Mills ratio."
            " Omitting $\\lambda(Z'\\gamma)$ from the regression — as naive OLS"
            " does — is a classical omitted-variable bias.\n"
            "\n"
            "### Heckman's two-step procedure\n"
            "\n"
            "1. **Probit** of $S$ on $Z$ gives $\\hat\\gamma$.\n"
            "2. **Inverse Mills ratio** $\\hat\\lambda_i = \\phi(Z_i'\\hat\\gamma)/\\Phi(Z_i'\\hat\\gamma)$"
            " is computed for selected observations.\n"
            "3. **OLS** of $y_i$ on $(X_i, \\hat\\lambda_i)$ on the selected"
            " subsample yields $\\hat\\beta$ and the coefficient $\\hat\\rho\\hat\\sigma$"
            " on the Mills ratio.\n"
            "\n"
            "Two-step naive second-stage SEs ignore the variability of"
            " $\\hat\\gamma$ from the probit step; a paired bootstrap"
            " (`model.bootstrap`) corrects for this.\n"
            "\n"
            "### Joint maximum likelihood\n"
            "\n"
            "The full log-likelihood is (Hansen, 2022, §27.10):\n"
            "\n"
            "$$\\ell = \\sum_{S_i=0} \\log[1 - \\Phi(Z_i'\\gamma)] +"
            " \\sum_{S_i=1}\\!\\left\\{\\log\\Phi\\!\\left(\\frac{Z_i'\\gamma +"
            " (\\rho/\\sigma)(Y_i - X_i'\\beta)}{\\sqrt{1 - \\rho^2}}\\right) -"
            " \\tfrac12\\log(2\\pi\\sigma^2) - \\frac{(Y_i - X_i'\\beta)^2}{2\\sigma^2}\\right\\}.$$\n"
            "\n"
            "Joint MLE is asymptotically efficient and somewhat slower than the"
            " two-step. The internal optimisation uses $\\rho = \\tanh(\\eta)$ to"
            " keep $|\\rho| < 1$ unconstrained.\n"
            "\n"
            "### Six predicted quantities\n"
            "\n"
            "`HeckitRegression.predict()` exposes six quantities through a"
            " one-letter `kind` argument; default `'snpohu'` returns all six in a"
            " DataFrame.\n"
            "\n"
            "**Selection equation** (uses $Z$):\n"
            "\n"
            "- **`s`** — selection probability $\\Phi(Z'\\hat\\gamma)$\n"
            "- **`n`** — non-selection probability $1 - \\Phi(Z'\\hat\\gamma)$\n"
            "- **`p`** — selection propensity (latent index) $Z'\\hat\\gamma$\n"
            "\n"
            "**Outcome equation** (uses $X$, sometimes $Z$):\n"
            "\n"
            "- **`o`** — observed conditional mean"
            " $\\mathbb E[Y \\mid X, Z, S=1] = X'\\hat\\beta + \\hat\\rho\\hat\\sigma\\,\\lambda(Z'\\hat\\gamma)$\n"
            "- **`h`** — hidden / latent mean $\\mathbb E[Y^* \\mid X] = X'\\hat\\beta$\n"
            "- **`u`** — unobserved conditional mean"
            " $\\mathbb E[Y^* \\mid X, Z, S=0] = X'\\hat\\beta - \\hat\\rho\\hat\\sigma\\,\\phi(Z'\\hat\\gamma)/[1-\\Phi(Z'\\hat\\gamma)]$\n"
            "\n"
            "### Marginal effects\n"
            "\n"
            "Greene's four standard quantities (Greene, *Econometric Analysis*,"
            " §19.5) are available via `model.get_margeff(kind=...)`:\n"
            "\n"
            "| `kind` | Differentiates |\n"
            "|---|---|\n"
            "| `'latent'` | $E[Y^*\\mid X] = X'\\beta$ — only X-side |\n"
            "| `'conditional'` | $E[Y\\mid X, Z, S=1]$ |\n"
            "| `'unconditional'` | $E[Y\\cdot S\\mid X, Z] = \\Phi(Z'\\gamma) X'\\beta + \\rho\\sigma\\,\\phi(Z'\\gamma)$ |\n"
            "| `'prob-selected'` | $P(S=1\\mid Z) = \\Phi(Z'\\gamma)$ — only Z-side |\n"
            "\n"
            "With $\\delta(t) = \\lambda(t)(t + \\lambda(t))$ (so $\\mathrm d\\lambda/\\mathrm dt = -\\delta$),"
            " the per-row conditional derivative for variable $v$ is\n"
            "\n"
            "$$\\frac{\\partial E[Y\\mid S=1]}{\\partial v} = \\beta_v\\,\\mathbf 1\\{v\\in X\\}"
            " \\;-\\; \\gamma_v\\,\\rho\\sigma\\,\\delta(Z'\\gamma)\\,\\mathbf 1\\{v\\in Z\\}.$$\n"
            "\n"
            "Variables that appear in both equations (matched by feature name)"
            " have their X- and Z-side derivatives added. Standard errors use"
            " the delta method on the joint MLE covariance."
        ),
    ]


# Markers in the Heckit notebook that switch between application and
# validation sections. Each entry is (substring, new_state).
HECKIT_SECTION_MARKERS = [
    # Validation block 1: cross-validation against external implementations.
    ("Cross-validation against",        "validation"),
    # Application resumes: predict / formula / bootstrap / margeff / Mroz.
    ("Six kinds of prediction",         "application"),
    # Validation block 2: finite-difference cross-check of derivative formulas.
    ("Cross-check via finite differences", "validation"),
    # Application resumes: Mroz real-data analysis.
    ("Real-data demonstration",         "application"),
]


def _split_heckit_cells(nb: nbf.NotebookNode) -> tuple[list, list]:
    """Slice Example 6 into (application_cells, validation_cells) by section
    headings. The very first cell of the source notebook is its own H1 title
    and is dropped (Part B already has its own heading)."""
    cells_in = nb.cells[1:]
    application: list = []
    validation: list = []
    state = "application"
    for cell in cells_in:
        if cell.cell_type == "markdown":
            for needle, new_state in HECKIT_SECTION_MARKERS:
                if needle in cell.source:
                    state = new_state
                    break
        (validation if state == "validation" else application).append(cell)
    return application, validation


# ----------------------------------------------------------------------
# build_report
# ----------------------------------------------------------------------


def build_report(execute: bool = True) -> Path:
    nb = nbf.v4.new_notebook()
    cells: list = []

    # --- Front matter: title, intro, Python landscape -----------------
    cells.extend(_intro_cells())

    # --- Part A: theory -----------------------------------------------
    cells.extend(_part_a_cells())

    # --- Part A: application examples (§3.2) --------------------------
    for heading, fname in PART_A_APPLICATION_NOTEBOOKS:
        cells.append(_md(f"---\n\n### {heading}"))
        src = nbf.read(HERE / fname, as_version=4)
        # drop the example's own H1 title cell
        src_cells = src.cells[1:] if (
            src.cells and src.cells[0].cell_type == "markdown"
        ) else src.cells
        cells.extend(src_cells)

    # --- Part A: validation (§3.3) ------------------------------------
    cells.append(_md("---\n\n" + PART_A_VALIDATION_HEADING))
    _, val_fname = PART_A_VALIDATION_NOTEBOOK
    src = nbf.read(HERE / val_fname, as_version=4)
    val_cells = src.cells[1:] if (
        src.cells and src.cells[0].cell_type == "markdown"
    ) else src.cells
    cells.extend(val_cells)

    # --- Part B: theory -----------------------------------------------
    cells.extend(_part_b_cells())

    # --- Part B: application + validation (split Example 6) -----------
    heckit_src = nbf.read(HERE / "06_heckit.ipynb", as_version=4)
    app_cells, val_cells = _split_heckit_cells(heckit_src)

    cells.append(_md(
        "## 4.2 Applications\n"
        "\n"
        "Demonstrations on simulated data, then on the canonical"
        " Mroz (1987) wage-equation dataset: how to call the two-step and"
        " MLE estimators, how to request the six prediction quantities,"
        " how to use the patsy-formula API, how to bootstrap the two-step"
        " standard errors, and how to read the marginal-effects table."
    ))
    cells.extend(app_cells)

    cells.append(_md(
        "---\n\n## 4.3 Empirical correctness checks\n"
        "\n"
        "Three independent checks on the Heckit estimator:\n"
        "\n"
        "1. **DGP recovery** is shown in §4.2 (the simulated data has known"
        " true parameters; the two-step and MLE estimates land close to them).\n"
        "2. **Cross-validation with independent implementations** —"
        " `py4etrics.Heckit` (two-step only) and R's"
        " `sampleSelection::selection` (two-step and joint MLE) on the very"
        " same data. Both must converge to the same point as our"
        " implementation up to optimiser tolerance.\n"
        "3. **Finite-difference cross-check** of the derivative formulas in"
        " `_heckit_effects.py` against the prediction code in `heckit.py`,"
        " which are written independently."
    ))
    cells.extend(val_cells)

    # --- Bibliography -------------------------------------------------
    cells.append(_md(
        "---\n\n## References\n\n"
        "[1] Tobin, J. (1958). Estimation of Relationships for Limited Dependent"
        " Variables. *Econometrica*, 26(1), 24–36.\n\n"
        "[2] Olsen, R. J. (1978). A Note on the Uniqueness of the Maximum"
        " Likelihood Estimator for the Tobit Model. *Econometrica*, 46(5),"
        " 1211–1215.\n\n"
        "[3] Heckman, J. J. (1979). Sample Selection Bias as a Specification"
        " Error. *Econometrica*, 47(1), 153–161.\n\n"
        "[4] Greene, W. H. (1981). On the Asymptotic Bias of the Ordinary Least"
        " Squares Estimator of the Tobit Model. *Econometrica*, 49(2), 505–513.\n\n"
        "[5] Fair, R. C. (1978). A Theory of Extramarital Affairs."
        " *Journal of Political Economy*, 86(1), 45–61.\n\n"
        "[6] Mroz, T. A. (1987). The Sensitivity of an Empirical Model of"
        " Married Women's Hours of Work to Economic and Statistical Assumptions."
        " *Econometrica*, 55(4), 765–799.\n\n"
        "[7] Toomet, O., & Henningsen, A. (2008). Sample Selection Models in R:"
        " Package sampleSelection. *Journal of Statistical Software*, 27(7).\n\n"
        "[8] Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and"
        " Panel Data* (2nd ed.). MIT Press.\n\n"
        "[9] Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.\n\n"
        "[10] Hansen, B. E. (2022). *Econometrics*. Princeton University Press.\n\n"
        "[11] Mincer, J. (1974). *Schooling, Experience, and Earnings*."
        " New York: Columbia University Press for the National Bureau of"
        " Economic Research."
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "censtrunc — Censored, Truncated, and Heckman Regression",
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
    parser.add_argument("--no-exec", action="store_true",
                        help="build the notebook without executing it")
    args = parser.parse_args()
    build_report(execute=not args.no_exec)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
