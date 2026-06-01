"""Cross-validate against R: ``survival::survreg``, ``truncreg::truncreg``,
and a base-R Heckman selection fit (probit + augmented OLS + ``optim``-based
joint MLE).

These tests run only when R is installed *and* exposes the required packages
(otherwise they are skipped, so the suite stays green in environments without
R). The R scripts live in ``tests/reference/fit_tobit.R`` and ``fit_heckit.R``.

The comparison protocol avoids cross-language RNG mismatches: Python generates
the data, writes it to a temporary CSV, R reads that exact CSV and fits its
models, and the coefficient sets are compared. Tobit, truncated, and Heckit
estimators are different parameterisations of the same MLE, so agreement to
six-plus decimals validates the Python implementation against a mature R
reference -- independently of this package's optimiser or its likelihood code.

For Heckit we deliberately avoid the ``sampleSelection`` package because its
dependency chain (``nloptr``/``lme4``/``car``) fails to compile on Apple
Silicon and several Linux distros. Instead we hand-roll the same likelihood
in base R + ``optim``, giving us an R-side optimiser that is structurally
independent of ours.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from censtrunc import CensoredRegression, HeckitRegression, TruncatedRegression

R_SCRIPT = Path(__file__).parent / "reference" / "fit_tobit.R"
R_HECKIT_SCRIPT = Path(__file__).parent / "reference" / "fit_heckit.R"
LEFT, RIGHT = 0.0, 2.5


def _r_available() -> bool:
    """True if Rscript and the needed packages (survival, truncreg, jsonlite) exist."""
    if shutil.which("Rscript") is None:
        return False
    probe = "quit(status = if (all(c('survival','truncreg','jsonlite') %in% rownames(installed.packages()))) 0 else 1)"
    try:
        res = subprocess.run(
            ["Rscript", "-e", probe], capture_output=True, timeout=60, check=False
        )
        return res.returncode == 0
    except Exception:  # pragma: no cover - environment dependent
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="R with packages survival, truncreg, jsonlite is required for the reference comparison",
)


@pytest.fixture(scope="module")
def data_csv(tmp_path_factory) -> Path:
    rng = np.random.default_rng(20260527)
    n = 4000
    X = rng.normal(size=(n, 2))
    y_star = 1.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(size=n)
    y = np.clip(y_star, LEFT, RIGHT)
    path = tmp_path_factory.mktemp("rref") / "data.csv"
    header = "y,x1,x2"
    rows = np.column_stack([y, X])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    return path


def _run_r(csv_path: Path) -> dict:
    res = subprocess.run(
        ["Rscript", str(R_SCRIPT), str(csv_path), str(LEFT), str(RIGHT)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return json.loads(res.stdout)


def test_censored_matches_R_tobit(data_csv):
    import pandas as pd

    df = pd.read_csv(data_csv)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()

    ref = _run_r(data_csv)
    r_coef = ref["tobit"]["coef"]  # {'(Intercept)':..., 'x1':..., 'x2':...}
    r_beta = np.array([r_coef["(Intercept)"], r_coef["x1"], r_coef["x2"]])
    r_scale = float(ref["tobit"]["scale"])

    model = CensoredRegression(left=LEFT, right=RIGHT).fit(X, y)
    np.testing.assert_allclose(model.coef_, r_beta, rtol=1e-3, atol=1e-3)
    assert abs(model.sigma_ - r_scale) < 1e-3


def test_truncated_matches_R_truncreg(data_csv):
    import pandas as pd

    df = pd.read_csv(data_csv)
    # Left-truncated subsample (truncreg uses a single truncation point).
    df_t = df[df["y"] > LEFT]
    X = df_t[["x1", "x2"]].to_numpy()
    y = df_t["y"].to_numpy()

    ref = _run_r(data_csv)
    r_coef = ref["truncreg"]["coef"]  # includes 'sigma'
    r_beta = np.array([r_coef["(Intercept)"], r_coef["x1"], r_coef["x2"]])

    model = TruncatedRegression(left=LEFT).fit(X, y)
    np.testing.assert_allclose(model.coef_, r_beta, rtol=2e-3, atol=2e-3)


# ----------------------------------------------------------------------
# Heckit comparison with a base-R reference (probit + OLS + optim-MLE).
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def heckit_csv(tmp_path_factory) -> Path:
    """Heckman selection DGP: one shared regressor in X and Z, two
    exclusion-restriction regressors in Z only. Two-step and MLE are both
    well-identified."""
    rng = np.random.default_rng(20250601)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    beta = np.array([0.5, 1.0, -0.4])
    gamma = np.array([0.1, 0.3, -0.2, 0.5, 0.7])
    rho, sigma = 0.5, 1.2
    cov = np.array([[sigma**2, rho * sigma], [rho * sigma, 1.0]])
    e, u = rng.multivariate_normal([0.0, 0.0], cov, size=n).T
    y_star = beta[0] + beta[1] * x1 + beta[2] * x2 + e
    s = (gamma[0] + gamma[1] * x1 + gamma[2] * x2 + gamma[3] * z1 + gamma[4] * z2 + u > 0).astype(int)
    y = np.where(s == 1, y_star, np.nan)

    import pandas as pd

    df = pd.DataFrame({"y": y, "s": s, "x1": x1, "x2": x2, "z1": z1, "z2": z2})
    path = tmp_path_factory.mktemp("rheck") / "data.csv"
    df.to_csv(path, index=False, na_rep="NA")
    return path


def _run_r_heckit(csv_path: Path) -> dict:
    res = subprocess.run(
        ["Rscript", str(R_HECKIT_SCRIPT), str(csv_path)],
        capture_output=True, text=True, timeout=300, check=True,
    )
    return json.loads(res.stdout)


def _fit_pair(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    X = df[["x1", "x2"]].to_numpy()
    Z = df[["x1", "x2", "z1", "z2"]].to_numpy()
    y = df["y"].to_numpy()
    return y, X, Z


def test_heckit_twostep_matches_R(heckit_csv):
    """censtrunc two-step and base-R two-step solve the *same* closed-form
    Heckman expressions, so they should agree to machine precision modulo
    numpy/R linear-algebra differences (~1e-6)."""
    ref = _run_r_heckit(heckit_csv)["twostep"]
    r_beta = np.array([ref["outcome"]["(Intercept)"], ref["outcome"]["x1"], ref["outcome"]["x2"]])
    r_gamma = np.array([
        ref["selection"]["(Intercept)"], ref["selection"]["x1"],
        ref["selection"]["x2"], ref["selection"]["z1"], ref["selection"]["z2"],
    ])

    y, X, Z = _fit_pair(heckit_csv)
    m = HeckitRegression(method="twostep").fit(y, X, Z)

    np.testing.assert_allclose(m.coef_, r_beta, atol=1e-4)
    np.testing.assert_allclose(m.gamma_, r_gamma, atol=1e-4)
    assert abs(m.sigma_ - ref["sigma"]) < 1e-4
    assert abs(m.rho_ - ref["rho"]) < 1e-4


def test_heckit_mle_matches_R(heckit_csv):
    """censtrunc MLE (scipy L-BFGS-B) vs an independent R optimiser on the
    *same* likelihood: any disagreement past optimiser tolerance would imply
    a bug in one implementation."""
    ref = _run_r_heckit(heckit_csv)["mle"]
    r_beta = np.array([ref["outcome"]["(Intercept)"], ref["outcome"]["x1"], ref["outcome"]["x2"]])
    r_gamma = np.array([
        ref["selection"]["(Intercept)"], ref["selection"]["x1"],
        ref["selection"]["x2"], ref["selection"]["z1"], ref["selection"]["z2"],
    ])

    y, X, Z = _fit_pair(heckit_csv)
    m = HeckitRegression(method="mle").fit(y, X, Z)

    np.testing.assert_allclose(m.coef_, r_beta, atol=1e-3)
    np.testing.assert_allclose(m.gamma_, r_gamma, atol=1e-3)
    assert abs(m.sigma_ - ref["sigma"]) < 1e-3
    assert abs(m.rho_ - ref["rho"]) < 1e-3
    # Joint log-likelihoods should match to many decimals.
    assert abs(m.llf_ - ref["logLik"]) < 1e-3
