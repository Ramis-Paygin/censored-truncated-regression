"""Cross-validate against R's ``AER::tobit`` and ``truncreg::truncreg``.

This test runs only when R is installed *and* exposes the required packages.
Otherwise it is skipped, so the suite stays green in environments without R
(including most CI runners). The R script lives in ``tests/reference/fit_tobit.R``.

The comparison protocol avoids cross-language RNG mismatches: Python generates
the data, writes it to a temporary CSV, R reads that exact CSV and fits its
models, and the two sets of coefficients are compared. Tobit and truncated
estimators are different parameterisations of the same MLE, so agreement to a
few decimals validates the Python implementation against a mature reference.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from censtrunc import CensoredRegression, TruncatedRegression

R_SCRIPT = Path(__file__).parent / "reference" / "fit_tobit.R"
LEFT, RIGHT = 0.0, 2.5


def _r_available() -> bool:
    """True if Rscript and the needed packages (AER, truncreg, jsonlite) exist."""
    if shutil.which("Rscript") is None:
        return False
    probe = "quit(status = if (all(c('AER','truncreg','jsonlite') %in% rownames(installed.packages()))) 0 else 1)"
    try:
        res = subprocess.run(
            ["Rscript", "-e", probe], capture_output=True, timeout=60, check=False
        )
        return res.returncode == 0
    except Exception:  # pragma: no cover - environment dependent
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="R with packages AER, truncreg, jsonlite is required for the reference comparison",
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
