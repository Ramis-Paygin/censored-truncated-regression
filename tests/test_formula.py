"""Tests for the formula-based ``from_formula`` constructors.

For each model class we verify that fitting via ``from_formula`` produces the
same parameter estimates as fitting on equivalent explicit ``(X, y)`` (and ``Z``
for Heckit) inputs. We also check that patsy transforms (``np.log``, ``I()``,
``C()``) propagate correctly through to the feature names and that calling
``fit()`` twice or without data raises clean errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

patsy = pytest.importorskip("patsy")

from censtrunc import CensoredRegression, HeckitRegression, TruncatedRegression  # noqa: E402


@pytest.fixture
def censored_df():
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame({
        "educ":  rng.normal(13, 3, n),
        "exper": rng.uniform(0, 40, n),
        "age":   rng.normal(40, 10, n),
    })
    df["expersq"] = df["exper"] ** 2
    ystar = 0.2*df["educ"] + 0.05*df["exper"] - 0.001*df["expersq"] + rng.normal(scale=0.5, size=n)
    df["lwage"] = np.clip(ystar, 0.0, None)
    return df


def test_censored_formula_matches_explicit(censored_df):
    df = censored_df
    formula = "lwage ~ 1 + educ + exper + expersq"
    m_form = CensoredRegression.from_formula(formula, data=df, left=0.0).fit()
    m_expl = CensoredRegression(left=0.0).fit(df[["educ", "exper", "expersq"]], df["lwage"])
    np.testing.assert_allclose(m_form.coef_, m_expl.coef_, atol=1e-9)
    np.testing.assert_allclose(m_form.sigma_, m_expl.sigma_, atol=1e-9)
    assert m_form.feature_names_ == ["const", "educ", "exper", "expersq"]


def test_censored_formula_with_transforms(censored_df):
    df = censored_df
    # patsy transforms: np.log and I() should work transparently
    m = CensoredRegression.from_formula(
        "lwage ~ educ + I(exper**2)", data=df, left=0.0
    ).fit()
    # Names reflect the patsy column labels (with 'Intercept' renamed to 'const')
    assert m.feature_names_[0] == "const"
    assert "educ" in m.feature_names_
    assert any("exper" in nm for nm in m.feature_names_)


def test_censored_formula_no_intercept(censored_df):
    """`-1` suppresses the intercept, just like in R/statsmodels."""
    df = censored_df
    m = CensoredRegression.from_formula(
        "lwage ~ -1 + educ + exper", data=df, left=0.0
    ).fit()
    assert "const" not in m.feature_names_


def test_censored_fit_without_data_after_formula(censored_df):
    df = censored_df
    m = CensoredRegression.from_formula("lwage ~ educ", data=df, left=0.0)
    m.fit()  # no args allowed because formula was given
    assert m._fitted


def test_censored_fit_no_args_no_formula_raises():
    m = CensoredRegression(left=0.0)
    with pytest.raises(ValueError, match="from_formula"):
        m.fit()


# ----------------------------------------------------------------------
# Truncated
# ----------------------------------------------------------------------


def test_truncated_formula_matches_explicit():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y_star = 1.0 + 0.5*df["x1"] - 0.3*df["x2"] + rng.normal(size=n)
    df["y"] = y_star
    mask = (y_star > 0) & (y_star < 2.5)
    df_t = df[mask]

    m_form = TruncatedRegression.from_formula("y ~ x1 + x2", data=df_t, left=0.0, right=2.5).fit()
    m_expl = TruncatedRegression(left=0.0, right=2.5).fit(df_t[["x1", "x2"]], df_t["y"])
    np.testing.assert_allclose(m_form.coef_, m_expl.coef_, atol=1e-6)


# ----------------------------------------------------------------------
# Heckit (two formulas)
# ----------------------------------------------------------------------


def test_heckit_formula_matches_explicit():
    rng = np.random.default_rng(42)
    n = 3000
    df = pd.DataFrame({
        "shared":  rng.normal(size=n),
        "x_only":  rng.normal(size=n),
        "z_only":  rng.normal(size=n),
    })
    beta_true = np.array([1.0, 0.5, -0.3])
    gamma_true = np.array([0.0, 0.3, 0.6])
    Sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    errs = rng.multivariate_normal([0.0, 0.0], Sigma, size=n)
    df["inlf"] = (
        (gamma_true[0] + df["shared"]*gamma_true[1] + df["z_only"]*gamma_true[2] + errs[:, 1]) > 0
    ).astype(int)
    df["lwage"] = beta_true[0] + df["shared"]*beta_true[1] + df["x_only"]*beta_true[2] + errs[:, 0]

    m_form = HeckitRegression.from_formula(
        outcome="lwage ~ 1 + shared + x_only",
        selection="inlf ~ 1 + shared + z_only",
        data=df,
        method="twostep",
    ).fit()
    y = np.where(df["inlf"] == 1, df["lwage"], np.nan)
    m_expl = HeckitRegression(method="twostep").fit(
        y, df[["shared", "x_only"]].to_numpy(), df[["shared", "z_only"]].to_numpy()
    )
    np.testing.assert_allclose(m_form.coef_, m_expl.coef_, atol=1e-9)
    np.testing.assert_allclose(m_form.gamma_, m_expl.gamma_, atol=1e-9)
    assert m_form.outcome_feature_names_ == ["const", "shared", "x_only"]
    assert m_form.selection_feature_names_ == ["const", "shared", "z_only"]


def test_heckit_formula_rejects_non_binary_selection_lhs():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [0.5, 0.7, 1.2],
        "s": [0.3, 0.7, 0.9],  # not 0/1
    })
    with pytest.raises(ValueError, match="binary"):
        HeckitRegression.from_formula(
            outcome="y ~ x", selection="s ~ x", data=df
        )
