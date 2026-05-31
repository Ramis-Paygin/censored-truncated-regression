"""Tests for the hypothesis-string parser and the model.lr_test(hypotheses) method.

The single-model LR test (`model.lr_test('x3 = 0')`) and the two-model variant
(`lr_test(model_full, model_restricted)`) should give the same statistic on
nested specifications, since they are testing the same restriction by two
routes: refitting with a linear constraint vs refitting on a sub-design matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from censtrunc import CensoredRegression, TruncatedRegression, lr_test
from censtrunc._restrictions import parse_hypotheses


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------


def test_parser_simple_string():
    names = ["sigma", "const", "x1", "x2", "x3"]
    R, r = parse_hypotheses("x1 = 0", names)
    assert R.shape == (1, 5)
    np.testing.assert_allclose(R[0], [0, 0, 1, 0, 0])
    assert r[0] == 0.0


def test_parser_multiple_parens():
    names = ["sigma", "const", "x1", "x2", "x3"]
    R, r = parse_hypotheses("(x1 = x2), (x3 = 2)", names)
    assert R.shape == (2, 5)
    np.testing.assert_allclose(R[0], [0, 0, 1, -1, 0])
    np.testing.assert_allclose(R[1], [0, 0, 0, 0, 1])
    np.testing.assert_allclose(r, [0.0, 2.0])


def test_parser_arithmetic():
    names = ["sigma", "const", "x1", "x2"]
    R, r = parse_hypotheses("(2*x1 + x2/4 = 1)", names)
    np.testing.assert_allclose(R[0], [0, 0, 2.0, 0.25])
    assert r[0] == 1.0


def test_parser_array_form():
    names = ["sigma", "const", "x1", "x2"]
    R, r = parse_hypotheses(np.array([[0, 0, 1, -1]]), names)
    np.testing.assert_allclose(R[0], [0, 0, 1, -1])
    assert r[0] == 0.0


def test_parser_tuple_form():
    names = ["sigma", "const", "x1", "x2"]
    R_in = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    R, r = parse_hypotheses((R_in, [0.5, -0.3]), names)
    np.testing.assert_allclose(R, R_in)
    np.testing.assert_allclose(r, [0.5, -0.3])


def test_parser_unknown_name():
    names = ["sigma", "const", "x1"]
    with pytest.raises(ValueError, match="Unknown parameter"):
        parse_hypotheses("x99 = 0", names)


def test_parser_nonlinear_product_rejected():
    names = ["sigma", "const", "x1", "x2"]
    with pytest.raises(ValueError, match="Non-linear"):
        parse_hypotheses("x1 * x2 = 0", names)


def test_parser_division_by_parameter_rejected():
    names = ["sigma", "const", "x1"]
    with pytest.raises(ValueError, match="divide by a parameter"):
        parse_hypotheses("1 / x1 = 0", names)


def test_parser_empty_string():
    names = ["sigma", "const", "x1"]
    with pytest.raises(ValueError, match="No constraints"):
        parse_hypotheses("   ", names)


def test_parser_missing_equals():
    names = ["sigma", "const", "x1"]
    with pytest.raises(ValueError, match="exactly one '='"):
        parse_hypotheses("x1 + 1", names)


def test_parser_only_constants():
    """Constraints that cancel out the parameter terms are degenerate."""
    names = ["sigma", "const", "x1"]
    with pytest.raises(ValueError, match="only constants"):
        parse_hypotheses("x1 = x1", names)


# ----------------------------------------------------------------------
# Model lr_test API
# ----------------------------------------------------------------------


@pytest.fixture
def censored_with_zero_slopes():
    """Censored model where the last two slopes are truly zero."""
    rng = np.random.default_rng(7)
    n = 2500
    X = rng.normal(size=(n, 4))
    beta_true = np.array([0.5, 0.4, -0.2, 0.0, 0.0])
    y_star = beta_true[0] + X @ beta_true[1:] + rng.normal(size=n)
    y = np.clip(y_star, 0.0, 2.5)
    return X, y, CensoredRegression(left=0.0, right=2.5).fit(X, y)


def test_lr_test_string_single_restriction(censored_with_zero_slopes):
    X, y, model = censored_with_zero_slopes
    res = model.lr_test("x4 = 0")
    assert res.df == 1
    assert 0 <= res.p_value <= 1
    # x4 truly zero in DGP -> should not be rejected
    assert res.p_value > 0.01


def test_lr_test_joint_string_restrictions(censored_with_zero_slopes):
    X, y, model = censored_with_zero_slopes
    res = model.lr_test("(x3 = 0), (x4 = 0)")
    assert res.df == 2
    assert res.p_value > 0.01  # both truly zero


def test_lr_test_single_matches_two_model_form(censored_with_zero_slopes):
    """model.lr_test('x4 = 0') ~ lr_test(full, restricted on first 3 columns)."""
    X, y, full = censored_with_zero_slopes
    res_method = full.lr_test("x4 = 0")
    restricted = CensoredRegression(left=0.0, right=2.5).fit(X[:, :3], y)
    res_func = lr_test(full, restricted)
    # Both routes test the same null; agree to optimiser tolerance
    assert abs(res_method.statistic - res_func.statistic) < 1e-2
    assert res_method.df == res_func.df


def test_lr_test_rejects_false_restriction(censored_with_zero_slopes):
    """A restriction the DGP violates should be rejected."""
    X, y, model = censored_with_zero_slopes
    res = model.lr_test("x1 = -x2")  # truth has x1=0.4, x2=-0.2 -> 0.4 ≠ 0.2
    assert res.p_value < 1e-3
    assert res.statistic > 10


def test_lr_test_tuple_form(censored_with_zero_slopes):
    """Tuple (R, r) input gives the same answer as the string form."""
    X, y, model = censored_with_zero_slopes
    names = ["sigma"] + list(model.feature_names_)
    R = np.zeros((1, len(names)))
    R[0, names.index("x4")] = 1.0
    res_tuple = model.lr_test((R, 0.0))
    res_str = model.lr_test("x4 = 0")
    assert abs(res_tuple.statistic - res_str.statistic) < 1e-6


def test_lr_test_unknown_parameter_raises(censored_with_zero_slopes):
    X, y, model = censored_with_zero_slopes
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.lr_test("xnope = 0")


def test_truncated_lr_test_runs():
    """The same lr_test API is available on TruncatedRegression."""
    rng = np.random.default_rng(11)
    n = 4000
    X = rng.normal(size=(n, 2))
    # both slopes truly non-zero, so the test should reject
    beta = np.array([1.0, 0.5, -0.3])
    y_star = beta[0] + X @ beta[1:] + rng.normal(size=n)
    mask = (y_star > 0) & (y_star < 2.5)
    model = TruncatedRegression(left=0.0, right=2.5).fit(X[mask], y_star[mask])
    res = model.lr_test("x2 = 0")
    # API works and returns sensible results
    assert res.df == 1
    assert 0 <= res.p_value <= 1
    # x2 truly non-zero -> should be rejected
    assert res.p_value < 1e-3
    # And the restricted log-likelihood should be lower than the unrestricted one
    assert res.ll_restricted < res.ll_full
