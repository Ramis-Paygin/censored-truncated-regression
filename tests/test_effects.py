"""Tests for marginal-effects helpers (AME and MEM)."""

from __future__ import annotations

import numpy as np

from censtrunc import CensoredRegression


def test_ame_latent_equals_beta(sim_two_sided):
    """Latent marginal effect equals beta for the slope coefficients (no shrinkage)."""
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    ame_lat = model.ame(kind="latent")
    # AME excludes the intercept by definition; compare against slopes only.
    slope_beta = model.coef_[1:]  # skip intercept (first column was the constant)
    np.testing.assert_allclose(ame_lat.effects, slope_beta, rtol=1e-10, atol=1e-10)


def test_ame_censored_smaller_than_latent(sim_two_sided):
    """Censoring shrinks marginal effects toward zero in absolute value."""
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    ame_lat = model.ame(kind="latent")
    ame_cen = model.ame(kind="censored")
    assert np.all(np.abs(ame_cen.effects) <= np.abs(ame_lat.effects))


def test_mem_returns_single_point(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    mem = model.mem(kind="censored")
    # length = number of slope coefficients
    assert mem.effects.shape == (sim_two_sided.X.shape[1],)


def test_marginal_effects_dataframe(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    df = model.ame(kind="censored").to_dataframe()
    assert list(df.columns) == ["dy/dx", "std err", "z", "P>|z|", "[0.025", "0.975]"]
    assert len(df) == sim_two_sided.X.shape[1]


# ----------------------------------------------------------------------
# get_margeff (statsmodels-style) API
# ----------------------------------------------------------------------


def test_get_margeff_at_options_match_wrappers(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    # at='overall' equals ame; at='mean' equals mem
    np.testing.assert_allclose(
        model.get_margeff(at="overall", kind="censored").margeff,
        model.ame(kind="censored").margeff,
    )
    np.testing.assert_allclose(
        model.get_margeff(at="mean", kind="censored").margeff,
        model.mem(kind="censored").margeff,
    )


def test_get_margeff_all_at_points_run(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    for at in ("overall", "mean", "median", "zero"):
        me = model.get_margeff(at=at, kind="censored")
        assert me.margeff.shape == (sim_two_sided.X.shape[1],)
        assert np.all(np.isfinite(me.margeff))


def test_get_margeff_elasticity_methods(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    for method in ("dydx", "eyex", "dyex", "eydx"):
        me = model.get_margeff(method=method, kind="censored")
        assert me.method == method
        # column label in the frame reflects the method
        label = {"dydx": "dy/dx", "eyex": "eyex", "dyex": "dyex", "eydx": "eydx"}[method]
        assert label in me.summary_frame().columns


def test_get_margeff_dummy_uses_discrete_difference():
    """For a binary regressor, dummy=True should change the reported effect."""
    rng = np.random.default_rng(3)
    n = 3000
    x_cont = rng.normal(size=n)
    x_bin = (rng.uniform(size=n) > 0.5).astype(float)
    X = np.column_stack([x_cont, x_bin])
    y_star = 1.0 + 0.6 * x_cont - 0.5 * x_bin + rng.normal(size=n)
    y = np.clip(y_star, 0.0, 2.5)
    model = CensoredRegression(left=0.0, right=2.5).fit(X, y, feature_names=["x_cont", "x_bin"])

    plain = model.get_margeff(method="dydx", kind="censored").margeff
    dummied = model.get_margeff(method="dydx", kind="censored", dummy=True).margeff
    # The continuous variable's effect is unchanged...
    assert abs(plain[0] - dummied[0]) < 1e-9
    # ...but the binary variable's discrete difference differs from the derivative.
    assert abs(plain[1] - dummied[1]) > 1e-4


def test_get_margeff_summary_is_statsmodels_like(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    text = model.get_margeff(at="overall", method="dydx").summary()
    assert "Marginal Effects" in text
    assert "Method:" in text and "dydx" in text
    assert "At:" in text and "overall" in text


def test_get_margeff_invalid_options_raise(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    import pytest

    with pytest.raises(ValueError, match="Unknown at"):
        model.get_margeff(at="nonsense")
    with pytest.raises(ValueError, match="Unknown method"):
        model.get_margeff(method="nonsense")


def test_prob_marginal_effects_sum_to_zero(sim_two_sided):
    """Effects on P(left), P(interior), P(right) must sum to zero per regressor."""
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    pl = model.get_margeff(kind="prob-left").margeff
    pi = model.get_margeff(kind="prob-interior").margeff
    pr = model.get_margeff(kind="prob-right").margeff
    np.testing.assert_allclose(pl + pi + pr, np.zeros_like(pl), atol=1e-8)


def test_prob_marginal_effect_matches_finite_difference(sim_two_sided):
    """Analytic dP(left)/dx at the mean equals the finite-difference of predict_proba."""
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    mem = model.get_margeff(at="mean", kind="prob-left").margeff
    mean_row = sim_two_sided.X.mean(axis=0, keepdims=True)
    eps = 1e-5
    for j in range(sim_two_sided.X.shape[1]):
        xp = mean_row.copy(); xp[0, j] += eps
        xm = mean_row.copy(); xm[0, j] -= eps
        num = (model.predict_proba(xp)["left"][0] - model.predict_proba(xm)["left"][0]) / (2 * eps)
        assert abs(num - mem[j]) < 1e-4


def test_prob_marginal_effects_have_standard_errors(sim_two_sided):
    model = CensoredRegression(left=sim_two_sided.left, right=sim_two_sided.right).fit(
        sim_two_sided.X, sim_two_sided.y
    )
    me = model.get_margeff(kind="prob-interior")
    assert me.margeff_se.shape == me.margeff.shape
    assert np.all(me.margeff_se > 0)
    assert np.all(np.isfinite(me.pvalues))


def test_prob_kinds_available_only_for_censored(sim_truncated):
    """Truncated models do not expose region-probability marginal effects."""
    from censtrunc import TruncatedRegression

    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    import pytest

    with pytest.raises(ValueError, match="Unknown kind"):
        model.get_margeff(kind="prob-left")


def test_truncated_get_margeff_rejects_censored_kind(sim_truncated):
    from censtrunc import TruncatedRegression

    model = TruncatedRegression(left=sim_truncated.left, right=sim_truncated.right).fit(
        sim_truncated.X, sim_truncated.y
    )
    import pytest

    # 'censored' is not a valid conditional mean for a truncated model
    with pytest.raises(ValueError, match="Unknown kind"):
        model.get_margeff(kind="censored")
    # but latent and truncated work
    assert model.get_margeff(kind="latent").margeff.shape[0] == sim_truncated.X.shape[1]
    assert model.get_margeff(kind="truncated").margeff.shape[0] == sim_truncated.X.shape[1]
