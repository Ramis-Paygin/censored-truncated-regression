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
