"""Tests for :meth:`HeckitRegression.get_margeff`.

Coverage goals:

1.  All four ``kind`` values run and return the right shape / variable list.
2.  Analytical sanity checks for the simplest formulas:

    - ``kind='latent'`` returns the outcome slopes verbatim.
    - ``kind='prob-selected'`` matches a probit's marginal effects on Z.
    - For an X-only regressor, ``kind='conditional'`` equals
      ``kind='latent'`` (the Mills-ratio correction has no Z component).

3.  Numerical derivative check: ``get_margeff(at='mean', kind=K)`` matches a
    central finite-difference of ``predict(kind=...)`` for each ``K``. This is
    the strongest correctness test because it reads our own predict formulae
    and our own derivative formulae from independent code paths.

4.  Delta-method SEs are positive and finite for MLE; ``NaN`` for two-step.

5.  The "shared regressor in X and Z" combines both effects.
"""

from __future__ import annotations

import numpy as np
import pytest

from censtrunc import HeckitRegression


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def heckit_named_data():
    """A DGP with one shared regressor and one X-only / Z-only regressor each."""
    rng = np.random.default_rng(2025_06_01)
    n = 4000
    shared = rng.normal(size=n)
    x_only = rng.normal(size=n)
    z_only = rng.normal(size=n)
    beta = np.array([1.0, 0.5, -0.3])  # const, shared, x_only
    gamma = np.array([0.0, 0.3, 0.6])  # const, shared, z_only
    rho, sigma = 0.6, 1.0
    cov = np.array([[sigma**2, rho * sigma], [rho * sigma, 1.0]])
    e, u = rng.multivariate_normal([0.0, 0.0], cov, size=n).T
    X = np.column_stack([shared, x_only])
    Z = np.column_stack([shared, z_only])
    y_star = beta[0] + X @ beta[1:] + e
    s = (gamma[0] + Z @ gamma[1:] + u > 0).astype(int)
    y = np.where(s == 1, y_star, np.nan)
    return {
        "y": y, "X": X, "Z": Z,
        "x_names": ["shared", "x_only"],
        "z_names": ["shared", "z_only"],
        "beta": beta, "gamma": gamma, "rho": rho, "sigma": sigma,
    }


@pytest.fixture
def fitted_mle(heckit_named_data):
    m = HeckitRegression(method="mle").fit(
        heckit_named_data["y"], heckit_named_data["X"], heckit_named_data["Z"],
        feature_names_outcome=heckit_named_data["x_names"],
        feature_names_selection=heckit_named_data["z_names"],
    )
    return m


@pytest.fixture
def fitted_twostep(heckit_named_data):
    m = HeckitRegression(method="twostep").fit(
        heckit_named_data["y"], heckit_named_data["X"], heckit_named_data["Z"],
        feature_names_outcome=heckit_named_data["x_names"],
        feature_names_selection=heckit_named_data["z_names"],
    )
    return m


# ----------------------------------------------------------------------
# Shape / API
# ----------------------------------------------------------------------


def test_all_kinds_run(fitted_mle):
    """Every supported kind returns a MarginalEffects with finite point
    estimates and matching variable list."""
    for kind, expected_vars in [
        ("latent",        ["shared", "x_only"]),
        ("conditional",   ["shared", "x_only", "z_only"]),
        ("unconditional", ["shared", "x_only", "z_only"]),
        ("prob-selected", ["shared", "z_only"]),
    ]:
        me = fitted_mle.get_margeff(kind=kind)
        assert me.names == expected_vars, kind
        assert me.margeff.shape == (len(expected_vars),)
        assert np.all(np.isfinite(me.margeff))


def test_unknown_kind_raises(fitted_mle):
    with pytest.raises(ValueError, match="Unknown kind"):
        fitted_mle.get_margeff(kind="nonsense")


def test_unknown_at_raises(fitted_mle):
    with pytest.raises(ValueError, match="Unknown at"):
        fitted_mle.get_margeff(at="nonsense")


def test_atexog_not_implemented(fitted_mle):
    with pytest.raises(NotImplementedError):
        fitted_mle.get_margeff(atexog={0: 1.0})


def test_unfitted_raises():
    m = HeckitRegression()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.get_margeff(kind="conditional")


# ----------------------------------------------------------------------
# Analytical sanity
# ----------------------------------------------------------------------


def test_latent_equals_outcome_slopes(fitted_mle):
    """``E[Y*|X] = X'beta`` so the marginal effects are exactly the slope coefs."""
    me = fitted_mle.ame(kind="latent")
    # slopes (drop the intercept from coef_)
    slope_idx = [
        fitted_mle.outcome_feature_names_.index(n) for n in me.names
    ]
    np.testing.assert_allclose(me.margeff, fitted_mle.coef_[slope_idx], atol=1e-12)


def test_conditional_x_only_equals_latent(fitted_mle):
    """For an X-only regressor (``x_only`` here) the Mills-ratio correction in
    ``E[Y|X,Z,S=1]`` doesn't contribute, so the conditional dy/dx equals the
    latent dy/dx (= beta)."""
    me_cond = fitted_mle.ame(kind="conditional")
    me_lat = fitted_mle.ame(kind="latent")
    j_cond = me_cond.names.index("x_only")
    j_lat = me_lat.names.index("x_only")
    assert abs(me_cond.margeff[j_cond] - me_lat.margeff[j_lat]) < 1e-12


def test_prob_selected_drops_x_only(fitted_mle):
    """``P(S=1|Z)`` does not depend on X-only variables, so they are absent
    from the prob-selected effects list."""
    me = fitted_mle.ame(kind="prob-selected")
    assert "x_only" not in me.names
    assert me.names == ["shared", "z_only"]


def test_latent_drops_z_only(fitted_mle):
    me = fitted_mle.ame(kind="latent")
    assert "z_only" not in me.names


# ----------------------------------------------------------------------
# Numerical-derivative check (the strongest correctness test)
# ----------------------------------------------------------------------


def _numerical_dydx_predict(model, kind: str, h: float = 1e-4) -> dict[str, float]:
    """Central finite difference of ``predict(kind=...)`` at the *sample mean*
    of each regressor, averaged across the data rows the same way ``at='mean'``
    builds its evaluation point.

    Returns ``{var_name: dy/dx}``.

    This reads the model's :meth:`predict` (which is itself a separate piece
    of code from the derivative formulas in ``_heckit_effects.py``) so we are
    cross-checking two independent implementations of the same quantity.
    """
    X_mean = model._X_train_.mean(axis=0, keepdims=True)
    Z_mean = model._Z_train_.mean(axis=0, keepdims=True)
    x_names = list(model.outcome_feature_names_)
    z_names = list(model.selection_feature_names_)

    out: dict[str, float] = {}
    all_vars = sorted(set(x_names) | set(z_names))
    for name in all_vars:
        if name == "const":
            continue
        X_p, X_m = X_mean.copy(), X_mean.copy()
        Z_p, Z_m = Z_mean.copy(), Z_mean.copy()
        if name in x_names:
            j = x_names.index(name)
            X_p[:, j] += h
            X_m[:, j] -= h
        if name in z_names:
            j = z_names.index(name)
            Z_p[:, j] += h
            Z_m[:, j] -= h

        # User-facing X / Z (the .predict pipeline re-prepends the intercept).
        def strip(M, names):
            if model.fit_intercept and names and names[0] == "const":
                return M[:, 1:]
            return M

        if kind == "latent":
            v_p = model.predict(X=strip(X_p, x_names), kind="outcome")
            v_m = model.predict(X=strip(X_m, x_names), kind="outcome")
        elif kind == "conditional":
            v_p = model.predict(
                X=strip(X_p, x_names), Z=strip(Z_p, z_names), kind="conditional"
            )
            v_m = model.predict(
                X=strip(X_m, x_names), Z=strip(Z_m, z_names), kind="conditional"
            )
        elif kind == "unconditional":
            X_clean_p = strip(X_p, x_names); X_clean_m = strip(X_m, x_names)
            Z_clean_p = strip(Z_p, z_names); Z_clean_m = strip(Z_m, z_names)
            cond_p = model.predict(X=X_clean_p, Z=Z_clean_p, kind="conditional")
            cond_m = model.predict(X=X_clean_m, Z=Z_clean_m, kind="conditional")
            pr_p = model.predict(Z=Z_clean_p, kind="selection_prob")
            pr_m = model.predict(Z=Z_clean_m, kind="selection_prob")
            v_p = pr_p * cond_p
            v_m = pr_m * cond_m
        elif kind == "prob-selected":
            v_p = model.predict(Z=strip(Z_p, z_names), kind="selection_prob")
            v_m = model.predict(Z=strip(Z_m, z_names), kind="selection_prob")
        else:
            raise ValueError(kind)

        out[name] = float((v_p.mean() - v_m.mean()) / (2 * h))
    return out


@pytest.mark.parametrize(
    "kind", ["latent", "conditional", "unconditional", "prob-selected"]
)
def test_get_margeff_matches_numerical_derivative(fitted_mle, kind):
    me = fitted_mle.get_margeff(at="mean", kind=kind)
    numeric = _numerical_dydx_predict(fitted_mle, kind)
    for name, analytic in zip(me.names, me.margeff):
        assert abs(analytic - numeric[name]) < 1e-5, (
            f"{kind}: analytic dydx for {name} = {analytic:.6f}, "
            f"numerical = {numeric[name]:.6f}"
        )


# ----------------------------------------------------------------------
# Delta-method SE behaviour
# ----------------------------------------------------------------------


def test_mle_has_positive_finite_se(fitted_mle):
    for kind in ("latent", "conditional", "unconditional", "prob-selected"):
        me = fitted_mle.get_margeff(kind=kind)
        assert np.all(np.isfinite(me.margeff_se)), kind
        assert np.all(me.margeff_se > 0), kind


def test_twostep_has_nan_se(fitted_twostep):
    """Two-step has no joint covariance, so SEs should be NaN (bootstrap is
    the right tool there)."""
    me = fitted_twostep.get_margeff(kind="conditional")
    assert np.all(np.isnan(me.margeff_se))
    # point estimates should still be finite, though
    assert np.all(np.isfinite(me.margeff))


# ----------------------------------------------------------------------
# at='mean'/'median'/'zero' all run and return single-vector results
# ----------------------------------------------------------------------


@pytest.mark.parametrize("at", ["overall", "mean", "median", "zero"])
def test_all_at_run(fitted_mle, at):
    me = fitted_mle.get_margeff(at=at, kind="conditional")
    assert me.margeff.shape == (3,)
    assert np.all(np.isfinite(me.margeff))


def test_ame_and_mem_shortcuts(fitted_mle):
    """`ame`/`mem` are pure shorthand for at='overall'/'mean'."""
    me_ame = fitted_mle.ame(kind="conditional")
    me_mem = fitted_mle.mem(kind="conditional")
    me_at_overall = fitted_mle.get_margeff(at="overall", kind="conditional")
    me_at_mean = fitted_mle.get_margeff(at="mean", kind="conditional")
    np.testing.assert_allclose(me_ame.margeff, me_at_overall.margeff)
    np.testing.assert_allclose(me_mem.margeff, me_at_mean.margeff)


# ----------------------------------------------------------------------
# Shared regressor: effects combine
# ----------------------------------------------------------------------


def test_shared_variable_combines_x_and_z_effects(fitted_mle):
    """For the conditional kind, ``shared`` carries the sum of the X-side
    (``beta_shared``) and Z-side (``-gamma_shared * rho * sigma * <delta>``)
    contributions. We check that ``conditional > latent`` or ``<`` depending
    on the sign of ``gamma_shared`` -- and quantitatively that the difference
    equals the Z-side contribution.
    """
    from scipy.stats import norm
    from censtrunc._heckit_effects import _inverse_mills

    me_lat = fitted_mle.ame(kind="latent")
    me_cond = fitted_mle.ame(kind="conditional")

    j_lat = me_lat.names.index("shared")
    j_cond = me_cond.names.index("shared")
    diff = me_cond.margeff[j_cond] - me_lat.margeff[j_lat]  # should be the Z-side term

    # Recompute the Z-side term independently.
    Z = fitted_mle._Z_train_
    Zg = Z @ fitted_mle.gamma_
    lam = _inverse_mills(Zg)
    delta = lam * (Zg + lam)
    j_z = fitted_mle.selection_feature_names_.index("shared")
    z_term = -fitted_mle.gamma_[j_z] * fitted_mle.rho_ * fitted_mle.sigma_ * delta.mean()

    assert abs(diff - z_term) < 1e-12


# ----------------------------------------------------------------------
# Recovery test: at large n, AME should be close to the true marginal effect.
# ----------------------------------------------------------------------


def test_conditional_ame_close_to_dgp_value():
    """On a single big simulation, the conditional AME for an X-only variable
    matches the true beta (within sampling error)."""
    rng = np.random.default_rng(7)
    n = 8000
    shared = rng.normal(size=n)
    x_only = rng.normal(size=n)
    z_only = rng.normal(size=n)
    beta_x_only_true = -0.4
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    e, u = rng.multivariate_normal([0.0, 0.0], cov, size=n).T
    y_star = 1.0 + 0.3 * shared + beta_x_only_true * x_only + e
    s = (0.2 * shared + 0.7 * z_only + u > 0).astype(int)
    y = np.where(s == 1, y_star, np.nan)

    m = HeckitRegression(method="mle").fit(
        y, np.column_stack([shared, x_only]), np.column_stack([shared, z_only]),
        feature_names_outcome=["shared", "x_only"],
        feature_names_selection=["shared", "z_only"],
    )
    me = m.ame(kind="conditional")
    j = me.names.index("x_only")
    # x_only is X-only, so conditional dydx = beta_x_only exactly
    assert abs(me.margeff[j] - beta_x_only_true) < 0.05
