"""Patsy-based formula parsing for the ``from_formula`` constructors.

Lets users specify a regression via the familiar string syntax of R /
statsmodels::

    'lwage ~ 1 + educ + exper + expersq'

instead of building ``X`` and ``y`` by hand. ``patsy`` already supports
intercepts (``+ 1`` or ``- 1``), transforms (``np.log(x)``, ``I(x**2)``),
categorical contrasts (``C(z)``), and interactions (``x:y``), so all of those
work out of the box.

The intercept column patsy emits is named ``'Intercept'``; we rename it to
``'const'`` to match the convention used elsewhere in this package
(``fit_intercept=True`` design matrices have ``'const'`` as the first column).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def parse_single_formula(
    formula: str, data: Any, *, keep_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Parse a one-equation formula and return ``(y, X, y_name, x_names)``.

    Parameters
    ----------
    formula : str
        Patsy formula, e.g. ``'y ~ 1 + x1 + x2'``.
    data : pandas.DataFrame or compatible mapping
        Source of the columns referenced in the formula.
    keep_missing : bool, default ``False``
        If ``True``, rows containing ``NaN`` in any referenced column are
        retained (patsy's NA-detection is disabled). The Heckman selection
        model needs this for its outcome equation, where ``y`` is missing
        exactly for the unselected rows but the corresponding ``X`` and ``Z``
        rows must still be kept.

    Returns
    -------
    y : ndarray of shape (n,)
    X : ndarray of shape (n, k)
    y_name : str
        Name of the left-hand-side column.
    x_names : list of str
        Column labels for ``X``; the intercept (if present) is named ``'const'``.
    """
    try:
        import patsy
    except ImportError as exc:  # pragma: no cover - patsy is a soft dep
        raise ImportError(
            "Formula parsing requires `patsy`. Install with: pip install patsy"
        ) from exc

    na_action = patsy.NAAction(NA_types=[]) if keep_missing else patsy.NAAction()
    y_df, X_df = patsy.dmatrices(
        formula, data, NA_action=na_action, return_type="dataframe",
    )
    if y_df.shape[1] != 1:
        raise ValueError(
            f"Formula {formula!r} must have exactly one left-hand-side column; "
            f"got {y_df.shape[1]}."
        )
    y_name = str(y_df.columns[0])
    x_names = [str(c) if c != "Intercept" else "const" for c in X_df.columns]
    return y_df.iloc[:, 0].to_numpy(), X_df.to_numpy(), y_name, x_names


def parse_rhs_only(formula: str, data: Any) -> tuple[np.ndarray, list[str]]:
    """Parse a right-hand-side-only formula (no LHS, just regressors).

    Accepts either ``'~ x1 + x2'`` or ``'x1 + x2'``; returns ``(X, x_names)``.
    Useful for the selection-equation regressors in Heckman models where the
    LHS (the selection indicator) is supplied separately.
    """
    try:
        import patsy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Formula parsing requires `patsy`. Install with: pip install patsy"
        ) from exc

    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    X_df = patsy.dmatrix(rhs, data, return_type="dataframe")
    x_names = [str(c) if c != "Intercept" else "const" for c in X_df.columns]
    return X_df.to_numpy(), x_names
