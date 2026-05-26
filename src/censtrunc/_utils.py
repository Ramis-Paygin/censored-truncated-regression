"""Internal utilities: input validation, threshold handling, observation classification."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # optional pandas support
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


ArrayLike = Any  # numpy array, pandas DataFrame/Series, or list


def _to_numpy(data: ArrayLike, name: str) -> tuple[np.ndarray, list[str] | None]:
    """Convert input data to ``np.ndarray`` and extract column names if present.

    Parameters
    ----------
    data
        Input data: numpy array, pandas DataFrame/Series, or array-like.
    name
        Name of the input (for error messages).

    Returns
    -------
    array
        2-D for matrices (X) — caller decides shape, here we just convert.
    columns
        List of column names if input was a pandas object; otherwise ``None``.
    """
    columns: list[str] | None = None

    if _HAS_PANDAS and isinstance(data, pd.DataFrame):
        columns = list(data.columns.astype(str))
        array = data.to_numpy(dtype=float)
    elif _HAS_PANDAS and isinstance(data, pd.Series):
        columns = [str(data.name)] if data.name is not None else None
        array = data.to_numpy(dtype=float)
    else:
        array = np.asarray(data, dtype=float)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values; please clean the data first.")

    return array, columns


def _prepare_design_matrix(
    X: ArrayLike,
    *,
    fit_intercept: bool,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Validate X, optionally prepend an intercept column, return matrix and column names."""
    X_arr, columns = _to_numpy(X, "X")
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got {X_arr.ndim}-D")

    if columns is None:
        columns = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i + 1}" for i in range(X_arr.shape[1])]
        )
    if len(columns) != X_arr.shape[1]:
        raise ValueError(
            f"Number of feature names ({len(columns)}) does not match number of columns ({X_arr.shape[1]})"
        )

    if fit_intercept:
        intercept_col = np.ones((X_arr.shape[0], 1))
        X_arr = np.hstack([intercept_col, X_arr])
        columns = ["const", *columns]

    return X_arr, columns


def _prepare_y(y: ArrayLike, n_expected: int) -> np.ndarray:
    """Validate y and return as 1-D numpy array."""
    y_arr, _ = _to_numpy(y, "y")
    y_arr = np.ravel(y_arr)
    if y_arr.shape[0] != n_expected:
        raise ValueError(f"X has {n_expected} rows but y has {y_arr.shape[0]} elements")
    return y_arr


def _resolve_thresholds(
    left: float | None,
    right: float | None,
) -> tuple[float, float, bool, bool]:
    """Resolve thresholds: ``None`` means unbounded.

    Returns
    -------
    left_val : float
        Effective left threshold (``-inf`` if unbounded).
    right_val : float
        Effective right threshold (``+inf`` if unbounded).
    has_left : bool
        True if left censoring is active.
    has_right : bool
        True if right censoring is active.
    """
    has_left = left is not None and np.isfinite(left)
    has_right = right is not None and np.isfinite(right)
    left_val = float(left) if has_left else -np.inf
    right_val = float(right) if has_right else np.inf
    if has_left and has_right and left_val >= right_val:
        raise ValueError(f"left ({left_val}) must be strictly less than right ({right_val})")
    return left_val, right_val, has_left, has_right


def _classify_observations(
    y: np.ndarray,
    left: float,
    right: float,
    has_left: bool,
    has_right: bool,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks (left_censored, right_censored, uncensored).

    An observation is considered left-censored if its value is at the left threshold
    (within ``tol``) and ``has_left`` is True. Similarly for right. All others are
    uncensored. Observations below ``left`` or above ``right`` are treated as censored
    at the corresponding threshold (defensive, with a warning-level diagnostic).
    """
    left_mask = np.zeros_like(y, dtype=bool)
    right_mask = np.zeros_like(y, dtype=bool)

    if has_left:
        # values at or below the left threshold are treated as left-censored
        left_mask = y <= left + tol
    if has_right:
        right_mask = y >= right - tol
    # ensure no double-classification (corner case: left == right which we rejected)
    uncensored_mask = ~(left_mask | right_mask)
    return left_mask, right_mask, uncensored_mask
