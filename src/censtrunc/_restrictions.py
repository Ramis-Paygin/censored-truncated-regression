"""Parse linear-restriction hypotheses in the statsmodels ``f_test`` style.

Input formats accepted:

- **String** — comma-separated constraints (parentheses optional),
  e.g. ``'(x1 = 0), (x2 - x3 = 0.5), (2*x1 + x4/10 = 1)'``.
  Each side of ``=`` may be a linear combination of parameter names and
  numeric constants. Multiplication is only allowed between a constant
  and a parameter, and division only by a constant — anything non-linear
  raises a clean ``ValueError``.

- **Array** of shape ``(q, p)`` — interpreted as ``R`` with ``r = 0``.

- **Tuple** ``(R, r)`` — used directly; ``r`` may be a scalar or a length-q vector.

The output is always a pair ``(R, r)`` with shapes ``(q, p)`` and ``(q,)`` such
that the restriction is ``R @ theta = r``, where ``theta`` is the parameter vector
``[sigma, beta_0, beta_1, ...]`` (matching ``model.params_``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class _LinearExpr:
    """Coefficient dict + scalar constant for a linear expression in parameters."""

    coefs: dict[str, float] = field(default_factory=dict)
    const: float = 0.0

    def scaled(self, k: float) -> "_LinearExpr":
        return _LinearExpr(
            coefs={n: c * k for n, c in self.coefs.items()},
            const=self.const * k,
        )

    def combined(self, other: "_LinearExpr", sign: int = 1) -> "_LinearExpr":
        out = _LinearExpr(coefs=dict(self.coefs), const=self.const + sign * other.const)
        for n, c in other.coefs.items():
            out.coefs[n] = out.coefs.get(n, 0.0) + sign * c
        return out


def _walk(node: ast.AST, allowed: set[str]) -> _LinearExpr:
    if isinstance(node, ast.Expression):
        return _walk(node.body, allowed)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return _LinearExpr(const=float(node.value))
    if isinstance(node, ast.Name):
        if node.id not in allowed:
            raise ValueError(
                f"Unknown parameter name {node.id!r}; known parameters: {sorted(allowed)}"
            )
        return _LinearExpr(coefs={node.id: 1.0})
    if isinstance(node, ast.UnaryOp):
        sub = _walk(node.operand, allowed)
        if isinstance(node.op, ast.USub):
            return sub.scaled(-1.0)
        if isinstance(node.op, ast.UAdd):
            return sub
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _walk(node.left, allowed)
        right = _walk(node.right, allowed)
        op = node.op
        if isinstance(op, ast.Add):
            return left.combined(right, +1)
        if isinstance(op, ast.Sub):
            return left.combined(right, -1)
        if isinstance(op, ast.Mult):
            if not left.coefs:
                return right.scaled(left.const)
            if not right.coefs:
                return left.scaled(right.const)
            raise ValueError("Non-linear restriction: product of two parameter terms")
        if isinstance(op, ast.Div):
            if right.coefs:
                raise ValueError("Cannot divide by a parameter (use a constant)")
            if right.const == 0.0:
                raise ValueError("Division by zero in restriction")
            return left.scaled(1.0 / right.const)
        raise ValueError(f"Unsupported binary operator: {type(op).__name__}")
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _parse_linear(expr_str: str, allowed: set[str]) -> _LinearExpr:
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty side of '=' in constraint")
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression {expr_str!r}: {exc}") from exc
    return _walk(tree, allowed)


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas not enclosed by parentheses."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses in hypothesis string")
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    if depth != 0:
        raise ValueError("Unbalanced parentheses in hypothesis string")
    parts.append(s[start:])
    return [p.strip() for p in parts if p.strip()]


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                # the opening '(' has its matching ')' before the end -> outer
                # parens do not enclose the entire expression
                return s
    return s[1:-1].strip()


def parse_hypotheses(
    hypotheses: Any, param_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a hypothesis specification to ``(R, r)`` matrices.

    See the module docstring for accepted input formats.
    """
    p = len(param_names)

    # Tuple form: (R, r)
    if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
        R = np.asarray(hypotheses[0], dtype=float)
        if R.ndim == 1:
            R = R.reshape(1, -1)
        rhs = hypotheses[1]
        if np.isscalar(rhs):
            r = np.full(R.shape[0], float(rhs))
        else:
            r = np.asarray(rhs, dtype=float).ravel()
        if R.shape[1] != p:
            raise ValueError(f"R must have {p} columns; got {R.shape[1]}")
        if r.shape[0] != R.shape[0]:
            raise ValueError(f"length of r ({r.shape[0]}) must equal rows of R ({R.shape[0]})")
        return R, r

    # Array form: (q, p) matrix, RHS assumed zero
    if hasattr(hypotheses, "__array__") or isinstance(hypotheses, (list,)):
        R = np.asarray(hypotheses, dtype=float)
        if R.ndim == 1:
            R = R.reshape(1, -1)
        if R.ndim != 2:
            raise ValueError(f"Array hypothesis must be 1-D or 2-D; got {R.ndim}-D")
        if R.shape[1] != p:
            raise ValueError(f"R must have {p} columns; got {R.shape[1]}")
        return R, np.zeros(R.shape[0])

    # String form
    if not isinstance(hypotheses, str):
        raise TypeError(
            f"hypotheses must be a string, array, or (R, r) tuple; got {type(hypotheses).__name__}"
        )

    allowed = set(param_names)
    raw_constraints = _split_top_level_commas(hypotheses)
    if not raw_constraints:
        raise ValueError("No constraints found in hypothesis string")

    R_rows: list[np.ndarray] = []
    r_vals: list[float] = []
    for raw in raw_constraints:
        c = _strip_outer_parens(raw)
        if c.count("=") != 1:
            raise ValueError(f"Constraint {raw!r} must contain exactly one '='")
        lhs_s, rhs_s = c.split("=", 1)
        lhs = _parse_linear(lhs_s, allowed)
        rhs = _parse_linear(rhs_s, allowed)
        # (LHS - RHS) terms · theta = (RHS const - LHS const)
        row = np.zeros(p)
        for name, coef in lhs.coefs.items():
            row[param_names.index(name)] += coef
        for name, coef in rhs.coefs.items():
            row[param_names.index(name)] -= coef
        R_rows.append(row)
        r_vals.append(rhs.const - lhs.const)

    R = np.asarray(R_rows)
    r = np.asarray(r_vals)
    # Sanity: each row must have at least one non-zero coefficient
    zero_rows = np.where(~np.any(R != 0, axis=1))[0]
    if len(zero_rows):
        raise ValueError(
            f"Constraint(s) #{zero_rows.tolist()} contain only constants; "
            "the parameter terms cancelled out"
        )
    return R, r
