"""censtrunc — censored and truncated normal regression with arbitrary thresholds.

A generalisation of the classical Tobit (1958) and the truncated regression model
to arbitrary left and right thresholds. Estimation is by maximum likelihood with
Olsen's globally concave reparameterisation for numerical reliability.

Quickstart
----------
>>> import numpy as np
>>> from censtrunc import CensoredRegression
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(500, 2))
>>> y_star = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=1.0, size=500)
>>> y = np.clip(y_star, 0.0, 2.0)  # two-sided censoring
>>> model = CensoredRegression(left=0.0, right=2.0).fit(X, y)
>>> print(model.coef_)  # doctest: +SKIP
"""

from __future__ import annotations

from .censored import CensoredRegression
from .effects import MarginalEffects
from .inference import LRTestResult, lr_test
from .truncated import TruncatedRegression

__all__ = [
    "CensoredRegression",
    "TruncatedRegression",
    "MarginalEffects",
    "LRTestResult",
    "lr_test",
    "__version__",
]
__version__ = "0.1.0"
