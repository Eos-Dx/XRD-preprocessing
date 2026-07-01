"""Shared numeric helpers."""

from __future__ import annotations

import numpy as np


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate ``y`` over ``x`` across NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
