"""Regularization terms applied to control-point arrangements."""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def control_variance(control: np.ndarray) -> tuple[float, np.ndarray]:
    """Variance of squared consecutive control-point spacings, plus its gradient.

    This regularizer encourages uniform spacing of control points along the
    curve. Given control points c_0, ..., c_{n-1} (columns), let
        d_i  = c_{i+1} - c_i
        s_i  = ||d_i||^2
        m    = mean(s_i)
        E    = sum_i (m - s_i)^2 / (m^2 + eps)

    Args:
        control: Array of shape (2, n) containing control-point coordinates.

    Returns:
        Tuple of (E, dE/dcontrol), with the gradient matching the input shape.
    """
    c = np.asarray(control, dtype=float).reshape(2, -1).T  # shape (n, 2)
    fd = c[1:] - c[:-1]                                    # shape (n-1, 2)
    s = (fd ** 2).sum(axis=1)                              # shape (n-1,)
    m = s.mean()
    denom = m ** 2 + _EPS
    diff = m - s                                           # shape (n-1,)
    energy = float((diff ** 2).sum() / denom)

    # dE/ds_i: chain through both the (m - s_i)^2 numerator and m^2 denominator.
    n_seg = s.shape[0]
    num = (diff ** 2).sum()
    dnum_ds = 2 * (s - m) * (1.0 - 1.0 / n_seg) + (2.0 / n_seg) * diff.sum()
    dnum_ds = 2.0 * (s - m) * (1.0 - 1.0 / n_seg)
    dm_ds = np.full(n_seg, 1.0 / n_seg)
    ddenom_ds = 2.0 * m * dm_ds
    dE_ds = (dnum_ds * denom - num * ddenom_ds) / (denom ** 2)

    # ds_i / d(fd_i) = 2 * fd_i; chain to control points via fd_i = c_{i+1} - c_i.
    dE_dfd = (dE_ds[:, None]) * (2.0 * fd)                 # shape (n-1, 2)

    grad_c = np.zeros_like(c)                              # shape (n, 2)
    grad_c[:-1] -= dE_dfd
    grad_c[1:] += dE_dfd

    return energy, grad_c.T                                # back to (2, n)
