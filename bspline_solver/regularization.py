"""Regularization terms applied to control-point arrangements."""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def tangent_speed(control: np.ndarray) -> tuple[float, np.ndarray]:
    """Reciprocal-speed barrier on endpoint tangent differences.

    At interior vertices the optimizer can satisfy C^1 continuity trivially by
    sliding both flanking control points toward the vertex (a → w, b → w), so
    that a + b = 2w holds with zero tangent speed.  This barrier blocks that
    path: it grows without bound as either endpoint difference collapses to zero
    and is negligible when tangent speeds are normal.

    For a segment with control points c_0, ..., c_{n-1} (n >= 4):
        d_start = c_1 - c_0,   d_end = c_{n-1} - c_{n-2}
        E = 1 / (||d_start||^2 + eps) + 1 / (||d_end||^2 + eps)

    Args:
        control: Array of shape (2, n) containing control-point coordinates.

    Returns:
        Tuple of (E, dE/dcontrol), with the gradient matching the input shape.
    """
    c = np.asarray(control, dtype=float).reshape(2, -1)

    d_start = c[:, 1] - c[:, 0]    # shape (2,)
    d_end   = c[:, -1] - c[:, -2]  # shape (2,)

    s_start = float(np.dot(d_start, d_start))
    s_end   = float(np.dot(d_end,   d_end))

    denom_start = s_start + _EPS
    denom_end   = s_end   + _EPS

    energy = 1.0 / denom_start + 1.0 / denom_end

    # dE / d(d) = -2 d / denom^2  (quotient rule on 1 / (||d||^2 + eps))
    dE_dstart = -2.0 * d_start / (denom_start ** 2)
    dE_dend   = -2.0 * d_end   / (denom_end   ** 2)

    grad = np.zeros_like(c)
    grad[:, 1]  += dE_dstart   # d_start = c_1 - c_0
    grad[:, 0]  -= dE_dstart
    grad[:, -1] += dE_dend     # d_end   = c_{n-1} - c_{n-2}
    grad[:, -2] -= dE_dend

    return energy, grad


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
