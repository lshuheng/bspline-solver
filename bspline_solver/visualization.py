"""Plotting utilities for B-spline segments and control polygons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from .config import DEGREE


def plot_spline_path(
    segments: list,
    knot: np.ndarray,
    control_visible: bool = True,
    resolution: int = 2000,
):
    """Plot one or more B-spline segments, optionally with control polygons.

    Args:
        segments: Iterable of (2, n_control) arrays, one per segment.
        knot: Shared knot vector for all segments.
        control_visible: Whether to overlay control polygons.
        resolution: Number of sample points per segment for plotting.

    Returns:
        The matplotlib Axes object.
    """
    _, ax = plt.subplots()

    all_xs = []
    all_ys = []

    for control in segments:
        c1, c2 = control
        uniq_knot = np.unique(knot)

        s1 = BSpline.construct_fast(knot, c1, DEGREE, extrapolate=False)
        s2 = BSpline.construct_fast(knot, c2, DEGREE, extrapolate=False)

        sample_t = np.setdiff1d(np.linspace(0.0, 1.0, int(resolution)), uniq_knot)
        sx = s1(sample_t)
        sy = s2(sample_t)

        mask = np.isfinite(sx) & np.isfinite(sy)
        sx_plot = sx[mask]
        sy_plot = sy[mask]

        ax.plot(sx_plot, sy_plot)

        all_xs.append(sx_plot)
        all_xs.append(c1)
        all_ys.append(sy_plot)
        all_ys.append(c2)

        if control_visible:
            n = len(c1)
            colors = plt.cm.coolwarm(np.linspace(0, 1, n))
            ax.plot(c1, c2, "--", color="gray", linewidth=0.8, alpha=0.5)
            ax.scatter(c1, c2, c=colors, s=10, edgecolors="black", linewidths=0.5, zorder=3)

    all_xs = np.concatenate(all_xs)
    all_ys = np.concatenate(all_ys)
    x_min, x_max = float(np.min(all_xs)), float(np.max(all_xs))
    y_min, y_max = float(np.min(all_ys)), float(np.max(all_ys))
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")
    return ax
