"""Plotting utilities for B-spline segments and control polygons."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from .config import DEGREE

if TYPE_CHECKING:
    from .experiment import ExperimentResult


def plot_spline_path(
    segments: list,
    knot: np.ndarray,
    control_visible: bool = False,
    resolution: int = 2000,
    ax=None,
):
    """Plot one or more B-spline segments, optionally with control polygons.

    Args:
        segments: Iterable of (2, n_control) arrays, one per segment.
        knot: Shared knot vector for all segments.
        control_visible: Whether to overlay control polygons.
        resolution: Number of sample points per segment for plotting.
        ax: Existing Axes to plot into; creates a new figure if None.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
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


def plot_result(result: "ExperimentResult", show: bool = True):
    """Plot initial/optimized paths and convergence diagnostics."""
    has_constraint = result.problem_metadata["constraint"] is not None
    n_cols = 3 + (2 if has_constraint else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    plot_spline_path(result.initial_controls, result.knot, ax=axes[0])
    _plot_reference_data(result, axes[0])
    axes[0].set_title("Initial path")

    plot_spline_path(result.optimized_controls, result.knot, ax=axes[1])
    _plot_reference_data(result, axes[1])
    axes[1].set_title("Optimized path")

    axes[2].plot(
        range(len(result.energy_history)),
        result.energy_history,
        marker="o",
        markersize=3,
    )
    axes[2].set_title("Energy")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Energy")

    if has_constraint:
        axes[3].plot(
            range(1, len(result.constraint_history) + 1),
            result.constraint_history,
            marker="o",
            markersize=3,
            color="tomato",
        )
        axes[3].set_title("Constraint violation")
        axes[3].set_xlabel("Outer iteration")
        axes[3].set_ylabel("g")

        axes[4].plot(
            range(1, len(result.multiplier_history) + 1),
            result.multiplier_history,
            marker="o",
            markersize=3,
            color="steelblue",
        )
        final_multiplier = (
            result.multiplier_history[-1]
            if len(result.multiplier_history)
            else 0.0
        )
        axes[4].set_title(
            f"Constraint multiplier\n(final = {final_multiplier:.4g})"
        )
        axes[4].set_xlabel("Outer iteration")
        axes[4].set_ylabel("lambda")

    fig.suptitle(result.title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _plot_reference_data(result: "ExperimentResult", ax) -> None:
    reference_points = [result.vertices]
    if result.trajectory is not None:
        ax.plot(
            result.trajectory[:, 0],
            result.trajectory[:, 1],
            "--",
            color="0.45",
            linewidth=1.0,
            label="Ground truth",
        )
        reference_points.append(result.trajectory)
    ax.scatter(
        result.vertices[:, 0],
        result.vertices[:, 1],
        color="black",
        s=18,
        zorder=4,
        label="Interpolation vertices",
    )
    if result.trajectory is not None:
        ax.legend()

    points = np.concatenate(reference_points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    current_x = ax.get_xlim()
    current_y = ax.get_ylim()
    ax.set_xlim(min(current_x[0], x_min), max(current_x[1], x_max))
    ax.set_ylim(min(current_y[0], y_min), max(current_y[1], y_max))
