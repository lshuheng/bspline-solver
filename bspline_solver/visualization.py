"""Plotting utilities for B-spline segments and control polygons."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

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

    for index, control in enumerate(segments):
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

        ax.plot(
            sx_plot,
            sy_plot,
            color="tab:orange",
            label="Interpolated trajectory" if index == 0 else "_nolegend_",
        )

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


def plot_linear_path(vertices: np.ndarray, ax=None, cyclic: bool = False):
    """Plot the piecewise-linear path through interpolation vertices."""
    if ax is None:
        _, ax = plt.subplots()

    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("vertices must have shape (n, 2) with n >= 2")
    if cyclic and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    ax.plot(
        points[:, 0],
        points[:, 1],
        color="tab:orange",
        label="Interpolated trajectory",
    )
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")
    return ax


def plot_scipy_spline_path(
    vertices: np.ndarray,
    resolution: int = 2000,
    ax=None,
    cyclic: bool = False,
):
    """Plot SciPy's parametric interpolating spline through the vertices."""
    if ax is None:
        _, ax = plt.subplots()

    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("vertices must have shape (n, 2)")
    if cyclic and len(points) >= 2 and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    parameter = np.concatenate([[0.0], np.cumsum(distances)])
    keep = np.concatenate([[True], np.diff(parameter) > 0.0])
    points = points[keep]
    parameter = parameter[keep]
    if len(points) < 2:
        raise ValueError("at least two distinct vertices are required")

    parameter = parameter / parameter[-1]
    degree = min(3, len(points) - 1)
    spline = make_interp_spline(
        parameter,
        points,
        k=degree,
        axis=0,
        bc_type="periodic" if cyclic and degree > 1 else None,
    )
    sample_t = np.linspace(0.0, 1.0, int(resolution))
    sample_points = spline(sample_t)

    ax.plot(
        sample_points[:, 0],
        sample_points[:, 1],
        color="tab:orange",
        label="Interpolated trajectory",
    )
    x_min, y_min = sample_points.min(axis=0)
    x_max, y_max = sample_points.max(axis=0)
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")
    return ax


def plot_result(result: "ExperimentResult", diagnostic_mode = False, show: bool = True):
    """Plot initial/optimized paths and optional convergence traces."""
    has_constraint = result.has_constraint
    n_path_plots = 3
    n_cols = n_path_plots + (1 if diagnostic_mode else 0) + (2 if has_constraint and diagnostic_mode else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    plot_linear_path(result.vertices, ax=axes[0], cyclic=result.cyclic)
    _plot_reference_data(result, axes[0], mark_endpoints=False)
    axes[0].set_title("Linear Interpolation")

    plot_scipy_spline_path(result.vertices, ax=axes[1], cyclic=result.cyclic)
    _plot_reference_data(result, axes[1], mark_endpoints=False)
    axes[1].set_title("SciPy spline interpolation")

    plot_spline_path(result.optimized_controls, result.knot, ax=axes[2])
    _plot_reference_data(result, axes[2], mark_endpoints=False)
    axes[2].set_title("Physics-based interpolation")

    if diagnostic_mode:

        axes[3].plot(
            range(len(result.energy_history)),
            result.energy_history,
            marker="o",
            markersize=3,
        )
        axes[3].set_title("Energy")
        axes[3].set_xlabel("Iteration")
        axes[3].set_ylabel("Energy")

        if has_constraint:
            axes[4].plot(
                range(1, len(result.constraint_history) + 1),
                result.constraint_history,
                marker="o",
                markersize=3,
                color="tomato",
            )
            axes[4].set_title("Constraint violation")
            axes[4].set_xlabel("Outer iteration")
            axes[4].set_ylabel("g")

            axes[5].plot(
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
            axes[5].set_title(
                f"Constraint multiplier\n(final = {final_multiplier:.4g})"
            )
            axes[5].set_xlabel("Outer iteration")
            axes[5].set_ylabel("lambda")

    if result.constraint_target is not None:
        fig.suptitle(f"Target length = {result.constraint_target:g}")
    _add_shared_path_legend(fig, axes[:n_path_plots])
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    if show:
        plt.show()
    return fig, axes


def plot_sampling_comparison(
    results: list["ExperimentResult"],
    show: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Plot the three interpolation views for each sampling level together."""
    if not results:
        raise ValueError("results must contain at least one ExperimentResult")

    if figsize is None:
        figsize = (12.0, max(3.2, 2.8 * len(results)))
    fig, axes = plt.subplots(len(results), 3, figsize=figsize)
    axes = np.atleast_2d(axes)

    column_titles = [
        "Linear Interpolation",
        "SciPy spline interpolation",
        "Physics-based interpolation",
    ]
    for row_index, (row, result) in enumerate(zip(axes, results)):
        plot_linear_path(result.vertices, ax=row[0])
        _plot_reference_data(result, row[0])

        plot_scipy_spline_path(result.vertices, ax=row[1])
        _plot_reference_data(result, row[1])

        plot_spline_path(result.optimized_controls, result.knot, ax=row[2])
        _plot_reference_data(result, row[2])

        if row_index == 0:
            for ax, title in zip(row, column_titles):
                ax.set_title(title, fontsize=10)
        row[0].set_ylabel(f"Points = {len(result.vertices)}", fontsize=9)
        for ax in row:
            ax.tick_params(labelsize=8)

    _add_shared_path_legend(fig, axes.ravel())
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), h_pad=1.0, w_pad=0.5)
    if show:
        plt.show()
    return fig, axes


def _plot_reference_data(
    result: "ExperimentResult",
    ax,
    mark_endpoints: bool = True,
) -> None:
    reference_points = [result.vertices]
    if result.trajectory is not None:
        trajectory = np.asarray(result.trajectory, dtype=float)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "--",
            color="tab:blue",
            linewidth=1.0,
            label="True trajectory",
        )
        reference_points.append(trajectory)
    ax.scatter(
        result.vertices[:, 0],
        result.vertices[:, 1],
        color="black",
        s=18,
        zorder=4,
        label="Interpolation points",
    )
    if mark_endpoints:
        ax.scatter(
            result.vertices[0, 0],
            result.vertices[0, 1],
            facecolors="none",
            edgecolors="tab:green",
            marker="o",
            s=80,
            linewidths=1.8,
            zorder=5,
            label="Start",
        )
        ax.scatter(
            result.vertices[-1, 0],
            result.vertices[-1, 1],
            color="tab:red",
            marker="x",
            s=80,
            linewidths=1.8,
            zorder=5,
            label="End",
        )

    points = np.concatenate(reference_points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    current_x = ax.get_xlim()
    current_y = ax.get_ylim()
    x_min = min(current_x[0], x_min)
    x_max = max(current_x[1], x_max)
    y_min = min(current_y[0], y_min)
    y_max = max(current_y[1], y_max)
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")


def _add_shared_path_legend(fig, axes) -> None:
    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                handles_by_label.setdefault(label, handle)
    if not handles_by_label:
        return

    ordered_labels = [
        "True trajectory",
        "Interpolated trajectory",
        "Interpolation points",
        "Start",
        "End",
        
    ]
    labels = [label for label in ordered_labels if label in handles_by_label]
    handles = [handles_by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
