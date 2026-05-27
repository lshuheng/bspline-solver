"""Diagnostic runner: optimize a spline path and display convergence plots."""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import sympy as sp

from .regularization import control_variance
from .solver import EnergyMinimizer2D
from .spline import SplinePath
from .visualization import plot_spline_path


def run_diagnostic(
    vertices: list,
    lagrangian: sp.Expr,
    constraint: Optional[sp.Expr] = None,
    theta: Optional[list[float]] = None,
    fix_location: Optional[list[bool]] = None,
    fix_angle: Optional[list[bool]] = None,
    cyclic: bool = False,
    n_bisections: int = 2,
    n_quad: int = 5,
    reg: Optional[Callable] = None,
    max_iteration: int = 50,
    constraint_tol: float = 1e-4,
    title: Optional[str] = None,
) -> EnergyMinimizer2D:
    """Build, optimize, and diagnose a spline path.

    Args:
        vertices: Ordered list of (x, y) waypoints.
        lagrangian: Sympy expression for the Lagrangian integrand.
        constraint: Optional sympy expression whose integral is constrained to zero.
        theta: Tangent angles at each waypoint; auto-generated if None.
        fix_location: Per-waypoint location-fixed flags; all True if None.
        fix_angle: Per-waypoint angle-fixed flags; all True if None.
        cyclic: Whether the path is closed.
        n_bisections: Knot refinement level (controls number of control points).
        n_quad: Gauss-Legendre quadrature points per knot interval.
        reg: Regularization callable (control_variance by default); pass None to disable.
        max_iteration: Maximum augmented-Lagrangian outer iterations.
        constraint_tol: Convergence tolerance on constraint violation.
        title: Optional suptitle for the figure.

    Returns:
        The EnergyMinimizer2D solver after optimization (control points in solver.control).
    """
    path = SplinePath(
        vertices,
        theta=theta,
        fix_location=fix_location,
        fix_angle=fix_angle,
        cyclic=cyclic,
    )
    control, knot = path.initial_controls(n_bisections)

    solver = EnergyMinimizer2D(path, control, knot, lagrangian, constraint, reg=reg, n_quad=n_quad)
    solver.minimize(max_iteration=max_iteration, constraint_tol=constraint_tol)

    has_constraint = solver.constraint is not None
    n_cols = 3 + (2 if has_constraint else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    plot_spline_path(list(control.values()), knot, ax=axes[0])
    axes[0].set_title("Initial path")

    plot_spline_path(list(solver.control.values()), knot, ax=axes[1])
    axes[1].set_title("Optimized path")

    iters = range(len(solver.energy_history))
    axes[2].plot(iters, solver.energy_history, marker="o", markersize=3)
    axes[2].set_title("Energy")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Energy")

    if has_constraint:
        axes[3].plot(
            range(1, len(solver.constraint_history) + 1),
            solver.constraint_history,
            marker="o", markersize=3, color="tomato",
        )
        axes[3].set_title("Constraint violation")
        axes[3].set_xlabel("Outer iteration")
        axes[3].set_ylabel("g")

        axes[4].plot(
            range(1, len(solver.multiplier_history) + 1),
            solver.multiplier_history,
            marker="o", markersize=3, color="steelblue",
        )
        final_lam = solver.multiplier_history[-1] if solver.multiplier_history else 0.0
        axes[4].set_title(f"Constraint multiplier\n(final = {final_lam:.4g})")
        axes[4].set_xlabel("Outer iteration")
        axes[4].set_ylabel("λ")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return solver
