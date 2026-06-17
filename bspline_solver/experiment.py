"""High-level experiment definitions and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import sympy as sp

from .datasets import TrajectoryDataset
from .regularization import control_variance
from .solver import EnergyMinimizer2D
from .spline import SplinePath

Regularizer = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class VariationalProblem:
    """Mathematical objective and optional problem parameters."""

    name: str
    lagrangian: sp.Expr
    constraint: Optional[sp.Expr] = None
    constraint_target: float = 0.0
    regularization: Optional[Regularizer] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.constraint is None and self.constraint_target != 0.0:
            raise ValueError("constraint_target requires a constraint integrand")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExperimentConfig:
    """Spline topology and numerical settings for one experiment run."""

    theta: Optional[list[float]] = None
    fix_location: Optional[list[bool]] = None
    fix_angle: Optional[list[bool]] = None
    cyclic: bool = False
    n_bisections: int = 2
    n_quad: int = 5
    max_iteration: int = 50
    constraint_tol: float = 1e-4
    geometric_init: bool = False


@dataclass
class ExperimentResult:
    """Numerical output from a variational experiment."""

    vertices: np.ndarray
    trajectory: Optional[np.ndarray]
    initial_controls: list[np.ndarray]
    optimized_controls: list[np.ndarray]
    knot: np.ndarray
    energy_history: np.ndarray
    constraint_history: np.ndarray
    multiplier_history: np.ndarray
    has_constraint: bool


def solve_experiment(
    dataset: TrajectoryDataset,
    problem: VariationalProblem,
    config: Optional[ExperimentConfig] = None,
) -> ExperimentResult:
    """Build and solve one interpolation experiment without plotting or I/O."""
    config = config or ExperimentConfig()
    path = SplinePath(
        dataset.vertices.tolist(),
        theta=config.theta,
        fix_location=config.fix_location,
        fix_angle=config.fix_angle,
        cyclic=config.cyclic,
    )
    initial_control, knot = path.initial_controls(config.n_bisections)
    control = {edge: values.copy() for edge, values in initial_control.items()}

    if config.geometric_init:
        utt, vtt = sp.symbols("utt vtt")
        geometric_solver = EnergyMinimizer2D(
            path,
            control,
            knot,
            utt**2 + vtt**2,
            reg=control_variance,
            n_quad=config.n_quad,
        )
        geometric_solver.minimize(max_iteration=config.max_iteration)
        control = {
            edge: values.copy()
            for edge, values in geometric_solver.control.items()
        }

    solver = EnergyMinimizer2D(
        path,
        control,
        knot,
        problem.lagrangian,
        (
            None
            if problem.constraint is None
            else problem.constraint - problem.constraint_target / len(path.edges)
        ),
        reg=problem.regularization,
        n_quad=config.n_quad,
    )
    solver.minimize(
        max_iteration=config.max_iteration,
        constraint_tol=config.constraint_tol,
    )

    return ExperimentResult(
        vertices=dataset.vertices.copy(),
        trajectory=(
            None if dataset.trajectory is None else dataset.trajectory.copy()
        ),
        initial_controls=[control[edge].copy() for edge in path.edges],
        optimized_controls=[
            solver.control[edge].copy() for edge in path.edges
        ],
        knot=np.asarray(knot).copy(),
        energy_history=np.asarray(solver.energy_history, dtype=float),
        constraint_history=np.asarray(solver.constraint_history, dtype=float),
        multiplier_history=np.asarray(solver.multiplier_history, dtype=float),
        has_constraint=problem.constraint is not None,
    )


def solve_sampling_experiments(
    datasets: Iterable[TrajectoryDataset],
    problem_factory: Callable[[TrajectoryDataset], VariationalProblem],
    config: Optional[ExperimentConfig] = None,
) -> list[ExperimentResult]:
    """Solve the same problem family across multiple sampled datasets."""
    config = config or ExperimentConfig()
    return [
        solve_experiment(
            dataset,
            problem_factory(dataset),
            config,
        )
        for dataset in datasets
    ]
