"""High-level experiment definitions, execution, and stable result persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional

import numpy as np
import sympy as sp

from .datasets import TrajectoryDataset
from .regularization import control_variance
from .solver import EnergyMinimizer2D
from .spline import SplinePath

Regularizer = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class VariationalProblem:
    """Mathematical objective and presentation metadata for an experiment."""

    name: str
    lagrangian: sp.Expr
    constraint: Optional[sp.Expr] = None
    constraint_target: float = 0.0
    regularization: Optional[Regularizer] = None
    title: Optional[str] = None
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
    """Solver-independent numerical output from a variational experiment."""

    dataset_name: str
    problem_name: str
    title: str
    vertices: np.ndarray
    trajectory: Optional[np.ndarray]
    initial_controls: list[np.ndarray]
    optimized_controls: list[np.ndarray]
    knot: np.ndarray
    energy_history: np.ndarray
    constraint_history: np.ndarray
    multiplier_history: np.ndarray
    dataset_metadata: dict[str, Any]
    problem_metadata: dict[str, Any]
    config: dict[str, Any]
    diagnostics: dict[str, Any]


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

    regularizer_name = (
        None
        if problem.regularization is None
        else getattr(problem.regularization, "__name__", repr(problem.regularization))
    )
    problem_metadata = {
        **problem.metadata,
        "lagrangian": str(problem.lagrangian),
        "constraint": (
            None if problem.constraint is None else str(problem.constraint)
        ),
        "constraint_target": problem.constraint_target,
        "regularization": regularizer_name,
    }

    return ExperimentResult(
        dataset_name=dataset.name,
        problem_name=problem.name,
        title=problem.title or problem.name,
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
        dataset_metadata=dict(dataset.metadata),
        problem_metadata=problem_metadata,
        config={
            "theta": config.theta,
            "fix_location": config.fix_location,
            "fix_angle": config.fix_angle,
            "cyclic": config.cyclic,
            "n_bisections": config.n_bisections,
            "n_quad": config.n_quad,
            "max_iteration": config.max_iteration,
            "constraint_tol": config.constraint_tol,
            "geometric_init": config.geometric_init,
        },
        diagnostics={
            "n_segments": len(path.edges),
            "n_controls_per_segment": int(
                next(iter(solver.control.values())).shape[1]
            ),
            "outer_iterations": len(solver.energy_history) - 1,
            "final_energy": float(solver.energy_history[-1]),
            "final_constraint_violation": (
                None
                if not solver.constraint_history
                else float(solver.constraint_history[-1])
            ),
            "constraint_satisfied": (
                None
                if problem.constraint is None
                else bool(
                    solver.constraint_history
                    and abs(solver.constraint_history[-1])
                    < config.constraint_tol
                )
            ),
        },
    )


def save_result(
    result: ExperimentResult,
    output_dir: str | Path = "outputs",
    name: Optional[str] = None,
) -> tuple[Path, Path]:
    """Save numerical arrays as NPZ and descriptive metadata as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(name or f"{result.dataset_name}_{result.problem_name}")
    arrays_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.json"

    arrays: dict[str, np.ndarray] = {
        "vertices": result.vertices,
        "knot": result.knot,
        "energy_history": result.energy_history,
        "constraint_history": result.constraint_history,
        "multiplier_history": result.multiplier_history,
    }
    if result.trajectory is not None:
        arrays["trajectory"] = result.trajectory
    for index, control in enumerate(result.initial_controls):
        arrays[f"initial_control_{index:03d}"] = control
    for index, control in enumerate(result.optimized_controls):
        arrays[f"optimized_control_{index:03d}"] = control
    np.savez_compressed(arrays_path, **arrays)

    metadata = {
        "schema_version": 1,
        "dataset_name": result.dataset_name,
        "problem_name": result.problem_name,
        "title": result.title,
        "n_segments": len(result.optimized_controls),
        "dataset_metadata": result.dataset_metadata,
        "problem_metadata": result.problem_metadata,
        "config": result.config,
        "diagnostics": result.diagnostics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    return arrays_path, metadata_path


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "experiment"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
