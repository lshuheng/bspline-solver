"""Ground-truth dataset generation interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

from .datasets import TrajectoryDataset
from .experiment import VariationalProblem


@dataclass(frozen=True)
class FixedMass:
    """A fixed attracting point mass for generated Kepler trajectories."""

    center: Sequence[float]
    mass: float


def make_kepler_ground_truth_problem(
    masses: Sequence[Mapping[str, Any] | FixedMass],
    gravitational_constant: float,
    name: str = "kepler",
) -> VariationalProblem:
    """Create the mechanical Kepler problem used by the shared IVP pipeline."""
    mass_records = _normalize_masses(masses)
    gravitational_constant = float(gravitational_constant)
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be a positive finite value")

    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = sum(
        -gravitational_constant
        * fixed_mass["mass"]
        / (
            (u - fixed_mass["center"][0]) ** 2
            + (v - fixed_mass["center"][1]) ** 2
        )
        ** sp.Rational(1, 2)
        for fixed_mass in mass_records
    )
    kinetic_energy = sp.Rational(1, 2) * (ut**2 + vt**2)
    return VariationalProblem(
        name=name,
        lagrangian=kinetic_energy - potential,
        metadata={
            "gravitational_constant": gravitational_constant,
            "masses": mass_records,
        },
    )


def ground_truth(
    problem: VariationalProblem,
    initial_position: Sequence[float],
    initial_velocity: Sequence[float],
    t_span: tuple[float, float],
    n_vertices: int | Sequence[int],
    n_dense: int = 2000,
    name: str | None = None,
    geometric_sampling: bool = True,
    max_position_norm: float | None = None,
) -> TrajectoryDataset | list[TrajectoryDataset]:
    """Generate a dense trajectory from a potential-derived second-order ODE.

    Supported problems are unit-mass mechanical Lagrangians of the form
    ``0.5 * (ut**2 + vt**2) - V(u, v)``. As a convenience, a position-only
    expression is also accepted and treated directly as ``V(u, v)``.
    """
    if n_dense < 2:
        raise ValueError("n_dense must be at least 2")
    single_dataset = isinstance(n_vertices, Integral)
    vertex_counts = _normalize_vertex_counts(n_vertices, n_dense)

    position = _vector2(initial_position, "initial_position")
    velocity = _vector2(initial_velocity, "initial_velocity")
    trajectory, energy, potential, actual_t_span, termination = _solve_variational_ivp(
        problem=problem,
        initial_position=position,
        initial_velocity=velocity,
        t_span=t_span,
        n_dense=n_dense,
        max_position_norm=max_position_norm,
    )
    trajectory = _validate_trajectory(trajectory, vertex_counts)

    dataset_name = name or f"generated_{problem.name}_trajectory"
    datasets = [
        _build_ground_truth_dataset(
            name=_sampling_dataset_name(dataset_name, vertex_count, vertex_counts),
            problem=problem,
            potential=potential,
            trajectory=trajectory,
            energy=energy,
            initial_position=position,
            initial_velocity=velocity,
            t_span=t_span,
            actual_t_span=actual_t_span,
            termination=termination,
            n_vertices=vertex_count,
            geometric_sampling=geometric_sampling,
        )
        for vertex_count in vertex_counts
    ]
    if single_dataset:
        return datasets[0]
    return datasets


def ground_truth_kepler(
    masses: Sequence[Mapping[str, Any] | FixedMass],
    gravitational_constant: float,
    initial_position: Sequence[float],
    initial_velocity: Sequence[float],
    t_span: tuple[float, float],
    n_vertices: int | Sequence[int],
    n_dense: int = 2000,
    name: str = "generated_kepler_orbit",
    geometric_sampling: bool = True,
) -> TrajectoryDataset | list[TrajectoryDataset]:
    """Generate a Kepler trajectory through the shared ground_truth pipeline."""
    problem = make_kepler_ground_truth_problem(
        masses=masses,
        gravitational_constant=gravitational_constant,
    )
    return ground_truth(
        problem=problem,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        t_span=t_span,
        n_vertices=n_vertices,
        n_dense=n_dense,
        name=name,
        geometric_sampling=geometric_sampling,
    )


def _build_ground_truth_dataset(
    name: str,
    problem: VariationalProblem,
    potential: sp.Expr,
    trajectory: np.ndarray,
    energy: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    t_span: tuple[float, float],
    actual_t_span: tuple[float, float],
    termination: dict[str, Any],
    n_vertices: int,
    geometric_sampling: bool,
) -> TrajectoryDataset:
    metadata: dict[str, Any] = {
        "description": "Generated trajectory from a VariationalProblem.",
        "source": "VariationalProblem IVP solver",
        "ground_truth_available": True,
        "problem_name": problem.name,
        "lagrangian": str(problem.lagrangian),
        "potential": str(potential),
        "ode": "q'' = -grad V(q)",
        "energy": float(energy),
        "initial_position": initial_position.tolist(),
        "initial_velocity": initial_velocity.tolist(),
        "t_span": [float(t_span[0]), float(t_span[1])],
        "actual_t_span": [float(actual_t_span[0]), float(actual_t_span[1])],
        "termination": termination,
        "n_dense": len(trajectory),
        "n_vertices": int(n_vertices),
        "geometric_sampling": bool(geometric_sampling),
    }
    metadata.update(problem.metadata)
    vertices, sample_indices, sampling_metadata = _sample_trajectory_vertices(
        trajectory,
        n_vertices,
        geometric_sampling,
    )
    metadata.update(sampling_metadata)
    metadata["sample_indices"] = sample_indices.tolist()

    return TrajectoryDataset(
        name=name,
        trajectory=trajectory,
        vertices=vertices,
        metadata=metadata,
    )


def _normalize_vertex_counts(
    n_vertices: int | Sequence[int],
    n_dense: int,
) -> list[int]:
    if isinstance(n_vertices, Integral) and not isinstance(n_vertices, bool):
        vertex_counts = [int(n_vertices)]
    else:
        if isinstance(n_vertices, str | bytes):
            raise ValueError("n_vertices must be an integer or a sequence of integers")
        vertex_counts = []
        try:
            counts = list(n_vertices)
        except TypeError as exc:
            raise ValueError(
                "n_vertices must be an integer or a sequence of integers"
            ) from exc
        for count in counts:
            if not isinstance(count, Integral) or isinstance(count, bool):
                raise ValueError("n_vertices values must be integers")
            vertex_counts.append(int(count))
        if not vertex_counts:
            raise ValueError("n_vertices must contain at least one sampling level")

    for vertex_count in vertex_counts:
        if not 2 <= vertex_count <= n_dense:
            raise ValueError("n_vertices values must be between 2 and n_dense")
    return vertex_counts


def _validate_trajectory(
    trajectory: np.ndarray,
    vertex_counts: Sequence[int],
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("IVP solver must return a trajectory with shape (m, 2)")
    if len(trajectory) < max(vertex_counts):
        raise ValueError("IVP solver trajectory must contain enough sampled points")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("IVP solver trajectory must contain only finite values")
    return trajectory


def _sampling_dataset_name(
    base_name: str,
    vertex_count: int,
    vertex_counts: Sequence[int],
) -> str:
    if len(vertex_counts) == 1:
        return base_name
    return f"{base_name}_{vertex_count}_points"


def _sample_trajectory_vertices(
    trajectory: np.ndarray,
    n_vertices: int,
    geometric_sampling: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arclength_data = _trajectory_arclength_data(trajectory)
    _, _, total_arclength = arclength_data
    if geometric_sampling:
        vertices, sample_indices, sample_arclengths, sample_segment_indices = (
            _sample_vertices_by_arclength(trajectory, n_vertices, arclength_data)
        )
        return vertices, sample_indices, {
            "sampling_method": "arclength",
            "total_arclength": float(total_arclength),
            "sample_arclengths": sample_arclengths.tolist(),
            "sample_segment_indices": sample_segment_indices.tolist(),
        }

    sample_indices = np.linspace(
        0,
        len(trajectory) - 1,
        n_vertices,
        dtype=int,
    )
    return trajectory[sample_indices], sample_indices, {
        "sampling_method": "index",
        "total_arclength": float(total_arclength),
    }


def _trajectory_arclength_data(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    segment_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative_lengths[-1])
    if not np.isfinite(total_length):
        raise ValueError("trajectory arclength must be finite")
    return segment_lengths, cumulative_lengths, total_length


def _sample_vertices_by_arclength(
    trajectory: np.ndarray,
    n_vertices: int,
    arclength_data: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if arclength_data is None:
        arclength_data = _trajectory_arclength_data(trajectory)
    segment_lengths, cumulative_lengths, total_length = arclength_data
    if total_length <= 0.0:
        raise ValueError("trajectory must have positive arclength")

    sample_arclengths = np.linspace(0.0, total_length, n_vertices)
    sample_segment_indices = np.searchsorted(
        cumulative_lengths,
        sample_arclengths,
        side="right",
    ) - 1
    sample_segment_indices = np.clip(
        sample_segment_indices,
        0,
        len(segment_lengths) - 1,
    )

    starts = cumulative_lengths[sample_segment_indices]
    lengths = segment_lengths[sample_segment_indices]
    weights = np.divide(
        sample_arclengths - starts,
        lengths,
        out=np.zeros_like(sample_arclengths),
        where=lengths > 0.0,
    )
    vertices = (
        (1.0 - weights[:, None]) * trajectory[sample_segment_indices]
        + weights[:, None] * trajectory[sample_segment_indices + 1]
    )
    vertices[0] = trajectory[0]
    vertices[-1] = trajectory[-1]

    sample_indices = np.searchsorted(
        cumulative_lengths,
        sample_arclengths,
        side="left",
    )
    sample_indices = np.clip(sample_indices, 0, len(trajectory) - 1)
    sample_indices[0] = 0
    sample_indices[-1] = len(trajectory) - 1
    return vertices, sample_indices, sample_arclengths, sample_segment_indices


def _solve_variational_ivp(
    problem: VariationalProblem,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    t_span: tuple[float, float],
    n_dense: int,
    max_position_norm: float | None,
) -> tuple[np.ndarray, float, sp.Expr, tuple[float, float], dict[str, Any]]:
    """Return a dense position trajectory, conserved energy, and potential."""
    if n_dense < 2:
        raise ValueError("n_dense must be at least 2")
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not np.isfinite([t0, t1]).all() or t1 <= t0:
        raise ValueError("t_span must contain two finite increasing values")
    if max_position_norm is not None:
        max_position_norm = float(max_position_norm)
        if not np.isfinite(max_position_norm) or max_position_norm <= 0.0:
            raise ValueError("max_position_norm must be a positive finite value")

    position = _vector2(initial_position, "initial_position")
    velocity = _vector2(initial_velocity, "initial_velocity")
    if (
        max_position_norm is not None
        and np.linalg.norm(position) >= max_position_norm
    ):
        raise ValueError("initial_position norm must be less than max_position_norm")
    potential = _potential_from_problem(problem)
    u, v = sp.symbols("u v")
    potential_func = sp.lambdify((u, v), potential, modules="numpy")
    grad_potential_func = sp.lambdify(
        (u, v),
        (sp.diff(potential, u), sp.diff(potential, v)),
        modules="numpy",
    )

    initial_potential = float(potential_func(position[0], position[1]))
    if not np.isfinite(initial_potential):
        raise ValueError("initial potential energy must be finite")
    energy = 0.5 * float(np.dot(velocity, velocity)) + initial_potential
    initial_state = np.concatenate([position, velocity])
    t_eval = np.linspace(t0, t1, int(n_dense))

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        point = state[:2]
        speed = state[2:]
        grad_potential = np.asarray(
            grad_potential_func(point[0], point[1]),
            dtype=float,
        ).reshape(2)
        if not np.all(np.isfinite(grad_potential)):
            raise ValueError("potential gradient must be finite along trajectory")
        acceleration = -grad_potential
        return np.concatenate([speed, acceleration])

    events = None
    if max_position_norm is not None:

        def position_norm_event(_time: float, state: np.ndarray) -> float:
            return max_position_norm - float(np.linalg.norm(state[:2]))

        position_norm_event.terminal = True
        position_norm_event.direction = -1
        events = position_norm_event

    solution = solve_ivp(
        rhs,
        (t0, t1),
        initial_state,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        events=events,
    )
    if not solution.success:
        raise RuntimeError(f"Variational IVP integration failed: {solution.message}")
    if solution.y.shape[0] != 4:
        raise RuntimeError("Variational IVP integration returned an unexpected shape")
    if solution.y.shape[1] < 2:
        raise RuntimeError("Variational IVP integration returned too few samples")

    position_limit_reached = (
        max_position_norm is not None
        and solution.t_events is not None
        and len(solution.t_events) == 1
        and len(solution.t_events[0]) > 0
    )
    if not position_limit_reached and solution.y.shape[1] != n_dense:
        raise RuntimeError("Variational IVP integration returned an unexpected shape")

    actual_t_span = (float(solution.t[0]), float(solution.t[-1]))
    if position_limit_reached:
        termination = {
            "reason": "max_position_norm",
            "time": float(solution.t_events[0][0]),
            "max_position_norm": float(max_position_norm),
        }
    else:
        termination = {"reason": "t_span_end", "time": float(t1)}

    return solution.y[:2].T, energy, potential, actual_t_span, termination


def _potential_from_problem(problem: VariationalProblem) -> sp.Expr:
    u, v, ut, vt, utt, vtt, t = sp.symbols("u v ut vt utt vtt t")
    derivative_symbols = {ut, vt, utt, vtt}
    unsupported_symbols = derivative_symbols | {t}
    lagrangian = problem.lagrangian
    kinetic_energy = sp.Rational(1, 2) * (ut**2 + vt**2)

    potential = sp.simplify(kinetic_energy - lagrangian)
    if not potential.free_symbols & unsupported_symbols:
        return potential

    if not lagrangian.free_symbols & unsupported_symbols:
        return lagrangian

    raise ValueError(
        "ground_truth supports mechanical Lagrangians of the form "
        "0.5 * (ut**2 + vt**2) - V(u, v), or direct potentials V(u, v)"
    )


def _vector2(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain two finite values")
    return vector


def _normalize_masses(
    masses: Sequence[Mapping[str, Any] | FixedMass],
) -> list[dict[str, Any]]:
    records = []
    for fixed_mass in masses:
        if isinstance(fixed_mass, Mapping):
            center_value = fixed_mass["center"]
            mass_value = fixed_mass["mass"]
        else:
            center_value = fixed_mass.center
            mass_value = fixed_mass.mass
        center = _vector2(center_value, "mass center")
        mass = float(mass_value)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be a positive finite value")
        records.append({"center": center.tolist(), "mass": mass})
    if not records:
        raise ValueError("at least one fixed mass is required")
    return records
