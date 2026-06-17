"""Ground-truth dataset generation interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .datasets import TrajectoryDataset


@dataclass(frozen=True)
class FixedMass:
    """A fixed attracting point mass for generated Kepler trajectories."""

    center: Sequence[float]
    mass: float


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
    """Generate a dense Kepler trajectory and subsample interpolation vertices."""
    if n_dense < 2:
        raise ValueError("n_dense must be at least 2")
    single_dataset = isinstance(n_vertices, Integral)
    vertex_counts = _normalize_vertex_counts(n_vertices, n_dense)

    position = _vector2(initial_position, "initial_position")
    velocity = _vector2(initial_velocity, "initial_velocity")
    mass_records = _normalize_masses(masses)

    trajectory, energy = _solve_kepler_ivp(
        masses=mass_records,
        gravitational_constant=float(gravitational_constant),
        initial_position=position,
        initial_velocity=velocity,
        t_span=t_span,
        n_dense=n_dense,
    )
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("IVP solver must return a trajectory with shape (m, 2)")
    if len(trajectory) < max(vertex_counts):
        raise ValueError("IVP solver trajectory must contain enough sampled points")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("IVP solver trajectory must contain only finite values")

    datasets = [
        _build_kepler_dataset(
            name=(
                name
                if len(vertex_counts) == 1
                else f"{name}_{vertex_count}_points"
            ),
            trajectory=trajectory,
            energy=energy,
            gravitational_constant=float(gravitational_constant),
            mass_records=mass_records,
            initial_position=position,
            initial_velocity=velocity,
            t_span=t_span,
            n_vertices=vertex_count,
            geometric_sampling=geometric_sampling,
        )
        for vertex_count in vertex_counts
    ]
    if single_dataset:
        return datasets[0]
    return datasets


def _build_kepler_dataset(
    name: str,
    trajectory: np.ndarray,
    energy: float,
    gravitational_constant: float,
    mass_records: list[dict[str, Any]],
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    t_span: tuple[float, float],
    n_vertices: int,
    geometric_sampling: bool,
) -> TrajectoryDataset:
    metadata: dict[str, Any] = {
        "description": "Generated Kepler trajectory.",
        "source": "Kepler IVP solver",
        "ground_truth_available": True,
        "energy": float(energy),
        "gravitational_constant": float(gravitational_constant),
        "masses": mass_records,
        "initial_position": initial_position.tolist(),
        "initial_velocity": initial_velocity.tolist(),
        "t_span": [float(t_span[0]), float(t_span[1])],
        "n_dense": len(trajectory),
        "n_vertices": int(n_vertices),
        "geometric_sampling": bool(geometric_sampling),
    }
    if geometric_sampling:
        vertices, sample_indices, sample_arclengths, sample_segment_indices = (
            _sample_vertices_by_arclength(trajectory, n_vertices)
        )
        metadata.update(
            {
                "sampling_method": "arclength",
                "sample_arclengths": sample_arclengths.tolist(),
                "sample_segment_indices": sample_segment_indices.tolist(),
            }
        )
    else:
        sample_indices = np.linspace(
            0,
            len(trajectory) - 1,
            n_vertices,
            dtype=int,
        )
        vertices = trajectory[sample_indices]
        metadata["sampling_method"] = "index"
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


def _sample_vertices_by_arclength(
    trajectory: np.ndarray,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segment_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_lengths[-1]
    if not np.isfinite(total_length) or total_length <= 0.0:
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


def _solve_kepler_ivp(
    masses: Sequence[Mapping[str, Any] | FixedMass],
    gravitational_constant: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    t_span: tuple[float, float],
    n_dense: int,
) -> tuple[np.ndarray, float]:
    """Return a dense position trajectory and its conserved energy."""
    if n_dense < 2:
        raise ValueError("n_dense must be at least 2")
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not np.isfinite([t0, t1]).all() or t1 <= t0:
        raise ValueError("t_span must contain two finite increasing values")

    mass_records = _normalize_masses(masses)
    centers = np.asarray([record["center"] for record in mass_records], dtype=float)
    mass_values = np.asarray([record["mass"] for record in mass_records], dtype=float)
    gravitational_constant = float(gravitational_constant)
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be a positive finite value")

    position = _vector2(initial_position, "initial_position")
    velocity = _vector2(initial_velocity, "initial_velocity")
    energy = 0.5 * float(np.dot(velocity, velocity)) + _kepler_potential(
        position,
        centers,
        mass_values,
        gravitational_constant,
    )
    initial_state = np.concatenate([position, velocity])
    t_eval = np.linspace(t0, t1, int(n_dense))

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        point = state[:2]
        speed = state[2:]
        displacement = point - centers
        distances = np.linalg.norm(displacement, axis=1)
        if np.any(distances == 0.0):
            raise ValueError("trajectory intersects a fixed mass center")
        acceleration = (
            -gravitational_constant
            * np.sum(
                mass_values[:, None]
                * displacement
                / distances[:, None] ** 3,
                axis=0,
            )
        )
        return np.concatenate([speed, acceleration])

    solution = solve_ivp(
        rhs,
        (t0, t1),
        initial_state,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Kepler IVP integration failed: {solution.message}")
    if solution.y.shape != (4, n_dense):
        raise RuntimeError("Kepler IVP integration returned an unexpected shape")

    return solution.y[:2].T, energy


def _kepler_potential(
    position: np.ndarray,
    centers: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float,
) -> float:
    displacement = position - centers
    distances = np.linalg.norm(displacement, axis=1)
    if np.any(distances == 0.0):
        raise ValueError("initial_position must not coincide with a fixed mass center")
    return -gravitational_constant * float(np.sum(masses / distances))


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
