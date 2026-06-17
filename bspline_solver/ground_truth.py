"""Ground-truth dataset generation interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    n_vertices: int,
    n_dense: int = 2000,
    name: str = "generated_kepler_orbit",
) -> TrajectoryDataset:
    """Generate a dense Kepler trajectory and evenly subsample its vertices."""
    if n_dense < 2:
        raise ValueError("n_dense must be at least 2")
    if not 2 <= n_vertices <= n_dense:
        raise ValueError("n_vertices must be between 2 and n_dense")

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
    if len(trajectory) < n_vertices:
        raise ValueError(
            "IVP solver trajectory must contain at least n_vertices points"
        )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("IVP solver trajectory must contain only finite values")

    sample_indices = np.linspace(
        0,
        len(trajectory) - 1,
        n_vertices,
        dtype=int,
    )

    return TrajectoryDataset(
        name=name,
        trajectory=trajectory,
        vertices=trajectory[sample_indices],
        metadata={
            "description": "Generated Kepler trajectory.",
            "source": "Kepler IVP solver",
            "ground_truth_available": True,
            "energy": float(energy),
            "gravitational_constant": float(gravitational_constant),
            "masses": mass_records,
            "initial_position": position.tolist(),
            "initial_velocity": velocity.tolist(),
            "t_span": [float(t_span[0]), float(t_span[1])],
            "n_dense": len(trajectory),
            "sample_indices": sample_indices.tolist(),
        },
    )


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
