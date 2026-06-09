"""Ground-truth dataset generation interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .datasets import TrajectoryDataset


def ground_truth_kepler(
    masses: Sequence[Mapping[str, Any]],
    gravitational_constant: float,
    initial_position: Sequence[float],
    initial_velocity: Sequence[float],
    t_span: tuple[float, float],
    n_vertices: int,
    n_dense: int = 10_000,
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
    masses: Sequence[Mapping[str, Any]],
    gravitational_constant: float,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    t_span: tuple[float, float],
    n_dense: int,
) -> tuple[np.ndarray, float]:
    """Return a dense position trajectory and its conserved energy."""
    raise NotImplementedError("Kepler IVP integration is not implemented")


def _vector2(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain two finite values")
    return vector


def _normalize_masses(
    masses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for fixed_mass in masses:
        center = _vector2(fixed_mass["center"], "mass center")
        mass = float(fixed_mass["mass"])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be a positive finite value")
        records.append({"center": center.tolist(), "mass": mass})
    if not records:
        raise ValueError("at least one fixed mass is required")
    return records
