"""Dataset interfaces for manually supplied and future generated trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class TrajectoryDataset:
    """Trajectory observations used by one interpolation experiment.

    Dense trajectory samples are optional while datasets are curated manually.
    Interpolation vertices are always required.
    """

    name: str
    vertices: np.ndarray
    trajectory: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vertices = _points_array(self.vertices, "vertices")
        trajectory = (
            None
            if self.trajectory is None
            else _points_array(self.trajectory, "trajectory")
        )

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _points_array(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{label} must have shape (n, 2)")
    if len(points) < 2:
        raise ValueError(f"{label} must contain at least two points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must contain only finite values")
    return points.copy()


def load_dataset(name: str) -> TrajectoryDataset:
    """Load a manually curated dataset bundled with the project."""
    if name == "kepler_orbit_1":
        from .data.kepler_orbit_1 import DATASET

        record = DATASET
    else:
        from .data.basic import DATASETS

        try:
            record = DATASETS[name]
        except KeyError as exc:
            available = sorted([*DATASETS, "kepler_orbit_1"])
            raise KeyError(
                f"unknown dataset {name!r}; available datasets: {available}"
            ) from exc

    return TrajectoryDataset(
        name=name,
        vertices=record["vertices"],
        trajectory=record.get("trajectory"),
        metadata=record.get("metadata", {}),
    )
