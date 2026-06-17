"""Variational B-spline ODE solver for boundary value problems."""

from .config import (
    CONSTRAINT_STIFFNESS,
    DEGREE,
    REGULARIZATION_STIFFNESS,
    TANGENT_INIT_SCALE,
)
from .datasets import TrajectoryDataset, load_dataset
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    VariationalProblem,
    save_result,
    solve_experiment,
)
from .lagrangian import Lagrangian2D
from .ground_truth import FixedMass, ground_truth_kepler
from .regularization import control_variance
from .solver import EnergyMinimizer2D, pack, unpack
from .spline import (
    SplinePath,
    control_len,
    initialize_segment,
    knot_interval,
    line_init,
)
from .visualization import plot_result, plot_sampling_comparison, plot_spline_path

__all__ = [
    "CONSTRAINT_STIFFNESS",
    "DEGREE",
    "EnergyMinimizer2D",
    "ExperimentConfig",
    "ExperimentResult",
    "FixedMass",
    "Lagrangian2D",
    "REGULARIZATION_STIFFNESS",
    "SplinePath",
    "TANGENT_INIT_SCALE",
    "TrajectoryDataset",
    "VariationalProblem",
    "control_len",
    "control_variance",
    "ground_truth_kepler",
    "initialize_segment",
    "knot_interval",
    "line_init",
    "load_dataset",
    "pack",
    "plot_result",
    "plot_sampling_comparison",
    "plot_spline_path",
    "save_result",
    "solve_experiment",
    "unpack",
]
