"""Variational B-spline ODE solver for boundary value problems."""

from importlib import import_module

from .datasets import TrajectoryDataset, load_dataset
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    VariationalProblem,
    solve_sampling_experiments,
    solve_experiment,
)
from .ground_truth import (
    FixedMass,
    ground_truth,
    ground_truth_kepler,
    make_kepler_ground_truth_problem,
)
from .problem_factories import (
    make_double_well_ground_truth_problem,
    make_double_well_problem,
    make_henon_heiles_ground_truth_problem,
    make_henon_heiles_problem,
    make_kepler_problem,
    make_polynomial_channel_ground_truth_problem,
    make_polynomial_channel_problem,
)

_VISUALIZATION_EXPORTS = {
    "plot_linear_path",
    "plot_result",
    "plot_sampling_comparison",
    "plot_spline_path",
}


def __getattr__(name: str):
    if name in _VISUALIZATION_EXPORTS:
        value = getattr(import_module(".visualization", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "FixedMass",
    "TrajectoryDataset",
    "VariationalProblem",
    "ground_truth",
    "ground_truth_kepler",
    "load_dataset",
    "make_kepler_ground_truth_problem",
    "make_double_well_ground_truth_problem",
    "make_double_well_problem",
    "make_henon_heiles_ground_truth_problem",
    "make_henon_heiles_problem",
    "make_kepler_problem",
    "make_polynomial_channel_ground_truth_problem",
    "make_polynomial_channel_problem",
    "plot_linear_path",
    "plot_result",
    "plot_sampling_comparison",
    "plot_spline_path",
    "solve_sampling_experiments",
    "solve_experiment",
]
