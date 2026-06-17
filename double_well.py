"""Demo: reconstruct a generated double-well potential trajectory."""

from __future__ import annotations

from bspline_solver import (
    ExperimentConfig,
    ground_truth,
    make_double_well_ground_truth_problem,
    make_double_well_problem,
    plot_sampling_comparison,
    solve_sampling_experiments,
)


def main() -> None:
    epsilon = 0.20
    omega = 0.75
    initial_position = (-0.85, 0.2)
    initial_velocity = (0.5, 0.85)
    t_span = (0.0, 20.0)
    n_vertices = [15, 20, 30]

    ground_truth_problem = make_double_well_ground_truth_problem(
        epsilon=epsilon,
        omega=omega,
    )
    datasets = ground_truth(
        problem=ground_truth_problem,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        t_span=t_span,
        n_vertices=n_vertices,
        n_dense=2000,
        name="generated_double_well_trajectory",
    )
    results = solve_sampling_experiments(
        datasets,
        make_double_well_problem,
        ExperimentConfig(geometric_init=False),
    )
    plot_sampling_comparison(results)


if __name__ == "__main__":
    main()
