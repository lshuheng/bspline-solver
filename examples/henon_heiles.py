"""Demo: reconstruct a generated Henon-Heiles trajectory."""

from __future__ import annotations

from bspline_solver import (
    ExperimentConfig,
    ground_truth,
    make_henon_heiles_ground_truth_problem,
    make_henon_heiles_problem,
    plot_sampling_comparison,
    solve_sampling_experiments,
)

from _figures import save_and_show


def main() -> None:
    lambda_value = 0.45
    initial_position = (0.0, 0.35)
    initial_velocity = (0.82, 0.0)
    t_span = (0.0, 30.0)
    n_vertices = [30, 40]

    ground_truth_problem = make_henon_heiles_ground_truth_problem(
        lambda_value=lambda_value,
    )
    datasets = ground_truth(
        problem=ground_truth_problem,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        t_span=t_span,
        n_vertices=n_vertices,
        n_dense=2000,
        name="generated_henon_heiles_trajectory",
    )
    results = solve_sampling_experiments(
        datasets,
        make_henon_heiles_problem,
        ExperimentConfig(geometric_init=False),
    )
    fig, _ = plot_sampling_comparison(results, show=False)
    save_and_show(fig, "henon_heiles.png")


if __name__ == "__main__":
    main()
