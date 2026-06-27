"""Demo: reconstruct a generated polynomial channel-scattering trajectory."""

from __future__ import annotations

from bspline_solver import (
    ExperimentConfig,
    ground_truth,
    make_polynomial_channel_ground_truth_problem,
    make_polynomial_channel_problem,
    plot_sampling_comparison,
    solve_sampling_experiments,
)


def main() -> None:
    kappa = 0.5
    alpha = 0.2
    mu = -0.1
    initial_position = (-2.6, 1.35)
    initial_velocity = (0.55, -0.20)
    t_span = (0.0, 10.0)
    n_vertices = [4, 5, 6]

    ground_truth_problem = make_polynomial_channel_ground_truth_problem(
        kappa=kappa,
        alpha=alpha,
        mu=mu,
    )
    datasets = ground_truth(
        problem=ground_truth_problem,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        t_span=t_span,
        n_vertices=n_vertices,
        n_dense=2000,
        name="generated_polynomial_channel_scattering_trajectory",
    )
    results = solve_sampling_experiments(
        datasets,
        make_polynomial_channel_problem,
        ExperimentConfig(geometric_init=False),
    )
    plot_sampling_comparison(results)


if __name__ == "__main__":
    main()
