"""Demo: reconstruct a generated three-center Kepler trajectory."""

from bspline_solver import (
    ExperimentConfig,
    FixedMass,
    ground_truth,
    make_kepler_ground_truth_problem,
    make_kepler_problem,
    plot_sampling_comparison,
    solve_sampling_experiments,
)


def main() -> None:
    masses = [
        FixedMass(center=[-1.1, -0.45], mass=0.90),
        FixedMass(center=[1.0, -0.35], mass=1.1),
        FixedMass(center=[0.0, 1.05], mass=0.75),
    ]
    gravitational_constant = 1.0
    initial_position = [-1.85, 0.35]
    initial_velocity = [0.95, 0.5]
    t_span = (0.0, 15)
    n_vertices = [8, 15]

    ground_truth_problem = make_kepler_ground_truth_problem(
        masses=masses,
        gravitational_constant=gravitational_constant,
    )
    datasets = ground_truth(
        problem=ground_truth_problem,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        t_span=t_span,
        n_vertices=n_vertices,
        n_dense=2000,
        name="generated_kepler_orbit",
    )
    results = solve_sampling_experiments(
        datasets,
        make_kepler_problem,
        ExperimentConfig(geometric_init=False),
    )
    plot_sampling_comparison(results)


if __name__ == "__main__":
    main()
