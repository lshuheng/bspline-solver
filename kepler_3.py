"""Demo: reconstruct a generated two-center Kepler trajectory."""

from bspline_solver import (
    ExperimentConfig,
    FixedMass,
    ground_truth_kepler,
    make_kepler_problem,
    plot_sampling_comparison,
    solve_experiment,
)


def main() -> None:
    masses = [
        FixedMass(center=[-1.1, 0.0], mass=1.0),
        FixedMass(center=[1.1, 0.0], mass=0.65),
    ]
    gravitational_constant = 1.0
    initial_position = [0, 2.8]
    initial_velocity = [0.72, 0.4]
    t_span = (0.0, 80)
    n_vertices = [10, 15]

    datasets = ground_truth_kepler(
        masses=masses,
        gravitational_constant=gravitational_constant,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        t_span=t_span,
        n_vertices=n_vertices,
        n_dense=2000,
        name="generated_kepler_orbit",
    )
    results = []
    for dataset in datasets:
        problem = make_kepler_problem(dataset)
        result = solve_experiment(
            dataset,
            problem,
            ExperimentConfig(geometric_init=False),
        )
        results.append(result)
    plot_sampling_comparison(results)


if __name__ == "__main__":
    main()
