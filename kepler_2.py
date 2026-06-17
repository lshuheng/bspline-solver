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
        FixedMass(center=[-1.1, -0.45], mass=0.90),
        FixedMass(center=[1.0, -0.35], mass=1.1),
        FixedMass(center=[0.0, 1.05], mass=0.75),
    ]
    gravitational_constant = 1.0
    initial_position = [-1.85, 0.35]
    initial_velocity = [0.95, 0.5]
    t_span = (0.0, 15)
    n_vertices = [8, 15]

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
