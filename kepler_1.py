"""Demo: reconstruct a generated two-center Kepler trajectory."""

from bspline_solver import (
    ExperimentConfig,
    FixedMass,
    ground_truth_kepler,
    make_kepler_problem,
    plot_sampling_comparison,
    solve_sampling_experiments,
)


def main() -> None:
    masses = [
        FixedMass(center=[-1.1, -0.6], mass=0.7),
        FixedMass(center=[1.1, -0.5], mass=0.85),
        FixedMass(center=[0.0, 1.05], mass=0.55),
    ]
    gravitational_constant = 1.0
    initial_position = [3.37, 0.15]
    initial_velocity = [-0.12, 0.78]
    t_span = (0.0, 40.0)
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
    results = solve_sampling_experiments(
        datasets,
        make_kepler_problem,
        ExperimentConfig(geometric_init=False),
    )
    plot_sampling_comparison(results)


if __name__ == "__main__":
    main()
