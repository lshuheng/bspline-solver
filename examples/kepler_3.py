"""Demo: reconstruct a generated two-center Kepler trajectory."""

from bspline_solver import (
    ExperimentConfig,
    FixedMass,
    ground_truth,
    make_kepler_ground_truth_problem,
    make_kepler_problem,
    plot_sampling_comparison,
    solve_sampling_experiments,
)

from _figures import save_and_show


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
    fig, _ = plot_sampling_comparison(results, show=False)
    save_and_show(fig, "kepler_3.png")


if __name__ == "__main__":
    main()
