"""Demo: reconstruct a manually supplied two-center Kepler trajectory."""

import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    TrajectoryDataset,
    VariationalProblem,
    load_dataset,
    plot_result,
    solve_experiment,
)


def make_kepler_problem(dataset: TrajectoryDataset) -> VariationalProblem:
    """Create the Jacobi-Maupertuis objective for fixed point masses."""
    energy = dataset.metadata["energy"]
    masses = dataset.metadata["masses"]
    gravitational_constant = dataset.metadata["gravitational_constant"]

    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = sum(
        -gravitational_constant
        * fixed_mass["mass"]
        / (
            (u - fixed_mass["center"][0]) ** 2
            + (v - fixed_mass["center"][1]) ** 2
        )
        ** sp.Rational(1, 2)
        for fixed_mass in masses
    )
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="kepler",
        lagrangian=(
            (2 * energy - 2 * potential) ** sp.Rational(1, 2) * speed
        ),
        title="Kepler orbit 1",
        metadata={
            "energy": energy,
            "gravitational_constant": gravitational_constant,
            "masses": masses,
        },
    )


def main() -> None:
    dataset = load_dataset("kepler_orbit_1")
    problem = make_kepler_problem(dataset)
    result = solve_experiment(
        dataset,
        problem,
        ExperimentConfig(geometric_init=True),
    )
    plot_result(result)


if __name__ == "__main__":
    main()
