"""Demo: reconstruct a generated two-center Kepler trajectory."""

import numpy as np
import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    FixedMass,
    TrajectoryDataset,
    VariationalProblem,
    ground_truth_kepler,
    plot_sampling_comparison,
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
    datasets = ground_truth_kepler(
        masses=[
            FixedMass(center=np.array([-1.1, -0.45]), mass=0.90),
            FixedMass(center=np.array([1.0, -0.35]), mass=1.1),
            FixedMass(center=np.array([0.0, 1.05]), mass=0.75)
        ],
        gravitational_constant=1.0,
        initial_position=[-1.85, 0.35],
        initial_velocity=[0.95, 0.5],
        t_span=(0.0, 15),
        n_vertices=[8,15],
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
