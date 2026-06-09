"""Demo: reconstruct a manually supplied two-center Kepler trajectory."""

import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    VariationalProblem,
    load_dataset,
    plot_result,
    solve_experiment,
)


def make_kepler_problem(
    energy: float,
    masses: list[tuple[float, float, float]],
    gravitational_constant: float = 1.0,
) -> VariationalProblem:
    """Create the Jacobi-Maupertuis objective for fixed point masses."""
    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = sum(
        -gravitational_constant
        * mass
        / ((u - x) ** 2 + (v - y) ** 2) ** sp.Rational(1, 2)
        for x, y, mass in masses
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
            "masses": [
                {"center": [x, y], "mass": mass}
                for x, y, mass in masses
            ],
        },
    )


def main() -> None:
    dataset = load_dataset("kepler_orbit_1")
    problem = make_kepler_problem(
        energy=-0.32766314,
        masses=[
            (-1.2, 0.0, 1.0),
            (1.2, 0.0, 1.0),
        ],
    )
    result = solve_experiment(
        dataset,
        problem,
        ExperimentConfig(geometric_init=True),
    )
    plot_result(result)


if __name__ == "__main__":
    main()
