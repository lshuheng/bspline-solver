"""Demo: maximize enclosed area at fixed total curve length."""

import math

import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    VariationalProblem,
    load_dataset,
    plot_result,
    solve_experiment,
)


def make_isoperimetric_problem(target_length: float) -> VariationalProblem:
    u, v, ut, vt = sp.symbols("u v ut vt")
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="isoperimetric",
        lagrangian=-(u * vt - v * ut) / 2,
        constraint=speed,
        constraint_target=target_length,
    )


def main() -> None:
    dataset = load_dataset("isoperimetric")
    problem = make_isoperimetric_problem(target_length=2.0 * math.pi)
    result = solve_experiment(
        dataset,
        problem,
        ExperimentConfig(
            fix_location=[False, False, False, False],
            cyclic=True,
        ),
    )
    plot_result(result)


if __name__ == "__main__":
    main()
