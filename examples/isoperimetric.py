"""Demo: maximize enclosed area at fixed total curve length."""

import math

import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    TrajectoryDataset,
    VariationalProblem,
    plot_result,
    solve_experiment,
)

from _figures import save_and_show


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
    dataset = TrajectoryDataset(
        name="isoperimetric",
        vertices=[
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],

        ],
    )
    problem = make_isoperimetric_problem(target_length=2.0 * math.pi)
    result = solve_experiment(
        dataset,
        problem,
        ExperimentConfig(
            fix_location=[True, True, True],
            cyclic=True,
        ),
    )
    fig, _ = plot_result(result, show=False)
    save_and_show(fig, "isoperimetric.png")


if __name__ == "__main__":
    main()
