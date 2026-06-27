"""Demo: minimize beam bending energy at fixed total length."""

import math

import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    TrajectoryDataset,
    VariationalProblem,
    plot_result,
    solve_experiment,
)


def make_beam_buckling_problem(target_length: float) -> VariationalProblem:
    ut, vt, utt, vtt = sp.symbols("ut vt utt vtt")
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="beam_buckling",
        lagrangian=sp.Rational(1, 2) * (utt**2 + vtt**2),
        constraint=speed,
        constraint_target=target_length,
    )


def main() -> None:
    dataset = TrajectoryDataset(
        name="beam_buckling",
        vertices=[
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 0.0],
        ],
    )
    problem = make_beam_buckling_problem(target_length=4.0)
    result = solve_experiment(
        dataset,
        problem,
        ExperimentConfig(
            theta=[0.0, math.pi / 4, 0.0],
            fix_angle=[True, False, True],
            fix_location=[True, True, True],
        ),
    )
    plot_result(result)


if __name__ == "__main__":
    main()
