"""Demo: minimize hanging-chain potential energy at fixed total length."""

import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    VariationalProblem,
    load_dataset,
    plot_result,
    solve_experiment,
)


def make_hanging_chain_problem(target_length: float) -> VariationalProblem:
    v, ut, vt = sp.symbols("v ut vt")
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="hanging_chain",
        lagrangian=v * speed,
        constraint=speed,
        constraint_target=target_length,
    )


def main() -> None:
    dataset = load_dataset("hanging_chain")
    problem = make_hanging_chain_problem(target_length=3.0)
    result = solve_experiment(
        dataset,
        problem,
        ExperimentConfig(fix_location=[True, False, True]),
    )
    plot_result(result)


if __name__ == "__main__":
    main()
