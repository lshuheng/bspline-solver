"""Demo: simulate a hanging chain by minimizing potential energy subject to a fixed-length integral constraint."""

import sympy as sp

from bspline_solver import run_diagnostic


def main() -> None:
    v = sp.Symbol("v")
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")

    vertices = [[0, 0], [0.5, 1], [2, 2]]
    target_length = 3.0
    n_edges = len(vertices) - 1

    lagrangian = v * (ut ** 2 + vt ** 2) ** sp.Rational(1, 2)
    constraint = (ut ** 2 + vt ** 2) ** sp.Rational(1, 2) - target_length / n_edges

    run_diagnostic(
        vertices=vertices,
        lagrangian=lagrangian,
        constraint=constraint,
        fix_location=[True, False, True],
        title="Hanging chain",
    )


if __name__ == "__main__":
    main()
