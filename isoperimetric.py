"""Demo: maximize area enclosed by a curve subject to a fixed-length integral constraint."""

import math

import sympy as sp

from bspline_solver import run_diagnostic

def isoperimetric_lagrangian(target_length: float, n_edges: int) -> sp.Expr:
    u = sp.Symbol("u")
    v = sp.Symbol("v")
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")

    lagrangian = -(u * vt - v * ut) / 2
    constraint = (ut ** 2 + vt ** 2) ** sp.Rational(1, 2) - target_length / n_edges
    return lagrangian, constraint


def main() -> None:
    vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
    target_length = 2.0 * math.pi
    n_edges = len(vertices)  
    lagrangian, constraint = isoperimetric_lagrangian(target_length, n_edges)

    run_diagnostic(
        vertices=vertices,
        lagrangian=lagrangian,
        constraint=constraint,
        fix_location=[False, False, False, False],
        cyclic=True,
        title="Isoperimetric",
    )


if __name__ == "__main__":
    main()
