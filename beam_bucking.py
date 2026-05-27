"""Demo: simulate a buckling beam of fixed length"""

import math

import sympy as sp

from bspline_solver import run_diagnostic

def beam_buckling_lagrangian(target_length: float, n_edges: int) -> sp.Expr:
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")
    utt = sp.Symbol("utt")
    vtt = sp.Symbol("vtt")

    return sp.Rational(1, 2) * (utt ** 2 + vtt ** 2), (ut ** 2 + vt ** 2) ** sp.Rational(1, 2) - target_length / n_edges


def main() -> None:
    vertices = [[0, 0], [0.5, 1], [1, 0]]
    target_length = 3
    n_edges = len(vertices) - 1
    lagrangian, constraint = beam_buckling_lagrangian(target_length, n_edges)
    
    run_diagnostic(
        vertices=vertices,
        lagrangian=lagrangian,
        constraint=constraint,
        theta=[0, math.pi / 4, 0],
        fix_angle=[True, False, True],
        fix_location=[True, False, True],
        title="Beam buckling",
    )


if __name__ == "__main__":
    main()
