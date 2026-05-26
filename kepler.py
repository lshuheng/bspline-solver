"""Demo: simulate kepler orbits"""

import numpy as np
import sympy as sp

from bspline_solver import run_diagnostic


class FixedMass:
    def __init__(self, center: np.array, mass: float) -> None:
        self.center = center
        self.mass = mass


def kepler_lagrangian(masses: list[FixedMass], E: float, G: float = 1.0) -> sp.Expr:
    u = sp.Symbol("u")
    v = sp.Symbol("v")
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")
    utt = sp.Symbol("utt")
    vtt = sp.Symbol("vtt")

    potential = sum(
        -G * m.mass / ((u - m.center[0]) ** 2 + (v - m.center[1]) ** 2) ** sp.Rational(1, 2)
        for m in masses
    )
    speed = (ut ** 2 + vt ** 2) ** sp.Rational(1, 2)
    return (2 * E - 2 * potential) ** sp.Rational(1, 2) * speed


def main() -> None:
    E = -1.68
    masses = [FixedMass(np.array([2.0, 0.0]), 2.0)]
    lagrangian = kepler_lagrangian(masses, E)

    run_diagnostic(
        vertices=[[1.0, 0], [2.01131479e+00, 3.12100909e-01]],
        lagrangian=lagrangian,
        reg=None,
        title="Kepler orbit",
    )


if __name__ == "__main__":
    main()
