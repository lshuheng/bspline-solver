"""Demo: simulate kepler orbits"""

import numpy as np
import sympy as sp

from bspline_solver import run_diagnostic
from bspline_solver import tangent_speed


class FixedMass:
    def __init__(self, center: np.array, mass: float) -> None:
        self.center = center
        self.mass = mass


def kepler_lagrangian(masses: list[FixedMass], E: float, G: float = 1.0) -> sp.Expr:
    u = sp.Symbol("u")
    v = sp.Symbol("v")
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")

    potential = sum(
        -G * m.mass / ((u - m.center[0]) ** 2 + (v - m.center[1]) ** 2) ** sp.Rational(1, 2)
        for m in masses
    )
    speed = (ut ** 2 + vt ** 2) ** sp.Rational(1, 2)
    return (2 * E - 2 * potential) ** sp.Rational(1, 2) * speed


def main() -> None:
    E = -2.18
    masses = [ FixedMass(center=np.array([2.0, 0.0]), mass=2.0), FixedMass(center=np.array([0.0, 0.0]), mass=0.5)]
    lagrangian = kepler_lagrangian(masses, E)
    vertices = [[np.float64(1.0), np.float64(0.0)],
 [np.float64(1.0855767098235112), np.float64(0.25415686994558795)],
 [np.float64(1.3703294893541382), np.float64(0.4276778888233724)],
 [np.float64(1.9637304071061117), np.float64(0.30414569797943036)],
 [np.float64(1.4987401257060948), np.float64(-0.22623076897768957)],
 [np.float64(1.1197410765293472), np.float64(0.02110573830491791)],
 [np.float64(1.003194316828438), np.float64(0.2737737664326215)],
 [np.float64(1.0436767373102793), np.float64(0.4600360277931118)],
 [np.float64(1.2264900948826924), np.float64(0.5423641906439897)],
 [np.float64(1.5976847175438265), np.float64(0.4542185203470346)]]

    run_diagnostic(
        vertices=vertices,
        lagrangian=lagrangian,
        title="Kepler orbit",
        reg = None
        
    )


if __name__ == "__main__":
    main()
