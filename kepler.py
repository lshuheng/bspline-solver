"""Demo: simulate kepler orbits"""

import math

import sympy as sp
import numpy as np

from bspline_solver import EnergyMinimizer2D, SplinePath, plot_spline_path
import matplotlib.pyplot as plt

class FixedMass:
    def __init__(self, center: np.array, mass:float)-> None:
        self.center = center
        self.mass = mass

def kepler_lagrangian(masses: list[FixedMass], E:float, G: float = 1.0) -> sp.Expr:
    # Symbolic variables matching the Lagrangian2D convention.
    t = sp.Symbol
    u = sp.Symbol("u")
    v = sp.Symbol("v")
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")
    utt = sp.Symbol("utt")
    vtt = sp.Symbol("vtt")

    potential = sum(-G * m.mass / ((u - m.center[0]) ** 2 + (v - m.center[1]) ** 2) ** sp.Rational(1, 2) for m in masses)
    speed = (ut ** 2 + vt ** 2) ** sp.Rational(1, 2)
    L = (2*E - 2*potential)**sp.Rational(1,2) * speed

    return  L


def main() -> None:


    # Waypoints: initial tangent angles are required for initialization;
    # fix_angle=False lets them optimize freely.
    vertices = [[1.0, 0], [2.01131479e+00, 3.12100909e-01]]
    thetas = [math.pi/4, 0]
    path = SplinePath(vertices, thetas, fix_angle=[False, False])
    control, knot = path.initial_controls()

    E = -1.68
    masses = [FixedMass(np.array([2.0, 0.0]), 2.0)]
    lagrangian = kepler_lagrangian(masses, E)
   
    solver = EnergyMinimizer2D(path, control, knot, lagrangian)

    plot_spline_path(list(control.values()), knot)
    plt.title("Initial path")

    solver.minimize()
    plot_spline_path(list(solver.control.values()), knot)
    plt.title("Optimized path")
    plt.show()


if __name__ == "__main__":
    main()
