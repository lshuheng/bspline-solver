"""Demo: maximize area enclosed by a curve subject to a fixed-length integral constraint."""

import math

import sympy as sp

from bspline_solver import EnergyMinimizer2D, SplinePath, plot_spline_path
import matplotlib.pyplot as plt


def main() -> None:
    # Symbolic variables matching the Lagrangian2D convention.
    t = sp.Symbol("t")
    u = sp.Symbol("u")
    v = sp.Symbol("v")
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")
    utt = sp.Symbol("utt")
    vtt = sp.Symbol("vtt")
    

    # Waypoints: initial tangent angles are required for initialization;
    # fix_angle=False lets them optimize freely.
    vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
    thetas = [math.pi/4, math.pi/4, math.pi/4, -math.pi/4]
    path = SplinePath(vertices, thetas, fix_angle=[False, False, False, False], fix_location=[False, False, False, False], cyclic=True)
    control, knot = path.initial_controls()

    target_length = 2.0 * math.pi / 4.0
    lagrangian = -(u * vt - v * ut)/2
    constraint = (ut ** 2 + vt ** 2) ** sp.Rational(1, 2) - target_length

    solver = EnergyMinimizer2D(path, control, knot, lagrangian, constraint)

    plot_spline_path(list(control.values()), knot)
    plt.title("Initial path")

    solver.minimize()
    plot_spline_path(list(solver.control.values()), knot)
    plt.title("Optimized path")
    plt.show()


if __name__ == "__main__":
    main()
