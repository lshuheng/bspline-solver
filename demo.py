"""Demo: minimize curve length subject to a fixed-length integral constraint."""

import sympy as sp

from bspline_solver import EnergyMinimizer2D, SplinePath, plot_spline_path
import matplotlib.pyplot as plt


def main() -> None:
    # Symbolic variables matching the Lagrangian2D convention.
    ut = sp.Symbol("ut")
    vt = sp.Symbol("vt")
    v = sp.Symbol("v")

    # Waypoints with tangent angles (None = no tangent constraint).
    vertices = [[0, 0], [1, 1], [2, 2]]
    thetas = [None, None, None]
    path = SplinePath(vertices, thetas)
    control, knot = path.initial_controls()

    # Lagrangian: integrand v * sqrt(ut^2 + vt^2). Constraint: arc length = 2.
    target_length = 2.0
    lagrangian = v * (ut ** 2 + vt ** 2) ** sp.Rational(1, 2)
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
