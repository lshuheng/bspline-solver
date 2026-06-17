"""Reusable variational problem factories for generated-trajectory demos."""

from __future__ import annotations

import sympy as sp

from .datasets import TrajectoryDataset
from .experiment import VariationalProblem


def make_kepler_problem(dataset: TrajectoryDataset) -> VariationalProblem:
    """Create the Jacobi-Maupertuis objective for fixed point masses."""
    energy = dataset.metadata["energy"]
    masses = dataset.metadata["masses"]
    gravitational_constant = dataset.metadata["gravitational_constant"]

    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = sum(
        -gravitational_constant
        * fixed_mass["mass"]
        / (
            (u - fixed_mass["center"][0]) ** 2
            + (v - fixed_mass["center"][1]) ** 2
        )
        ** sp.Rational(1, 2)
        for fixed_mass in masses
    )
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="kepler",
        lagrangian=(
            (2 * energy - 2 * potential) ** sp.Rational(1, 2) * speed
        ),
    )


def make_double_well_ground_truth_problem(
    epsilon: float = 0.20,
    omega: float = 0.75,
) -> VariationalProblem:
    """Create the unit-mass mechanical Lagrangian for the double-well IVP."""
    epsilon_value = float(epsilon)
    omega_value = float(omega)
    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = _double_well_potential(u, v, epsilon_value, omega_value)
    kinetic_energy = sp.Rational(1, 2) * (ut**2 + vt**2)
    return VariationalProblem(
        name="double_well",
        lagrangian=kinetic_energy - potential,
        metadata={
            "epsilon": epsilon_value,
            "omega": omega_value,
        },
    )


def make_double_well_problem(dataset: TrajectoryDataset) -> VariationalProblem:
    """Create the Jacobi-Maupertuis objective for the double-well potential."""
    energy = dataset.metadata["energy"]
    epsilon = dataset.metadata["epsilon"]
    omega = dataset.metadata["omega"]

    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = _double_well_potential(u, v, epsilon, omega)
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="double_well",
        lagrangian=(
            (2 * energy - 2 * potential) ** sp.Rational(1, 2) * speed
        ),
    )


def _double_well_potential(
    u: sp.Symbol,
    v: sp.Symbol,
    epsilon: float,
    omega: float,
) -> sp.Expr:
    return (
        sp.Rational(1, 4) * (u**2 - 1) ** 2
        + sp.Rational(1, 2) * omega**2 * v**2
        + epsilon * u * v**2
    )
