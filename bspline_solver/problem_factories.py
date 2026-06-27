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
    """Create the fixed-energy geometric objective for the double-well potential."""
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


def make_henon_heiles_ground_truth_problem(
    lambda_value: float = 0.10,
) -> VariationalProblem:
    """Create the unit-mass mechanical Lagrangian for the Henon-Heiles IVP."""
    lambda_value = float(lambda_value)
    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = _henon_heiles_potential(u, v, lambda_value)
    kinetic_energy = sp.Rational(1, 2) * (ut**2 + vt**2)
    return VariationalProblem(
        name="henon_heiles",
        lagrangian=kinetic_energy - potential,
        metadata={"lambda_value": lambda_value},
    )


def make_henon_heiles_problem(dataset: TrajectoryDataset) -> VariationalProblem:
    """Create the fixed-energy geometric objective for the Henon-Heiles potential."""
    energy = dataset.metadata["energy"]
    lambda_value = dataset.metadata["lambda_value"]

    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = _henon_heiles_potential(u, v, lambda_value)
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="henon_heiles",
        lagrangian=(
            (2 * energy - 2 * potential) ** sp.Rational(1, 2) * speed
        ),
    )


def make_polynomial_channel_ground_truth_problem(
    kappa: float = 1.2,
    alpha: float = 0.18,
    mu: float = -0.18,
) -> VariationalProblem:
    """Create the unit-mass mechanical Lagrangian for channel scattering."""
    kappa_value = float(kappa)
    alpha_value = float(alpha)
    mu_value = float(mu)
    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = _polynomial_channel_potential(
        u,
        v,
        kappa_value,
        alpha_value,
        mu_value,
    )
    kinetic_energy = sp.Rational(1, 2) * (ut**2 + vt**2)
    return VariationalProblem(
        name="polynomial_channel_scattering",
        lagrangian=kinetic_energy - potential,
        metadata={
            "kappa": kappa_value,
            "alpha": alpha_value,
            "mu": mu_value,
        },
    )


def make_polynomial_channel_problem(dataset: TrajectoryDataset) -> VariationalProblem:
    """Create the fixed-energy geometric objective for channel scattering."""
    energy = dataset.metadata["energy"]
    kappa = dataset.metadata["kappa"]
    alpha = dataset.metadata["alpha"]
    mu = dataset.metadata["mu"]

    u, v, ut, vt = sp.symbols("u v ut vt")
    potential = _polynomial_channel_potential(u, v, kappa, alpha, mu)
    speed = (ut**2 + vt**2) ** sp.Rational(1, 2)
    return VariationalProblem(
        name="polynomial_channel_scattering",
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


def _henon_heiles_potential(
    u: sp.Symbol,
    v: sp.Symbol,
    lambda_value: float,
) -> sp.Expr:
    return (
        sp.Rational(1, 2) * (u**2 + v**2)
        + lambda_value * (u**2 * v - sp.Rational(1, 3) * v**3)
    )


def _polynomial_channel_potential(
    u: sp.Symbol,
    v: sp.Symbol,
    kappa: float,
    alpha: float,
    mu: float,
) -> sp.Expr:
    return sp.Rational(1, 2) * kappa * (v - alpha * u**2) ** 2 + mu * u
