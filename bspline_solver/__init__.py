"""Variational B-spline ODE solver for boundary value problems."""

from .diagnostic import run_diagnostic
from .config import (
    CONSTRAINT_STIFFNESS,
    DEGREE,
    REGULARIZATION_STIFFNESS,
    TANGENT_INIT_SCALE,
)
from .lagrangian import Lagrangian2D
from .regularization import control_variance, tangent_speed
from .solver import EnergyMinimizer2D, pack, unpack
from .spline import (
    SplinePath,
    control_len,
    initialize_segment,
    knot_interval,
    line_init,
)
from .visualization import plot_spline_path

__all__ = [
    "CONSTRAINT_STIFFNESS",
    "DEGREE",
    "EnergyMinimizer2D",
    "Lagrangian2D",
    "REGULARIZATION_STIFFNESS",
    "SplinePath",
    "TANGENT_INIT_SCALE",
    "control_len",
    "control_variance",
    "tangent_speed",
    "initialize_segment",
    "knot_interval",
    "line_init",
    "pack",
    "plot_spline_path",
    "run_diagnostic",
    "unpack",
]
