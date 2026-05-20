"""B-spline geometry: knot construction, control-point initialization, and path structure."""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .config import DEGREE, TANGENT_INIT_SCALE, rng


def control_len(n_bisections: int) -> int:
    """Number of control points implied by the uniform-bisection knot scheme.

    Relationship: len(controls) = len(knots) - 1 - DEGREE.
    """
    return DEGREE + 2 ** n_bisections


def knot_interval(start: float, end: float, n_bisections: int = 4) -> np.ndarray:
    """Build a clamped uniform-bisection knot vector on [start, end]."""
    return np.concatenate(
        [
            np.full(DEGREE, start),
            np.linspace(start, end, 2 ** n_bisections + 1),
            np.full(DEGREE, end),
        ]
    )


def line_init(
    boundary_pts: np.ndarray,
    n_bisections: int = 4,
    std: float = 0.1,
) -> np.ndarray:
    """Initialize control points along the chord between two endpoints, with normal jitter.

    Args:
        boundary_pts: Pair of (x, y) endpoints.
        n_bisections: Controls the number of control points.
        std: Standard deviation of normal-direction perturbations.

    Returns:
        Array of shape (2, n_control) with control point coordinates.
    """
    p_start, p_end = np.array(boundary_pts)
    velocity = p_end - p_start

    sample = np.linspace(0, 1, control_len(n_bisections))
    points = np.array([p_start + s * velocity for s in sample])
    xs = points[:, 0]
    ys = points[:, 1]

    noise = rng.normal(0.0, std, len(xs))
    noise[0] = 0.0
    noise[-1] = 0.0
    xs = xs - velocity[1] * noise
    ys = ys + velocity[0] * noise

    return np.array([xs, ys])


def initialize_segment(
    boundary_pts: np.ndarray,
    boundary_theta: tuple[Optional[float], Optional[float]],
    n_bisections: int = 2,
    method: Callable = line_init,
) -> np.ndarray:
    """Initialize one segment's control points, clamping endpoints and tangent directions."""
    control = method(boundary_pts, n_bisections)
    h = 1.0 / (2 ** n_bisections)

    control[0, 0] = boundary_pts[0][0]
    control[0, -1] = boundary_pts[-1][0]
    control[1, 0] = boundary_pts[0][1]
    control[1, -1] = boundary_pts[-1][1]

    theta1, theta2 = boundary_theta
    if theta1 is not None:
        du1 = math.cos(theta1)
        dv1 = math.sin(theta1)
        control[0, 1] = control[0, 0] + TANGENT_INIT_SCALE * du1 * h / 3
        control[1, 1] = control[1, 0] + TANGENT_INIT_SCALE * dv1 * h / 3

    if theta2 is not None:
        du2 = math.cos(theta2)
        dv2 = math.sin(theta2)
        control[0, -2] = control[0, -1] - TANGENT_INIT_SCALE * du2 * h / 3
        control[1, -2] = control[1, -1] - TANGENT_INIT_SCALE * dv2 * h / 3

    return control


def normal_projection(theta: float, v1: float, v2: float) -> np.ndarray:
    """Project the 2D vector (v1, v2) onto the unit normal of direction theta."""
    n1 = -math.sin(theta)
    n2 = math.cos(theta)
    proj = n1 * v1 + n2 * v2
    return np.array([proj * n1, proj * n2])


class SplinePath:
    """A sequence of B-spline segments joined at waypoints.

    A path is defined by an ordered list of waypoints (vertices). Between
    consecutive waypoints there is a B-spline segment; each waypoint
    optionally carries a tangent angle that constrains the curve's slope
    at that point.

    Attributes:
        vertex: List of waypoint coordinates as tuples.
        theta: Dict mapping each vertex to its tangent angle (or None).
        edges: List of (start_vertex, end_vertex) tuples.
    """

    def __init__(
        self,
        vertex: list,
        theta: list[Optional[float]],
        n_bisections: int = 2,
    ) -> None:
        self.vertex = [tuple(v) for v in vertex]
        self.theta = dict(zip(self.vertex, theta))
        self.edges = [
            (self.vertex[i], self.vertex[i + 1]) for i in range(len(self.vertex) - 1)
        ]
        self.vertex_to_edge_head = {
            v: [e for e in self.edges if e[0] == v] for v in self.vertex
        }
        self.vertex_to_edge_tail = {
            v: [e for e in self.edges if e[-1] == v] for v in self.vertex
        }

    def initial_controls(
        self,
        n_bisections: int = 2,
    ) -> tuple[dict, np.ndarray]:
        """Build initial control-point dict and shared knot vector for all edges."""
        control_dict = {
            e: initialize_segment(e, (self.theta[e[0]], self.theta[e[1]]), n_bisections)
            for e in self.edges
        }
        knot = knot_interval(0.0, 1.0, n_bisections)
        return control_dict, knot

    def grad_mask(self, grad: dict) -> None:
        """Zero/project gradients at clamped endpoints and tangent-constrained waypoints.

        Endpoints are pinned (gradient set to zero), tangent-constrained points
        only allow movement along their tangent direction (normal component
        projected out), and at internal waypoints the gradients on adjacent
        edges are coupled to enforce C^1 continuity.
        """
        for v in self.vertex:
            e_head = self.vertex_to_edge_head[v]
            e_tail = self.vertex_to_edge_tail[v]
            theta = self.theta[v]

            if e_head:
                e = e_head[0]
                grad_array = grad[e]
                grad_array[:, 0] = 0
                if theta is not None:
                    d = grad_array[:, 1].copy()
                    grad_array[:, 1] = d - normal_projection(theta, *d)

            if e_tail:
                e = e_tail[0]
                grad_array = grad[e]
                grad_array[:, -1] = 0
                if theta is not None:
                    d = grad_array[:, -2].copy()
                    grad_array[:, -2] = d - normal_projection(theta, *d)

            if e_head and e_tail:
                e_t = e_tail[0]
                e_h = e_head[0]
                g_t = grad[e_t][:, -2].copy()
                g_h = grad[e_h][:, 1].copy()
                effective = g_t - g_h
                grad[e_t][:, -2] = effective
                grad[e_h][:, 1] = -effective
