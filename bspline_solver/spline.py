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
    std: float = 0.0,
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
    consecutive waypoints there is a B-spline segment; each waypoint carries
    a tangent angle used to initialize control-point positions, and independent
    flags controlling whether its location and angle are held fixed during
    optimization.

    Attributes:
        vertex: List of waypoint coordinates as tuples.
        theta: Dict mapping each vertex to its tangent angle.
        fix_location: Dict mapping each vertex to whether its location is fixed.
        fix_angle: Dict mapping each vertex to whether its tangent angle is fixed.
        cyclic: Whether the path is closed (last vertex connected back to first).
        edges: List of (start_vertex, end_vertex) tuples.
    """

    def __init__(
        self,
        vertex: list,
        theta: list[float] | None = None,
        n_bisections: int = 2,
        fix_location: list[bool] | None = None,
        fix_angle: list[bool] | None = None,
        cyclic: bool = False,
    ) -> None:
        self.vertex = [tuple(v) for v in vertex]
        n = len(self.vertex)

        if theta is None:
            pts = [np.array(v) for v in self.vertex]
            angles = []
            for i in range(n):
                if cyclic:
                    prev = pts[(i - 1) % n]
                    nxt = pts[(i + 1) % n]
                    d = nxt - prev
                elif i == 0:
                    d = pts[1] - pts[0]
                elif i == n - 1:
                    d = pts[-1] - pts[-2]
                else:
                    d = pts[i + 1] - pts[i - 1]
                angles.append(math.atan2(d[1], d[0]))
            theta = angles

        self.theta = dict(zip(self.vertex, theta))
        self.fix_location = dict(zip(self.vertex, fix_location if fix_location is not None else [True] * n))
        self.fix_angle = dict(zip(self.vertex, fix_angle if fix_angle is not None else [False] * n))
        self.cyclic = cyclic
        self.edges = [
            (self.vertex[i], self.vertex[i + 1]) for i in range(len(self.vertex) - 1)
        ]
        if cyclic:
            self.edges.append((self.vertex[-1], self.vertex[0]))
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
        """Zero/project gradients according to per-waypoint fix_location and fix_angle flags.

        Fixed-location waypoints have their endpoint gradient zeroed. Fixed-angle
        waypoints have the normal component of their adjacent interior control-point
        gradient projected out. At internal waypoints, free-location endpoints are
        coupled so both adjacent edges move together, and C^1 continuity is always
        enforced by coupling the second/second-to-last control-point gradients.
        """
        for v in self.vertex:
            e_head = self.vertex_to_edge_head[v]
            e_tail = self.vertex_to_edge_tail[v]
            theta = self.theta[v]
            fix_loc = self.fix_location[v]
            fix_ang = self.fix_angle[v]

            if e_head:
                e = e_head[0]
                grad_array = grad[e]
                if fix_loc:
                    grad_array[:, 0] = 0
                if fix_ang:
                    d = grad_array[:, 1].copy()
                    grad_array[:, 1] = d - normal_projection(theta, *d)

            if e_tail:
                e = e_tail[0]
                grad_array = grad[e]
                if fix_loc:
                    grad_array[:, -1] = 0
                if fix_ang:
                    d = grad_array[:, -2].copy()
                    grad_array[:, -2] = d - normal_projection(theta, *d)

            if e_head and e_tail:
                e_t = e_tail[0]
                e_h = e_head[0]
                if fix_loc:
                    # Waypoint is fixed; anti-symmetric coupling maintains a + b = 2w = const.
                    g_t = grad[e_t][:, -2].copy()
                    g_h = grad[e_h][:, 1].copy()
                    effective = g_t - g_h
                    grad[e_t][:, -2] = effective
                    grad[e_h][:, 1] = -effective
                else:
                    # Waypoint is free. Project (a, b, w) jointly onto C^1 surface a + b = 2w.
                    # grad f = (1, 1, -2), |grad f|^2 = 6; lambda = (g_a + g_b - 2*g_w) / 6.
                    g_a = grad[e_t][:, -2].copy()
                    g_b = grad[e_h][:, 1].copy()
                    g_w = grad[e_t][:, -1].copy() + grad[e_h][:, 0].copy()
                    lam = (g_a + g_b - 2 * g_w) / 6
                    grad[e_t][:, -2] = g_a - lam
                    grad[e_h][:, 1] = g_b - lam
                    g_w_eff = g_w + 2 * lam
                    grad[e_t][:, -1] = g_w_eff
                    grad[e_h][:, 0] = g_w_eff
