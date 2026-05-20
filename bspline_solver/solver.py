"""Variational B-spline energy minimizer with augmented-Lagrangian constraints."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import sympy as sp
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import roots_legendre

from .config import CONSTRAINT_STIFFNESS, DEGREE, REGULARIZATION_STIFFNESS
from .lagrangian import Lagrangian2D
from .regularization import control_variance
from .spline import SplinePath


def unpack(D: dict) -> np.ndarray:
    """Flatten a dict of control arrays into a single 1D vector for the optimizer."""
    return np.concatenate([D[k].ravel() for k in D])


def pack(arr: np.ndarray, keys: list) -> dict:
    """Inverse of unpack: rebuild the dict of (2, n) control arrays."""
    n = len(arr) // (2 * len(keys))
    return {k: arr[i * 2 * n : (i + 1) * 2 * n].reshape(2, n) for i, k in enumerate(keys)}


class EnergyMinimizer2D:
    """Augmented-Lagrangian solver for variational B-spline boundary-value problems.

    Given a SplinePath skeleton, control-point initialization, a knot vector,
    a Lagrangian, and an optional integral constraint, this class minimizes
        ∫ L(t, u, u', u'', v, v', v'') dt + reg(c)
    subject to
        ∫ G(t, u, u', u'', v, v', v'') dt = 0
    using L-BFGS-B inner loops and an augmented-Lagrangian outer loop.
    """

    def __init__(
        self,
        skeleton: SplinePath,
        init_control: dict,
        init_knot: np.ndarray,
        lagrangian: sp.Expr,
        constraint: Optional[sp.Expr] = None,
        reg: Optional[Callable] = control_variance,
        n_quad: int = 5,
    ) -> None:
        self.spline = skeleton
        self.lagrangian = Lagrangian2D(lagrangian)
        self.constraint = Lagrangian2D(constraint) if constraint is not None else None
        self.constraint_multiplier = 0.0
        self.constraint_stiffness = CONSTRAINT_STIFFNESS
        self.control = init_control
        self.knot = init_knot
        self.n_quad = n_quad
        (
            self.quad_pts,
            self.quad_wts,
            self.B,
            self.B_1,
            self.B_2,
            self.B_3,
        ) = self._compute_quadrature(self.n_quad)
        self.reg = reg
        self.reg_stiffness = REGULARIZATION_STIFFNESS

    def _compute_quadrature(self, n_quad: int):
        """Precompute Gauss-Legendre quadrature nodes/weights and basis evaluations."""
        knots = np.unique(self.knot)
        pts, wts = roots_legendre(n_quad)

        all_t = []
        all_w = []
        for i in range(len(knots) - 1):
            t_a = knots[i]
            t_b = knots[i + 1]
            scale = (t_b - t_a) / 2
            center = (t_b + t_a) / 2
            all_t.append(pts * scale + center)
            all_w.append(wts * scale)

        quad_pts = np.concatenate(all_t)
        quad_wts = np.concatenate(all_w)

        B, B_1, B_2, B_3 = [], [], [], []
        n_basis = len(self.knot) - 1 - DEGREE
        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            basis = BSpline.construct_fast(self.knot, coeffs, DEGREE, extrapolate=False)
            B.append(basis(quad_pts))
            B_1.append(basis.derivative(nu=1)(quad_pts))
            B_2.append(basis.derivative(nu=2)(quad_pts))
            B_3.append(basis.derivative(nu=3)(quad_pts))

        return (
            quad_pts,
            quad_wts,
            np.asarray(B),
            np.asarray(B_1),
            np.asarray(B_2),
            np.asarray(B_3),
        )

    def _evaluate_functional(self, control: np.ndarray, lag: Lagrangian2D):
        """Compute the integral and its control-point gradient for one Lagrangian."""
        c = np.asarray(control, dtype=float).reshape(2, -1)
        c1, c2 = c
        t = self.quad_pts
        u = c1 @ self.B
        ut = c1 @ self.B_1
        utt = c1 @ self.B_2
        v = c2 @ self.B
        vt = c2 @ self.B_1
        vtt = c2 @ self.B_2

        L_quad = lag.L_func(t, u, ut, utt, v, vt, vtt)
        L_total = np.dot(L_quad, self.quad_wts)

        u_grad_quad = (
            lag.Lu_func(t, u, ut, utt, v, vt, vtt) * self.B
            + lag.Lut_func(t, u, ut, utt, v, vt, vtt) * self.B_1
            + lag.Lutt_func(t, u, ut, utt, v, vt, vtt) * self.B_2
        )
        v_grad_quad = (
            lag.Lv_func(t, u, ut, utt, v, vt, vtt) * self.B
            + lag.Lvt_func(t, u, ut, utt, v, vt, vtt) * self.B_1
            + lag.Lvtt_func(t, u, ut, utt, v, vt, vtt) * self.B_2
        )
        grad = np.array(
            [u_grad_quad @ self.quad_wts, v_grad_quad @ self.quad_wts]
        )
        return L_total, grad

    def grad(self, control: np.ndarray):
        """Lagrangian and (optional) constraint energy/gradients for one segment."""
        L_total, grad_total = self._evaluate_functional(control, self.lagrangian)
        if self.constraint is not None:
            g_total, dg_total = self._evaluate_functional(control, self.constraint)
        else:
            g_total = np.float64(0.0)
            dg_total = np.zeros_like(grad_total)
        return L_total, grad_total, g_total, dg_total

    def grad_total(self, control_dict: dict):
        """Apply grad() to every segment in the path."""
        energy, gradient, g, dg = {}, {}, {}, {}
        for e in control_dict:
            energy[e], gradient[e], g[e], dg[e] = self.grad(control_dict[e])
        return energy, gradient, g, dg

    def minimize(self, max_iteration: int = 50, constraint_tol: float = 1e-4) -> None:
        """Run the augmented-Lagrangian outer loop with L-BFGS-B inner solves."""
        for _ in range(max_iteration):
            _, _, g_prev, _ = self.grad_total(self.control)
            g_prev_acc = sum(g_prev.values())

            def objective(control_flat):
                control_dict = pack(control_flat, self.spline.edges)
                energy, gradient, g, dg = self.grad_total(control_dict)
                for e in self.spline.edges:
                    if self.constraint is not None:
                        energy[e] += (
                            self.constraint_multiplier * g[e]
                            + self.constraint_stiffness * (g[e] ** 2) / 2
                        )
                        gradient[e] += (
                            self.constraint_multiplier * dg[e]
                            + self.constraint_stiffness * g[e] * dg[e]
                        )
                    if self.reg is not None:
                        energy_reg, gradient_reg = self.reg(control_dict[e])
                        energy[e] += self.reg_stiffness * energy_reg
                        gradient[e] += self.reg_stiffness * gradient_reg

                self.spline.grad_mask(gradient)
                return sum(energy.values()), unpack(gradient)

            res = minimize(
                fun=objective,
                x0=unpack(self.control),
                jac=True,
                method="L-BFGS-B",
            )
            self.control = pack(res.x, self.spline.edges)

            if self.constraint is None:
                break

            _, _, g_final, _ = self.grad_total(self.control)
            g_final_acc = sum(g_final.values())

            if abs(g_final_acc) < constraint_tol:
                break

            if g_prev_acc != 0 and g_final_acc / g_prev_acc > 0.75:
                self.constraint_stiffness *= 5

            self.constraint_multiplier += self.constraint_stiffness * g_final_acc
