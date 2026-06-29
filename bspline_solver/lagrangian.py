"""Symbolic Lagrangian with precomputed partial derivatives."""

from __future__ import annotations

import sympy as sp


class Lagrangian2D:
    """Wraps a symbolic Lagrangian and its first partials.

    The symbolic variables used internally are:
        t                -- curve parameter
        u, ut, utt       -- first component and its first/second derivatives
        v, vt, vtt       -- second component and its first/second derivatives

    Attributes:
        L_func: Callable evaluating L(t, u, ut, utt, v, vt, vtt).
        Lu_func, Lut_func, Lutt_func: Partials with respect to u, u', u''.
        Lv_func, Lvt_func, Lvtt_func: Partials with respect to v, v', v''.
    """

    def __init__(self, lagrangian: sp.Expr) -> None:
        self.t = sp.Symbol("t")
        self.u = sp.Symbol("u")
        self.ut = sp.Symbol("ut")
        self.utt = sp.Symbol("utt")
        self.v = sp.Symbol("v")
        self.vt = sp.Symbol("vt")
        self.vtt = sp.Symbol("vtt")

        self.L = lagrangian
        self.Lu = sp.diff(self.L, self.u)
        self.Lut = sp.diff(self.L, self.ut)
        self.Lutt = sp.diff(self.L, self.utt)
        self.Lv = sp.diff(self.L, self.v)
        self.Lvt = sp.diff(self.L, self.vt)
        self.Lvtt = sp.diff(self.L, self.vtt)

        args = (self.t, self.u, self.ut, self.utt, self.v, self.vt, self.vtt)
        self.L_func = sp.lambdify(args, self.L, modules="numpy")
        self.Lu_func = sp.lambdify(args, self.Lu, modules="numpy")
        self.Lut_func = sp.lambdify(args, self.Lut, modules="numpy")
        self.Lutt_func = sp.lambdify(args, self.Lutt, modules="numpy")
        self.Lv_func = sp.lambdify(args, self.Lv, modules="numpy")
        self.Lvt_func = sp.lambdify(args, self.Lvt, modules="numpy")
        self.Lvtt_func = sp.lambdify(args, self.Lvtt, modules="numpy")
