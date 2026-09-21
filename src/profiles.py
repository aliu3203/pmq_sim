"""Longitudinal gradient profiles G(z) for a quadrupole.

Every profile is written in the dimensionless coordinate

    s = (z - z0) / L

so that G(z) = G0 * f(s) and the n-th derivative with respect to z picks up a
chain-rule factor:

    d^n G / dz^n = (G0 / L^n) * f^(n)(s)

Profiles supply f through f''' in closed form, because the Maxwell-consistent
field expansion in field.py needs G'' (for Bx, By) and G''' (for Bz).
"""

import numpy as np


class GradientProfile:
    """Base class: a dimensionless shape function f(s) and its derivatives."""

    def f(self, s):
        raise NotImplementedError

    def d1(self, s):
        raise NotImplementedError

    def d2(self, s):
        raise NotImplementedError

    def d3(self, s):
        raise NotImplementedError

    def deriv(self, n, s):
        """f^(n)(s) for n in 0..3."""
        return (self.f, self.d1, self.d2, self.d3)[n](s)


class HardEdge(GradientProfile):
    """Step function: 1 inside |s| < 1/2, 0 outside.

    This is the model already used in quadrupole/tracking2.m and is here as the
    analytic baseline -- the field integral of a hard edge is exactly G0 * L.

    Its derivatives are delta functions, not ordinary functions, so d1/d2/d3
    raise rather than returning zero.  Returning zero would look exactly like a
    valid soft-edge model with vanishing fringe curvature and would silently
    hide the fact that only the leading-order field is available here.
    """

    def f(self, s):
        s = np.asarray(s, dtype=float)
        return np.where(np.abs(s) < 0.5, 1.0, 0.0)

    def _no_derivative(self, order):
        raise NotImplementedError(
            f"HardEdge has no ordinary {order} derivative (it is a distribution). "
            "Use a smooth profile (Tanh, Gaussian, Enge) for the G'' expansion, "
            "or restrict yourself to the leading-order field."
        )

    def d1(self, s):
        self._no_derivative("first")

    def d2(self, s):
        self._no_derivative("second")

    def d3(self, s):
        self._no_derivative("third")


class Tanh(GradientProfile):
    """Smoothed flat-top of unit length with fringe scale d.

        f(s) = [tanh((s + 1/2)/d) - tanh((s - 1/2)/d)] / 2

    Approaches HardEdge as d -> 0.  Derivatives follow from
    (d/dx) tanh(x) = sech^2(x) = 1 - tanh^2(x).
    """

    def __init__(self, d=0.1):
        if d <= 0:
            raise ValueError("fringe scale d must be positive")
        self.d = float(d)

    def _edges(self, s):
        s = np.asarray(s, dtype=float)
        return (s + 0.5) / self.d, (s - 0.5) / self.d

    def f(self, s):
        a, b = self._edges(s)
        return 0.5 * (np.tanh(a) - np.tanh(b))

    def d1(self, s):
        a, b = self._edges(s)
        # (1/2) * sech^2 * (1/d) for each edge
        return 0.5 / self.d * (_sech2(a) - _sech2(b))

    def d2(self, s):
        a, b = self._edges(s)
        # d/dx sech^2(x) = -2 tanh(x) sech^2(x)
        ga = -2.0 * np.tanh(a) * _sech2(a)
        gb = -2.0 * np.tanh(b) * _sech2(b)
        return 0.5 / self.d**2 * (ga - gb)

    def d3(self, s):
        a, b = self._edges(s)
        # With S = sech^2, T = tanh:  dS/dx = -2 T S, so
        # d^2 S / dx^2 = -2 (S*S + T*(-2 T S)) = S (4 T^2 - 2 S)
        ha = _sech2(a) * (4.0 * np.tanh(a) ** 2 - 2.0 * _sech2(a))
        hb = _sech2(b) * (4.0 * np.tanh(b) ** 2 - 2.0 * _sech2(b))
        return 0.5 / self.d**3 * (ha - hb)


class Gaussian(GradientProfile):
    """f(s) = exp(-s^2 / (2 d^2)).

    Peak gradient G0 at the center; useful for short magnets where the fringe
    regions overlap and there is no real flat top.
    """

    def __init__(self, d=0.3):
        if d <= 0:
            raise ValueError("width d must be positive")
        self.d = float(d)

    def f(self, s):
        s = np.asarray(s, dtype=float)
        return np.exp(-(s**2) / (2.0 * self.d**2))

    def d1(self, s):
        s = np.asarray(s, dtype=float)
        return -s / self.d**2 * self.f(s)

    def d2(self, s):
        s = np.asarray(s, dtype=float)
        return (s**2 / self.d**4 - 1.0 / self.d**2) * self.f(s)

    def d3(self, s):
        s = np.asarray(s, dtype=float)
        return (-(s**3) / self.d**6 + 3.0 * s / self.d**4) * self.f(s)


class Enge(GradientProfile):
    """Standard accelerator-physics fringe model.

        f(s) = f_edge(s + 1/2) * f_edge(-(s - 1/2))
        f_edge(t) = 1 / (1 + exp(P(t))),   P(t) = a0 + a1 t + a2 t^2 + ...

    Each edge is a logistic in a polynomial; the flat top is their product, so
    the profile is symmetric about s = 0 and the coefficients describe a single
    edge.  With the default coefficients the transition takes place over roughly
    one tenth of the magnet length.

    Derivatives use the logistic recursion (see _logistic_derivs) composed with
    the polynomial via Faa di Bruno up to third order, then the product rule
    across the two edges.  Everything is closed form -- no sympy needed.
    """

    def __init__(self, coeffs=(0.0, -20.0)):
        self.coeffs = np.asarray(coeffs, dtype=float)
        if self.coeffs.size < 2:
            raise ValueError("need at least a0 and a1")

    def _edge_derivs(self, t):
        """Return (g, g', g'', g''') of one edge evaluated at t, w.r.t. t."""
        t = np.asarray(t, dtype=float)
        u, u1, u2, u3 = _poly_derivs(self.coeffs, t)
        g, gu1, gu2, gu3 = _logistic_derivs(u)
        # Faa di Bruno for g(u(t))
        f1 = gu1 * u1
        f2 = gu2 * u1**2 + gu1 * u2
        f3 = gu3 * u1**3 + 3.0 * gu2 * u1 * u2 + gu1 * u3
        return g, f1, f2, f3

    def _both_edges(self, s):
        """Derivatives of the rising and falling edge, both w.r.t. s."""
        s = np.asarray(s, dtype=float)
        # Rising edge at s = -1/2: argument t = s + 1/2, dt/ds = +1
        a0, a1, a2, a3 = self._edge_derivs(s + 0.5)
        # Falling edge at s = +1/2: argument t = -(s - 1/2), dt/ds = -1,
        # so odd-order derivatives flip sign.
        b0, b1, b2, b3 = self._edge_derivs(-(s - 0.5))
        return (a0, a1, a2, a3), (b0, -b1, b2, -b3)

    def f(self, s):
        (a0, _, _, _), (b0, _, _, _) = self._both_edges(s)
        return a0 * b0

    def d1(self, s):
        (a0, a1, _, _), (b0, b1, _, _) = self._both_edges(s)
        return a1 * b0 + a0 * b1

    def d2(self, s):
        (a0, a1, a2, _), (b0, b1, b2, _) = self._both_edges(s)
        return a2 * b0 + 2.0 * a1 * b1 + a0 * b2

    def d3(self, s):
        (a0, a1, a2, a3), (b0, b1, b2, b3) = self._both_edges(s)
        return a3 * b0 + 3.0 * a2 * b1 + 3.0 * a1 * b2 + a0 * b3


def _sech2(x):
    """sech^2(x), written so that large |x| underflows to 0 rather than overflowing."""
    return 1.0 / np.cosh(np.clip(x, -350.0, 350.0)) ** 2


def _poly_derivs(coeffs, t):
    """P(t) and its first three derivatives, coeffs given low order first."""
    t = np.asarray(t, dtype=float)
    p = np.zeros_like(t)
    p1 = np.zeros_like(t)
    p2 = np.zeros_like(t)
    p3 = np.zeros_like(t)
    for n, a in enumerate(coeffs):
        p = p + a * t**n
        if n >= 1:
            p1 = p1 + a * n * t ** (n - 1)
        if n >= 2:
            p2 = p2 + a * n * (n - 1) * t ** (n - 2)
        if n >= 3:
            p3 = p3 + a * n * (n - 1) * (n - 2) * t ** (n - 3)
    return p, p1, p2, p3


def _logistic_derivs(u):
    """g = 1/(1+e^u) and dg/du, d2g/du2, d3g/du3, expressed through g itself.

        dg/du   = -g(1-g)
        d2g/du2 =  g(1-g)(1-2g)
        d3g/du3 = -g(1-g)[(1-g)(1-2g) - g(1-2g) - 2g(1-g)]

    Writing the derivatives in terms of g keeps them bounded even where e^u
    overflows, which matters because the Enge polynomial is deliberately steep.
    """
    u = np.asarray(u, dtype=float)
    g = 1.0 / (1.0 + np.exp(np.clip(u, -700.0, 700.0)))
    gm = 1.0 - g
    g1 = -g * gm
    g2 = g * gm * (1.0 - 2.0 * g)
    g3 = -g * gm * (gm * (1.0 - 2.0 * g) - g * (1.0 - 2.0 * g) - 2.0 * g * gm)
    return g, g1, g2, g3


PROFILES = {
    "hard": HardEdge,
    "tanh": Tanh,
    "gauss": Gaussian,
    "enge": Enge,
}


def make_profile(name, **kwargs):
    """Look up a profile class by name and construct it."""
    try:
        cls = PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; choose from {sorted(PROFILES)}"
        ) from None
    return cls(**kwargs)
