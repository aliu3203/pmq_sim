"""Maxwell-consistent soft-edge quadrupole field.

The field is built from a scalar potential rather than written down component by
component, so that Maxwell's equations hold by construction rather than by luck.
In the current-free bore B = grad(psi) with laplacian(psi) = 0, and for a normal
quadrupole with longitudinal gradient profile G(z):

    psi(x,y,z) = G(z) x y  -  (G''(z)/12) x y (x^2 + y^2)

Taking the gradient:

    Bx = G(z) y   - (G''(z)/12)  (3 x^2 y + y^3)
    By = G(z) x   - (G''(z)/12)  (x^3 + 3 x y^2)
    Bz = G'(z) x y - (G'''(z)/12) (x^3 y + x y^3)

Two properties of this truncation are worth stating explicitly, because the
self-checks in checks.py assert them:

  * curl(B) = 0 identically, since B is a gradient of a scalar.  A numerical
    curl test therefore checks that the three expressions above really are
    consistent partial derivatives of one potential -- a typo catcher.

  * div(B) is NOT exactly zero.  The G'' term was chosen to cancel the G'' x y
    coming from d^2 psi / dz^2, but the second z-derivative of the G'' term
    itself survives:

        div(B) = -(G''''(z)/12) x y (x^2 + y^2)

    so the field is exact to O(r^3) and the residual grows as r^4.  Asserting
    div(B) == 0 would be wrong; asserting the r^4 scaling is the real test.
"""

import numpy as np

from profiles import GradientProfile, make_profile


class PMQ:
    """A single permanent magnet quadrupole.

    Parameters
    ----------
    G0 : float
        Peak field gradient [T/m].  Positive G0 focuses in x for a negatively
        charged particle travelling in +z, matching the convention in
        quadrupole/tracking2.m.
    L : float
        Magnet length [m].  Sets the scale of the longitudinal profile; for
        soft-edge profiles the effective length differs slightly (see
        integrals.effective_length).
    z0 : float
        Longitudinal position of the magnet center [m].
    profile : GradientProfile or str
        Shape of G(z).  A string is looked up in profiles.PROFILES and built
        with `profile_kwargs`.
    dx, dy : float
        Transverse offset of the magnetic center from the reference axis [m].
        This is the quantity being scanned: the wire runs along the reference
        axis and the magnet is translated relative to it.
    theta : float
        Roll angle about the z axis [rad].
    """

    def __init__(
        self,
        G0,
        L,
        z0=0.0,
        profile="tanh",
        dx=0.0,
        dy=0.0,
        theta=0.0,
        **profile_kwargs,
    ):
        self.G0 = float(G0)
        self.L = float(L)
        self.z0 = float(z0)
        self.dx = float(dx)
        self.dy = float(dy)
        self.theta = float(theta)
        if isinstance(profile, GradientProfile):
            self.profile = profile
        else:
            self.profile = make_profile(profile, **profile_kwargs)

    # -- longitudinal gradient and its z-derivatives -----------------------

    def _s(self, z):
        return (np.asarray(z, dtype=float) - self.z0) / self.L

    def G(self, z):
        """G(z) [T/m]."""
        return self.G0 * self.profile.f(self._s(z))

    def dG(self, z):
        """dG/dz [T/m^2]."""
        return self.G0 / self.L * self.profile.d1(self._s(z))

    def d2G(self, z):
        """d2G/dz2 [T/m^3]."""
        return self.G0 / self.L**2 * self.profile.d2(self._s(z))

    def d3G(self, z):
        """d3G/dz3 [T/m^4]."""
        return self.G0 / self.L**3 * self.profile.d3(self._s(z))

    # -- field --------------------------------------------------------------

    def B(self, x, y, z, order=2):
        """Field at (x, y, z) in lab coordinates.

        Returns (Bx, By, Bz) broadcast against the inputs.

        `order=2` includes the G'' correction terms; `order=0` keeps only the
        leading term, which is the hard-edge model of quadrupole/tracking2.m
        and is the only option available for the HardEdge profile.
        """
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )

        # Translate into the magnet's transverse frame.
        u = x - self.dx
        v = y - self.dy

        # Rotate into the magnet's rolled frame.
        c, s = np.cos(self.theta), np.sin(self.theta)
        xl = c * u + s * v
        yl = -s * u + c * v

        g = self.G(z)
        bx = g * yl
        by = g * xl
        bz = self.dG(z) * xl * yl if order >= 2 else np.zeros_like(xl)

        if order >= 2:
            g2 = self.d2G(z) / 12.0
            g3 = self.d3G(z) / 12.0
            bx = bx - g2 * (3.0 * xl**2 * yl + yl**3)
            by = by - g2 * (xl**3 + 3.0 * xl * yl**2)
            bz = bz - g3 * (xl**3 * yl + xl * yl**3)
        elif order != 0:
            raise ValueError("order must be 0 or 2")

        # Rotate the field vector back into lab coordinates.
        return c * bx - s * by, s * bx + c * by, bz


class Lattice:
    """A collection of PMQs along the z axis.

    Superposition is exact: Maxwell's equations are linear and each magnet's
    field is a vacuum solution, so the sum is too.  A single magnet is the
    common case; the list makes the triplet configurations a matter of
    configuration rather than a rewrite.
    """

    def __init__(self, magnets):
        if isinstance(magnets, PMQ):
            magnets = [magnets]
        self.magnets = list(magnets)
        if not self.magnets:
            raise ValueError("lattice needs at least one magnet")

    def __len__(self):
        return len(self.magnets)

    def __iter__(self):
        return iter(self.magnets)

    def B(self, x, y, z, order=2):
        bx = by = bz = 0.0
        for m in self.magnets:
            mx, my, mz = m.B(x, y, z, order=order)
            bx = bx + mx
            by = by + my
            bz = bz + mz
        return bx, by, bz

    def z_extent(self, pad=10.0):
        """(z_min, z_max) covering every magnet plus `pad` lengths of fringe."""
        lo = min(m.z0 - pad * m.L for m in self.magnets)
        hi = max(m.z0 + pad * m.L for m in self.magnets)
        return lo, hi
