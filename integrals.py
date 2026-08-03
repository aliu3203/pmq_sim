"""Field integrals along the wire, and the offset -> kick relationship.

The pulsed wire runs along a line of fixed (x, y) through the bore, so what a
long-pulse measurement responds to is the first field integral

    I_y(x, y) = integral of By dz,      I_x(x, y) = integral of Bx dz

taken along that line.  For a beam of momentum p the same integral is a
deflection angle, which is what `kick` converts to.

There is a result here worth keeping in mind when interpreting the numbers.
Writing u = x - dx and v = y - dy for the wire position in the magnet frame,

    integral of By dz = (integral of G dz) u  -  (1/12)(integral of G'' dz)(u^3 + 3 u v^2)

and for any profile that decays at both ends, integral of G'' dz = [G']_-inf^+inf
= 0.  So the field integral is *exactly* linear in the PMQ offset regardless of
the fringe shape, and the whole position-to-kick relationship collapses to one
number: the integrated gradient (integral of G dz) = G0 * L_eff.

Fringe shape changes the shape of B(z), hence the time-domain trace and its
sensitivity to pulse width -- but not the integrated kick.  Any curvature seen
in a measured kick-versus-position scan has to come from somewhere else:
higher multipoles from Halbach segmentation, fringe overlap between adjacent
magnets, a truncated integration range, or the wire and detector response.
"""

import numpy as np
from scipy.integrate import simpson

from field import Lattice, PMQ


def _as_lattice(obj):
    return obj if isinstance(obj, Lattice) else Lattice(obj)


def b_along_wire(lattice, x=0.0, y=0.0, z_range=None, n=4001, order=2):
    """Sample the field along a line of fixed (x, y).

    Returns (z, Bx, By, Bz).  This is the simulated analogue of a pulsed-wire
    trace before any wire dynamics: the spatial field profile the current pulse
    actually sees.
    """
    lattice = _as_lattice(lattice)
    if z_range is None:
        z_range = lattice.z_extent()
    z = np.linspace(z_range[0], z_range[1], int(n))
    bx, by, bz = lattice.B(np.full_like(z, x), np.full_like(z, y), z, order=order)
    return z, bx, by, bz


def field_integral(lattice, x=0.0, y=0.0, z_range=None, n=4001, order=2):
    """(integral of Bx dz, integral of By dz) along the wire, in T*m.

    The default z_range spans ten magnet lengths beyond the outermost magnet so
    the fringe field is fully captured -- important, because truncating the
    range is one of the few ways to make an ideal quadrupole look nonlinear in
    offset.
    """
    z, bx, by, _ = b_along_wire(
        lattice, x=x, y=y, z_range=z_range, n=n, order=order
    )
    return simpson(bx, x=z), simpson(by, x=z)


def effective_length(pmq, n=4001, pad=10.0):
    """L_eff = (integral of G dz) / G0 [m].

    Equals L exactly for a hard edge; for soft-edge profiles it measures how
    much integrated gradient the fringe adds or removes relative to the
    nominal length.
    """
    z = np.linspace(pmq.z0 - pad * pmq.L, pmq.z0 + pad * pmq.L, int(n))
    return simpson(pmq.G(z), x=z) / pmq.G0


def integrated_gradient(pmq, n=4001, pad=10.0):
    """integral of G dz [T], the single number that sets the offset-to-kick slope."""
    return pmq.G0 * effective_length(pmq, n=n, pad=pad)


def offset_scan(lattice, offsets, axis="x", x=0.0, y=0.0, n=4001, order=2):
    """Sweep the PMQ transverse position and return the resulting field integrals.

    Every magnet in the lattice is translated together, which is what happens on
    the bench when the magnet mount is moved relative to a fixed wire.  The wire
    itself stays at (x, y).

    Parameters
    ----------
    offsets : array of float
        Magnet displacements [m] along `axis`.
    axis : {"x", "y"}

    Returns
    -------
    ix, iy : arrays of float
        integral of Bx dz and integral of By dz [T*m] at each offset.

    The output is the physical counterpart of the `peaks` array built in
    pmq_measurements/analysis.py -- same sweep, but in field units rather than
    photodiode volts.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    lattice = _as_lattice(lattice)
    offsets = np.atleast_1d(np.asarray(offsets, dtype=float))

    original = [(m.dx, m.dy) for m in lattice.magnets]
    ix = np.empty(offsets.size)
    iy = np.empty(offsets.size)
    try:
        for i, d in enumerate(offsets):
            for m, (dx0, dy0) in zip(lattice.magnets, original):
                if axis == "x":
                    m.dx = dx0 + d
                else:
                    m.dy = dy0 + d
            ix[i], iy[i] = field_integral(
                lattice, x=x, y=y, n=n, order=order
            )
    finally:
        for m, (dx0, dy0) in zip(lattice.magnets, original):
            m.dx, m.dy = dx0, dy0
    return ix, iy


def kick(field_integral_tm, p_ev=None, charge=-1.0):
    """Convert a field integral [T*m] to a deflection angle [rad].

    dx' = q * (integral of B dz) / p

    `p_ev` is the momentum in eV/c; for the 7.6 MeV beam in
    quadrupole/tracking2.m that is about 8.1e6.  `charge` is in units of the
    elementary charge.
    """
    if p_ev is None:
        raise ValueError("p_ev (momentum in eV/c) is required")
    # p [kg m/s] = p_ev * e / c, and q = charge * e, so q/p = charge * c / p_ev.
    c = 2.99792458e8
    return charge * c / float(p_ev) * np.asarray(field_integral_tm, dtype=float)


def offset_sensitivity(lattice, axis="x", span=1e-3, n_points=11, **kwargs):
    """Slope of the field integral with respect to magnet offset [T*m per m].

    Fits a straight line through a small symmetric scan.  For an ideal
    quadrupole this equals -(integral of G dz) exactly: moving the magnet by +d
    is equivalent to moving the wire by -d.
    """
    offsets = np.linspace(-span, span, int(n_points))
    ix, iy = offset_scan(lattice, offsets, axis=axis, **kwargs)
    response = iy if axis == "x" else ix
    slope, _ = np.polyfit(offsets, response, 1)
    return slope
