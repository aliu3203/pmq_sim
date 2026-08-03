"""Analytic self-checks for the magnetics layer.

Run with:  python checks.py

These print numbers rather than just passing silently, because the point is to
see *how well* each identity holds, not merely that it holds to some tolerance
somebody picked.  Checks 2 and 4 are the ones that matter: check 2 proves the
Maxwell expansion is correct to the claimed order, and check 4 proves the
offset-to-kick relationship is the single number to extract from bench data.
"""

import numpy as np

from field import PMQ, Lattice
from integrals import (
    effective_length,
    field_integral,
    integrated_gradient,
    offset_scan,
    offset_sensitivity,
)
from profiles import Enge, Gaussian, HardEdge, Tanh

# Representative magnet: numbers are placeholders, but every check below is
# parameter-independent.
G0 = 500.0      # T/m
LMAG = 0.02     # m
BORE = 3e-3     # m, radius


def _fd_grad(fn, x, y, z, h):
    """Central-difference gradient of a scalar field at (x, y, z)."""
    dfdx = (fn(x + h, y, z) - fn(x - h, y, z)) / (2 * h)
    dfdy = (fn(x, y + h, z) - fn(x, y - h, z)) / (2 * h)
    dfdz = (fn(x, y, z + h) - fn(x, y, z - h)) / (2 * h)
    return dfdx, dfdy, dfdz


def _report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail.strip().splitlines():
        print(f"       {line}")
    print()
    return ok


# ---------------------------------------------------------------------------
# 1. curl(B) = 0 identically, since B is the gradient of a scalar potential.
# ---------------------------------------------------------------------------
def check_curl():
    q = PMQ(G0, LMAG, profile="tanh", d=0.15)
    rng = np.random.default_rng(0)
    npts = 400
    r = BORE * np.sqrt(rng.uniform(0, 1, npts))
    phi = rng.uniform(0, 2 * np.pi, npts)
    x, y = r * np.cos(phi), r * np.sin(phi)
    z = rng.uniform(-1.5 * LMAG, 1.5 * LMAG, npts)

    h = 1e-6
    bx = lambda a, b, c: q.B(a, b, c)[0]
    by = lambda a, b, c: q.B(a, b, c)[1]
    bz = lambda a, b, c: q.B(a, b, c)[2]

    _, dbx_dy, dbx_dz = _fd_grad(bx, x, y, z, h)
    dby_dx, _, dby_dz = _fd_grad(by, x, y, z, h)
    dbz_dx, dbz_dy, _ = _fd_grad(bz, x, y, z, h)

    curl = np.stack([dbz_dy - dby_dz, dbx_dz - dbz_dx, dby_dx - dbx_dy])
    # curl(B) has units of T/m, so the natural scale to compare against is the
    # gradient G0 itself, not the field magnitude G0 * bore.
    rel = np.abs(curl).max() / G0

    ok = rel < 1e-6
    return _report(
        "1. curl(B) = 0  (coded components are consistent partials of one potential)",
        ok,
        f"max |curl B| / G0 = {rel:.3e}   (finite-difference floor, h = {h:g})",
    )


# ---------------------------------------------------------------------------
# 2. div(B) is the O(r^4) truncation residual -(G''''/12) x y (x^2 + y^2),
#    NOT zero.  Verify both the r^4 scaling and the closed form.
# ---------------------------------------------------------------------------
def check_divergence_scaling():
    q = PMQ(G0, LMAG, profile="tanh", d=0.15)
    z_probe = 0.5 * LMAG  # in the fringe, where G'''' is large
    ht, hz = BORE / 50.0, LMAG / 500.0

    def divergence(x, y, z):
        """div(B) via fourth-order central differences.

        div(B) is a small residual left over after three O(G0) derivatives
        cancel, so a second-order stencil leaves a truncation error comparable
        to the residual itself.  The fourth-order stencil is in fact exact for
        the two transverse terms, since Bx and By are cubic polynomials in x
        and y.
        """
        def deriv(comp, axis, h):
            def shift(k):
                p = [x, y, z]
                p[axis] = p[axis] + k * h
                return q.B(*p)[comp]

            return (shift(-2) - 8 * shift(-1) + 8 * shift(1) - shift(2)) / (12 * h)

        return deriv(0, 0, ht) + deriv(1, 1, ht) + deriv(2, 2, hz)

    radii = BORE * np.array([0.2, 0.3, 0.45, 0.7, 1.0])
    div_rms = []
    for r in radii:
        phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        x, y = r * np.cos(phi), r * np.sin(phi)
        div_rms.append(np.sqrt(np.mean(divergence(x, y, np.full_like(x, z_probe)) ** 2)))
    div_rms = np.array(div_rms)

    slope, _ = np.polyfit(np.log(radii), np.log(div_rms), 1)

    # Closed form: div(B) = -(G''''/12) x y (x^2 + y^2), with G'''' from a
    # fourth-order seven-point stencil.
    gz = lambda k: q.G(z_probe + k * hz)
    g4 = (
        -gz(-3) + 12 * gz(-2) - 39 * gz(-1) + 56 * gz(0)
        - 39 * gz(1) + 12 * gz(2) - gz(3)
    ) / (6 * hz**4)

    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x, y = BORE * np.cos(phi), BORE * np.sin(phi)
    div_num = divergence(x, y, np.full_like(x, z_probe))
    div_pred = -(g4 / 12.0) * x * y * (x**2 + y**2)
    form_err = np.abs(div_num - div_pred).max() / np.abs(div_pred).max()

    ok = abs(slope - 4.0) < 0.05 and form_err < 1e-3
    return _report(
        "2. div(B) is the O(r^4) truncation residual, not zero",
        ok,
        f"fitted power of r in |div B|:  {slope:.4f}   (expected 4)\n"
        f"agreement with -(G''''/12) x y (x^2+y^2):  {form_err:.3e} relative\n"
        f"|div B| at r = bore, relative to |grad B| ~ G0:  "
        f"{np.abs(div_num).max() / G0:.3e}",
    )


# ---------------------------------------------------------------------------
# 3. Analytic profile derivatives against high-accuracy finite differences.
# ---------------------------------------------------------------------------
def check_profile_derivatives():
    profiles = {
        "Tanh(d=0.15)": Tanh(0.15),
        "Gaussian(d=0.3)": Gaussian(0.3),
        "Enge((0,-20))": Enge((0.0, -20.0)),
        "Enge((0,-14,2))": Enge((0.0, -14.0, 2.0)),
    }
    s = np.linspace(-1.2, 1.2, 241)
    worst = 0.0
    lines = []
    for name, p in profiles.items():
        # h = 1e-3 balances the h^4 truncation of these fourth-order stencils
        # against the roundoff amplification of dividing by h^3.
        h = 1e-3
        f = p.f
        d1 = (f(s - 2 * h) - 8 * f(s - h) + 8 * f(s + h) - f(s + 2 * h)) / (12 * h)
        d2 = (
            -f(s - 2 * h) + 16 * f(s - h) - 30 * f(s) + 16 * f(s + h) - f(s + 2 * h)
        ) / (12 * h**2)
        d3 = (
            f(s - 3 * h)
            - 8 * f(s - 2 * h)
            + 13 * f(s - h)
            - 13 * f(s + h)
            + 8 * f(s + 2 * h)
            - f(s + 3 * h)
        ) / (8 * h**3)

        errs = []
        for analytic, numeric in ((p.d1(s), d1), (p.d2(s), d2), (p.d3(s), d3)):
            scale = max(np.abs(numeric).max(), 1e-30)
            errs.append(np.abs(analytic - numeric).max() / scale)
        worst = max(worst, max(errs))
        lines.append(
            f"{name:<18} f' {errs[0]:.2e}   f'' {errs[1]:.2e}   f''' {errs[2]:.2e}"
        )

    ok = worst < 1e-5
    return _report(
        "3. Analytic profile derivatives match finite differences",
        ok,
        "\n".join(lines) + f"\nworst relative error: {worst:.2e}",
    )


# ---------------------------------------------------------------------------
# 4. The field integral is exactly linear in PMQ offset, with slope
#    -(integral of G dz), for every smooth profile.
# ---------------------------------------------------------------------------
def check_offset_linearity():
    lines = []
    worst_resid = 0.0
    worst_slope = 0.0
    for name, kw in (
        ("tanh", dict(profile="tanh", d=0.15)),
        ("gauss", dict(profile="gauss", d=0.3)),
        ("enge", dict(profile="enge", coeffs=(0.0, -20.0))),
    ):
        q = PMQ(G0, LMAG, **kw)
        offsets = np.linspace(-1e-3, 1e-3, 21)
        _, iy = offset_scan(q, offsets, axis="x")

        coeffs = np.polyfit(offsets, iy, 1)
        resid = np.abs(iy - np.polyval(coeffs, offsets)).max()
        rel_resid = resid / np.abs(iy).max()

        gl = integrated_gradient(q)
        slope_err = abs(coeffs[0] - (-gl)) / abs(gl)

        worst_resid = max(worst_resid, rel_resid)
        worst_slope = max(worst_slope, slope_err)
        lines.append(
            f"{name:<6} L_eff/L = {effective_length(q) / LMAG:.6f}   "
            f"int G dz = {gl:.6f} T   "
            f"nonlinearity {rel_resid:.2e}   slope error {slope_err:.2e}"
        )

    ok = worst_resid < 1e-10 and worst_slope < 1e-9
    return _report(
        "4. Field integral is exactly linear in offset, slope = -(int G dz)",
        ok,
        "\n".join(lines)
        + f"\nworst nonlinearity {worst_resid:.2e}, worst slope error {worst_slope:.2e}",
    )


# ---------------------------------------------------------------------------
# 5. Tanh with a small fringe scale reproduces the hard edge.
# ---------------------------------------------------------------------------
def check_hard_edge_limit():
    hard = PMQ(G0, LMAG, profile="hard")
    x_probe = 1e-3
    # order=0: the hard edge has no ordinary derivatives, so only the leading
    # term is defined for it.
    _, iy_hard = field_integral(hard, x=x_probe, n=200001, order=0)
    expected = G0 * LMAG * x_probe

    lines = [f"hard edge:  int By dz = {iy_hard:.9e}  (exact {expected:.9e})"]
    int_errs = []
    shape_errs = []
    for d in (0.05, 0.02, 0.01):
        soft = PMQ(G0, LMAG, profile="tanh", d=d)
        _, iy = field_integral(soft, x=x_probe, n=200001)
        int_errs.append(abs(iy - expected) / abs(expected))

        # Pointwise convergence on a *fixed* window of the flat top.  The
        # window must not shrink with d: measured a fixed number of fringe
        # widths from the edge, the G'' correction grows as 1/d^2 and the
        # error would appear to diverge.
        s = np.linspace(-0.4, 0.4, 501)
        z = LMAG * s
        by_soft = soft.B(x_probe, 0.0, z)[1]
        shape_errs.append(np.abs(by_soft / (G0 * x_probe) - 1.0).max())

        lines.append(
            f"tanh d={d:<5} int By dz = {iy:.9e}   integral err {int_errs[-1]:.2e}"
            f"   flat-top err {shape_errs[-1]:.2e}"
        )

    # The tanh profile is a difference of two shifted tanh steps, and the
    # integral of such a difference is exactly (shift) x (total jump) = L,
    # independent of d.  So the field integral is exact for every d rather than
    # converging to it -- the fringe removes precisely as much area as it adds.
    # What actually converges with d is the pointwise field on the flat top.
    monotone = shape_errs == sorted(shape_errs, reverse=True)
    ok = (
        abs(iy_hard - expected) / expected < 1e-9
        and max(int_errs) < 1e-12
        and monotone
    )
    return _report(
        "5. Tanh -> HardEdge as the fringe scale shrinks",
        ok,
        "\n".join(lines)
        + f"\nintegral exact for every d (max err {max(int_errs):.2e}); "
        f"flat-top field converges monotonically: {monotone}",
    )


# ---------------------------------------------------------------------------
# 6. Superposition: well-separated magnets add.
# ---------------------------------------------------------------------------
def check_superposition():
    a = PMQ(G0, LMAG, z0=-0.10, profile="tanh", d=0.15)
    b = PMQ(-G0, LMAG, z0=+0.10, profile="gauss", d=0.3)
    x_probe = 1e-3
    z_range = (-0.5, 0.5)

    _, iy_a = field_integral(a, x=x_probe, z_range=z_range, n=100001)
    _, iy_b = field_integral(b, x=x_probe, z_range=z_range, n=100001)
    _, iy_ab = field_integral(
        Lattice([a, b]), x=x_probe, z_range=z_range, n=100001
    )

    err = abs(iy_ab - (iy_a + iy_b)) / max(abs(iy_a), abs(iy_b))
    ok = err < 1e-12
    return _report(
        "6. Superposition of magnets in a lattice",
        ok,
        f"int By dz:  A {iy_a:.6e}   B {iy_b:.6e}   A+B {iy_ab:.6e}\n"
        f"relative error {err:.2e}",
    )


# ---------------------------------------------------------------------------
# 7. HardEdge refuses to supply derivatives rather than silently returning 0.
# ---------------------------------------------------------------------------
def check_hard_edge_refuses_derivatives():
    q = PMQ(G0, LMAG, profile="hard")
    try:
        q.B(1e-3, 0.0, 0.0, order=2)
    except NotImplementedError as exc:
        return _report(
            "7. HardEdge raises instead of faking a zero G''",
            True,
            f"raised NotImplementedError: {str(exc).splitlines()[0]}",
        )
    return _report(
        "7. HardEdge raises instead of faking a zero G''",
        False,
        "order=2 silently succeeded on a hard edge -- G'' is being treated as zero",
    )


# ---------------------------------------------------------------------------
# 8. Roll angle: matches the closed form in tracking2.m at leading order, and
#    is a genuine rotation at full order.
# ---------------------------------------------------------------------------
def check_roll():
    theta = 0.3
    q = PMQ(G0, LMAG, profile="tanh", d=0.15, theta=theta, dx=2e-4, dy=-1e-4)
    x, y, z = 1.5e-3, -0.7e-3, 0.004

    # Leading order against quadrupole/tracking2.m:237, which writes the rolled
    # quadrupole field directly in terms of cos(2*theta) and sin(2*theta).
    bx, by, _ = q.B(x, y, z, order=0)
    u, v = x - q.dx, y - q.dy
    g = q.G(z)
    bx_ref = g * (v * np.cos(2 * theta) - u * np.sin(2 * theta))
    by_ref = g * (u * np.cos(2 * theta) + v * np.sin(2 * theta))
    ref_err = max(abs(bx - bx_ref) / abs(bx_ref), abs(by - by_ref) / abs(by_ref))

    # At full order, rolling the magnet and rotating the probe point together
    # must leave |B| unchanged -- the G'' terms have to rotate too.
    plain = PMQ(G0, LMAG, profile="tanh", d=0.15)
    rolled = PMQ(G0, LMAG, profile="tanh", d=0.15, theta=theta)
    c, s = np.cos(theta), np.sin(theta)
    b_plain = np.array(plain.B(x, y, z))
    b_rolled = np.array(rolled.B(c * x - s * y, s * x + c * y, z))
    n_plain = np.linalg.norm(b_plain)
    inv_err = abs(np.linalg.norm(b_rolled) - n_plain) / n_plain

    ok = ref_err < 1e-14 and inv_err < 1e-14
    return _report(
        "8. Roll angle matches tracking2.m and is a true rotation at order 2",
        ok,
        f"vs tracking2.m closed form (order 0):  {ref_err:.2e} relative\n"
        f"|B| invariance under roll (order 2):   {inv_err:.2e} relative",
    )


def main():
    print(f"\nPMQ magnetics self-checks   G0 = {G0} T/m, L = {LMAG * 1e3} mm, "
          f"bore radius = {BORE * 1e3} mm\n")
    results = [
        check_curl(),
        check_divergence_scaling(),
        check_profile_derivatives(),
        check_offset_linearity(),
        check_hard_edge_limit(),
        check_superposition(),
        check_hard_edge_refuses_derivatives(),
        check_roll(),
    ]
    n_pass = sum(results)
    print(f"{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
