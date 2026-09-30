"""Worked example: fix one PMQ, scan the roll of the other.

Same doublet as example_roll_scan.py -- a 3 mm focusing magnet followed by a
6 mm defocusing magnet, hard-edge, probed by a wire at x = 1 mm -- but here
PMQ 1 is held at theta = 0 while PMQ 2 is rolled about the beam axis.  This is
the *differential roll* that a real mount error produces: one magnet turned
relative to its neighbour, not the pair turned together.

Because the two magnets sit at different z and (hard-edge) barely overlap, the
wire sees two separate bumps and the field integrals simply add:

    int By dz / x = G1 L1            +  G2 L2 cos(2 theta)      (normal)
    int Bx dz / x =                  -  G2 L2 sin(2 theta)      (skew)

With G1 L1 = +1.5 T and G2 L2 = -3.0 T that is

    normal = 1.5 - 3 cos(2 theta) T ,    skew = 3 sin(2 theta) T .

Two things to notice, and this script shows both:

  A.  Along the wire, PMQ 1's bump stays put in B_y while PMQ 2's bump rotates
      out of B_y and into B_x as it is rolled -- at 45 deg PMQ 2 contributes
      only skew, at 90 deg it has flipped sign in B_y.

  B.  The *net* integrated gradient is NOT conserved (unlike rolling a single
      magnet): the fixed PMQ 1 term breaks the quadrature sum, so |net| swings
      from 1.5 T (the two partially cancel) up to 4.5 T (they add).

Run:  python examples/example_roll_differential.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make the library in src/ importable when run as `python examples/<name>.py`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from field import PMQ, Lattice
from integrals import b_along_wire, field_integral

# --- Magnet parameters (identical to example_roll_scan.py) ------------------
G1, L1, Z1, P1 = 500, 0.003, 0.2, "hard"
THETA1 = 0.0                       # PMQ 1 is the fixed reference magnet

L_sep = 0.013523

G2, L2, P2 = -500, 0.006, "hard"
Z2 = Z1 + L_sep

X_PROBE = 1e-3                      # wire off-axis; on the x axis By reads
Y_PROBE = 0.0                      # normal and Bx reads skew directly
ORDER = 0                          # hard edge has no G'' expansion

ROLLS_DEG = (0.0, 45.0, 90.0)      # PMQ 2 rolls to draw in panel A

# Categorical slots 1-3 from the validated reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#8a8985"


def main():
    m1 = PMQ(G1, L1, Z1, P1, theta=THETA1)
    m2 = PMQ(G2, L2, Z2, P2, theta=0.0)
    lat = Lattice([m1, m2])

    z_lo = Z1 - 4 * L1
    z_hi = Z2 + 4 * L2

    print(f"\ndoublet:  PMQ1 {G1} T/m x {L1 * 1e3:g} mm (fixed, theta=0)")
    print(f"          PMQ2 {G2} T/m x {L2 * 1e3:g} mm (rolled)")
    print(f"  G1 L1 = {G1 * L1:+.2f} T   G2 L2 = {G2 * L2:+.2f} T")
    print(f"  wire at (x, y) = ({X_PROBE * 1e3:g}, {Y_PROBE * 1e3:g}) mm\n")
    print("  PMQ2 roll   net int By/x   net int Bx/x     |net|")
    print("   [deg]       (normal)        (skew)          [T]")
    for deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        m2.theta = np.deg2rad(deg)
        ix, iy = field_integral(lat, x=X_PROBE, y=Y_PROBE, order=ORDER)
        n, s = iy / X_PROBE, ix / X_PROBE
        print(f"   {deg:5.1f}    {n:+9.4f}     {s:+9.4f}     {np.hypot(n, s):6.3f}")
    m2.theta = 0.0
    print()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Panel A: field along the wire at several PMQ 2 rolls ---------------
    for m, label in ((m1, "PMQ 1 (fixed)"), (m2, "PMQ 2 (rolled)")):
        ax_a.axvspan((m.z0 - m.L / 2) * 1e3, (m.z0 + m.L / 2) * 1e3,
                     color="#f0efec", zorder=0)
        ax_a.annotate(label, xy=(m.z0 * 1e3, 1.01),
                      xycoords=("data", "axes fraction"),
                      ha="center", va="bottom", fontsize=8.5, color=MUTED)

    ax_a.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    for color, deg in zip(SERIES, ROLLS_DEG):
        m2.theta = np.deg2rad(deg)
        z, bx, by, _ = b_along_wire(
            lat, x=X_PROBE, y=Y_PROBE, z_range=(z_lo, z_hi), n=4001, order=ORDER
        )
        ax_a.plot(z * 1e3, by * 1e3, color=color, lw=2.0, zorder=3,
                  label=f"PMQ 2 roll {deg:g}°")
        ax_a.plot(z * 1e3, bx * 1e3, color=color, lw=2.0, ls="--", zorder=3)
    m2.theta = 0.0

    ax_a.annotate(
        "solid $B_y$ (normal) · dashed $B_x$ (skew)",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8.5, color=MUTED,
    )
    ax_a.set_xlabel("z [mm]")
    ax_a.set_ylabel(r"$B$ along the wire [mT]")
    ax_a.set_title("A.  PMQ 1 fixed, PMQ 2's field rotates", loc="left",
                   fontsize=11, pad=18)
    ax_a.set_xlim(z_lo * 1e3, z_hi * 1e3)
    ax_a.margins(y=0.15)
    ax_a.legend(frameon=False, fontsize=8.5, loc="lower right")
    _recede(ax_a)

    # --- Panel B: net normal / skew integrated gradient vs PMQ 2 roll -------
    rolls = np.linspace(0.0, 90.0, 91)
    normal = np.empty_like(rolls)
    skew = np.empty_like(rolls)
    for i, deg in enumerate(rolls):
        m2.theta = np.deg2rad(deg)
        ix, iy = field_integral(lat, x=X_PROBE, y=Y_PROBE, order=ORDER)
        normal[i] = iy / X_PROBE
        skew[i] = ix / X_PROBE
    m2.theta = 0.0

    ax_b.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax_b.plot(rolls, normal, color=SERIES[0], lw=2.0, zorder=3,
              label=r"net normal  $=G_1L_1 + G_2L_2\cos 2\theta$")
    ax_b.plot(rolls, skew, color=SERIES[1], lw=2.0, zorder=3,
              label=r"net skew  $=-G_2L_2\sin 2\theta$")
    ax_b.plot(rolls, np.hypot(normal, skew), color=MUTED, lw=1.5, ls=":",
              zorder=2, label=r"$|$net$|$ — not conserved")

    ax_b.axvline(45.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax_b.annotate("45° — PMQ 2\npure skew", xy=(45.0, 0.6), xytext=(46.5, 0.4),
                  fontsize=8.5, color=INK, va="center", ha="left")
    ax_b.set_xlabel("PMQ 2 roll angle θ [deg]")
    ax_b.set_ylabel(r"net integrated gradient [T]")
    ax_b.set_title("B.  Fixed magnet breaks strength conservation",
                   loc="left", fontsize=11)
    ax_b.set_xlim(0, 90)
    ax_b.set_xticks([0, 22.5, 45, 67.5, 90])
    ax_b.legend(frameon=False, fontsize=8.5, loc="upper left")
    _recede(ax_b)

    fig.tight_layout()
    fig.savefig("roll_differential.png", dpi=200)
    print("wrote roll_differential.png")
    plt.show()


def _recede(ax):
    """Push the grid and frame into the background where they belong."""
    ax.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


if __name__ == "__main__":
    main()
