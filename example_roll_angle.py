"""Worked example: what a roll angle does to the quadrupole field profile.

A roll is a rotation of the magnet about the z (beam) axis.  Because the field
is a quadrupole, a mechanical roll of theta rotates the whole field *pattern* by
2*theta.  The leading-order lab field for a wire at (x, y) with the magnet
on-axis is

    Bx = G(z) [ -sin(2 theta) x + cos(2 theta) y ]
    By = G(z) [  cos(2 theta) x + sin(2 theta) y ]

Two things follow, and this script shows both:

  A.  Along a wire at fixed (x, y) the By trace scales as cos(2 theta) while a
      Bx trace grows as sin(2 theta).  The magnet is progressively turned from a
      *normal* quadrupole into a *skew* one; at 45 deg it is purely skew.

  B.  The transverse field magnitude |Bt| = |G| sqrt(x^2 + y^2) at any point is
      independent of theta -- the roll only turns the field, it does not weaken
      it.  The normal and skew integrated gradients trace out cos(2 theta) and
      sin(2 theta), summing in quadrature to a constant.

Run:  python example_roll_angle.py
"""

import matplotlib.pyplot as plt
import numpy as np

from field import PMQ
from integrals import b_along_wire, field_integral, integrated_gradient

# --- Magnet parameters ------------------------------------------------------
G0 = 500.0       # peak gradient [T/m]
LMAG = 0.02      # magnet length [m]
FRINGE = 0.15    # tanh fringe scale as a fraction of LMAG

# Wire held off-axis so the quadrupole shows a profile at all (on the axis a
# quadrupole field is identically zero).  On the x axis, By reads the normal
# response and Bx reads the skew response directly.
X_PROBE = 1e-3
Y_PROBE = 0.0

ROLLS_DEG = (0.0, 22.5, 45.0)     # traces to draw in panel A

# Categorical slots 1-3 from the validated reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#8a8985"


def main():
    quad = PMQ(G0, LMAG, profile="tanh", d=FRINGE)
    gl = integrated_gradient(quad)  # int G dz = G0 * L_eff

    print(f"\nPMQ:  G0 = {G0} T/m, L = {LMAG * 1e3:.1f} mm, tanh fringe d = {FRINGE}")
    print(f"  integrated gradient   int G dz = {gl:.4f} T")
    print(f"  wire at (x, y) = ({X_PROBE * 1e3:g}, {Y_PROBE * 1e3:g}) mm\n")
    print("  roll     int By dz / x   int Bx dz / x     quadrature")
    print("  [deg]      (normal)         (skew)          [T]")
    for deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        quad.theta = np.deg2rad(deg)
        ix, iy = field_integral(quad, x=X_PROBE, y=Y_PROBE)
        normal, skew = iy / X_PROBE, ix / X_PROBE
        print(f"  {deg:5.1f}   {normal:+10.4f}     {skew:+10.4f}      "
              f"{np.hypot(normal, skew):8.4f}")
    quad.theta = 0.0
    print()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Panel A: field along the wire at several roll angles ---------------
    ax_a.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    for color, deg in zip(SERIES, ROLLS_DEG):
        quad.theta = np.deg2rad(deg)
        z, bx, by, _ = b_along_wire(quad, x=X_PROBE, y=Y_PROBE, n=2001)
        ax_a.plot(z * 1e3, by * 1e3, color=color, lw=2.0, zorder=3)
        ax_a.plot(z * 1e3, bx * 1e3, color=color, lw=2.0, ls="--", zorder=3)
        i = np.argmax(np.abs(by) + np.abs(bx))
        ax_a.annotate(
            f"{deg:g}°",
            xy=(z[i] * 1e3, max(by[i], bx[i]) * 1e3),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=9,
            color=color,
            va="bottom",
        )
    quad.theta = 0.0

    ax_a.annotate(
        "solid $B_y$ (normal) · dashed $B_x$ (skew)",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8.5, color=MUTED,
    )
    ax_a.set_xlabel("z [mm]")
    ax_a.set_ylabel(r"$B$ along the wire [mT]")
    ax_a.set_title("A.  Roll swaps $B_y$ into $B_x$", loc="left", fontsize=11)
    ax_a.set_xlim(-2 * LMAG * 1e3, 2 * LMAG * 1e3)
    ax_a.margins(y=0.18)
    _recede(ax_a)

    # --- Panel B: normal / skew integrated gradient vs roll -----------------
    rolls = np.linspace(0.0, 90.0, 91)
    normal = np.empty_like(rolls)
    skew = np.empty_like(rolls)
    for i, deg in enumerate(rolls):
        quad.theta = np.deg2rad(deg)
        ix, iy = field_integral(quad, x=X_PROBE, y=Y_PROBE)
        normal[i] = iy / X_PROBE
        skew[i] = ix / X_PROBE
    quad.theta = 0.0

    ax_b.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax_b.plot(rolls, normal, color=SERIES[0], lw=2.0, zorder=3,
              label=r"normal  $\int B_y\,dz/x \;=\; \int G\,dz\,\cos 2\theta$")
    ax_b.plot(rolls, skew, color=SERIES[1], lw=2.0, zorder=3,
              label=r"skew  $\int B_x\,dz/x \;=\; -\int G\,dz\,\sin 2\theta$")
    ax_b.plot(rolls, np.hypot(normal, skew), color=MUTED, lw=1.5, ls=":",
              zorder=2, label=r"quadrature $=\int G\,dz$ (constant)")

    ax_b.axvline(45.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax_b.annotate(
        "45° — pure skew,\nnormal null",
        xy=(45.0, 0.0), xytext=(46, 2.0),
        fontsize=8.5, color=INK, va="bottom",
    )
    ax_b.set_xlabel("roll angle θ [deg]")
    ax_b.set_ylabel(r"integrated gradient [T]")
    ax_b.set_title("B.  Strength conserved, split by $2\\theta$", loc="left", fontsize=11)
    ax_b.set_xlim(0, 90)
    ax_b.set_xticks([0, 22.5, 45, 67.5, 90])
    ax_b.legend(frameon=False, fontsize=8.5, loc="lower left")
    _recede(ax_b)

    fig.tight_layout()
    fig.savefig("roll_angle.png", dpi=200)
    print("wrote roll_angle.png")
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
