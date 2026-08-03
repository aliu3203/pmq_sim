"""Worked example: how PMQ transverse position sets the kick on the wire.

Runs the same +/-0.8 mm displacement sweep as the 5-29-26-* bench campaign and
produces two panels:

  A.  By(z) along the wire at several magnet offsets -- the spatial field
      profile the current pulse actually sees, and the null at zero offset that
      defines the magnetic center.

  B.  The first field integral versus magnet offset, with the analytic
      prediction -(int G dz) * offset overlaid.  These lie on top of each other
      to machine precision, which is the point: the whole position-to-kick
      relationship is one number.

Run:  python example_offset_scan.py
"""

import matplotlib.pyplot as plt
import numpy as np

from field import PMQ
from integrals import (
    b_along_wire,
    effective_length,
    integrated_gradient,
    kick,
    offset_scan,
    offset_sensitivity,
)

# --- Magnet parameters ------------------------------------------------------
# TODO: replace with the measured numbers for the PMQ on the bench.  Every
# result below scales with these; none of the physics depends on them.
G0 = 500.0       # peak gradient [T/m]
LMAG = 0.02      # magnet length [m]
FRINGE = 0.15    # fringe scale as a fraction of LMAG
P_EV = 8.1e6     # beam momentum [eV/c], matching the 7.6 MeV in tracking2.m

# Bench sweep: +/-0.8 mm, the range covered by the 5-29-26-* runs.
SWEEP_MM = 0.8

# Categorical slots 1-3 from the validated reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#8a8985"


def main():
    quad = PMQ(G0, LMAG, profile="tanh", d=FRINGE)

    l_eff = effective_length(quad)
    gl = integrated_gradient(quad)
    slope = offset_sensitivity(quad, axis="x", span=SWEEP_MM * 1e-3)

    print(f"\nPMQ:  G0 = {G0} T/m, L = {LMAG * 1e3:.1f} mm, tanh fringe d = {FRINGE}")
    print(f"  effective length      L_eff = {l_eff * 1e3:.4f} mm  ({l_eff / LMAG:.6f} L)")
    print(f"  integrated gradient   int G dz = {gl:.4f} T")
    print(f"  offset sensitivity    d(int By dz)/d(offset) = {slope:.4f} T")
    print(f"    ... which is -(int G dz) to {abs(slope + gl) / abs(gl):.1e} relative")
    iy_100um = slope * 100e-6
    dxp, _ = kick(0.0, iy_100um, P_EV)
    print(
        f"  a 100 um misalignment gives int By dz = {iy_100um * 1e3:+.4f} mT*m"
        f"  =  {dxp * 1e3:+.4f} mrad at {P_EV / 1e6:.1f} MeV/c"
    )
    print("    (thin-lens; a magnet this strong deflects ~3x more once tracked)\n")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Panel A: field along the wire at several offsets -------------------
    ax_a.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    # The zero-offset trace is the flat line itself: on the magnetic center the
    # wire sees no field anywhere, which is the null the alignment looks for.
    ax_a.annotate(
        "0.0 mm — wire on the magnetic center",
        xy=(0.02, 0.93),
        xycoords="axes fraction",
        fontsize=8.5,
        color=MUTED,
        va="center",
    )

    for color, off_mm in zip(SERIES, (0.2, 0.4, 0.8)):
        quad.dx = off_mm * 1e-3
        z, _, by, _ = b_along_wire(quad, x=0.0, y=0.0, n=2001)
        ax_a.plot(z * 1e3, by * 1e3, color=color, lw=2.0, zorder=3)
        # Direct labels: the aqua slot sits below 3:1 contrast on a light
        # surface, so visible labels are required rather than optional.
        i = np.argmin(by)
        ax_a.annotate(
            f"{off_mm} mm",
            xy=(z[i] * 1e3, by[i] * 1e3),
            xytext=(4, -2),
            textcoords="offset points",
            fontsize=9,
            color=color,
            va="top",
        )
    quad.dx = 0.0

    ax_a.set_xlabel("z [mm]")
    ax_a.set_ylabel(r"$B_y$ along the wire [mT]")
    ax_a.set_title("A.  Field seen by the wire, by magnet offset", loc="left", fontsize=11)
    ax_a.set_xlim(-2 * LMAG * 1e3, 2 * LMAG * 1e3)
    ax_a.margins(y=0.12)
    _recede(ax_a)

    # --- Panel B: field integral vs offset ----------------------------------
    offsets = np.linspace(-SWEEP_MM * 1e-3, SWEEP_MM * 1e-3, 17)
    _, iy = offset_scan(quad, offsets, axis="x")

    ax_b.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax_b.axvline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax_b.plot(
        offsets * 1e3,
        -gl * offsets * 1e3,
        color=MUTED,
        lw=2.0,
        zorder=2,
        label=r"analytic  $-(\int G\,dz)\cdot\Delta$",
    )
    ax_b.plot(
        offsets * 1e3,
        iy * 1e3,
        "o",
        color=SERIES[0],
        ms=6,
        mew=1.5,
        mec="white",
        zorder=3,
        label="simulated field integral",
    )

    resid = np.abs(iy - (-gl * offsets)).max() / np.abs(iy).max()
    ax_b.annotate(
        f"deviation from linear: {resid:.0e}",
        xy=(0.04, 0.06),
        xycoords="axes fraction",
        fontsize=9,
        color=INK,
    )

    ax_b.set_xlabel("PMQ transverse offset [mm]")
    ax_b.set_ylabel(r"$\int B_y\,dz$ [mT$\cdot$m]")
    ax_b.set_title("B.  Kick versus position is exactly linear", loc="left", fontsize=11)
    ax_b.legend(frameon=False, fontsize=9, loc="upper right")
    _recede(ax_b)

    fig.tight_layout()
    fig.savefig("offset_scan.png", dpi=200)
    print("wrote offset_scan.png")
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
