import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make the library in src/ importable when run as `python examples/<name>.py`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from field import PMQ, Lattice
from integrals import (
    b_along_wire,
    effective_length,
    integrated_gradient,
    kick,
    offset_scan,
    offset_sensitivity,
)

#PMQ 1

G1 = 500
L1 = 0.003
Z1 = 0.2
P1 = 'hard'
THETA1 = 0

L_sep = 0.013523

#PMQ 2

G2 = -500
L2 = 0.006
Z2 = Z1 + L_sep
P2 = 'hard'
THETA2 = 0

lat = Lattice([PMQ(G1, L1, Z1, P1, theta=THETA1), PMQ(G2, L2, Z2, P2, theta=THETA2)])

# --- Probe line -------------------------------------------------------------
# A quadrupole field is identically zero on its own axis, so probing at
# (0, 0) returns Bx = By = 0 everywhere.  To see a profile at all the wire must
# be off-axis -- either move the probe (below) or give the magnets dx/dy.
X_PROBE = 0   # [m] wire position relative to the magnetic centers
Y_PROBE = 1e-3

# 'hard' has no ordinary derivatives, so only the leading-order field exists
# for it.  order=2 would raise.  Switch P1/P2 to 'tanh' to use the full model.
ORDER = 0 if "hard" in (P1, P2) else 2

# Window: tight around the two magnets rather than the default 10 fringe
# lengths, since a hard edge has no fringe to show.
z_lo = Z1 - 4 * L1
z_hi = Z2 + 4 * L2

z, bx, by, bz = b_along_wire(
    lat, x=X_PROBE, y=Y_PROBE, z_range=(z_lo, z_hi), n=4001, order=ORDER
)

ix, iy = np.trapezoid(bx, z), np.trapezoid(by, z)
print(f"probe at (x, y) = ({X_PROBE * 1e3:.3f}, {Y_PROBE * 1e3:.3f}) mm, order={ORDER}")
print(f"  int Bx dz = {ix * 1e3:+.4f} mT*m")
print(f"  int By dz = {iy * 1e3:+.4f} mT*m")
print(f"  peak |Bx| = {np.abs(bx).max() * 1e3:.2f} mT   "
      f"peak |By| = {np.abs(by).max() * 1e3:.2f} mT")

# --- Plot -------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834"]
MUTED = "#8a8985"

fig, ax = plt.subplots(figsize=(8.5, 4.4))

# Shade where each magnet sits.
for m, label in zip(lat, ("PMQ 1", "PMQ 2")):
    ax.axvspan(
        (m.z0 - m.L / 2) * 1e3, (m.z0 + m.L / 2) * 1e3,
        color="#f0efec", zorder=0,
    )
    # Above the axes, so the labels cannot collide with the traces.
    ax.annotate(
        label, xy=(m.z0 * 1e3, 1.01), xycoords=("data", "axes fraction"),
        ha="center", va="bottom", fontsize=9, color=MUTED,
    )

ax.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
ax.plot(z * 1e3, bx * 1e3, color=SERIES[0], lw=2.0, zorder=3, label=r"$B_x$")
ax.plot(z * 1e3, by * 1e3, color=SERIES[1], lw=2.0, zorder=3, label=r"$B_y$")

ax.set_xlabel("z [mm]")
ax.set_ylabel("B [mT]")
ax.set_title(
    f"Field along the wire at x = {X_PROBE * 1e3:g} mm, "
    f"roll {np.rad2deg(THETA1):.2f}$^\\circ$ / {np.rad2deg(THETA2):.2f}$^\\circ$",
    loc="left", fontsize=11, pad=18,
)
ax.set_xlim(z_lo * 1e3, z_hi * 1e3)
ax.margins(y=0.15)
ax.legend(frameon=False, fontsize=10)
ax.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#d5d4d0")
ax.tick_params(colors=MUTED, labelsize=9)

fig.tight_layout()
fig.savefig("roll_field_profile.png", dpi=200)
print("wrote roll_field_profile.png")
plt.show()

