"""Worked example: building a global reference frame for the pulsed-wire bench.

The bench has three independent x-y stages -- one under each wire anchor and one
under the PMQ -- and each reads out in its own coordinates.  Nothing ties those
coordinates together:

  * each stage sits at its own small rotation about z, set by how it was bolted
    down (a few degrees is easy to get by eye);
  * a stage turned around to face the operator has its x axis running
    *backwards* relative to the others -- a mirror image, not a rotation;
  * micrometer heads differ in scale, and the two axes of a stacked stage are
    not exactly perpendicular;
  * the zero of every micrometer is arbitrary.

So "move 0.1 mm in x" means three different physical motions, and the null
readings in the run notes -- (6.10, 3.2), (6.2, 5.2), (4.09, 4.92) in
pmq_measurements/5-15-26, 5-20-26-2, 5-29-26-* -- cannot be compared from one
session to the next: the wire moved, the magnet moved, or both.

What a frame needs
------------------
An origin, an orientation, and a length scale.  Each has a physical source on
this bench:

  origin, z    the two anchor wire seats.  They are the only parts of the bench
               that define a straight line, so the *reference axis* is the
               straight line through them at their home readings, and the
               origin is that axis at the PMQ's z.  (A kinematic V-groove seat
               is what makes it survive re-stringing.)
  orientation  gravity.  A digital inclinometer on one stage gives its tilt to
               ~0.05 deg.  The wire's own sag also points along gravity, which
               makes it a free cross-check, though a weak one.
  scale        one trusted micrometer, here the PMQ stage's.

Every other stage is then *cross-calibrated* against that master using the
wire and magnet as the sensor.  That step needs no new hardware, and it is the
one this script simulates in detail.

Why the wire can calibrate the stages
-------------------------------------
The first field integral is exactly linear in the wire's offset from the
magnetic center (see README), so the photodiode signals are

    s = D K (w(z_M) - c)

with c the magnetic center, w(z_M) the wire at the magnet's z, K the magnet's
2x2 response (set by int G dz and its roll) and D the detector's (diode gains
and angles).  The wire runs straight between the anchor seats, so
w(z_M) = (1 - lam) A + lam B + sag, with lam = (z_M - z_A) / (z_B - z_A).  If
each stage maps its reading r to a position as M r + offset, then bumping one
stage axis at a time measures

    J_A = (1 - lam) D K M_A,    J_B = lam D K M_B,    J_PMQ = -D K M_PMQ

and the ratio

    M_PMQ^-1 M_A = -J_PMQ^-1 J_A / (1 - lam)

contains neither D nor K.  The detector gain, diode angles, magnet roll and
field strength all cancel.  The magnet is used as a comparator, never as a
standard.  From outside, only lam is needed, and a tape measure gives it.

What the wire cannot tell you
-----------------------------
Only the wire's position *at the magnet* enters s.  Two things are therefore
invisible to one magnet at one z, and must be fixed by convention or by another
measurement:

  * the separate offsets of the two anchors, since only (1 - lam) A + lam B is
    seen.  Defining the reference axis through the anchor homes is exactly the
    convention that fixes this.
  * the wire's angle through the magnet.  A pivot about z_M leaves the first
    integral unchanged for a symmetric magnet, so a slope sweep made by moving
    one anchor is, to first order, an offset scan with lever arm lam or
    (1 - lam) (panel C).  The magnet's pitch and yaw need a second z (the
    magnet moved along the rail, or a second magnet) or the short-pulse trace
    shape.

The simulation
--------------
A hidden `Bench` holds the truth: how each stage is really mounted, where the
magnet is, the detector, the sag, and the noise.  The experimenter's code
reaches it only through what the real bench reports.  Stage readings go in,
photodiode volts come out, and there are also an inclinometer, a tape measure
and the measured wave speed.  The truth is used only to score the result.

  A.  Each stage's +x/+y in the global frame, recovered vs true.
  B.  A commanded rigid +100 um horizontal shift of the wire: the naive
      command (same micrometer turn on both anchors) vs the calibrated one.
  C.  A slope sweep pivoting about the PMQ, naive vs calibrated, read back as
      the apparent wire offset at the magnet.
  D.  Six sessions with the anchors re-set and the wire re-strung: raw null
      readings wander by hundreds of um, while the magnet's position in the
      global frame holds to a couple of um -- the anchor re-seat floor.

Run:  python examples/example_global_frame.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import simpson

# Make the library in src/ importable when run as `python examples/<name>.py`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from field import PMQ
from integrals import integrated_gradient

# --- Bench geometry ---------------------------------------------------------
# TODO: replace with tape-measured positions.  z runs from the upstream anchor's
# wire seat toward the downstream one.
Z_A = 0.0        # upstream anchor wire seat [m]
Z_M = 0.35       # PMQ center [m]
Z_B = 1.02       # downstream anchor wire seat [m]  (the 1.02 m string, 5-26-26-1)
C0 = 222.0       # wire wave speed [m/s]  (pmq/PMQ_measurements/pmq_analysis.py)
G_ACC = 9.81     # [m/s^2]

# --- Magnet (same placeholders as example_offset_scan.py) -------------------
G0 = 500.0       # peak gradient [T/m]
LMAG = 0.02      # magnet length [m]
FRINGE = 0.15    # tanh fringe scale as a fraction of LMAG

# --- Operator settings ------------------------------------------------------
# Micrometer readings [mm] the stages normally sit at.  Not hidden: these are
# just the numbers the operator dials in.
HOME = {"PMQ": (5.0, 5.0), "A": (6.0, 5.0), "B": (6.0, 5.0)}
STAGES = ("PMQ", "A", "B")

# --- Hidden truth: how each stage is really mounted -------------------------
# TODO: invented but plausible for stages placed by eye; the calibration never
# sees them.  `angle_deg` is the direction of +x travel above true horizontal.
# `origin_um` is where the carried point (the wire seat, or for the PMQ stage
# the magnetic center) sits in the lab when the stage reads HOME.
TRUE_STAGES = {
    "PMQ": dict(angle_deg=2.5, origin_um=(180.0, -120.0)),
    "A": dict(angle_deg=-4.0, origin_um=(40.0, -25.0)),
    # The same stage model turned around to face the operator, so +x runs
    # backwards.  It also carries a different micrometer on y and is not
    # quite square.
    "B": dict(angle_deg=1.5, flip_x=True, scale=(1.0, 0.98), skew_deg=0.8,
              origin_um=(-60.0, 35.0)),
}
TRUE_MAGNET_ROLL_DEG = 1.5
TRUE_DIODE_ANGLE_DEG = (7.0, -4.0)   # x / y diode sensing axes, off true x / y
TRUE_DIODE_GAIN = (100.0, 70.0)      # [V per T*m of field integral]

# --- Measurement quality ----------------------------------------------------
SIGNAL_NOISE_V = 1e-3    # per averaged trace; ~1 um of wire offset at this gain
STAGE_REPEAT_UM = 1.0    # micrometer repeatability, per axis
RESTRING_UM = 2.0        # where an anchor seat lands after re-stringing
INCLINOMETER_DEG = 0.05
TAPE_MM = 1.0
C0_REL = 0.01            # wave-speed measurement, relative

BUMPS_MM = np.linspace(-0.8, 0.8, 9)   # +/-80 divisions, as in the 5-29-26 runs
N_REPEAT = 20                           # repeated calibrations, for error bars
N_SESSIONS = 6
SESSION_SPREAD_MM = 1.0                 # how far the anchors get re-set
SEED = 1

# Categorical slots 1-2 from the validated reference palette.
CAL = "#2a78d6"      # calibrated / global frame
NAIVE = "#eb6834"    # naive / raw readings
INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e6e5e1"


# ============================================================================
# The hidden truth
# ============================================================================

class StageFrame:
    """How one stage really maps its reading [mm] to a lab position [m]."""

    def __init__(self, home, angle_deg, origin_um, flip_x=False,
                 scale=(1.0, 1.0), skew_deg=0.0):
        phi = np.deg2rad(angle_deg)
        psi = phi + np.deg2rad(skew_deg)
        ex = scale[0] * np.array([np.cos(phi), np.sin(phi)])
        ey = scale[1] * np.array([-np.sin(psi), np.cos(psi)])
        if flip_x:
            ex = -ex
        self.M = 1e-3 * np.column_stack([ex, ey])     # m per mm of reading
        self.home = np.asarray(home, dtype=float)
        self.origin = 1e-6 * np.asarray(origin_um, dtype=float)

    def to_lab(self, reading):
        return self.origin + self.M @ (np.asarray(reading, dtype=float) - self.home)


class Bench:
    """The real bench.  Lab frame: x horizontal, y up, z along the wire.

    Experimenter code may call only measure(), inclinometer(), tape_measure(),
    wave_speed() and restring().  Everything else is truth, for scoring.
    """

    def __init__(self, seed, magnet_roll_deg=TRUE_MAGNET_ROLL_DEG,
                 diode_angle_deg=TRUE_DIODE_ANGLE_DEG,
                 diode_gain=TRUE_DIODE_GAIN, g0=G0, noise=True):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.n_measure = 0
        self.stages = {k: StageFrame(HOME[k], **TRUE_STAGES[k]) for k in STAGES}
        self.seat = {"A": np.zeros(2), "B": np.zeros(2)}   # re-stringing error [m]
        self.quad = PMQ(g0, LMAG, z0=Z_M, profile="tanh", d=FRINGE,
                        theta=np.deg2rad(magnet_roll_deg))
        ax, ay = np.deg2rad(diode_angle_deg)
        # Rows are the two diodes' sensing directions, each scaled by its gain.
        self.D = np.array([
            diode_gain[0] * np.array([np.cos(ax), np.sin(ax)]),
            diode_gain[1] * np.array([-np.sin(ay), np.cos(ay)]),
        ])
        # Ten magnet lengths of fringe either side, as field_integral does.
        self.z = np.linspace(max(Z_M - 10 * LMAG, Z_A), min(Z_M + 10 * LMAG, Z_B), 4001)

    def _gauss(self, sigma, n=2):
        return sigma * self.rng.standard_normal(n) if self.noise else np.zeros(n)

    # -- truth --------------------------------------------------------------

    def true_matrix(self, name):
        """The stage's reading -> lab map [um per mm]."""
        return 1e6 * self.stages[name].M

    def seat_lab(self, name, reading):
        return self.stages[name].to_lab(reading) + self.seat[name]

    def wire(self, a, b, z):
        """The wire's lab (x, y) [m] at z: straight between the seats, plus sag.

        Sag at fixed tension is g z (L - z) / 2 c0^2 whatever the wire is made
        of -- the wave speed already carries tension over linear density.
        """
        pa, pb = self.seat_lab("A", a), self.seat_lab("B", b)
        t = (z - Z_A) / (Z_B - Z_A)
        sag = G_ACC * (z - Z_A) * (Z_B - z) / (2.0 * C0**2)
        return pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t - sag

    # -- what the bench reports ---------------------------------------------

    def measure(self, r):
        """Photodiode signals [V] with the stages at readings r["A"], r["B"], r["PMQ"]."""
        self.n_measure += 1
        rep = 1e-3 * STAGE_REPEAT_UM
        a, b, m = (np.asarray(r[k], dtype=float) + self._gauss(rep) for k in ("A", "B", "PMQ"))
        self.quad.dx, self.quad.dy = self.stages["PMQ"].to_lab(m)
        # The full field, integrated along the actual tilted, sagging wire.
        x, y = self.wire(a, b, self.z)
        bx, by, _ = self.quad.B(x, y, self.z)
        ix, iy = simpson(bx, x=self.z), simpson(by, x=self.z)
        # Current along +z: force per length I z x B = I (-By, Bx), the same
        # pairing as integrals.kick.
        return self.D @ np.array([-iy, ix]) + self._gauss(SIGNAL_NOISE_V)

    def inclinometer(self):
        """Tilt of the PMQ stage's +x travel above horizontal [deg], as read."""
        ex = self.stages["PMQ"].M[:, 0]
        return np.rad2deg(np.arctan2(ex[1], ex[0])) + self._gauss(INCLINOMETER_DEG, 1)[0]

    def tape_measure(self):
        """(z_A, z_M, z_B) [m], as read."""
        return np.array([Z_A, Z_M, Z_B]) + self._gauss(1e-3 * TAPE_MM, 3)

    def wave_speed(self):
        return C0 * (1.0 + self._gauss(C0_REL, 1)[0])

    def restring(self):
        """Re-seat the wire in both anchors; each seat lands a few um off."""
        for k in self.seat:
            self.seat[k] = self._gauss(1e-6 * RESTRING_UM)


# ============================================================================
# The experimenter: stage readings and signals only
# ============================================================================

def bump_records(bench, center, stage, bumps=BUMPS_MM):
    """Step one stage along its x then its y; everything else held at `center`."""
    records = []
    for axis in (0, 1):
        for d in bumps:
            r = {k: np.array(v, dtype=float) for k, v in center.items()}
            r[stage][axis] += d
            records.append((r, bench.measure(r)))
    return records


def fit_response(records, center, stages=STAGES):
    """Least-squares fit of  s = s0 + sum_k J_k (r_k - center_k).

    Returns s0, {stage: 2x2 J [V per mm]} and the residual rms [V].
    """
    X = np.array([
        np.concatenate([[1.0]] + [r[k] - center[k] for k in stages])
        for r, _ in records
    ])
    S = np.array([s for _, s in records])
    beta, *_ = np.linalg.lstsq(X, S, rcond=None)
    J = beta[1:].T
    jac = {k: J[:, 2 * i:2 * i + 2] for i, k in enumerate(stages)}
    return beta[0], jac, np.std(S - X @ beta)


def find_null(bench, readings, span=0.2, iters=2):
    """Null both signals with the PMQ stage -- how (Xeq, Yeq) is found now."""
    r = {k: np.asarray(v, dtype=float) for k, v in readings.items()}
    for _ in range(iters):
        recs = bump_records(bench, r, "PMQ", np.linspace(-span, span, 5))
        s0, jac, _ = fit_response(recs, r, stages=("PMQ",))
        r["PMQ"] = r["PMQ"] - np.linalg.solve(jac["PMQ"], s0)
    return r["PMQ"]


class GlobalFrame:
    """The experimenter's global frame, built only from bench measurements.

    x horizontal and y up (gravity, from the inclinometer on the PMQ stage).
    z runs along the reference axis, the straight line through the two anchor
    wire seats at their home readings.  The origin is that axis at the PMQ's z.
    Positions are in um; every stage map M[k] is in um per mm of reading.
    """

    def __init__(self, r0, jac, z, c0, pmq_tilt_deg, resid):
        z_a, z_m, z_b = z
        self.z = z
        self.lam = (z_m - z_a) / (z_b - z_a)
        self.sag = 1e6 * G_ACC * (z_m - z_a) * (z_b - z_m) / (2.0 * c0**2)
        self.home = {k: np.asarray(v, dtype=float) for k, v in r0.items()}
        self.resid = resid
        # The master: trusted micrometer scale, square axes, tilt from the level.
        # This is the one stage the wire cannot calibrate -- everything else
        # inherits its scale.
        t = np.deg2rad(pmq_tilt_deg)
        self.M = {"PMQ": 1e3 * np.array([[np.cos(t), -np.sin(t)],
                                         [np.sin(t), np.cos(t)]])}
        # The anchors, through the wire.  D and K cancel in J_PMQ^-1 J_k.
        self.J_pmq = jac["PMQ"]
        to_pmq = -np.linalg.inv(self.J_pmq)
        self.M["A"] = self.M["PMQ"] @ to_pmq @ jac["A"] / (1.0 - self.lam)
        self.M["B"] = self.M["PMQ"] @ to_pmq @ jac["B"] / self.lam

    def wire_at_magnet(self, r):
        """Wire position at the PMQ's z [um], from the anchor readings alone."""
        da = self.M["A"] @ (np.asarray(r["A"]) - self.home["A"])
        db = self.M["B"] @ (np.asarray(r["B"]) - self.home["B"])
        return (1.0 - self.lam) * da + self.lam * db + np.array([0.0, -self.sag])

    def magnet(self, r_null, m):
        """Magnetic center [um] with the PMQ stage at m, from any nulled setting."""
        return self.wire_at_magnet(r_null) + self.M["PMQ"] @ (m - r_null["PMQ"])

    def predict_null(self, a, b):
        """PMQ reading that will null the signals, before measuring it."""
        shift = self.wire_at_magnet({"A": a, "B": b}) - self.wire_at_magnet(self.home)
        return self.home["PMQ"] + np.linalg.solve(self.M["PMQ"], shift)

    def anchor_readings(self, da_um, db_um):
        """Anchor readings that move the two seats by da, db [um] in this frame."""
        return (self.home["A"] + np.linalg.solve(self.M["A"], da_um),
                self.home["B"] + np.linalg.solve(self.M["B"], db_um))

    def wire_offset(self, s):
        """Wire position relative to the magnetic center [um], read from signals."""
        return self.M["PMQ"] @ -np.linalg.solve(self.J_pmq, s)


def calibrate(bench, home):
    """The whole procedure: null at home, bump all six axes, then build the frame."""
    r0 = {k: np.asarray(v, dtype=float) for k, v in home.items()}
    r0["PMQ"] = find_null(bench, r0)
    records = []
    for stage in STAGES:
        records += bump_records(bench, r0, stage)
    _, jac, resid = fit_response(records, r0)
    return GlobalFrame(r0, jac, bench.tape_measure(), bench.wave_speed(),
                       bench.inclinometer(), resid)


def _axes(M):
    """(+x angle, +y angle [deg from global x], x scale, y scale) of a stage map."""
    ang = [((np.rad2deg(np.arctan2(v[1], v[0])) + 90.0) % 360.0) - 90.0 for v in M.T]
    return np.array([ang[0], ang[1],
                     np.linalg.norm(M[:, 0]) / 1e3, np.linalg.norm(M[:, 1]) / 1e3])


# ============================================================================

def main():
    home = {k: np.asarray(v, dtype=float) for k, v in HOME.items()}
    bench = Bench(SEED)
    lam = (Z_M - Z_A) / (Z_B - Z_A)
    sag = 1e6 * G_ACC * (Z_M - Z_A) * (Z_B - Z_M) / (2.0 * C0**2)

    print(f"\nbench: anchor seats at z = {Z_A:.2f} and {Z_B:.2f} m, PMQ at "
          f"{Z_M:.2f} m  (lever arm lam = {lam:.3f})")
    print(f"  PMQ int G dz = {integrated_gradient(bench.quad):.3f} T;  "
          f"wire sag at the PMQ = {sag:.1f} um")
    print(f"  noise: {SIGNAL_NOISE_V * 1e3:g} mV per trace, {STAGE_REPEAT_UM:g} um "
          f"stage repeatability, {RESTRING_UM:g} um anchor re-seat\n")

    # --- 1. Cross-calibration -------------------------------------------------
    frame = calibrate(bench, home)
    n_cal = bench.n_measure
    reps = [calibrate(Bench(SEED + 1 + k), home) for k in range(N_REPEAT)]
    spread = {k: np.std([_axes(f.M[k]) for f in reps], axis=0) for k in STAGES}

    print(f"1. Cross-calibration: {n_cal} traces (null at home, then "
          f"{BUMPS_MM.size}-point bumps on all six axes)")
    print(f"   fit residual {frame.resid * 1e3:.2f} mV rms: {SIGNAL_NOISE_V * 1e3:g} mV "
          f"signal noise plus {STAGE_REPEAT_UM:g} um stage repeatability")
    print(f"   from outside: lam = {frame.lam:.4f} (tape), sag = {frame.sag:.1f} um "
          f"(measured c0), PMQ tilt {_axes(frame.M['PMQ'])[0]:+.2f} deg (level)")
    print(f"   recovered +/- spread over {N_REPEAT} repeat calibrations  [true]")
    print("   stage        +x axis [deg]            +y axis [deg]          "
          "scale x / y             handed")
    for k in STAGES:
        rec, true, err = _axes(frame.M[k]), _axes(bench.true_matrix(k)), spread[k]
        mirrored = np.linalg.det(frame.M[k]) < 0
        print(f"   {k:<5} {rec[0]:8.2f} ±{err[0]:.2f} [{true[0]:7.2f}]   "
              f"{rec[1]:7.2f} ±{err[1]:.2f} [{true[1]:6.2f}]   "
              f"{rec[2]:.4f}/{rec[3]:.4f} [{true[2]:.3f}/{true[3]:.3f}]   "
              f"{'MIRRORED' if mirrored else 'right'}")

    # --- 2. The detector and magnet really do drop out ------------------------
    # With the noise off, whatever is left is the method, not the statistics.
    ref = calibrate(Bench(SEED, noise=False), home)
    exact = max(np.abs(ref.M[k] - bench.true_matrix(k)).max() for k in ("A", "B")) / 1e3
    print(f"\n2. Noise off: fit residual {ref.resid:.1e} V (the response is linear), "
          f"and the anchor maps match\n   the true mounting to {exact:.1e} relative.  "
          "Changing what the frame should not depend on:")
    variants = (
        ("diodes at (20, -15) deg, 3x gain", dict(diode_angle_deg=(20.0, -15.0),
                                                  diode_gain=(300.0, 210.0))),
        ("magnet rolled 20 deg", dict(magnet_roll_deg=20.0)),
        ("half-strength magnet", dict(g0=G0 / 2)),
    )
    for label, kw in variants:
        f = calibrate(Bench(SEED, noise=False, **kw), home)
        d = max(np.abs(f.M[k] - ref.M[k]).max() for k in ("A", "B")) / 1e3
        print(f"   {label:<34} anchor maps change by {d:.1e} relative")

    # --- 3. Commands: a rigid shift, and a pivot about the PMQ ----------------
    delta = np.array([100.0, 0.0])                     # um, global frame
    commands = {
        "naive": (home["A"] + delta / 1e3, home["B"] + delta / 1e3),
        "calibrated": frame.anchor_readings(delta, delta),
    }
    zz = np.linspace(Z_A, Z_B, 300)
    x0, y0 = bench.wire(home["A"], home["B"], zz)
    shift_err = {}
    print(f"\n3. Command: shift the wire rigidly by (+{delta[0]:g}, {delta[1]:g}) um")
    for name, (a, b) in commands.items():
        x, y = bench.wire(a, b, zz)
        ex, ey = (x - x0) * 1e6 - delta[0], (y - y0) * 1e6 - delta[1]
        shift_err[name] = (ex, ey)
        i_m = np.argmin(np.abs(zz - Z_M))
        tilt = np.hypot(ex[-1] - ex[0], ey[-1] - ey[0]) / (Z_B - Z_A)
        print(f"   {name:<11} at the PMQ: ({delta[0] + ex[i_m]:+7.1f}, "
              f"{delta[1] + ey[i_m]:+6.1f}) um;  unwanted tilt {tilt:6.1f} urad")

    # Pivot about the PMQ: a pure slope sweep, commanded with tape-measured z.
    slopes = np.linspace(-1000.0, 1000.0, 9)            # urad, horizontal
    z_a, z_m, z_b = frame.z
    pivot = {"naive": [], "calibrated": []}
    for sl in slopes:
        da = np.array([-sl * (z_m - z_a), 0.0])        # urad * m = um
        db = np.array([+sl * (z_b - z_m), 0.0])
        cmd = {
            "naive": (home["A"] + da / 1e3, home["B"] + db / 1e3),
            "calibrated": frame.anchor_readings(da, db),
        }
        for name, (a, b) in cmd.items():
            s = bench.measure({"A": a, "B": b, "PMQ": frame.home["PMQ"]})
            pivot[name].append(frame.wire_offset(s))
    pivot = {k: np.array(v) for k, v in pivot.items()}
    print("   pivot about the PMQ, +/-1 mrad: apparent wire offset at the magnet")
    for name, off in pivot.items():
        gx, gy = (np.polyfit(slopes * 1e-3, off[:, j], 1)[0] for j in (0, 1))
        print(f"   {name:<11} ({gx:+7.1f}, {gy:+6.1f}) um per mrad of slope")

    # --- 4. Sessions: re-set the anchors, re-string, re-null ------------------
    # The truth to score against: the magnet at the PMQ home reading, relative
    # to the reference axis as it stood during calibration.
    axis_m = (1 - lam) * bench.seat_lab("A", home["A"]) + lam * bench.seat_lab("B", home["B"])
    c_true = 1e6 * (bench.stages["PMQ"].to_lab(home["PMQ"]) - axis_m)

    rng = np.random.default_rng(SEED + 1000)
    rows = []
    for _ in range(N_SESSIONS):
        bench.restring()
        a = home["A"] + rng.uniform(-SESSION_SPREAD_MM, SESSION_SPREAD_MM, 2)
        b = home["B"] + rng.uniform(-SESSION_SPREAD_MM, SESSION_SPREAD_MM, 2)
        # Predict first, then measure.
        naive_pred = frame.home["PMQ"] + (1 - frame.lam) * (a - home["A"]) \
            + frame.lam * (b - home["B"])
        cal_pred = frame.predict_null(a, b)
        m_null = find_null(bench, {"A": a, "B": b, "PMQ": home["PMQ"]})
        c_hat = frame.magnet({"A": a, "B": b, "PMQ": m_null}, home["PMQ"])
        rows.append((m_null, naive_pred, cal_pred, c_hat))

    raw = np.array([r[0] for r in rows])
    naive_err = 1e3 * np.linalg.norm(np.array([r[1] for r in rows]) - raw, axis=1)
    cal_err = 1e3 * np.linalg.norm(np.array([r[2] for r in rows]) - raw, axis=1)
    glob = np.array([r[3] for r in rows])
    raw_dev = 1e3 * np.linalg.norm(raw - raw.mean(axis=0), axis=1)
    glob_dev = np.linalg.norm(glob - glob.mean(axis=0), axis=1)

    print(f"\n4. {N_SESSIONS} sessions: anchors re-set by up to "
          f"±{SESSION_SPREAD_MM:g} mm, wire re-strung each time")
    print("   session   null (Xeq, Yeq) [mm]   predicted null, error [um]    "
          "magnet @ PMQ home, global [um]")
    print("                                        naive    calibrated")
    for i, (m_null, _, _, c_hat) in enumerate(rows):
        print(f"      {i + 1}      ({m_null[0]:.3f}, {m_null[1]:.3f})        "
              f"{naive_err[i]:7.1f}     {cal_err[i]:5.1f}          "
              f"({c_hat[0]:+7.1f}, {c_hat[1]:+7.1f})")
    print(f"   rms spread: raw null {np.sqrt(np.mean(raw_dev**2)):.0f} um,  "
          f"global magnet {np.sqrt(np.mean(glob_dev**2)):.1f} um")
    print(f"   global magnet vs truth ({c_true[0]:+.1f}, {c_true[1]:+.1f}) um:  "
          f"mean error {np.linalg.norm(glob.mean(axis=0) - c_true):.1f} um\n")

    # --- Figure -----------------------------------------------------------------
    fig = plt.figure(figsize=(13.5, 8.8), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0])

    titles = {"PMQ": "A.  PMQ stage (master)", "A": "Anchor A (upstream)",
              "B": "Anchor B (downstream)"}
    for j, k in enumerate(STAGES):
        _compass(fig.add_subplot(gs[0, j]), frame.M[k], bench.true_matrix(k),
                 spread[k], titles[k], legend=(j == 0))

    # --- Panel B: rigid shift -----------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    _mark_bench(ax_b)
    ax_b.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    for name, color in (("naive", NAIVE), ("calibrated", CAL)):
        ex, ey = shift_err[name]
        ax_b.plot(zz, ex, color=color, lw=2.0, zorder=3, label=f"{name}, x")
        ax_b.plot(zz, ey, color=color, lw=2.0, ls="--", zorder=3, label=f"{name}, y")
    ax_b.set_xlabel("z along the wire [m]")
    ax_b.set_ylabel("achieved − intended wire shift [µm]")
    ax_b.set_title(f"B.  Command: shift the wire +{delta[0]:g} µm in x",
                   loc="left", fontsize=11, pad=18)
    ax_b.set_xlim(Z_A, Z_B)
    ax_b.margins(y=0.12)
    ax_b.legend(frameon=False, fontsize=8.5, loc="lower left")
    _recede(ax_b)

    # --- Panel C: pivot -----------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    for name, color in (("naive", NAIVE), ("calibrated", CAL)):
        off = pivot[name]
        ax_c.plot(slopes, off[:, 0], "o-", color=color, lw=2.0, ms=6, mec="white",
                  mew=1.5, zorder=3, label=f"{name}, x")
        ax_c.plot(slopes, off[:, 1], "o--", color=color, lw=2.0, ms=6, mec="white",
                  mew=1.5, zorder=3, label=f"{name}, y")
    ax_c.annotate("calibrated pivot: first integral\nstays null, as it should",
                  xy=(slopes[-3], 0.0), xytext=(0, 14), textcoords="offset points",
                  ha="center", va="bottom", fontsize=8.5, color=INK)
    ax_c.set_xlabel("commanded horizontal slope about the PMQ [µrad]")
    ax_c.set_ylabel("wire offset at the magnet, from signals [µm]")
    ax_c.set_title("C.  Slope sweep pivoting about the PMQ", loc="left",
                   fontsize=11, pad=18)
    ax_c.margins(y=0.12)
    # A long handle, so the ringed marker is not mistaken for a dash.
    ax_c.legend(frameon=False, fontsize=8.5, loc="lower left", handlelength=3.5)
    _recede(ax_c)

    # --- Panel D: session to session ----------------------------------------
    ax_d = fig.add_subplot(gs[1, 2])
    groups = (
        ("raw null reading: spread between sessions", raw_dev, NAIVE),
        ("magnet in global frame: spread", glob_dev, CAL),
        ("null predicted from anchors, naive: error", naive_err, NAIVE),
        ("null predicted, calibrated: error", cal_err, CAL),
    )
    for row, (label, vals, color) in enumerate(groups):
        yrow = -row
        ax_d.plot(vals, np.full_like(vals, yrow), "o", color=color, ms=8,
                  mec="white", mew=1.5, zorder=3)
        rms = np.sqrt(np.mean(vals**2))
        ax_d.plot([rms, rms], [yrow - 0.18, yrow + 0.18], color=INK, lw=1.5, zorder=4)
        ax_d.annotate(f"{label}   (rms {rms:.3g} µm)", xy=(0.12, yrow + 0.24),
                      fontsize=8.5, color=INK, va="bottom")
    ax_d.set_xscale("log")
    ax_d.set_xlim(0.1, 5000)
    ax_d.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    ax_d.minorticks_off()
    ax_d.set_ylim(-len(groups) + 0.5, 0.75)
    ax_d.set_yticks([])
    ax_d.set_xlabel("distance [µm], log scale")
    ax_d.set_title(f"D.  {N_SESSIONS} sessions, anchors re-set and re-strung",
                   loc="left", fontsize=11, pad=18)
    _recede(ax_d)
    ax_d.spines["left"].set_visible(False)
    ax_d.grid(True, axis="x", which="major", color=GRID, lw=0.8, zorder=0)
    ax_d.grid(False, axis="y")

    fig.savefig("global_frame.png", dpi=200)
    print("wrote global_frame.png")
    plt.show()


def _compass(ax, rec, true, err, title, legend=False):
    """One stage's +x/+y, recovered (arrows) vs true (circles), in the global frame."""
    # adjustable="datalim" widens the x range to fill the slot instead of
    # shrinking the box, which keeps constrained layout from clipping titles.
    # The extent is set through the data limits, since fixed limits would
    # fight the aspect.
    ax.update_datalim([(-1.55, -1.1), (1.55, 1.55)])
    ax.margins(0)
    ax.set_aspect("equal", adjustable="datalim")
    for v, lab in (((1.35, 0.0), "global x"), ((0.0, 1.35), "global y")):
        ax.annotate("", xy=v, xytext=(0, 0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=GRID, lw=1.2,
                                    shrinkA=0, shrinkB=0))
        # Past the arrow end, clear of the stage arrows and their true circles.
        ax.annotate(lab, xy=v, xytext=(4, 0) if v[1] == 0 else (0, 4),
                    textcoords="offset points", fontsize=8, color=MUTED,
                    ha="left" if v[1] == 0 else "center",
                    va="center" if v[1] == 0 else "bottom")
    for col, lab in ((0, "+x"), (1, "+y")):
        v, t = rec[:, col] / 1e3, true[:, col] / 1e3
        ax.plot(*t, "o", ms=13, mfc="none", mec=INK, mew=1.2, zorder=2)
        ax.annotate("", xy=v, xytext=(0, 0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=CAL, lw=2.0,
                                    mutation_scale=14, shrinkA=0, shrinkB=0))
        # Label beside the shaft, on its counter-clockwise side, clear of the
        # global axis labels at the ends.
        u = v / np.linalg.norm(v)
        ax.annotate(lab, xy=0.6 * v, xytext=(-11 * u[1], 11 * u[0]),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=10, color=INK)
    r, tr = _axes(rec), _axes(true)
    lines = [
        f"+x  {r[0]:7.2f}° ±{err[0]:.2f}   true {tr[0]:7.2f}°",
        f"+y  {r[1]:7.2f}° ±{err[1]:.2f}   true {tr[1]:7.2f}°",
        f"scale {r[2]:.3f} / {r[3]:.3f}   true {tr[2]:.3f} / {tr[3]:.3f}",
    ]
    if np.linalg.det(rec) < 0:
        lines.append("mirrored: +x runs backwards")
    ax.text(0.5, 0.02, "\n".join(lines), transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.5, color=INK, family="monospace")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if legend:
        handles = [
            Line2D([], [], color=CAL, lw=2.0, label="recovered (wire + level)"),
            Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK,
                   label="true mounting"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")


def _mark_bench(ax):
    """Shade the anchor and PMQ positions along z, labelled above the axes."""
    for z, label in ((Z_A, "anchor A"), (Z_M, "PMQ"), (Z_B, "anchor B")):
        ax.axvline(z, color=GRID, lw=1.5, zorder=0)
        ax.annotate(label, xy=(z, 1.01), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8.5, color=MUTED)


def _recede(ax):
    """Push the grid and frame into the background where they belong."""
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
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
