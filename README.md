# pmq_sim — magnetics layer

A from-scratch, Maxwell-consistent soft-edge quadrupole field model, built to answer
one question quantitatively: **how does PMQ transverse position affect the kick
generated on a wire threading the bore?**

This is the first layer only. No wire dynamics, no detector model — see
[Not modelled here](#not-modelled-here).

## Coordinate system

Right-handed Cartesian `(x, y, z)`, with **z along the wire** — the propagation
direction, and the beam direction in the eventual application. `x` is horizontal,
`y` vertical. All lengths in metres, fields in tesla, gradients in T/m; the plotting
scripts convert to mm and mT at the last moment.

```
        y
        |          . z  (wire / beam direction, into the magnet)
        |        .
        |      .
        |    .
        |  .
        +--------------- x
```

There are **two transverse frames**, and keeping them apart is what the `dx`/`dy`
parameters are for:

| frame | coords | meaning |
|---|---|---|
| **lab / reference axis** | `x, y` | fixed by the wire. `(0, 0)` is where the wire runs. |
| **magnet** | `u = x − dx`, `v = y − dy`, then rolled by `θ` | centred on the magnet's magnetic center |

`dx, dy` are the **magnetic center's position in lab coordinates** — i.e. how far the
magnet has been moved off the wire, which is exactly the bench knob being scanned.
The field formulas are evaluated in the magnet frame and the resulting vector is
rotated back to the lab frame.

This is why the sign works out the way it does. With the wire at `(0, 0)` and the
magnet displaced to `+dx`, the wire sits at `u = −dx` in the magnet frame, so

```
∫By dz = (∫G dz)·u = −(∫G dz)·dx
```

and `offset_sensitivity` returns **−∫G dz**, not `+∫G dz`. Moving the magnet right is
equivalent to moving the wire left.

`z0` is the magnet center along z; `z` is measured in the same lab frame as `z0`, so
a lattice just places magnets at different `z0`.

### Roll

`theta` is a right-handed rotation about **+z**. Internally the probe point is
rotated *into* the magnet frame by `−θ` and the resulting field vector rotated back
by `+θ`. At leading order this reproduces the closed form in
`quadrupole/tracking2.m:237` exactly (check 8):

```
Bx = G·(v·cos2θ − u·sin2θ)
By = G·(u·cos2θ + v·sin2θ)
```

Deriving it as a coordinate rotation rather than hard-coding `cos2θ`/`sin2θ` means
the `G''` terms rotate correctly too, which the `2θ` form does not generalise to.

### Field and force signs

`G0` is defined by **`By = G0·x`** on the midplane. From the Lorentz force with
`v = (0, 0, vz)` and `B = (Bx, By, 0)`:

```
Fx = q(v×B)_x = −q·vz·By        dx' = −q·(∫By dz)/p
Fy = q(v×B)_y = +q·vz·Bx        dy' = +q·(∫Bx dz)/p
```

Note the two planes carry **opposite signs**, and each is driven by the *other*
field component. `integrals.kick` therefore takes both integrals at once so the
pairing cannot be got wrong.

Consequences worth stating outright, because they are easy to assume backwards:

- **Positive `G0` focuses a *positively* charged particle in x.** For an electron
  the sense reverses — positive `G0` is **defocusing in x**, focusing in y. (The
  `% Quad 1 (Focusing in X)` comment beside `G1 = +500` in `quadrupole/tracking2.m`
  has this backwards; tracking that field with the same file's own equations of
  motion defocuses in x.)
- `kick()` is a **thin-lens** result. At `G0 = 500 T/m`, `L = 20 mm` the tracked
  deflection is ~3× larger, because the deflected particle drifts into the
  defocusing field and is deflected further. It converges to the thin-lens value as
  the gradient weakens (ratio 1.0014 at `G0 = 0.5 T/m`). The *field integral* is
  exact regardless; only its interpretation as an angle is approximate.

## The model

The field is derived from a scalar potential rather than written down component by
component, so Maxwell consistency is structural rather than something checked after
the fact. In the current-free bore **B** = ∇ψ with ∇²ψ = 0, and for a normal
quadrupole with longitudinal gradient profile `G(z)`:

$$\psi(x,y,z) = G(z)\,xy \;-\; \frac{G''(z)}{12}\,xy\,(x^2+y^2)$$

The second term is fixed by requiring ∇²ψ = 0: the first term contributes
`G''(z)·xy` through `∂²/∂z²`, and `∇²⊥[xy(x²+y²)] = 12xy`, so the coefficient must
be `−1/12`. Taking the gradient:

```
Bx = G(z)·y   − (G''(z)/12)·(3x²y + y³)
By = G(z)·x   − (G''(z)/12)·(x³ + 3xy²)
Bz = G'(z)·xy − (G'''(z)/12)·(x³y + xy³)
```

### Truncation order

Two properties matter, and `checks.py` asserts both:

- **∇×B ≡ 0 identically**, because **B** is a gradient. A numerical curl test
  therefore verifies only that the three coded expressions really are consistent
  partial derivatives of one potential — which is exactly the typo it needs to catch.

- **∇·B is *not* exactly zero.** The `G''` term cancels the `G''·xy` from `∂²ψ/∂z²`,
  but the second z-derivative of the `G''` term itself survives:

  ```
  ∇·B = −(G''''(z)/12)·xy·(x² + y²)
  ```

  The field is exact to O(r³) and the residual grows as r⁴. Asserting `∇·B == 0`
  would be wrong; asserting the r⁴ scaling is the real test. At r = 3 mm with the
  example parameters the residual is ~1e-6 of `G0` — far below any physical effect.

Carrying more terms means continuing the same recursion: each new term cancels the
`∂²/∂z²` of the previous one.

### Gradient profiles

`G(z) = G0·f(s)` with `s = (z − z0)/L`, so `dⁿG/dzⁿ = (G0/Lⁿ)·f⁽ⁿ⁾(s)`. Every
profile supplies `f` through `f'''` in closed form — needed because `Bx`, `By` use
`G''` and `Bz` uses `G'''`.

| name | `f(s)` | notes |
|---|---|---|
| `hard` | `1` for `|s| < 1/2`, else `0` | leading order only; matches `quadrupole/tracking2.m` |
| `tanh` | `[tanh((s+½)/d) − tanh((s−½)/d)] / 2` | smoothed flat top, fringe scale `d` |
| `gauss` | `exp(−s²/2d²)` | for short magnets with no real flat top |
| `enge` | product of two logistic edges | standard accelerator fringe model |

`hard` **raises** on any derivative request rather than returning zero. A zero `G''`
looks exactly like a valid soft-edge model with no fringe curvature and would
silently hide the fact that only the leading-order field is available.

The `tanh` profile has a property worth knowing: its integral is *exactly* `L` for
any `d`, because the integral of a difference of two shifted `tanh` steps is
(shift) × (total jump). The fringe removes precisely as much area as it adds. What
converges as `d → 0` is the pointwise field on the flat top, not the integral.

## The result this layer exists to produce

With the wire at `(x, y)` and the magnet center at `(dx, dy)`, writing `u = x − dx`
and `v = y − dy`:

$$\int B_y\,dz = \left(\int G\,dz\right)u \;-\; \frac{1}{12}\left(\int G''\,dz\right)(u^3 + 3uv^2)$$

For any profile that decays at both ends, `∫G'' dz = [G']₋∞^∞ = 0`.

> **The field integral is exactly linear in PMQ offset, for any fringe shape.**
> The entire position-to-kick relationship collapses to one number, the integrated
> gradient `∫G dz = G0·L_eff`.

`checks.py` confirms this numerically at the 1e-16 level for `tanh`, `gauss`, and
`enge` alike.

The result survives more than it looks like it should. A triplet with *overlapping*
fringes (L = 6 mm, 13.5 mm spacing) is still exactly linear, with slope equal to the
summed integrated gradients — because superposition holds and each magnet
contributes linearly, so no amount of overlap can produce curvature. The same is
true if the magnets have different individual offsets, or if the wire is tilted or
sagging: `By = G(z)·x` is linear in `x`, so integrating along any path leaves the
dependence on a rigid displacement linear.

That makes the statement of what this layer **cannot** explain quite sharp. Fringe
shape changes the shape of `B(z)` — hence the time-domain trace and its sensitivity
to pulse width — but not the integrated kick. Curvature in a measured
kick-versus-position scan has only three places left to come from:

- **higher multipoles** from Halbach segmentation — not in this model, which is a
  pure quadrupole, and the leading candidate
- **a truncated integration range**, which makes `∫G'' dz ≠ 0` and revives the cubic
  term (this is why `field_integral` defaults to ±10 magnet lengths of fringe)
- **the wire dynamics or detector response** — finite pulse width, dispersion, the
  photodiode's linear range

Worth knowing before attributing a measured nonlinearity to magnet alignment.

## Files

| file | contents |
|---|---|
| `profiles.py` | `G(z)` shape functions and their first three derivatives |
| `field.py` | `PMQ` (offset, roll, order-0 or order-2 field) and `Lattice` (superposition) |
| `integrals.py` | field integrals, effective length, `offset_scan`, `kick`, `offset_sensitivity` |
| `checks.py` | eight analytic self-checks — run this first |
| `example_offset_scan.py` | the ±0.8 mm bench sweep, with figure |

## Running

This project owns its environment in `.venv/` (Python 3.14, numpy 2.5.1,
scipy 1.18.0, matplotlib 3.11.1):

```bash
cd pmq_sim
.venv/bin/python checks.py                 # 8/8 checks pass, with numbers printed
.venv/bin/python example_offset_scan.py    # writes offset_scan.png
```

To rebuild it from scratch:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

(The repo-root `.venv` has only pip/setuptools and is unusable; `pmq_measurements`
has its own separate environment.)

`checks.py` prints numbers rather than passing silently, because the point is to see
*how well* each identity holds. Checks 2 and 4 are the substantive ones: check 2
proves the Maxwell expansion is correct to the claimed order, and check 4 proves the
offset-to-kick relationship is the single number to extract from bench data.

## Magnet parameters

`G0`, `L`, and the bore radius in `example_offset_scan.py` and `checks.py` are
**placeholders** marked with `TODO`. Substitute the real numbers for the magnet on
the bench and everything scales. The physics and all eight checks are
parameter-independent.

## Not modelled here

Deferred to later layers, listed so the interfaces leave room:

- the current pulse and the Lorentz impulse density on the wire
- the dispersive damped wire equation — `c₀ ≈ 222 m/s` and the `EIwT` dispersion
  machinery already exist in `pmq/PMQ_measurements/pmq_analysis.py:147`
- photodiode response, the 127 µm wire diameter, diode angle (`6-11-26-diodeangle`)
- wire sag and the deliberate slope sweeps (`6-8-26-slopes`)
- higher multipoles from Halbach segmentation
- comparison against measured CSVs or the Radia exports in `pmq/pmq_radia/`
