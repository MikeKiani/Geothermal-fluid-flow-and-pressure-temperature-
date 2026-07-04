# Methodology

This document sets out the physics, equations, and modelling choices behind the
`geothermal_welltest` package. It is aimed at a reservoir engineer or geoscientist who wants to
understand — and trust — what each step computes.

---

## 1. Data and units

The input is a downhole **pressure–temperature profile** logged after the well has been shut and
heating for a period long enough that the profile is approaching its stable, undisturbed state.

| column | meaning | unit |
|---|---|---|
| `depth_m` | measured depth below the casing-head flange | m |
| `whp_barg` | well-head pressure | bar gauge |
| `pres_barg` | downhole pressure | bar gauge |
| `temp_degC` | downhole temperature | °C |

**Gauge to absolute pressure.** Steam-table calculations require *absolute* pressure. Bar-gauge
is referenced to atmospheric, so

```
p_abs = p_gauge + 1.01325 bar
```

The original tutorial subtracted 1 bar; that is corrected here (`io_utils.load_profile` and
`steam_tables.bara_from_barg`). The sign matters most near the boiling front, where the BPD curve
and the measured temperature nearly coincide.

---

## 2. Boiling point for depth (BPD)

The BPD curve is the temperature at which water would boil at the measured downhole pressure.
Where the logged temperature meets the BPD curve the fluid is at the boiling front and may be
two-phase — a common signature of a productive feed zone.

We compute saturation temperature from pressure with the **IAPWS-IF97 Region 4 backward
equation**, `T_sat(p)`. With `β = (p / 1 MPa)^{1/4}` and coefficients `n₁…n₁₀`:

```
E = β² + n₃β + n₆
F = n₁β² + n₄β + n₇
G = n₂β² + n₅β + n₈
D = 2G / (−F − √(F² − 4EG))
T_sat = ½ ( n₁₀ + D − √[(n₁₀ + D)² − 4(n₉ + n₁₀D)] )      [K]
```

This is exact to ~0.01 °C against IAPWS reference points over 611 Pa – 22.06 MPa and removes the
need for the external `iapws` package. The **below-boiling margin** `T − T_sat` is reported as a
continuous feed-zone indicator (values approaching zero indicate proximity to boiling).

---

## 3. Temperature gradient and curvature

Feed zones are read from the *shape* of the profile:

- **Conductive** (impermeable) intervals show a steep, roughly constant gradient `dT/dz`.
- **Convective / permeable** intervals are **near-isothermal**, `dT/dz ≈ 0`.
- Gradient **breaks** — spikes in the curvature `d²T/dz²` — mark the boundaries where a feed zone
  starts or stops controlling the profile.

Because logging depth spacing is uneven and direct differentiation amplifies noise, the profile
is resampled onto an even grid (`grid_step_m`, default 2 m) and smoothed with a Savitzky–Golay
filter (`sg_window`, `sg_polyorder`) before the first and second derivatives are taken. Each
depth is then labelled `conductive` or `convective/isothermal` using the
`isothermal_grad_degC_per_m` threshold (default 0.15 °C/m).

### Temperature feed-zone score

The open-hole gradient is mapped to a 0–1 likelihood: isothermal intervals score high, steep
intervals score low, plus a small bonus where the profile is near boiling. Peaks in this score
are the temperature-only feed-zone picks.

---

## 4. Fluid-velocity analysis (spinner cross-plot)

### 4.1 The measurement

A PTS tool carries an impeller (spinner) whose rotation frequency responds to the fluid moving
past it. Because the tool itself moves on wireline, the spinner reads the **relative** velocity:

```
f = ( v_tool − v_fluid ) / k
```

where `k` is the spinner calibration (m/s per Hz). Logging the same depth on several passes at
different tool speeds and regressing **tool speed (y)** on **spinner frequency (x)** gives a
straight line whose **y-intercept at f = 0 is the fluid velocity** — the tool speed at which the
spinner stops turning is exactly the fluid velocity. The fit's R² is a natural quality filter,
and the number of passes in the window guards against under-determined fits.

`cross_plot_analysis` performs the fit for one depth window; `calc_fluid_velocity` sweeps it over
the open hole in windows of `xplot_step_m`; `qc_velocity` keeps intervals with
`R² > qc_min_r2` and at least `qc_min_obs` points.

### 4.2 Feed-zone signal

During an **injection** test the pumped fluid leaves the well at each feed zone, so the in-well
velocity is highest at the top of the open hole and **steps down** with depth, reaching ≈ 0 below
the deepest feed. The derivative `dV/dz` therefore peaks at feed zones, and the integral of each
step is proportional to the mass that zone accepts — its **relative injectivity**
(`injectivity_from_velocity`).

### 4.3 Forward model ("perform a similar model")

When raw spinner logs are unavailable, `build_synthetic_pts` generates a physically consistent
PTS dataset:

1. Feed zones are specified as `(top_m, bottom_m, relative_injectivity)` and normalised.
2. The in-well flow fraction is built as a product of smooth `tanh` steps, one per zone
   (`inwell_flow_fraction`), so flow decreases from ~1 at the top of the open hole to ~0 below
   the deepest zone.
3. Flow is converted to velocity through the liner area, and the spinner response for each pass
   is `f = (v_tool − v_fluid)/k` plus heteroscedastic noise (turbulent zones read noisier).

Inverting this synthetic data with the cross-plot method recovers the input velocity — the
package's tests assert the recovery to within ~5 cm/s — which validates the whole chain. To run
on measured data, replace this forward model with your PTS import; nothing downstream changes.

---

## 5. Combining the evidence

`combine_and_detect` places both datasets on the same depth grid, normalises each to 0–1, and
forms a weighted sum:

```
combined(z) = w_temp · E_temp(z) + w_vel · E_vel(z)      (open hole only)
```

with weights `weight_temperature` and `weight_velocity`. `E_temp` is the temperature feed-zone
score; `E_vel` is the mean positive `dV/dz` across pump rates. Confident feed zones are peaks in
the combined score that are supported by **both** datasets (temperature evidence > 0.30 and
velocity evidence > 0.20). The result is a ranked table of feed zones with their combined score,
the supporting evidence values, and the relative injectivity from the velocity model.

Requiring agreement between two independent physical measurements is what makes the picks robust:
a temperature isotherm alone can be ambiguous, and a velocity step alone can be noise, but their
coincidence is a strong indicator of a permeable feed zone.

---

## 6. Configuration

All tunable parameters live in `config.py` (`WellGeometry`, `AnalysisConfig`) and are passed
explicitly through the pipeline, so a different well or logging campaign is described by editing
one place rather than hunting through the code.
