"""Spinner (fluid-velocity) analysis.

Two halves:

1. A **forward model** (:func:`build_synthetic_pts`) that generates a physically
   consistent pressure-temperature-spinner (PTS) dataset from a set of feed zones.
   Use this to demonstrate the workflow when raw spinner logs are unavailable, or
   as a sensitivity tool.
2. The **cross-plot inversion** (:func:`cross_plot_analysis`,
   :func:`calc_fluid_velocity`) that recovers fluid velocity from real or
   synthetic PTS data.

Physics of the cross-plot method
--------------------------------
A PTS impeller reads the *relative* velocity between fluid and tool,
``f = (v_tool - v_fluid) / k``. Logging the same depth on several passes at
different tool speeds and regressing tool speed (y) on spinner frequency (x)
gives a line whose y-intercept (at ``f = 0``) is the fluid velocity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig, DEFAULT_CONFIG, DEFAULT_GEOMETRY, WellGeometry

# A feed zone is (top_m, bottom_m, relative_injectivity)
FeedZone = Tuple[float, float, float]


def normalise_feedzones(feedzones: Sequence[FeedZone]) -> List[FeedZone]:
    """Rescale the injectivity weights so they sum to 1.0."""
    total = sum(w for _, _, w in feedzones)
    if total <= 0:
        raise ValueError("Feed-zone injectivity weights must sum to a positive number.")
    return [(a, b, w / total) for a, b, w in feedzones]


def inwell_flow_fraction(z: np.ndarray, feedzones: Sequence[FeedZone]) -> np.ndarray:
    """Fraction of surface flow still inside the well at depth ``z`` (injection).

    Starts near 1 at the top of the open hole and steps smoothly down through
    each feed zone toward ~0 below the deepest zone.
    """
    z = np.asarray(z, dtype=float)
    frac = np.ones_like(z)
    for a, b, w in feedzones:
        mid, width = 0.5 * (a + b), (b - a)
        frac -= w * 0.5 * (1 + np.tanh((z - mid) / (width * 0.35)))
    return np.clip(frac, 0, 1)


def build_synthetic_pts(
    pump_tph: float,
    feedzones: Sequence[FeedZone],
    profile: pd.DataFrame,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
    n_passes: int = 7,
    spinner_k: float = 0.06,
    noise_hz: float = 2.2,
    rng: np.random.Generator | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-model a PTS dataset for one pump rate.

    Parameters
    ----------
    pump_tph
        Surface pump rate [t/hr].
    feedzones
        Sequence of ``(top_m, bottom_m, injectivity)`` (need not be normalised).
    profile
        Temperature/pressure profile used to attach realistic pressure/temp columns.
    n_passes
        Number of logging passes (tool speeds sampled between -0.9 and +0.9 m/s).
    spinner_k
        Spinner calibration [m/s per Hz].
    noise_hz
        Base spinner-frequency noise; heteroscedastic (turbulent zones read noisier).

    Returns
    -------
    (pts, v_true)
        ``pts``: long-form dataframe (one row per pass per depth) with
        ``depth_m, speed_mps, frequency_hz, pressure_bara, temp_degC``.
        ``v_true``: the ground-truth fluid velocity used to build ``pts``.
    """
    rng = rng or np.random.default_rng(7)
    feedzones = normalise_feedzones(feedzones)

    q_top = pump_tph * 1000 / 3600 / 1000.0  # t/hr -> m^3/s (rho ~ 1000 kg/m^3)
    zg = np.arange(geometry.casing_shoe_m - 2, geometry.terminal_depth_m - 18, 0.25)
    v_fluid = q_top * inwell_flow_fraction(zg, feedzones) / geometry.liner_area_m2

    passes = []
    for v_tool in np.linspace(-0.9, 0.9, n_passes):
        local = noise_hz * (1 + 0.9 * rng.random(zg.size))  # heteroscedastic noise
        freq = (v_tool - v_fluid) / spinner_k + rng.normal(0, local)
        passes.append(pd.DataFrame({
            "depth_m": zg,
            "speed_mps": v_tool + rng.normal(0, 0.004, zg.size),
            "frequency_hz": freq,
            "pressure_bara": np.interp(zg, profile.depth_m, profile.pressure_bara),
            "temp_degC": np.interp(zg, profile.depth_m, profile.temp_degC),
        }))
    pts = pd.concat(passes, ignore_index=True)
    v_true = pd.DataFrame({"depth_m": zg, "v_true": v_fluid})
    return pts, v_true


def cross_plot_analysis(
    pts_df: pd.DataFrame, top: float, bottom: float
) -> Tuple[np.ndarray, np.ndarray, float, float, float, int]:
    """Fit tool-speed vs spinner-frequency within one depth interval.

    Returns ``(freq, speed, fluid_velocity, slope, r_squared, n_obs)``. Degenerate
    intervals (too few points or zero frequency spread) return NaN model values.
    """
    d = pts_df[(pts_df.depth_m > top) & (pts_df.depth_m < bottom)]
    d = d[d.frequency_hz.notna() & d.pressure_bara.notna()]
    fx = d.frequency_hz.to_numpy()
    sy = d.speed_mps.to_numpy()
    if fx.size > 2 and np.ptp(fx) > 1e-6:
        lm = stats.linregress(fx, sy)
        return fx, sy, lm.intercept, lm.slope, lm.rvalue ** 2, fx.size
    return fx, sy, np.nan, np.nan, np.nan, int(fx.size)


def calc_fluid_velocity(
    pts_df: pd.DataFrame, top: float, bottom: float, step: float
) -> pd.DataFrame:
    """Sweep the cross-plot over ``[top, bottom]`` in windows of ``step`` metres."""
    tops = np.arange(top, bottom - step, step)
    rows = []
    for a in tops:
        b = a + step
        _, _, v, s, r, n = cross_plot_analysis(pts_df, a, b)
        rows.append(((a + b) / 2, v, s, r, n))
    return pd.DataFrame(
        rows, columns=["depth_m", "intercept_velocity_mps", "slope", "r_squared", "obs_num"]
    )


def qc_velocity(
    dfv: pd.DataFrame, config: AnalysisConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Keep well-fit intervals and add the ``dv_dz`` feed-zone signal."""
    g = dfv[(dfv.r_squared > config.qc_min_r2) & (dfv.obs_num >= config.qc_min_obs)]
    g = g.copy().sort_values("depth_m")
    g["dv_dz"] = (
        np.gradient(g.intercept_velocity_mps.to_numpy(), g.depth_m.to_numpy())
        if len(g) > 1 else 0.0
    )
    return g


@dataclass
class VelocityResult:
    """Fluid-velocity outputs for a set of pump rates."""

    raw: Dict[str, pd.DataFrame]      #: unfiltered cross-plot results per rate
    clean: Dict[str, pd.DataFrame]    #: QC-filtered results per rate
    v_true: Dict[str, pd.DataFrame]   #: ground truth (synthetic runs only)


def run_velocity_analysis(
    pts_by_rate: Dict[str, pd.DataFrame],
    config: AnalysisConfig = DEFAULT_CONFIG,
    v_true: Dict[str, pd.DataFrame] | None = None,
) -> VelocityResult:
    """Invert every pump rate and apply QC."""
    raw, clean = {}, {}
    for name, pts in pts_by_rate.items():
        r = calc_fluid_velocity(pts, config.xplot_top_m, config.xplot_bottom_m, config.xplot_step_m)
        raw[name] = r
        clean[name] = qc_velocity(r, config)
    return VelocityResult(raw=raw, clean=clean, v_true=v_true or {})
