"""Combine independent evidence to locate and rank feed zones."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import AnalysisConfig, DEFAULT_CONFIG, DEFAULT_GEOMETRY, WellGeometry
from .spinner import VelocityResult


@dataclass
class FeedZoneResult:
    """Combined feed-zone interpretation."""

    combined: pd.DataFrame  #: depth_m, E_temp, E_vel, combined
    picks_m: np.ndarray     #: confident feed-zone depths [m]
    table: pd.DataFrame     #: ranked summary with evidence + relative injectivity


def _velocity_evidence(grid: np.ndarray, velocity: VelocityResult) -> np.ndarray:
    """Mean positive ``dV/dz`` across rates, normalised to 0..1 on ``grid``."""
    stack = []
    for g in velocity.clean.values():
        if len(g) < 2:
            continue
        stack.append(np.interp(grid, g.depth_m, g.dv_dz.clip(lower=0), left=0, right=0))
    if not stack:
        return np.zeros_like(grid)
    e = np.mean(stack, axis=0)
    pos = e[e > 0]
    ref = np.nanpercentile(pos, 85) + 1e-9 if pos.size else 1.0
    return np.clip(e / ref, 0, 1)


def injectivity_from_velocity(
    velocity: VelocityResult,
    feedzones,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
    rate_name: str | None = None,
) -> pd.DataFrame:
    """Estimate each zone's relative injectivity from the velocity step across it.

    Uses the highest-rate clean profile by default (largest, clearest steps).
    """
    if rate_name is None:
        rate_name = max(velocity.clean, key=lambda k: velocity.clean[k].intercept_velocity_mps.max())
    hr = velocity.clean[rate_name].sort_values("depth_m")
    v_of = lambda d: np.interp(d, hr.depth_m, hr.intercept_velocity_mps)

    rows = []
    for i, (a, b, _) in enumerate(feedzones, 1):
        drop = max(v_of(a - 3) - v_of(b + 3), 0.0)
        rows.append((f"FZ{i}", a, b, drop))
    out = pd.DataFrame(rows, columns=["zone", "top_m", "bottom_m", "v_drop_mps"])
    total = out.v_drop_mps.sum()
    out["mass_fraction"] = (out.v_drop_mps / total).round(3) if total > 0 else np.nan
    return out


def combine_and_detect(
    temp_grad: pd.DataFrame,
    velocity: VelocityResult,
    feedzones=None,
    config: AnalysisConfig = DEFAULT_CONFIG,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> FeedZoneResult:
    """Fuse temperature and velocity evidence into a combined feed-zone score.

    Feed zones are peaks in the weighted-sum score that are supported by *both*
    datasets (temperature evidence > 0.30 and velocity evidence > 0.20).
    """
    grid = temp_grad.depth_m.to_numpy()
    open_hole = grid > geometry.casing_shoe_m

    e_temp = temp_grad["temp_fz_score"].to_numpy()
    e_vel = _velocity_evidence(grid, velocity)

    combined = (config.weight_temperature * e_temp
                + config.weight_velocity * e_vel) * open_hole
    combined_df = pd.DataFrame(
        {"depth_m": grid, "E_temp": e_temp, "E_vel": e_vel, "combined": combined}
    )

    peaks, _ = find_peaks(combined, height=0.42, distance=14, prominence=0.10)
    picks = [grid[p] for p in peaks if e_temp[p] > 0.30 and e_vel[p] > 0.20]
    picks = np.array(picks)

    inj = injectivity_from_velocity(velocity, feedzones, geometry) if feedzones else None

    def match_fraction(depth: float) -> float:
        if inj is None:
            return np.nan
        for _, r in inj.iterrows():
            if r.top_m - 10 <= depth <= r.bottom_m + 10:
                return r.mass_fraction
        return np.nan

    rows = []
    for j, d in enumerate(picks, 1):
        p = int(np.argmin(np.abs(grid - d)))
        rows.append({
            "feed_zone": f"FZ{j}",
            "depth_m": round(float(d), 0),
            "combined_score": round(float(combined[p]), 2),
            "temp_evidence": round(float(e_temp[p]), 2),
            "vel_evidence": round(float(e_vel[p]), 2),
            "rel_injectivity": match_fraction(d),
        })
    table = (pd.DataFrame(rows)
             .sort_values("combined_score", ascending=False)
             .reset_index(drop=True)) if rows else pd.DataFrame()

    return FeedZoneResult(combined=combined_df, picks_m=picks, table=table)
