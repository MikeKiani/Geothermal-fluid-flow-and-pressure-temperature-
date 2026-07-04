"""Temperature-profile models: gradient, curvature, regime, feed-zone score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from .config import AnalysisConfig, DEFAULT_CONFIG, DEFAULT_GEOMETRY, WellGeometry


@dataclass
class TemperatureResult:
    """Container for the temperature-model outputs."""

    grad: pd.DataFrame           #: even-grid frame: depth_m, T_sm, dTdz, d2Tdz2, regime, temp_fz_score
    feedzone_depths: np.ndarray  #: temperature-only feed-zone picks [m]


def compute_gradient(
    df: pd.DataFrame,
    config: AnalysisConfig = DEFAULT_CONFIG,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> pd.DataFrame:
    """Resample onto an even grid and compute smoothed derivatives.

    Differentiating unevenly spaced logs directly amplifies noise, so we
    interpolate onto a regular grid and apply a Savitzky-Golay filter before
    taking the first (gradient) and second (curvature) derivatives.

    Returns
    -------
    pandas.DataFrame
        Columns ``depth_m``, ``T_sm``, ``dTdz`` [degC/m], ``d2Tdz2`` [degC/m^2],
        ``regime`` (``"conductive"`` / ``"convective/isothermal"``).
    """
    z = np.arange(df.depth_m.min(), df.depth_m.max(), config.grid_step_m)
    t = np.interp(z, df.depth_m, df.temp_degC)

    win = config.sg_window
    if win >= z.size:
        win = (z.size // 2) * 2 - 1  # largest odd number < size
    win = max(win, config.sg_polyorder + 2 | 1)
    t_sm = savgol_filter(t, window_length=win, polyorder=config.sg_polyorder)

    grad = pd.DataFrame({"depth_m": z, "T_sm": t_sm})
    grad["dTdz"] = np.gradient(grad["T_sm"].to_numpy(), grad["depth_m"].to_numpy())
    grad["d2Tdz2"] = np.gradient(grad["dTdz"].to_numpy(), grad["depth_m"].to_numpy())
    grad["regime"] = np.where(
        grad["dTdz"].abs() < config.isothermal_grad_degC_per_m,
        "convective/isothermal",
        "conductive",
    )
    return grad


def temperature_feedzone_score(
    grad: pd.DataFrame,
    df: pd.DataFrame,
    config: AnalysisConfig = DEFAULT_CONFIG,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> TemperatureResult:
    """Score each open-hole depth for feed-zone likelihood from temperature.

    Isothermal (low ``|dT/dz|``) intervals below the casing shoe score high; a
    small bonus is added where the profile is close to boiling. Peaks in the
    score are the temperature-only feed-zone picks.
    """
    grad = grad.copy()
    open_hole = (grad.depth_m > geometry.casing_shoe_m).to_numpy()

    absg = grad["dTdz"].abs().to_numpy()
    ref = np.nanpercentile(absg[open_hole], 80) + 1e-9

    score = np.zeros(len(grad))
    score[open_hole] = np.clip(1.0 - absg[open_hole] / ref, 0, 1)

    # boiling-proximity bonus (0..0.3) mapped onto the gradient grid
    tmt = np.interp(grad.depth_m, df.depth_m, df.t_minus_tsat)
    boil_bonus = np.clip((tmt + 20) / 20, 0, 1) * 0.3
    score = np.clip(score + boil_bonus * open_hole, 0, 1)
    grad["temp_fz_score"] = score

    peaks, _ = find_peaks(score, height=0.55, distance=8, prominence=0.10)
    return TemperatureResult(grad=grad, feedzone_depths=grad.depth_m.to_numpy()[peaks])


def run_temperature_analysis(
    df: pd.DataFrame,
    config: AnalysisConfig = DEFAULT_CONFIG,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> TemperatureResult:
    """Convenience wrapper: gradient model + feed-zone score in one call."""
    grad = compute_gradient(df, config, geometry)
    return temperature_feedzone_score(grad, df, config, geometry)
