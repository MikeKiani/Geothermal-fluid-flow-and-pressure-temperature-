"""Publication-quality figures for the well-test workflow.

Every function returns a :class:`matplotlib.figure.Figure` so callers can save,
display, or embed them. Import this module only when plotting is needed.
"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig, DEFAULT_CONFIG, DEFAULT_GEOMETRY, WellGeometry
from .feedzones import FeedZoneResult
from .spinner import VelocityResult
from .temperature import TemperatureResult

_RATE_COLORS = ["tab:green", "tab:red", "tab:orange", "tab:blue", "tab:purple"]


def apply_house_style() -> None:
    """Set a clean, consistent matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    })


def _rate_color_map(names) -> Dict[str, str]:
    return {n: _RATE_COLORS[i % len(_RATE_COLORS)] for i, n in enumerate(names)}


def plot_temperature_dashboard(
    df: pd.DataFrame,
    temp: TemperatureResult,
    config: AnalysisConfig = DEFAULT_CONFIG,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> plt.Figure:
    """Four-panel temperature interpretation: T+BPD, gradient, below-boiling, score."""
    grad = temp.grad
    fig, axes = plt.subplots(1, 4, figsize=(16, 9.5), sharey=True)
    fig.suptitle("Temperature interpretation dashboard — heating profile", fontsize=15, y=0.98)

    ax = axes[0]
    ax.plot(df.temp_degC, df.depth_m, color="tab:red", lw=2.2, label="Measured T")
    ax.plot(df.tsat_degC, df.depth_m, "--", color="tab:blue", lw=1.4, label="BPD")
    conv = grad[grad.regime == "convective/isothermal"]
    ax.scatter(np.interp(conv.depth_m, df.depth_m, df.temp_degC), conv.depth_m,
               s=14, color="tab:green", alpha=0.5, label="convective/isothermal")
    ax.set(xlabel="T [°C]", ylabel="Depth [m]", title="(a) T + BPD + regime")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    g = config.isothermal_grad_degC_per_m
    ax.plot(grad.dTdz, grad.depth_m, color="k", lw=1.4)
    ax.axvspan(-g, g, color="tab:green", alpha=0.12)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set(xlabel="dT/dz [°C/m]", title="(b) Gradient")

    ax = axes[2]
    ax.plot(df.t_minus_tsat, df.depth_m, color="tab:purple", lw=1.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="T − Tsat [°C]", title="(c) Below-boiling")

    ax = axes[3]
    ax.fill_betweenx(grad.depth_m, 0, grad.temp_fz_score, color="tab:green", alpha=0.35)
    for d in temp.feedzone_depths:
        p = int(np.argmin(np.abs(grad.depth_m.to_numpy() - d)))
        ax.plot(grad.temp_fz_score.iloc[p], d, "v", color="darkgreen", ms=9)
    ax.set(xlabel="FZ score", title="(d) Temp FZ score", xlim=(0, 1))

    for ax in axes:
        ax.axhspan(geometry.top_of_liner_m, geometry.terminal_depth_m, color="grey", alpha=0.05)
        ax.invert_yaxis()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_crossplot_example(
    pts_df: pd.DataFrame, top: float = 700, bottom: float = 701
) -> plt.Figure:
    """Illustrate the cross-plot inversion on one depth interval."""
    seg = pts_df[(pts_df.depth_m > top) & (pts_df.depth_m < bottom)]
    lm = stats.linregress(seg.frequency_hz, seg.speed_mps)
    xx = np.linspace(seg.frequency_hz.min(), seg.frequency_hz.max(), 50)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(seg.frequency_hz, seg.speed_mps, facecolors="none", edgecolors="k", s=55, label="passes")
    ax.plot(xx, lm.slope * xx + lm.intercept, color="tab:orange", lw=3, alpha=0.7, label="linear fit")
    ax.scatter(0, lm.intercept, color="tab:orange", s=130, zorder=5,
               label=f"fluid velocity = {lm.intercept:.3f} m/s")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set(xlabel="Spinner frequency [Hz]", ylabel="Tool speed [m/s]",
           title=f"Cross-plot @ ~{top:.0f} m  (R² = {lm.rvalue**2:.3f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_velocity_qc(
    velocity: VelocityResult,
    feedzones=None,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> plt.Figure:
    """Raw inversion coloured by R² alongside the cleaned profiles (vs truth)."""
    colors = _rate_color_map(velocity.clean.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)

    im = None
    for name, r in velocity.raw.items():
        im = axes[0].scatter(r.intercept_velocity_mps, r.depth_m, c=r.r_squared,
                             s=12, cmap="viridis", vmin=0.5, vmax=1)
    axes[0].set(title="Raw inversion (colour = R²)", xlabel="Fluid velocity [m/s]", ylabel="Depth [m]")
    if im is not None:
        fig.colorbar(im, ax=axes[0], label="R²")

    for name, g in velocity.clean.items():
        if name in velocity.v_true:
            vt = velocity.v_true[name]
            axes[1].plot(vt.v_true, vt.depth_m, "--", color=colors[name], lw=1, alpha=0.6)
        axes[1].plot(g.intercept_velocity_mps, g.depth_m, ".", color=colors[name], ms=5, label=name)
    axes[1].set(title="Cleaned velocity vs ground truth (dashed)", xlabel="Fluid velocity [m/s]")
    axes[1].legend(fontsize=8, loc="lower right")

    if feedzones:
        for a, b, _ in feedzones:
            for ax in axes:
                ax.axhspan(a, b, color="tab:blue", alpha=0.08)
    for ax in axes:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_feedzone_signal(
    velocity: VelocityResult,
    feedzones=None,
) -> plt.Figure:
    """Velocity and its dV/dz feed-zone signal for every pump rate."""
    colors = _rate_color_map(velocity.clean.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    for name, g in velocity.clean.items():
        axes[0].plot(g.intercept_velocity_mps, g.depth_m, "-", color=colors[name], lw=1.6, label=name)
        axes[1].plot(g.dv_dz, g.depth_m, "-", color=colors[name], lw=1.4, label=name)
    axes[0].set(title="Fluid velocity", xlabel="V [m/s]", ylabel="Depth [m]")
    axes[1].set(title="dV/dz (feed-zone signal)", xlabel="dV/dz [1/s]")
    axes[1].axvline(0, color="grey", lw=0.6)
    if feedzones:
        for a, b, _ in feedzones:
            for ax in axes:
                ax.axhspan(a, b, color="tab:blue", alpha=0.10)
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right")
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_combined_interpretation(
    df: pd.DataFrame,
    temp: TemperatureResult,
    velocity: VelocityResult,
    feedzone_result: FeedZoneResult,
    geometry: WellGeometry = DEFAULT_GEOMETRY,
) -> plt.Figure:
    """The synthesis figure: T+BPD, evidence tracks, combined score, velocity."""
    grid = feedzone_result.combined.depth_m.to_numpy()
    e_temp = feedzone_result.combined.E_temp.to_numpy()
    e_vel = feedzone_result.combined.E_vel.to_numpy()
    combined = feedzone_result.combined.combined.to_numpy()
    picks = feedzone_result.picks_m
    colors = _rate_color_map(velocity.clean.keys())

    fig, axes = plt.subplots(1, 4, figsize=(17, 10), sharey=True)
    fig.suptitle("Combined feed-zone interpretation — temperature + fluid velocity",
                 fontsize=15, y=0.98)

    ax = axes[0]
    ax.plot(df.temp_degC, df.depth_m, color="tab:red", lw=2, label="T")
    ax.plot(df.tsat_degC, df.depth_m, "--", color="tab:blue", lw=1.3, label="BPD")
    ax.set(xlabel="T [°C]", ylabel="Depth [m]", title="(a) Temperature")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.fill_betweenx(grid, 0, e_temp, color="tab:green", alpha=0.4, label="temperature")
    ax.plot(e_vel, grid, color="tab:red", lw=1.5, label="velocity dV/dz")
    ax.set(xlabel="normalised evidence", title="(b) Evidence", xlim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    ax.fill_betweenx(grid, 0, combined, color="tab:purple", alpha=0.4)
    ax.plot(combined, grid, color="tab:purple", lw=1.6)
    ax.set(xlabel="combined score", title="(c) Combined", xlim=(0, 1))

    ax = axes[3]
    for name, g in velocity.clean.items():
        ax.plot(g.intercept_velocity_mps, g.depth_m, "-", color=colors[name], lw=1.4, label=name)
    ax.set(xlabel="V [m/s]", title="(d) Fluid velocity")
    ax.legend(fontsize=8, loc="lower right")

    for j, d in enumerate(picks, 1):
        for ax in axes:
            ax.axhline(d, color="k", lw=0.6, ls="--", alpha=0.6)
        axes[0].text(df.temp_degC.min() + 3, d - 3, f"FZ{j}", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.axhspan(geometry.top_of_liner_m, geometry.terminal_depth_m, color="grey", alpha=0.05)
        ax.invert_yaxis()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
