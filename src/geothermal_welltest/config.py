"""Well geometry, fluid, and default analysis constants.

All depths are measured depth (mMD) below the casing-head flange (CHF).
Edit :class:`WellGeometry` to describe a different well.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class WellGeometry:
    """Static description of the well completion."""

    casing_shoe_m: float = 462.5      #: bottom of blank production casing
    top_of_liner_m: float = 425.0     #: top of perforated liner
    terminal_depth_m: float = 946.0   #: deepest drilled depth
    liner_id_m: float = 0.2477        #: ~10-3/4" perforated liner inside diameter

    @property
    def liner_area_m2(self) -> float:
        """Cross-sectional flow area of the perforated liner [m^2]."""
        return math.pi * (self.liner_id_m / 2.0) ** 2


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable parameters for the temperature and spinner workflows."""

    # temperature gradient model
    grid_step_m: float = 2.0          #: even-grid resample spacing
    sg_window: int = 15               #: Savitzky-Golay window (odd)
    sg_polyorder: int = 2             #: Savitzky-Golay polynomial order
    isothermal_grad_degC_per_m: float = 0.15  #: |dT/dz| below this = convective

    # spinner cross-plot model
    xplot_step_m: float = 2.0         #: cross-plot window length
    xplot_top_m: float = 460.0        #: shallowest analysis depth
    xplot_bottom_m: float = 926.0     #: deepest analysis depth
    qc_min_r2: float = 0.90           #: keep intervals with R^2 above this
    qc_min_obs: int = 5               #: keep intervals with at least this many points

    # evidence-combination weights
    weight_temperature: float = 0.45
    weight_velocity: float = 0.55

    # atmospheric pressure used for barg -> bara conversion
    atmospheric_bar: float = 1.01325


DEFAULT_GEOMETRY = WellGeometry()
DEFAULT_CONFIG = AnalysisConfig()
