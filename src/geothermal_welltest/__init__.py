"""Geothermal well-test analysis: temperature, fluid velocity, and feed zones.

A small, dependency-light toolkit for completion-test interpretation, rebuilt and
extended from the Software Underground *Transform 2021* tutorial by Irene Wallis
and Katie McLean.

Typical use
-----------
>>> from geothermal_welltest import load_profile, run_temperature_analysis
>>> df = load_profile("data/Data-Temp-Heating37days.csv")
>>> temp = run_temperature_analysis(df)

See ``scripts/run_analysis.py`` for the full end-to-end pipeline.
"""
from __future__ import annotations

from .config import AnalysisConfig, WellGeometry, DEFAULT_CONFIG, DEFAULT_GEOMETRY
from .steam_tables import tsat_from_p_mpa, bara_from_barg
from .io_utils import load_profile, profile_summary
from .temperature import (
    TemperatureResult,
    compute_gradient,
    temperature_feedzone_score,
    run_temperature_analysis,
)
from .spinner import (
    FeedZone,
    VelocityResult,
    build_synthetic_pts,
    cross_plot_analysis,
    calc_fluid_velocity,
    qc_velocity,
    run_velocity_analysis,
    normalise_feedzones,
    inwell_flow_fraction,
)
from .feedzones import FeedZoneResult, combine_and_detect, injectivity_from_velocity

__version__ = "1.0.0"

__all__ = [
    "AnalysisConfig", "WellGeometry", "DEFAULT_CONFIG", "DEFAULT_GEOMETRY",
    "tsat_from_p_mpa", "bara_from_barg",
    "load_profile", "profile_summary",
    "TemperatureResult", "compute_gradient", "temperature_feedzone_score",
    "run_temperature_analysis",
    "FeedZone", "VelocityResult", "build_synthetic_pts", "cross_plot_analysis",
    "calc_fluid_velocity", "qc_velocity", "run_velocity_analysis",
    "normalise_feedzones", "inwell_flow_fraction",
    "FeedZoneResult", "combine_and_detect", "injectivity_from_velocity",
    "__version__",
]
