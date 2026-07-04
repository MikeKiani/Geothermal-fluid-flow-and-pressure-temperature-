"""Data loading and quality control for downhole temperature/pressure profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .config import AnalysisConfig, DEFAULT_CONFIG
from .steam_tables import bara_from_barg, tsat_from_p_mpa

REQUIRED_COLUMNS = {"depth_m", "whp_barg", "pres_barg", "temp_degC"}


def load_profile(
    path: Union[str, Path],
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load a heating temperature/pressure profile and derive analysis columns.

    The input CSV must contain ``depth_m``, ``whp_barg``, ``pres_barg`` and
    ``temp_degC``. The returned frame adds absolute pressure, the boiling point
    for depth (``tsat_degC``) and the below-boiling margin (``t_minus_tsat``).

    Parameters
    ----------
    path
        Path to the CSV file.
    config
        Analysis configuration (used for the atmospheric-pressure constant).

    Returns
    -------
    pandas.DataFrame
        Depth-sorted profile with derived columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.sort_values("depth_m").reset_index(drop=True)

    # bar gauge -> bar absolute is +1 atm (a common data-prep error is to subtract)
    df["pressure_bara"] = bara_from_barg(df["pres_barg"], config.atmospheric_bar)
    df["pressure_mpa"] = df["pressure_bara"] * 0.1          # IAPWS wants MPa
    df["tsat_degC"] = tsat_from_p_mpa(df["pressure_mpa"])    # boiling point for depth
    df["t_minus_tsat"] = df["temp_degC"] - df["tsat_degC"]   # +ve => at/over boiling
    return df


def profile_summary(df: pd.DataFrame) -> dict:
    """Return a small dict of headline numbers for logging/printing."""
    return {
        "n_points": int(len(df)),
        "depth_min_m": float(df.depth_m.min()),
        "depth_max_m": float(df.depth_m.max()),
        "temp_max_degC": float(df.temp_degC.max()),
        "temp_max_depth_m": float(df.loc[df.temp_degC.idxmax(), "depth_m"]),
    }
