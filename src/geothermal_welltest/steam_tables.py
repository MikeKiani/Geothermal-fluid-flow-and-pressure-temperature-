"""Steam-table utilities.

A dependency-free implementation of the IAPWS-IF97 Region 4 *backward* equation
for saturation temperature as a function of pressure, T_sat(p). This replaces the
external ``iapws`` package for the one quantity the well-test workflow needs: the
boiling-point-for-depth (BPD) curve.

Accuracy is ~0.01 degC against the IAPWS-IF97 reference values across the valid
range 611.213 Pa to 22.064 MPa.

Reference
---------
Wagner, W. et al. (2000). *The IAPWS Industrial Formulation 1997 for the
Thermodynamic Properties of Water and Steam*. J. Eng. Gas Turbines Power, 122.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

# IAPWS-IF97 Region 4 coefficients (n1..n10)
_N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
)

_KELVIN = 273.15
_P_MIN_MPA = 611.213e-6   # triple point
_P_MAX_MPA = 22.064       # critical point


def tsat_from_p_mpa(p_mpa: Union[float, ArrayLike]) -> Union[float, np.ndarray]:
    """Saturation (boiling) temperature from pressure.

    Parameters
    ----------
    p_mpa
        Absolute pressure in megapascals (MPa). Scalar or array-like.

    Returns
    -------
    float or numpy.ndarray
        Saturation temperature in degrees Celsius. Values outside the valid
        pressure range return NaN.

    Examples
    --------
    >>> round(float(tsat_from_p_mpa(0.101325)), 2)
    99.97
    """
    p = np.asarray(p_mpa, dtype=float)
    valid = (p >= _P_MIN_MPA) & (p <= _P_MAX_MPA)
    with np.errstate(invalid="ignore"):
        beta = np.where(valid, p, np.nan) ** 0.25
        n = _N
        e = beta ** 2 + n[2] * beta + n[5]
        f = n[0] * beta ** 2 + n[3] * beta + n[6]
        g = n[1] * beta ** 2 + n[4] * beta + n[7]
        d = 2 * g / (-f - np.sqrt(f ** 2 - 4 * e * g))
        t = (n[9] + d - np.sqrt((n[9] + d) ** 2 - 4 * (n[8] + n[9] * d))) / 2.0
        tsat = t - _KELVIN
    tsat = np.where(valid, tsat, np.nan)
    return float(tsat) if tsat.ndim == 0 else tsat


def bara_from_barg(p_barg: Union[float, ArrayLike], atmospheric_bar: float = 1.01325):
    """Convert bar-gauge to bar-absolute (adds one atmosphere)."""
    return np.asarray(p_barg, dtype=float) + atmospheric_bar
