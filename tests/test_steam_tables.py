"""Tests for steam tables and profile loading."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geothermal_welltest import bara_from_barg, load_profile, tsat_from_p_mpa

DATA = Path(__file__).resolve().parents[1] / "data" / "Data-Temp-Heating37days.csv"


class TestSteamTables(unittest.TestCase):
    def test_reference_boiling_points(self):
        # IAPWS-IF97 reference saturation temperatures (degC)
        cases = [(0.101325, 99.97), (1.0, 179.88), (5.0, 263.94), (10.0, 311.00)]
        for p, ref in cases:
            self.assertAlmostEqual(tsat_from_p_mpa(p), ref, delta=0.1)

    def test_vectorised(self):
        out = tsat_from_p_mpa(np.array([0.1, 1.0, 5.0]))
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(np.diff(out) > 0))  # monotonic increasing

    def test_out_of_range_is_nan(self):
        self.assertTrue(np.isnan(tsat_from_p_mpa(100.0)))  # above critical pressure

    def test_barg_to_bara(self):
        self.assertAlmostEqual(float(bara_from_barg(0.0)), 1.01325, places=5)


class TestLoadProfile(unittest.TestCase):
    def test_loads_and_derives_columns(self):
        df = load_profile(DATA)
        for col in ("pressure_bara", "pressure_mpa", "tsat_degC", "t_minus_tsat"):
            self.assertIn(col, df.columns)
        self.assertTrue(df.depth_m.is_monotonic_increasing)

    def test_bara_is_barg_plus_atm(self):
        df = load_profile(DATA)
        np.testing.assert_allclose(df.pressure_bara, df.pres_barg + 1.01325, rtol=0, atol=1e-9)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_profile("does/not/exist.csv")


if __name__ == "__main__":
    unittest.main()
