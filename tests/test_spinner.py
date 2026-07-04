"""Tests for the spinner forward/inverse model and feed-zone detection."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geothermal_welltest import (
    DEFAULT_CONFIG, DEFAULT_GEOMETRY, build_synthetic_pts, calc_fluid_velocity,
    combine_and_detect, cross_plot_analysis, inwell_flow_fraction, load_profile,
    normalise_feedzones, run_temperature_analysis, run_velocity_analysis,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "Data-Temp-Heating37days.csv"
FEEDZONES = [(560, 600, 0.10), (665, 690, 0.08), (720, 745, 0.14), (748, 775, 0.24),
             (800, 835, 0.28), (852, 872, 0.11), (905, 925, 0.05)]


class TestForwardModel(unittest.TestCase):
    def test_weights_normalise(self):
        fz = normalise_feedzones(FEEDZONES)
        self.assertAlmostEqual(sum(w for _, _, w in fz), 1.0, places=9)

    def test_flow_fraction_monotonic_decreasing(self):
        z = np.linspace(460, 926, 200)
        frac = inwell_flow_fraction(z, normalise_feedzones(FEEDZONES))
        self.assertLessEqual(frac[-1], frac[0])          # decreases with depth
        self.assertTrue(np.all((frac >= -1e-9) & (frac <= 1 + 1e-9)))
        self.assertLess(frac[-1], 0.1)                    # ~empty below deepest zone


class TestCrossPlotInversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_profile(DATA)
        rng = np.random.default_rng(1)
        cls.pts, cls.v_true = build_synthetic_pts(110.0, FEEDZONES, cls.df, rng=rng)

    def test_intercept_recovers_fluid_velocity(self):
        # in a 1 m window, the cross-plot intercept should match the true velocity
        top, bottom = 700.0, 701.0
        _, _, v, _, r2, n = cross_plot_analysis(self.pts, top, bottom)
        true_v = float(np.interp(700.5, self.v_true.depth_m, self.v_true.v_true))
        self.assertGreater(r2, 0.8)
        self.assertAlmostEqual(v, true_v, delta=0.05)

    def test_full_profile_tracks_truth(self):
        fv = calc_fluid_velocity(self.pts, 470, 920, 2.0).dropna()
        pred = np.interp(fv.depth_m, self.v_true.depth_m, self.v_true.v_true)
        err = np.abs(fv.intercept_velocity_mps - pred)
        self.assertLess(np.median(err), 0.05)            # median error < 5 cm/s

    def test_degenerate_interval_returns_nan(self):
        _, _, v, _, r2, n = cross_plot_analysis(self.pts, 10.0, 11.0)  # no data here
        self.assertTrue(np.isnan(v))


class TestFeedZoneDetection(unittest.TestCase):
    def test_detects_zones_near_model(self):
        df = load_profile(DATA)
        temp = run_temperature_analysis(df)
        rng = np.random.default_rng(7)
        pts_by_rate = {}
        for name, rate in {"low": 45.0, "high": 110.0, "mid": 75.0}.items():
            pts_by_rate[name], _ = build_synthetic_pts(rate, FEEDZONES, df, rng=rng)
        velocity = run_velocity_analysis(pts_by_rate, DEFAULT_CONFIG)
        fz = combine_and_detect(temp.grad, velocity, FEEDZONES)

        # every modelled major zone should have a pick within 20 m
        for a, b, w in FEEDZONES:
            if w < 0.15:
                continue
            mid = 0.5 * (a + b)
            self.assertTrue(np.any(np.abs(fz.picks_m - mid) < 20),
                            f"no pick near modelled zone at {mid:.0f} m")
        self.assertFalse(fz.table.empty)


if __name__ == "__main__":
    unittest.main()
