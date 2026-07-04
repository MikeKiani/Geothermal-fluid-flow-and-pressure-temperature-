"""Command-line pipeline for the geothermal well-test workflow.

Installed as the ``welltest-analyze`` console command (see pyproject.toml) and
also invoked by ``scripts/run_analysis.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe

from . import (
    DEFAULT_CONFIG, DEFAULT_GEOMETRY, build_synthetic_pts, combine_and_detect,
    load_profile, profile_summary, run_temperature_analysis, run_velocity_analysis,
)
from . import plotting

# Feed zones (top_m, bottom_m, relative injectivity) for the forward model.
# These sit on the temperature-derived permeable intervals; edit to match a well.
FEEDZONES = [
    (560, 600, 0.10),
    (665, 690, 0.08),
    (720, 745, 0.14),
    (748, 775, 0.24),   # major
    (800, 835, 0.28),   # major
    (852, 872, 0.11),
    (905, 925, 0.05),
]

PUMP_RATES_TPH = {"low (45 t/hr)": 45.0, "high (110 t/hr)": 110.0, "mid (75 t/hr)": 75.0}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="welltest-analyze", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/Data-Temp-Heating37days.csv",
                   help="path to the heating temperature/pressure CSV")
    p.add_argument("--outdir", default="figures", help="directory for figures and tables")
    p.add_argument("--seed", type=int, default=7, help="RNG seed for the synthetic PTS model")
    p.add_argument("--no-synthetic-truth", action="store_true",
                   help="do not overlay the synthetic ground truth on plots")
    p.add_argument("--dpi", type=int, default=130, help="figure resolution")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plotting.apply_house_style()

    # 1. temperature ----------------------------------------------------------
    df = load_profile(args.data, DEFAULT_CONFIG)
    s = profile_summary(df)
    print(f"[data] {s['n_points']} points, {s['depth_min_m']:.0f}-{s['depth_max_m']:.0f} m, "
          f"Tmax {s['temp_max_degC']:.1f} °C at {s['temp_max_depth_m']:.0f} m")
    temp = run_temperature_analysis(df, DEFAULT_CONFIG, DEFAULT_GEOMETRY)
    print(f"[temperature] feed-zone picks: "
          f"{', '.join(f'{d:.0f} m' for d in temp.feedzone_depths) or 'none'}")

    # 2. fluid velocity (forward model -> cross-plot inversion) ----------------
    rng = np.random.default_rng(args.seed)
    pts_by_rate, v_true = {}, {}
    for name, rate in PUMP_RATES_TPH.items():
        pts, vt = build_synthetic_pts(rate, FEEDZONES, df, DEFAULT_GEOMETRY, rng=rng)
        pts_by_rate[name] = pts
        v_true[name] = vt
    velocity = run_velocity_analysis(
        pts_by_rate, DEFAULT_CONFIG, v_true=None if args.no_synthetic_truth else v_true)
    for name in velocity.raw:
        print(f"[velocity] {name}: {len(velocity.raw[name])} intervals -> "
              f"{len(velocity.clean[name])} after QC")

    # 3. combine and detect ---------------------------------------------------
    fz = combine_and_detect(temp.grad, velocity, FEEDZONES, DEFAULT_CONFIG, DEFAULT_GEOMETRY)
    print(f"[feed zones] confident picks: "
          f"{', '.join(f'{d:.0f} m' for d in fz.picks_m) or 'none'}")

    # 4. save figures + table -------------------------------------------------
    figs = {
        "01_temperature_dashboard.png": plotting.plot_temperature_dashboard(df, temp),
        "02_crossplot_example.png": plotting.plot_crossplot_example(pts_by_rate["high (110 t/hr)"]),
        "03_velocity_qc.png": plotting.plot_velocity_qc(velocity, FEEDZONES),
        "04_feedzone_signal.png": plotting.plot_feedzone_signal(velocity, FEEDZONES),
        "05_combined_interpretation.png": plotting.plot_combined_interpretation(df, temp, velocity, fz),
    }
    for name, fig in figs.items():
        fig.savefig(outdir / name, dpi=args.dpi, bbox_inches="tight")
    if not fz.table.empty:
        fz.table.to_csv(outdir / "feed_zones.csv", index=False)
        print("\nRanked feed zones:")
        print(fz.table.to_string(index=False))
    print(f"\n[done] wrote {len(figs)} figures + feed_zones.csv to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
