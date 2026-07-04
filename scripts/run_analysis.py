#!/usr/bin/env python3
"""Thin wrapper to run the analysis pipeline from a source checkout.

Adds ``src/`` to the path (so no install is required) and delegates to
``geothermal_welltest.cli.main``. After ``pip install .`` you can instead run the
installed console command ``welltest-analyze`` with the same arguments.

Examples
--------
    python scripts/run_analysis.py
    python scripts/run_analysis.py --data data/Data-Temp-Heating37days.csv --outdir figures
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geothermal_welltest.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
