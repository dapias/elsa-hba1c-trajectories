"""Redraw Figures 4-6, S3 and S4 from the saved output tables and the cached drift fits (seconds; no ELSA data).

Figure S1 (mixed-model diagnostics) refits the trajectory model: use tools/redraw_diagnostics.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

from elsa_hba1c.cli import setup
from elsa_hba1c.figures import figures_only

if __name__ == "__main__":
    setup(__doc__)
    figures_only()
