"""Redraw Figure S1 (mixed-model diagnostics) and Figure S2 (residual correlation) without the full run.
Uses nurse_long.pkl and interviews.pkl from --work and the cached drift fit 06_drift_fits.json.

    python tools/redraw_diagnostics.py --work ../work_local --out ../outputs/paper
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import argparse
import json

import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.checks import lmm_diagnostics
from elsa_hba1c.cohort import learning_visits, mark_diabetes
from elsa_hba1c.output import log

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="../work_local")
ap.add_argument("--out", default="../outputs/paper")
a = ap.parse_args()
config.WORK_DIR, config.OUT_DIR = Path(a.work), Path(a.out)
config.OUT_DIR.mkdir(parents=True, exist_ok=True)
fits_file = config.WORK_DIR / "06_drift_fits.json"
if not fits_file.exists():
    raise SystemExit(f"{fits_file} not found - pass the folder with the cached drift fits as --work")
FITS = json.load(open(fits_file))
nl = pd.read_pickle(config.WORK_DIR / "nurse_long.pkl")
iv = pd.read_pickle(config.WORK_DIR / "interviews.pkl")
lmm = learning_visits(mark_diabetes(nl, iv))
log(f"{len(lmm):,} non-diabetic visits; fitting the mixed model once ...")
lmm_diagnostics(lmm, FITS)
print(f"written: {config.OUT_DIR / 'FIG_fig_lmm_diagnostics.png'}")
