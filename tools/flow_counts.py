"""Recompute the Figure 3 loss-to-follow-up boxes and check them against the extended outcome set.

    python tools/flow_counts.py --data ../tab --work ../work_local
Prints aggregate counts (local use; do not share counts below 10).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import argparse

import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.association import death_status, extended_set

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../tab")
ap.add_argument("--work", default="../work_local")
a = ap.parse_args()
config.DATA_DIR, config.WORK_DIR = Path(a.data), Path(a.work)

f = pd.read_pickle(config.WORK_DIR / "features.pkl")
iv = pd.read_pickle(config.WORK_DIR / "interviews.pkl")
allv = f[f["z_personal"].notna()].copy()
allv["status"] = death_status(allv, iv)
lost = allv[~allv["primary"]]
print(f"Eligible baseline visits without HbA1c at the next round: {len(lost):,}")
print(lost["status"].value_counts().to_string(), "\n")
ext = extended_set(f, iv)
added = allv.loc[ext.index[~ext["primary"]]]
print(f"Visits added to the extended set: {len(added):,}, cases {int(added['incident_dm'].sum())}")
print(pd.crosstab(added["status"], added["incident_dm"], margins=True).to_string())
