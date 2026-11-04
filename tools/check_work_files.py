"""Check that interviews.pkl, nurse_long.pkl and features.pkl in --work match the current code: rerun steps 1-3
into a fresh folder and compare column by column.

    python tools/check_work_files.py --data ../tab --work ../work_local --fresh ../work_check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import argparse
import time

import numpy as np
import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.cohort import step2
from elsa_hba1c.data import step1
from elsa_hba1c.features import step3
from elsa_hba1c.output import log

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../tab")
ap.add_argument("--work", default="../work_local", help="folder with the existing pickles")
ap.add_argument("--fresh", default="../work_check", help="new, empty folder for the regenerated pickles")
a = ap.parse_args()
old_dir, new_dir = Path(a.work), Path(a.fresh)
new_dir.mkdir(parents=True, exist_ok=True)
(new_dir / "out").mkdir(exist_ok=True)
for f in ["interviews.pkl", "nurse_long.pkl", "features.pkl"]:
    p = old_dir / f
    print(f"{f:16s} {'last modified: ' + time.ctime(p.stat().st_mtime) if p.exists() else 'MISSING'}")

config.DATA_DIR, config.WORK_DIR, config.OUT_DIR = Path(a.data), new_dir, new_dir / "out"
log("step 1 ..."); _, nl, iv = step1()
nlm, lmm, cohort = step2(nl, iv)
log("step 3 (cross-fitted mixed models) ..."); step3(cohort, lmm)


def compare(name, keys):
    old, new = pd.read_pickle(old_dir / name), pd.read_pickle(new_dir / name)
    print(f"\n=== {name}: old {old.shape}, new {new.shape}")
    common = [c for c in new.columns if c in old.columns]
    for side, cols in [("old", set(old) - set(new)), ("new", set(new) - set(old))]:
        if cols:
            print(f"  columns only in {side}:", sorted(cols))
    o = old[common].sort_values(keys).reset_index(drop=True)
    n = new[common].sort_values(keys).reset_index(drop=True)
    if len(o) != len(n):
        print("  DIFFERENT number of rows")
        return False
    same = True
    for c in common:
        x, y = o[c], n[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            ok = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            ok = x.astype(str).equals(y.astype(str))
        if not ok:
            print(f"  {c}: values differ")
        same &= ok
    print("  IDENTICAL" if same else "  not identical (see above)")
    return same


ok = [compare("interviews.pkl", ["idauniq", "wave"]), compare("nurse_long.pkl", ["idauniq", "round"]),
      compare("features.pkl", ["idauniq", "round"])]
print("\nRESULT:", "all three match the current code" if all(ok) else "at least one file differs")
