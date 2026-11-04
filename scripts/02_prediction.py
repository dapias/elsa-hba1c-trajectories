"""Section 4: out-of-fold prediction models A-E (Table 3, Table S3), screening counts, final models (Table S4).

Needs features.pkl from 01_prepare_data.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.cli import setup
from elsa_hba1c.features import derive
from elsa_hba1c.output import log
from elsa_hba1c.prediction import alert_comparison, export_final_models, step4

if __name__ == "__main__":
    setup(__doc__)
    features = derive(pd.read_pickle(config.WORK_DIR / "features.pkl"))
    pred, perf = step4(features)
    alert_comparison(pred)
    export_final_models(features)
    log(f"done: tables in {config.OUT_DIR}")
