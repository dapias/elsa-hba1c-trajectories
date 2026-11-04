"""Section 5: associations of the four features (Table S7, Figure 4A) and the sensitivity analyses of the slope
association (Tables S8-S9), with Table 1, Tables S1-S2 and the flow chart (Figure 3).

Needs the work files from 01_prepare_data.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import numpy as np
import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.association import (
    death_status, death_vs_slope, deaths_scenarios, hba1c_only_outcome, interaction, long_history_checks,
    multiple_imputation, slope_covariate_steps, slope_crude, slope_history_and_age, slope_reversal_diagnostics,
    step5_associations,
)
from elsa_hba1c.cli import setup
from elsa_hba1c.cohort import learning_visits, mark_diabetes
from elsa_hba1c.descriptives import descriptives
from elsa_hba1c.features import derive
from elsa_hba1c.output import log, save

def load_work():
    """Health visits, interviews and features written by 01_prepare_data.py."""
    nl = pd.read_pickle(config.WORK_DIR / "nurse_long.pkl")
    iv = pd.read_pickle(config.WORK_DIR / "interviews.pkl")
    nlm = mark_diabetes(nl, iv)
    lmm = learning_visits(nlm)                       # as in step2()
    features = derive(pd.read_pickle(config.WORK_DIR / "features.pkl"))
    return nl, iv, nlm, lmm, features


if __name__ == "__main__":
    setup(__doc__)
    nl, iv, nlm, lmm, features = load_work()
    df = features[features["primary"] & features["z_personal"].notna()].copy()        # analysis set
    log(f"analysis set: {len(df):,} baseline visits, {int(df['incident_dm'].sum())} cases")
    assoc = step5_associations(features, df, iv)
    slope_crude(df)
    slope_reversal_diagnostics(df)
    slope_covariate_steps(df)
    slope_history_and_age(df)
    inter = pd.concat([interaction(df).assign(population="all"),
                       interaction(df[df["n_hist"] >= 2]).assign(population=">= 2 earlier values")], ignore_index=True)
    save(inter, "S1_slope_hba1c_interaction")
    log("multiple imputation ..."); multiple_imputation(df)

    extra_nl = nl[["idauniq", "round", "wstval", "fglu", "fasted"]].drop_duplicates(["idauniq", "round"])
    allv = features[features["z_personal"].notna()].merge(extra_nl, on=["idauniq", "round"], how="left",
                                                          suffixes=("", "_nl"))
    allv.index = features[features["z_personal"].notna()].index
    allv["status"] = death_status(allv, iv)
    pdf = allv[allv["primary"]]
    descriptives(nlm, features, allv, pdf, lmm)
    death_vs_slope(allv)
    dead = allv[allv["status"] == "died before next round"].copy()
    dead["years_to_next"] = dead["round"].map(pdf.groupby("round")["years_to_next"].median())
    dead["log_years_next"] = np.log(dead["years_to_next"])
    log("deaths scenarios ..."); deaths_scenarios(pdf, dead)
    long_history_checks(pdf, lmm)
    hba1c_only_outcome(pdf, nl)
    log(f"done: tables in {config.OUT_DIR}")
