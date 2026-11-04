"""Sections 6-8: drift model (Table 5, Figure 5), cap sweep (Table S11, Figure S4), slope adjusted for the
drift-model level and the simulation (Table S10, Figure 4B), observability (Figure 6, Table S12) and model checks
(Figures S1-S3).

Needs the work files from 01_prepare_data.py and the tables T2_variance_components_by_fold (01) and
T3_S1_adjusted_associations (03) in --out. The five cross-fitted trajectory models are refitted (a few minutes;
same folds and data as in step 3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.association import adjusted, pre, slope_including_today
from elsa_hba1c.checks import lmm_diagnostics, semivariogram_figure
from elsa_hba1c.cli import setup
from elsa_hba1c.cohort import learning_visits, mark_diabetes
from elsa_hba1c.config import CROSSING_CAP, N_FOLDS
from elsa_hba1c.current_level import LEVELS, add_level, simulate_selftest, slope_vs_level
from elsa_hba1c.drift import (
    CAPPED, add_crossing_visits, cap_sweep, censored_fits, check_censored_likelihood, crossing_value_numbers,
    drift_fits, drift_table,
)
from elsa_hba1c.features import assign_folds, derive, fit_population
from elsa_hba1c.figures import figure1, figure2_tau_profile
from elsa_hba1c.observability import observability
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
    _, lmm_f = assign_folds(features, lmm)                 # the fold of each person, as in step 3
    log("refitting the five cross-fitted trajectory models ...")
    models = {k: fit_population(lmm_f[lmm_f["fold"] != k]) for k in range(N_FOLDS)}
    vc = pd.read_csv(config.OUT_DIR / "T2_variance_components_by_fold.csv")
    assoc = pd.read_csv(config.OUT_DIR / "T3_S1_adjusted_associations.csv")

    # ---- step 6: drift model
    datasets = {"non-diabetic visits": lmm[["idauniq", "round", "age", "female", "hba1c"]],
                "plus crossing visit": add_crossing_visits(lmm, nl)}
    datasets[CAPPED] = datasets["plus crossing visit"].assign(
        hba1c=datasets["plus crossing visit"]["hba1c"].clip(upper=CROSSING_CAP))
    base_cross = datasets["plus crossing visit"]
    rounds6 = sorted(base_cross["round"].unique())
    FITS, extra = drift_fits(datasets, rounds6)
    CENS, prof_c, f_c, Gs_c, nX_c = censored_fits(base_cross, rounds6, FITS, extra)
    check_censored_likelihood(f_c, Gs_c, nX_c)
    drift_table(FITS, extra, CENS, prof_c)
    crossing_value_numbers(base_cross, nl, lmm)
    log("cap sweep ..."); cap_sweep(base_cross, rounds6, CENS, prof_c)
    figure2_tau_profile(FITS, extra, CENS, prof_c)

    # ---- 6b: slope adjusted for the drift-model level; slope including today's value; self-test
    df = add_level(df, lmm, FITS["plus crossing visit"]["drift"], rounds6, "level_drift")
    df = add_level(df, lmm, f_c, rounds6, "level_drift_censored")
    sc = save(slope_vs_level(df, LEVELS), "S1_slope_vs_drift_level", ["n_baseline_visits", "n_events"])
    df["slope_incl_today"] = slope_including_today(df, lmm_f, models)
    rows = []
    for popl, d in [("all", df), ("HbA1c 42-47.9", pre(df))]:
        for lev in ["hba1c", "level_drift"]:
            r = adjusted(d.dropna(subset=["level_drift"]) if lev != "hba1c" else d, "slope_incl_today", level=lev)
            rows.append(dict(population=popl, adjusted_for=LEVELS[lev], **r))
    save(pd.DataFrame(rows), "S1_slope_including_baseline_value", ["n_baseline_visits", "n_events"])
    simulate_selftest()
    figure1(assoc, sc)

    # ---- 7-8: observability and model checks
    observability(vc, FITS, f_c)
    semivariogram_figure(datasets, {k: FITS[k] for k in FITS if k in datasets}, rounds6)
    lmm_diagnostics(lmm, FITS)
    log(f"done: tables and figures in {config.OUT_DIR}")
