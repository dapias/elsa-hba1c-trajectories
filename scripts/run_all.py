"""Run all analyses of the paper except the Transformer, in one process (the reference run).

    python scripts/run_all.py --data ../tab --work ../work_local --out ../outputs/paper

    --quick           100 bootstrap repetitions instead of 1,000
    --refit           ignore cached likelihood fits
    --from-features   only Table S4, the association analyses and the slope diagnostics, from features.pkl and
                      interviews.pkl of an earlier full run (minutes)
    --figures-only    only redraw Figures 4-6, S3 and S4 from saved outputs and cached fits (seconds)

Inputs  UKDS SN 5050 .tab files (licensed; never share) in --data.
Work    person-level files and cached likelihood fits in --work (never share).
Output  aggregate tables (CSV, counts < 10 suppressed) and figures in --out; README.md maps them to the paper.

Run time: about 1-2 hours on a laptop the first time. Likelihood fits are cached in --work
(06_drift_fits.json, 08d_cap_sweep.json, 08e_censored_drift.json, 08r_profile_extra.json) and reused.
The same analyses can be run section by section with scripts/01_*.py to 06_*.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from elsa_hba1c import config
from elsa_hba1c.config import CROSSING_CAP
from elsa_hba1c.output import log, save
from elsa_hba1c.data import step1
from elsa_hba1c.cohort import step2
from elsa_hba1c.features import derive, step3
from elsa_hba1c.prediction import alert_comparison, export_final_models, step4
from elsa_hba1c.association import (
    adjusted, death_status, death_vs_slope, deaths_scenarios, hba1c_only_outcome, interaction,
    long_history_checks, multiple_imputation, pre, slope_covariate_steps, slope_crude,
    slope_history_and_age, slope_including_today, slope_reversal_diagnostics, step5_associations,
)
from elsa_hba1c.descriptives import descriptives
from elsa_hba1c.drift import (
    CAPPED, add_crossing_visits, cap_sweep, censored_fits, check_censored_likelihood,
    crossing_value_numbers, drift_fits, drift_table,
)
from elsa_hba1c.current_level import LEVELS, add_level, simulate_selftest, slope_vs_level
from elsa_hba1c.observability import observability
from elsa_hba1c.checks import lmm_diagnostics, semivariogram_figure
from elsa_hba1c.figures import figure1, figure2_tau_profile, figures_only


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../tab", help="folder with the UKDS .tab files")
    ap.add_argument("--work", default="../work_local", help="person-level files and fit caches (local only)")
    ap.add_argument("--out", default="../outputs/paper", help="aggregate tables and figures")
    ap.add_argument("--quick", action="store_true", help="100 bootstrap repetitions instead of 1,000")
    ap.add_argument("--refit", action="store_true", help="ignore cached likelihood fits")
    ap.add_argument("--figures-only", action="store_true",
                    help="only redraw Figures 4, 5, 6, S1 and S2 from saved outputs and cached fits (seconds)")
    ap.add_argument("--from-features", action="store_true",
                    help="skip the slow steps: rerun only Table S8, the association analyses (Table 4, S1), the\n                         crude slope ORs and the sign-reversal diagnostics from features.pkl and interviews.pkl of an earlier full run (minutes)")
    a = ap.parse_args()
    config.DATA_DIR, config.WORK_DIR, config.OUT_DIR = Path(a.data), Path(a.work), Path(a.out)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True); config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.figures_only:
        figures_only()
        return
    if a.from_features:
        features = derive(pd.read_pickle(config.WORK_DIR / "features.pkl"))
        iv = pd.read_pickle(config.WORK_DIR / "interviews.pkl")
        export_final_models(features)
        df = features[features["primary"] & features["z_personal"].notna()].copy()
        step5_associations(features, df, iv)
        slope_crude(df)
        slope_reversal_diagnostics(df)
        slope_covariate_steps(df)
        slope_history_and_age(df)
        log(f"done (from saved features): tables in {config.OUT_DIR}")
        return
    config.N_BOOT, config.REFIT = (100 if a.quick else 1000), a.refit

    # ---- steps 1-3: data, cohort, cross-fitted personal features
    log("step 1: reading ELSA ..."); _, nl, iv = step1()
    nlm, lmm, cohort = step2(nl, iv)
    features, vc, models, lmm_f = step3(cohort, lmm)
    features = derive(features)

    # ---- step 4: prediction
    pred, perf = step4(features)
    alert_comparison(pred)
    export_final_models(features)

    # ---- step 5: associations and sensitivity analyses
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


if __name__ == "__main__":
    main()
