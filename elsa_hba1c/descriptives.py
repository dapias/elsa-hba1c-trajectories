"""Descriptive tables: Table 1, characteristics by HbA1c range, history structure, flow."""
from __future__ import annotations

import numpy as np
import pandas as pd
from . import config
from .config import HBA1C_PREDIAB, MIN_GAP_YEARS, NEXT_ROUND, SLOPE
from .output import n_str, save
from .association import pre


# Descriptive statistics: flow chart, Table 1, history structure
def mean_sd(x):
    x = x.dropna()
    return f"{x.mean():.2f} ({x.std():.2f})" if len(x) else ""


def med_iqr(x, d=1):
    x = x.dropna()
    return f"{x.median():.{d}f} ({x.quantile(.25):.{d}f}-{x.quantile(.75):.{d}f})" if len(x) else ""


def pct(mask, base):
    k, n = int(mask.sum()), int(base.sum())
    return f"{n_str(k)} ({100 * k / n:.1f})" if n else ""


def ifcc_to_pct(x):
    return 0.09148 * x + 2.152


def table1_col(d):
    b, hb = pd.Series(True, index=d.index), d["hba1c"]
    row = {"baseline visits, n": n_str(len(d)), "persons, n": n_str(d["idauniq"].nunique()),
           "age, years, mean (SD)": mean_sd(d["age"]), "women, n (%)": pct(d["female"] == 1, d["female"].notna()),
           "HbA1c, mmol/mol, mean (SD)": mean_sd(hb), "HbA1c, mmol/mol, median (IQR)": med_iqr(hb),
           "HbA1c, %, mean (SD)": mean_sd(ifcc_to_pct(hb)),
           "HbA1c 42-47.9 mmol/mol, n (%)": pct(hb >= HBA1C_PREDIAB, hb.notna()),
           "personal slope, mean (SD)": mean_sd(d[SLOPE]), "BMI, mean (SD)": mean_sd(d["bmi"]),
           "waist, cm, mean (SD)": mean_sd(d["wstval"]) if "wstval" in d else "",
           "systolic BP, mean (SD)": mean_sd(d["sysval"]),
           "BP medication, n (%)": pct(d["bp_med"] == 1, d["bp_med"].notna()),
           "HDL, mean (SD)": mean_sd(d["hdl"]), "triglycerides, median (IQR)": med_iqr(d["trig"], 2),
           "current smoker, n (%)": pct(d["smoker"] == 1, d["smoker"].notna()),
           "vigorous activity weekly, n (%)": pct(d["vig_pa_weekly"] == 1, d["vig_pa_weekly"].notna()),
           "fasting sample, n (%)": pct(d["fasted"] == 1, b) if "fasted" in d else "",
           "earlier HbA1c values 1, n (%)": pct(d["n_hist"] == 1, b),
           "earlier HbA1c values 2, n (%)": pct(d["n_hist"] == 2, b),
           "earlier HbA1c values >= 3, n (%)": pct(d["n_hist"] >= 3, b),
           "years to next visit, median (IQR)": med_iqr(d["years_to_next"])}
    for c in ["bmi", "wstval", "sysval", "bp_med", "hdl", "trig", "smoker", "vig_pa_weekly"]:
        if c in d:
            row[f"missing {c}, %"] = f"{100 * d[c].isna().mean():.1f}"
    return pd.Series(row)


def descriptives(nlm, features, allv, pdf, lmm):
    cand = nlm[nlm["round"].isin(list(NEXT_ROUND))]
    c3 = cand[cand["hba1c"].notna() & cand["age"].notna() & cand["female"].notna()]
    steps = [("health-visit records, rounds 2, 4, 6, 8/9", cand), ("valid HbA1c, age and sex", c3),
             ("free of diabetes at the visit", c3[c3["dm_cum"] == 0]),
             ("eligible baseline visits (>= 1 earlier non-diabetic HbA1c)", features),
             ("analysis set: HbA1c at the next round", features[features["primary"]])]
    flow = [dict(step=s, n_visits=len(d), n_persons=d["idauniq"].nunique()) for s, d in steps]
    flow += [dict(step=f"not followed up: {s}", n_visits=int(n), n_persons=np.nan)
             for s, n in allv.loc[~allv["primary"], "status"].value_counts().items()]
    save(pd.DataFrame(flow), "T1_flow", ["n_visits", "n_persons"])

    lost = allv[~allv["primary"]]
    by_outcome = pd.DataFrame({"all": table1_col(pdf), "incident diabetes": table1_col(pdf[pdf["incident_dm"] == 1]),
                               "no diabetes": table1_col(pdf[pdf["incident_dm"] == 0]),
                               "not followed up": table1_col(lost)})
    by_band = pd.DataFrame({"HbA1c < 42": table1_col(pdf[pdf["hba1c"] < HBA1C_PREDIAB]),
                            "HbA1c 42-47.9": table1_col(pre(pdf))})
    by_outcome.rename_axis("characteristic").reset_index().to_csv(config.OUT_DIR / "T1_characteristics_by_outcome.csv", index=False)
    by_band.rename_axis("characteristic").reset_index().to_csv(config.OUT_DIR / "T1_characteristics_by_band.csv", index=False)

    hh = pdf[["idauniq", "t"]].reset_index(names="row").merge(lmm[["idauniq", "t"]].rename(columns={"t": "t_e"}),
                                                              on="idauniq")
    hh = hh[hh["t_e"] < hh["t"] - MIN_GAP_YEARS]
    first = hh.groupby("row")["t_e"].min()
    gaps = pd.concat([hh[["row", "t_e"]], pdf[["t"]].reset_index(names="row").rename(columns={"t": "t_e"})]
                     ).sort_values(["row", "t_e"])
    gaps["gap"] = gaps.groupby("row")["t_e"].diff()
    ph = pdf.assign(history_years=pdf["t"] - pdf.index.map(first),
                    median_gap=pdf.index.map(gaps.groupby("row")["gap"].median()),
                    n_hist_cat=np.where(pdf["n_hist"] >= 3, "3+", pdf["n_hist"].astype(int).astype(str)),
                    round_lab=pdf["round"].map({4: "4", 6: "6", 8: "8/9"}), all="all")
    rows = []
    for by, label in [("n_hist_cat", "earlier HbA1c values"), ("round_lab", "baseline round"), ("all", "all")]:
        for k, g in ph.groupby(by):
            rows.append(dict(grouping=label, group=k, baseline_visits=len(g), persons=g["idauniq"].nunique(),
                             cases=int(g["incident_dm"].sum()), history_years=med_iqr(g["history_years"]),
                             interval_between_values=med_iqr(g["median_gap"]),
                             years_to_next=med_iqr(g["years_to_next"]),
                             hba1c_42_48_pct=100 * (g["hba1c"] >= HBA1C_PREDIAB).mean()))
    save(pd.DataFrame(rows), "T1_history_structure", ["baseline_visits", "persons", "cases"])
