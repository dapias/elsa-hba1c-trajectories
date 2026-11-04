"""Section 2. Baseline visits and outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .config import FGLU_DIABETES, HBA1C_DIABETES, MIN_GAP_YEARS, MIN_HISTORY, NEXT_ROUND
from .output import log


def mark_diabetes(nl, iv):
    """lab_dm: diabetes by lab values; dm_cum: diabetic at this visit or before (once diabetic, always)."""
    nl = nl.sort_values(["idauniq", "t"]).reset_index(drop=True)
    nl["lab_dm"] = ((nl["hba1c"] >= HBA1C_DIABETES) | (nl["fglu"] >= FGLU_DIABETES)).astype(float)
    nl.loc[nl["hba1c"].isna() & nl["fglu"].isna(), "lab_dm"] = np.nan
    first_report = nl["idauniq"].map(iv[iv["dm_report"] == 1].groupby("idauniq")["wave"].min())
    dm_now = (nl["lab_dm"] == 1) | (first_report <= nl["core_wave"])
    nl["dm_cum"] = dm_now.astype(float).groupby(nl["idauniq"]).cummax()
    return nl


def learning_visits(nl):
    lmm = nl[(nl["dm_cum"] == 0) & nl["hba1c"].notna() & nl["age"].notna() & nl["female"].notna()]
    return lmm[["idauniq", "round", "t", "age", "female", "hba1c"]].copy()


def collect_candidates(nl):
    cand = nl[nl["round"].isin(list(NEXT_ROUND))]
    by_person = {pid: g for pid, g in nl.groupby("idauniq")}
    rows = []
    for _, v in cand.iterrows():
        visits = by_person[v["idauniq"]]
        prior = visits[visits["t"] < v["t"] - MIN_GAP_YEARS]
        nr = NEXT_ROUND[int(v["round"])]
        nxt = visits[visits["round"] == nr]
        has = len(nxt) > 0
        rows.append(dict(
            idauniq=v["idauniq"], round=int(v["round"]), core_wave=int(v["core_wave"]), t=v["t"],
            ok_hba1c=pd.notna(v["hba1c"]), ok_age=pd.notna(v["age"]) and pd.notna(v["female"]),
            dm_now=v["dm_cum"] == 1, n_hist=int((prior["hba1c"].notna() & (prior["dm_cum"] == 0)).sum()),
            next_round=nr, next_core_wave=int(nxt["core_wave"].iloc[0]) if has else (9 if nr == 8 else nr),
            t_next=nxt["t"].iloc[0] if has else np.nan, next_hba1c=nxt["hba1c"].iloc[0] if has else np.nan,
            next_lab_dm=nxt["lab_dm"].iloc[0] if has else np.nan))
    ix = pd.DataFrame(rows)
    ix["idauniq"] = ix["idauniq"].astype(nl["idauniq"].dtype)
    return ix


def apply_inclusion_rules(ix):
    return ix[ix["ok_hba1c"] & ix["ok_age"] & ~ix["dm_now"] & (ix["n_hist"] >= MIN_HISTORY)]


def add_outcome(ix, iv):
    ix = ix.copy()
    reports = iv[iv["dm_report"] == 1][["idauniq", "wave"]]
    m = ix[["idauniq", "core_wave", "next_core_wave"]].reset_index().merge(reports, on="idauniq")
    m = m[(m["wave"] > m["core_wave"]) & (m["wave"] <= m["next_core_wave"])]
    ix["report_incident"] = ix.index.isin(m["index"]).astype(float)
    ix["incident_dm"] = ((ix["next_lab_dm"] == 1) | (ix["report_incident"] == 1)).astype(float)
    ix["primary"] = ix["next_hba1c"].notna()
    return ix


INDEX_COVARIATES = ["idauniq", "round", "age", "female", "educ", "hba1c", "bmi", "sysval", "hdl", "trig",
                    "smoker", "vig_pa_weekly", "bp_med", "working", "job_phys", "job_stress", "nssec8"]


def add_index_covariates(ix, nl):
    keep = [k for k in INDEX_COVARIATES if k in nl.columns]
    ix = ix.merge(nl[keep].drop_duplicates(["idauniq", "round"]), on=["idauniq", "round"], how="left",
                  validate="many_to_one")
    ix["years_to_next"] = ix["t_next"] - ix["t"]
    return ix.reset_index(drop=True)


def step2(nl, iv):
    nlm = mark_diabetes(nl, iv)
    lmm = learning_visits(nlm)
    ix = add_index_covariates(add_outcome(apply_inclusion_rules(collect_candidates(nlm)), iv), nlm)
    log(f"step 2: {int(ix['primary'].sum()):,} primary baseline visits, "
        f"{int(ix.loc[ix['primary'], 'incident_dm'].sum())} cases")
    return nlm, lmm, ix
