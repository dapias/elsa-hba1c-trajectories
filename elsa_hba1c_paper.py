#!/usr/bin/env python3
"""
Personal HbA1c trajectories and incident diabetes in ELSA: all analyses of the paper except the Transformer.

Consolidates ELSA_personal_baseline_all_steps.ipynb (steps 1-6), 07_reviewer_checks.ipynb and
08_cedric_feedback.ipynb into one script. Only the code behind results reported in mainv6 is kept;
exploratory cells and superseded analyses (e.g. the capped drift fit as a main result) are not.

Run:   python elsa_hba1c_paper.py --data ../tab --work ../work_local --out ../outputs
       (--quick: 100 bootstrap repetitions; --refit: ignore cached drift fits;
        --from-features: only Table S8, the association analyses and the crude slope ORs, from the
        features saved by an earlier full run; --figures-only: redraw Figures 4-6, S1 and S2 from saved outputs)

Inputs  UKDS SN 5050 .tab files (licensed; never share) in --data.
Work    person-level files and the cached likelihood fits in --work (never share).
Output  aggregate tables (CSV, counts < 10 suppressed) and figures in --out.

Where the results go in the paper (output file prefixes):
  T1_*        Table 1 (characteristics), Supplementary tables by HbA1c band and history structure, flow chart
  T2_*        Table 2 (variance components, ICC, index of individuality, RCV)
  T3_*        Table 3 / Figure 1A (adjusted ORs of the four features)
  AUC_*       Table of out-of-fold AUC, ΔAUC, calibration, screening counts
  S1_*        Supplementary Table S1 (sensitivity analyses of the slope association)
  T4_*        Table 4 (independent-noise and drift models: non-diabetic, plus crossing value, censored)
  S2_*        Supplementary Table S2 / Figure 5 (slope observability)
  S5_*        Supplementary cap-sweep table and figure
  S8_*        Supplementary Table S8 (final models A and B; the full-precision CSV is for local use only)
  FIG_*       Figures (fig1_odds_ratios, fig2_tau_profile, fig_observability, fig_variogram,
              fig_cap_sweep, fig_lmm_diagnostics, fig_residual_correlation)
Not produced here: the Transformer (s07_strats.py) and the illustrative figures (fig_personal_baseline, fig_drift).

Run time: about 1-2 hours on a laptop the first time (cross-fitted mixed models, bootstraps, drift fits,
censored ML fits). The likelihood fits are cached in --work under the file names the notebooks used
(06_drift_fits.json, 08d_cap_sweep.json, 08e_censored_drift.json, 08r_profile_extra.json), so existing
caches are reused.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =====================================================================================================
# 0. Settings
# =====================================================================================================
DATA_DIR = WORK_DIR = OUT_DIR = Path(".")         # set in main()
N_BOOT = 1000
REFIT = False
RANDOM_SEED = 20261008

HBA1C_DIABETES = 48.0          # mmol/mol
HBA1C_PREDIAB = 42.0           # mmol/mol
FGLU_DIABETES = 7.0            # mmol/l, fasting glucose
MIN_HISTORY = 1                # earlier non-diabetic HbA1c values required at a baseline visit
MIN_GAP_YEARS = 0.25           # "earlier" = at least 3 months before
AGE_CENTRE = 65                # age enters as (age - 65) / 10
ROUNDS = [2, 4, 6, 8, 11]      # health-visit rounds; 8 stands for 8/9
NEXT_ROUND = {2: 4, 4: 6, 6: 8, 8: 11}
HOMOGENEOUS_ROUNDS = [4, 6]    # baseline rounds with a ~4-year interval
N_FOLDS = 5
MIN_CELL = 10                  # disclosure control: counts below 10 are shown as "<10"

# drift model
TAU_GRID = [0.5, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12.5, 15, 17.5, 20, 25, 30, 50, 100]
EXTRA_TAUS_NONDIABETIC = [35, 40, 60, 70, 80, 90, 95]          # where the non-diabetic profile is flat
CENSORED_PROFILE_TAUS = [2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 35, 50]
EXTRA_TAUS_CENSORED = [13, 14, 17.5, 22.5, 30, 40, 45, 60]
CROSSING_CAP = 55.0            # only as starting values for the censored fit (and one point of the cap sweep)
CAPS = [48, 50, 52, 55, 58, 60, 65, 70, 80, None]               # cap sweep (supplement); None = uncapped
SWEEP_TAUS = [1, 2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 35, 50, 100]
N_BOOT_VARIOGRAM = 200
LAG_BINS = [0, 3, 5, 7, 9.5, 12.5, 15, 25]

# sensitivity analyses
N_IMP = 20                     # multiple imputation
N_DRAW = 50                    # deaths: draws per scenario
K_RISK = [1, 2, 3]             # deaths: multiplier of the predicted risk

# observability
OBS_INTERVALS = [0.25, 0.5, 1, 2, 4]
OBS_LENGTHS = [4, 8, 12, 16, 20, 25, 30]

# ELSA files
HARMONIZED_FILE = "gh_elsa_h.tab"              # Gateway harmonised ELSA (waves 1-10)
W11_CORE_FILE = "wave_11_elsa_data_eul.tab"    # wave 11 is not in the harmonised file
W11_EOL_FILE = "wave_11_elsa_eol_eul.tab"      # end-of-life interviews after wave 10
NURSE_ROUNDS = {   # waves 8 and 9 are ONE health-visit round; the stand-alone wave-8 nurse file is not used
    2: dict(file="wave_2_nurse_data_v2.tab", fasting="fasteli", core_wave=2),
    4: dict(file="wave_4_nurse_data.tab", fasting="fastelig", core_wave=4),
    6: dict(file="wave_6_elsa_nurse_data_v2.tab", fasting="fastelig", core_wave=6),
    8: dict(file="elsa_nurse_w8w9_data_eul.tab", fasting="fastelig", core_wave="from_wave_column"),
    11: dict(file="wave_11_elsa_nurse_data_eul.tab", fasting="fastelig2", core_wave=11),
}
NURSE_VARS = ["hba1c", "fglu", "chol", "hdl", "ldl", "trig", "hscrp", "rtin", "hgb", "cfib", "wbc",
              "bmival", "wstval", "sysval", "diaval", "vismon", "visyear", "confage", "indager", "wave"]
PLAUSIBLE = {"hba1c": (15, 200), "fglu": (1.5, 40), "chol": (1, 20), "hdl": (0.1, 6), "ldl": (0.1, 15),
             "trig": (0.1, 40), "hscrp": (0.01, 300), "rtin": (1, 5000), "hgb": (4, 25), "cfib": (0.3, 12),
             "wbc": (0.5, 100), "bmival": (12, 80), "wstval": (40, 200), "sysval": (60, 260), "diaval": (30, 160)}
W = range(1, 11)               # interview waves in the harmonised file
HARM_STEMS = ["iwy", "iwm", "iwstat", "diabe", "rxdiab", "work", "jphysl", "jobsum", "nssec8", "smoken",
              "vgactx_e", "rxhibp", "mbmi", "mweight", "mheight"]

# models and covariates
HORIZON = ["log_years_next"]
BASE = ["hba1c", "age", "female"] + HORIZON
CLIN = BASE + ["bmi", "sysval", "hdl", "log_trig", "smoker", "vig_pa_weekly", "bp_med"]
MODELS = {"A_population": BASE, "B_clinical": CLIN, "C_naive_change": CLIN + ["delta_last"],
          "D_personal_baseline": CLIN + ["z_personal", "slope_personal"], "E_forecast": CLIN + ["logit_p_cross48"]}
COMPARISONS = [("D_personal_baseline", "B_clinical"), ("C_naive_change", "B_clinical"),
               ("D_personal_baseline", "C_naive_change"), ("E_forecast", "B_clinical"), ("B_clinical", "A_population")]
ADJ_NO_HORIZON = ["age", "female", "bmi", "sysval", "hdl", "log_trig", "smoker", "vig_pa_weekly", "bp_med"]
ADJ = ADJ_NO_HORIZON + ["log_years_next"]
EXPOSURES = {"z_personal": "deviation from personal baseline (z)",
             "delta_last": "change since last HbA1c, round-corrected",
             "logit_p_cross48": "forecast logit P(next HbA1c >= 48)",
             "slope_personal": "personal HbA1c slope (per decade)",
             "slope_incl_today": "personal slope incl. today's value (per decade)"}
SLOPE = "slope_personal"


# =====================================================================================================
# Output helpers (aggregate only)
# =====================================================================================================
def n_str(n):
    n = int(n)
    return f"<{MIN_CELL}" if 0 < n < MIN_CELL else f"{n:,}"


def save(t: pd.DataFrame, name: str, count_cols=()) -> pd.DataFrame:
    """Write an aggregate table to OUT_DIR/<name>.csv; counts in count_cols below MIN_CELL become '<10'."""
    if "idauniq" in t.columns:
        raise ValueError(f"{name}: refusing to write the person identifier")
    t = t.copy()
    for c in count_cols:
        if c in t:
            t[c] = t[c].map(lambda v: n_str(v) if pd.notna(v) and not isinstance(v, str) else v)
    t.to_csv(OUT_DIR / f"{name}.csv", index=False)
    return t


def or_text(r):
    return f"{r['or_per_sd']:.2f} ({r['or_lo']:.2f}-{r['or_hi']:.2f})"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================================================
# 1. Data: one row per health visit (nl) and per interview (iv)
# =====================================================================================================
def header(path: Path) -> list[str]:
    with open(path, "r", encoding="latin-1") as fh:
        return fh.readline().rstrip("\r\n").split("\t")


def read_tab(path: Path, wanted: list[str] | None = None) -> pd.DataFrame:
    """Read a UKDS .tab file, keeping `wanted` columns (case-insensitive); all values numeric."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - check --data")
    cols = header(path)
    keep = cols if wanted is None else [c for c in cols if c.lower() in {w.lower() for w in wanted}]
    df = pd.read_csv(path, sep="\t", usecols=keep, dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
    return df


def clean_negative(s):
    return s.where(s >= 0)                       # UKDS / harmonised special-missing codes are negative


def binary(s, yes=(1,), no=(0,)):
    out = pd.Series(np.nan, index=s.index)
    out[s.isin(yes)] = 1.0
    out[s.isin(no)] = 0.0
    return out


def decimal_date(year, month):
    """Decimal year; each date at the middle of its month, unknown month = mid-year."""
    y, m = clean_negative(year), clean_negative(month)
    m = m.where((m >= 1) & (m <= 12))
    return y + (m.fillna(6.5) - 0.5) / 12.0


def load_harmonized():
    wanted = ["idauniq", "rabyear", "ragender", "raeducl"] + [f"r{w}{s}" for w in W for s in HARM_STEMS]
    return read_tab(DATA_DIR / HARMONIZED_FILE, wanted)


def interviews_long(h):
    """One row per person and interview wave: diabetes reported (diagnosis or medication)."""
    rows = []
    for w in W:
        g = lambda s: h.get(f"r{w}{s}", pd.Series(np.nan, index=h.index))
        diag, med = binary(g("diabe")), binary(g("rxdiab"))
        dm = diag.where(med != 1, 1.0)
        d = pd.DataFrame({"idauniq": h["idauniq"], "wave": w, "t": decimal_date(g("iwy"), g("iwm")), "dm_report": dm})
        rows.append(d[(g("iwstat") == 1).values])
    c = read_tab(DATA_DIR / W11_CORE_FILE, ["idauniq", "iintdaty", "iintdatm", "hediaddi", "hedids12", "hehavedi"])
    dm = pd.Series(np.nan, index=c.index)
    dm[(c["hedids12"] == 0) | (c["hediaddi"] == 2)] = 0.0
    dm[(c["hediaddi"] == 1) | (c["hedids12"] == 1) | c["hehavedi"].isin([1, 2])] = 1.0
    rows.append(pd.DataFrame({"idauniq": c["idauniq"], "wave": 11,
                              "t": decimal_date(c["iintdaty"], c["iintdatm"]), "dm_report": dm}))
    return pd.concat(rows, ignore_index=True).sort_values(["idauniq", "wave"])


def detect_units(x, analyte):
    """Unit of a whole round from its median: HbA1c % (DCCT) -> mmol/mol (IFCC); haemoglobin g/l -> g/dl."""
    v = x[x > 0]
    med = float(v.median()) if len(v) else np.nan
    if analyte == "hba1c" and med < 15:
        return (x - 2.15) * 10.929, "% (DCCT) -> mmol/mol", med
    if analyte == "hgb" and med > 30:
        return x / 10.0, "g/l -> g/dl", med
    return x, "as recorded", med


def nurse_long(h):
    """One row per person and health visit, all rounds, units harmonised, interview covariates attached."""
    out, unit_rows = [], []
    for rnd, spec in NURSE_ROUNDS.items():
        n = read_tab(DATA_DIR / spec["file"], ["idauniq", spec["fasting"], "sex", "indsex"] + NURSE_VARS)
        d = pd.DataFrame({"idauniq": n["idauniq"], "round": rnd})
        d["t"] = decimal_date(n["visyear"], n.get("vismon", pd.Series(np.nan, index=n.index)))
        if spec["core_wave"] == "from_wave_column":
            cw = clean_negative(n["wave"]) if "wave" in n else pd.Series(np.nan, index=n.index)
            cw = cw.where(cw.isin([8, 9])).fillna(pd.Series(np.where(d["t"] < 2018, 8, 9), index=n.index))
            d["core_wave"] = cw.astype(int)
        else:
            d["core_wave"] = spec["core_wave"]
        d["fasted"] = binary(n.get(spec["fasting"], pd.Series(np.nan, index=n.index)), yes=(1,), no=(2,))
        for a in PLAUSIBLE:
            if a not in n:
                continue
            raw = n[a].where(n[a] >= 0).where(lambda s: s < 9990)
            conv, unit, med = detect_units(raw, a)
            lo, hi = PLAUSIBLE[a]
            d[a] = conv.where((conv >= lo) & (conv <= hi))
            if a in ("hba1c", "hgb", "fglu"):
                unit_rows.append(dict(round=rnd, analyte=a, unit_decision=unit, raw_median=med,
                                      n_valid=int(d[a].notna().sum())))
        if "fglu" in d:
            d.loc[d["fasted"] != 1, "fglu"] = np.nan          # fasting glucose only if fasted
        for c in ("confage", "indager"):
            if c in n:
                d[f"_{c}"] = n[c].where(n[c] >= 0)
        sx = n.get("sex", n.get("indsex"))
        d["_nsex"] = sx.where(sx.isin([1, 2])) if sx is not None else np.nan
        out.append(d)
    nl = pd.concat(out, ignore_index=True)

    p = h[["idauniq", "rabyear", "ragender", "raeducl"]].copy()
    p["rabyear"] = clean_negative(p["rabyear"])
    nl = nl.merge(p, on="idauniq", how="left")
    nl["age"] = nl["t"] - nl["rabyear"] - 0.5
    fb = nl["_confage"] if "_confage" in nl else pd.Series(np.nan, index=nl.index)
    if "_indager" in nl:
        fb = fb.fillna(nl["_indager"])
    nl["age"] = nl["age"].fillna(fb)
    nl["female"] = np.where(nl["ragender"].isin([1, 2]), nl["ragender"] == 2,
                            np.where(nl["_nsex"].isin([1, 2]), nl["_nsex"] == 2, np.nan)).astype(float)
    nl["educ"] = clean_negative(nl["raeducl"])

    cov_rows = []
    for w in W:
        g = lambda s: h.get(f"r{w}{s}", pd.Series(np.nan, index=h.index))
        cov_rows.append(pd.DataFrame({
            "idauniq": h["idauniq"], "core_wave": w,
            "smoker": binary(g("smoken")),
            "vig_pa_weekly": clean_negative(g("vgactx_e")).map({2: 1, 3: 1, 4: 0, 5: 0}),
            "bp_med": binary(g("rxhibp")).where(g("rxhibp") != -24, 0.0),
            "working": binary(g("work")), "job_phys": clean_negative(g("jphysl")),
            "job_stress": clean_negative(g("jobsum")), "nssec8": clean_negative(g("nssec8")),
            "h_bmi": clean_negative(g("mbmi")), "h_weight": clean_negative(g("mweight")),
            "h_height": clean_negative(g("mheight"))}))
    cov = pd.concat(cov_rows, ignore_index=True).sort_values(["idauniq", "core_wave"])
    cov["h_height_last"] = cov.groupby("idauniq")["h_height"].ffill()
    nl = nl.merge(cov, on=["idauniq", "core_wave"], how="left")
    bmi_calc = nl["h_weight"] / nl["h_height_last"] ** 2
    nl["bmi"] = nl["bmival"] if "bmival" in nl else np.nan
    nl["bmi"] = nl["bmi"].fillna(nl["h_bmi"]).fillna(bmi_calc.where((bmi_calc > 12) & (bmi_calc < 80)))
    nl = nl.drop(columns=[c for c in nl.columns if c.startswith("_")] +
                 ["ragender", "raeducl", "h_bmi", "h_weight", "h_height", "h_height_last"])
    return nl, pd.DataFrame(unit_rows)


def one_visit_per_round(nl):
    """One health visit per person and round: valid HbA1c first, then the earlier visit (waves 8/9 only)."""
    return (nl.assign(_no=nl["hba1c"].isna()).sort_values(["idauniq", "round", "_no", "t"])
              .drop_duplicates(["idauniq", "round"]).drop(columns="_no")
              .sort_values(["round", "idauniq"]).reset_index(drop=True))


def step1():
    h = load_harmonized()
    iv = interviews_long(h)
    nl, units = nurse_long(h)
    nl = one_visit_per_round(nl)
    nl.to_pickle(WORK_DIR / "nurse_long.pkl"); iv.to_pickle(WORK_DIR / "interviews.pkl")
    save(units, "D_units")
    return h, nl, iv


# =====================================================================================================
# 2. Baseline visits and outcome
# =====================================================================================================
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


# =====================================================================================================
# 3. Personal baselines: cross-fitted linear mixed model and the four features
# =====================================================================================================
def design(df):
    """X: const, a, a^2, female, round dummies (ref. round 2); Z: 1, a; a = (age - 65) / 10."""
    a = (df["age"].to_numpy() - 65.0) / 10.0
    cols, names = [np.ones(len(df)), a, a ** 2, df["female"].to_numpy()], ["const", "age10", "age10_sq", "female"]
    for r in ROUNDS[1:]:
        cols.append((df["round"].to_numpy() == r).astype(float)); names.append(f"round_{r}")
    return np.column_stack(cols), np.column_stack([np.ones(len(df)), a]), names


def fit_population(train):
    """REML fit of the random intercept and slope model; intercept-only fallback if G is singular."""
    X, Z, names = design(train)
    y, groups = train["hba1c"].to_numpy(), train["idauniq"].to_numpy()
    try:
        res = sm.MixedLM(y, X, groups=groups, exog_re=Z).fit(reml=True, method="lbfgs", maxiter=300)
        G = np.asarray(res.cov_re)
        if np.any(np.linalg.eigvalsh(G) <= 1e-8):
            raise np.linalg.LinAlgError("singular G")
        kind = "intercept+slope"
    except Exception:
        res = sm.MixedLM(y, X, groups=groups, exog_re=Z[:, :1]).fit(reml=True, method="lbfgs", maxiter=300)
        G = np.diag([float(np.asarray(res.cov_re)[0, 0]), 1e-6])
        kind = "intercept only"
    return dict(beta=np.asarray(res.fe_params), G=G, s2=float(res.scale), names=names, kind=kind,
                converged=bool(res.converged))


def round_offset(m, r):
    r = int(r)
    return 0.0 if r == ROUNDS[0] else float(m["beta"][m["names"].index(f"round_{r}")])


def posterior(m, Xh, Zh, yh):
    """Empirical Bayes posterior of (level, slope) given a person's values."""
    S = np.linalg.inv(np.linalg.inv(m["G"]) + Zh.T @ Zh / m["s2"])
    return S @ Zh.T @ (yh - Xh @ m["beta"]) / m["s2"], S


def predict(m, x, z, mu, S):
    return x @ m["beta"] + z @ mu, np.sqrt(z @ S @ z + m["s2"])


def visit_design(age, female, rnd):
    X, Z, _ = design(pd.DataFrame([dict(age=age, female=female, round=rnd)]))
    return X[0], Z[0]


def features_for_visit(m, r, history):
    """z_population, z_personal, delta_last, slope_personal, mu_next, p_cross48 for one baseline visit."""
    h = history[history["t"] < r["t"] - MIN_GAP_YEARS] if history is not None else None
    x, z = visit_design(r["age"], r["female"], r["round"])
    mean, sd = predict(m, x, z, np.zeros(2), m["G"])
    out = dict(z_population=(r["hba1c"] - mean) / sd)
    if h is None or not len(h):
        return out
    Xh, Zh, _ = design(h)
    mu, S = posterior(m, Xh, Zh, h["hba1c"].to_numpy())
    mean, sd = predict(m, x, z, mu, S)
    last = h.iloc[-1]
    out.update(mu_personal=mean, sd_personal=sd, z_personal=(r["hba1c"] - mean) / sd,
               delta_last=(r["hba1c"] - round_offset(m, r["round"])) - (last["hba1c"] - round_offset(m, last["round"])),
               delta_last_raw=r["hba1c"] - last["hba1c"], slope_personal=m["beta"][1] + mu[1])
    # forecast of the next visit, updated with today's value
    hh = pd.concat([h[["age", "female", "round", "hba1c"]],
                    pd.DataFrame([dict(age=r["age"], female=r["female"], round=r["round"], hba1c=r["hba1c"])])],
                   ignore_index=True)
    Xa, Za, _ = design(hh)
    mu, S = posterior(m, Xa, Za, hh["hba1c"].to_numpy())
    years = r["years_to_next"] if pd.notna(r["years_to_next"]) else 4.0
    xn, zn = visit_design(r["age"] + years, r["female"], r["next_round"])
    mean, sd = predict(m, xn, zn, mu, S)
    out.update(mu_next=mean, p_cross48=1 - norm.cdf((HBA1C_DIABETES - mean) / sd))
    return out


def assign_folds(cohort, lmm):
    rng = np.random.default_rng(RANDOM_SEED)
    persons = np.array(sorted(set(lmm["idauniq"]) | set(cohort["idauniq"])))
    fold_of = dict(zip(persons, rng.integers(0, N_FOLDS, len(persons))))
    return cohort.assign(fold=cohort["idauniq"].map(fold_of)), lmm.assign(fold=lmm["idauniq"].map(fold_of))


def step3(cohort, lmm):
    """Cross-fitting: the model for fold k is fitted on the other folds. Returns features, vc and the 5 models."""
    cohort, lmm_f = assign_folds(cohort, lmm)
    history_of = {pid: g.sort_values("t") for pid, g in lmm_f.groupby("idauniq")}
    feats, fits, models = [], [], {}
    for k in range(N_FOLDS):
        m = fit_population(lmm_f[lmm_f["fold"] != k])
        models[k] = m
        fits.append(dict(fold=k, model=m["kind"], converged=m["converged"],
                         sd_intercept=np.sqrt(m["G"][0, 0]), sd_slope_per_decade=np.sqrt(m["G"][1, 1]),
                         corr_int_slope=m["G"][0, 1] / np.sqrt(m["G"][0, 0] * m["G"][1, 1]),
                         sd_residual=np.sqrt(m["s2"]), **{f"beta_{n}": b for n, b in zip(m["names"], m["beta"])}))
        for i, r in cohort[cohort["fold"] == k].iterrows():
            feats.append(dict(row=i, **features_for_visit(m, r, history_of.get(r["idauniq"]))))
        log(f"step 3: fold {k} done")
    features = cohort.join(pd.DataFrame(feats).set_index("row"))
    vc = pd.DataFrame(fits)
    features.to_pickle(WORK_DIR / "features.pkl")
    save(vc, "T2_variance_components_by_fold")
    sb, sw = vc["sd_intercept"].mean(), vc["sd_residual"].mean()
    save(pd.DataFrame([dict(sd_between_persons=sb, sd_within_person=sw, icc=sb ** 2 / (sb ** 2 + sw ** 2),
                            index_of_individuality=sw / sb, reference_change_value_95=1.96 * np.sqrt(2) * sw,
                            **{c: vc[c].mean() for c in vc if c.startswith("beta_round_")})]), "T2_individuality")
    return features, vc, models, lmm_f


def derive(df):
    """Derived predictors used by the prediction and association models."""
    df = df.copy()
    df["log_trig"] = np.log(df["trig"])
    df["log_years_next"] = np.log(df["years_to_next"].where(df["years_to_next"] > 0))
    p = df["p_cross48"].clip(1e-6, 1 - 1e-6)
    df["logit_p_cross48"] = np.log(p / (1 - p))
    df["hba1c_mean2"] = df["hba1c"] - df["delta_last"] / 2     # mean of today's and the last earlier value
    return df


# =====================================================================================================
# 4. Prediction: out-of-fold logistic models A-E, AUC, calibration, screening counts
# =====================================================================================================
def oof_predict(df, cols, y="incident_dm"):
    """Out-of-fold risks; imputation, scaling and the ridge-penalised fit are learned on the training folds."""
    cols = [c for c in cols if c in df and df[c].notna().any()]
    pred = pd.Series(np.nan, index=df.index)
    for k in sorted(df["fold"].unique()):
        tr, te = df[df["fold"] != k], df[df["fold"] == k]
        if tr[y].nunique() < 2 or len(te) == 0:
            continue
        mdl = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        mdl.fit(tr[cols], tr[y])
        pred[te.index] = mdl.predict_proba(te[cols])[:, 1]
    return pred


# =====================================================================================================
# 4b. Final models A and B refitted on all baseline visits -> Supplementary Table S8
# =====================================================================================================
S8_MODELS = ["A_population", "B_clinical"]
S8_LABELS = {   # column -> (label in Table S8, transformation)
    "hba1c": ("HbA1c, \\mmol", "none"),
    "age": ("Age, years", "none"),
    "female": ("Female sex", "0/1"),
    "log_years_next": ("Years to next visit", "$\\ln$"),
    "bmi": ("BMI, kg/m$^2$", "none"),
    "sysval": ("Systolic BP, mmHg", "none"),
    "hdl": ("HDL cholesterol, mmol/l", "none"),
    "log_trig": ("Triglycerides, mmol/l", "$\\ln$"),
    "smoker": ("Current smoker", "0/1"),
    "vig_pa_weekly": ("Weekly vigorous activity", "0/1"),
    "bp_med": ("BP medication", "0/1"),
}


def fit_final(df, cols, y="incident_dm"):
    """The pipeline of oof_predict(), fitted once on all rows; returns the fitted pipeline and its input columns."""
    cols = [c for c in cols if c in df and df[c].notna().any()]          # same column rule as oof_predict()
    mdl = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=2000))
    mdl.fit(df[cols], df[y])
    return mdl, cols


def final_model_parameters(mdl, cols, name):
    """One row per model column (in the pipeline's order): median, scaler mean/SD, coefficient."""
    imp, sc, lr = mdl[0], mdl[1], mdl[2]
    miss_idx = list(imp.indicator_.features_) if imp.indicator_ is not None else []
    rows = [dict(model=name, order=0, column="intercept", kind="intercept", median=np.nan,
                 mean=np.nan, sd=np.nan, coef_std=lr.intercept_[0])]
    # imputer output = the cols in order, then one indicator per column with missing values (in col order)
    out_cols = [(c, "predictor") for c in cols] + [(cols[i], "missing_indicator") for i in miss_idx]
    for j, (c, kind) in enumerate(out_cols):
        rows.append(dict(model=name, order=j + 1, column=c, kind=kind,
                         median=imp.statistics_[cols.index(c)] if kind == "predictor" and cols.index(c) in miss_idx
                         else np.nan,
                         mean=sc.mean_[j], sd=sc.scale_[j], coef_std=lr.coef_[0, j]))
    t = pd.DataFrame(rows)
    # the same model on the original scale of each column (after transformation and imputation)
    t["coef_per_unit"] = t["coef_std"] / t["sd"]
    t.loc[t["kind"] == "intercept", "coef_per_unit"] = lr.intercept_[0] - (t["coef_std"] * t["mean"] / t["sd"]).sum()
    return t


def check_reproduction(df, mdl, cols, t):
    """Recompute risks from the exported numbers alone and compare with the fitted pipeline."""
    X = df[cols].copy()
    lp = np.full(len(X), t.loc[t["kind"] == "intercept", "coef_std"].iloc[0])
    med = t[t["kind"] == "predictor"].set_index("column")["median"]
    for _, r in t[t["kind"] != "intercept"].iterrows():
        x = X[r["column"]]
        x = x.isna().astype(float) if r["kind"] == "missing_indicator" else x.fillna(med[r["column"]])
        lp = lp + r["coef_std"] * (x.to_numpy() - r["mean"]) / r["sd"]
    diff = np.abs(1 / (1 + np.exp(-lp)) - mdl.predict_proba(df[cols])[:, 1]).max()
    if not diff < 1e-8:
        raise RuntimeError(f"Table S8 export does not reproduce the fitted model (max |diff| = {diff:.1e})")
    return diff


def s8_latex(t):
    """Body rows of Supplementary Table S8 (columns: predictor, transformation, median, mean, SD, A, B)."""
    def f(v, d=3):
        return "--" if pd.isna(v) else f"{v:.{d}f}".replace("-", "$-$")
    first = t[t["model"] == S8_MODELS[-1]]                       # model B has every column of model A
    lines = []
    for _, r in first.iterrows():
        coefs = []
        for m in S8_MODELS:
            hit = t[(t["model"] == m) & (t["column"] == r["column"]) & (t["kind"] == r["kind"])]
            coefs.append(f(hit["coef_std"].iloc[0]) if len(hit) else "--")
        if r["kind"] == "intercept":
            lines.append(f"Intercept & -- & -- & -- & -- & {coefs[0]} & {coefs[1]} \\\\")
        elif r["kind"] == "predictor":
            lab, tr = S8_LABELS.get(r["column"], (r["column"].replace("_", "\\_"), "none"))
            lines.append(f"{lab} & {tr} & {f(r['median'], 2)} & {f(r['mean'], 2)} & {f(r['sd'], 2)} "
                         f"& {coefs[0]} & {coefs[1]} \\\\")
        else:
            if not any("Missingness" in x for x in lines):
                lines.append("\\multicolumn{7}{l}{\\textit{Missingness indicators (1 = value missing)}} \\\\")
            lab = S8_LABELS.get(r["column"], (r["column"].replace("_", "\\_"),))[0]
            lines.append(f"\\quad {lab} & 0/1 & -- & {f(r['mean'], 3)} & {f(r['sd'], 3)} "
                         f"& {coefs[0]} & {coefs[1]} \\\\")
    return "\n".join(lines) + "\n"


def export_final_models(features):
    """Refit models A and B on the full analysis set of step 4 and write Supplementary Table S8.

    Writes S8_final_model_parameters.csv (every number needed to reproduce the risks, also per original unit)
    and S8_final_models_rows.tex (table body to paste between \\midrule and \\bottomrule)."""
    df = derive(features[features["primary"] & features["z_personal"].notna()]).reset_index(drop=True)
    tabs = []
    for m in S8_MODELS:
        mdl, cols = fit_final(df, MODELS[m])
        t = final_model_parameters(mdl, cols, m)
        diff = check_reproduction(df, mdl, cols, t)
        log(f"S8: {m} refitted on {len(df):,} visits, {len(t) - 1} columns, reproduction max |diff| = {diff:.1e}")
        tabs.append(t)
    t = pd.concat(tabs, ignore_index=True)
    # medians, means and SDs of a column must be the same in both models (same data, same pipeline)
    g = t[t["kind"] != "intercept"].groupby(["column", "kind"])[["median", "mean", "sd"]]
    spread = (g.max() - g.min()).abs() / g.max().abs().clip(lower=1e-12)
    if (spread.fillna(0) > 1e-9).any().any():
        raise RuntimeError("Table S8: preprocessing parameters differ between models A and B")
    save(t, "S8_final_model_parameters")
    (OUT_DIR / "S8_final_models_rows.tex").write_text(s8_latex(t))
    return t


def fast_auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def calib(y, p):
    """Calibration intercept and slope from a logistic regression of y on logit(p)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def cluster_boot(df, pred_cols, rng, n_boot):
    persons, idx = df["idauniq"].unique(), df.groupby("idauniq").indices
    out = []
    for _ in range(n_boot):
        sub = df.iloc[np.concatenate([idx[p] for p in rng.choice(persons, len(persons), replace=True)])]
        out.append({c: fast_auc(sub["incident_dm"], sub[c]) for c in pred_cols})
    return pd.DataFrame(out)


def perf_table(df, label, rng):
    y = df["incident_dm"].to_numpy()
    boot = cluster_boot(df, [f"p_{m}" for m in MODELS], rng, N_BOOT)
    rows, comp = [], []
    for m in MODELS:
        p = df[f"p_{m}"].to_numpy()
        ci, cs = calib(y, p)
        rows.append(dict(population=label, model=m, n_baseline_visits=len(df), n_events=int(y.sum()),
                         auc=fast_auc(y, p), auc_lo=boot[f"p_{m}"].quantile(.025), auc_hi=boot[f"p_{m}"].quantile(.975),
                         brier=brier_score_loss(y, p), calib_intercept=ci, calib_slope=cs))
    for a, b in COMPARISONS:
        d = boot[f"p_{a}"] - boot[f"p_{b}"]
        comp.append(dict(population=label, comparison=f"{a} - {b}", n_baseline_visits=len(df), n_events=int(y.sum()),
                         delta_auc=fast_auc(y, df[f"p_{a}"]) - fast_auc(y, df[f"p_{b}"]),
                         lo=d.quantile(.025), hi=d.quantile(.975), p_boot_le0=(d <= 0).mean()))
    return rows, comp


def step4(features):
    rng = np.random.default_rng(RANDOM_SEED)
    df = derive(features[features["primary"] & features["z_personal"].notna()]).reset_index(drop=True)
    for m, cols in MODELS.items():
        df[f"p_{m}"] = oof_predict(df, cols)
    df = df.dropna(subset=[f"p_{m}" for m in MODELS]).reset_index(drop=True)
    df.to_pickle(WORK_DIR / "predictions.pkl")
    groups = [("all", df), ("HbA1c < 42", df[df["hba1c"] < HBA1C_PREDIAB]),
              ("history = 1 earlier value", df[df["n_hist"] == 1]),
              ("history >= 2 earlier values", df[df["n_hist"] >= 2]),
              ("~4-year interval", df[df["round"].isin(HOMOGENEOUS_ROUNDS)])]
    perf, comp = [], []
    for label, s in groups:
        s = s.reset_index(drop=True)
        if s["incident_dm"].nunique() < 2:
            continue
        r, c = perf_table(s, label, rng)
        perf += r; comp += c
        log(f"step 4: bootstrap {label} done")
    perf = save(pd.DataFrame(perf), "AUC_model_performance", ["n_baseline_visits", "n_events"])
    save(pd.DataFrame(comp), "AUC_delta", ["n_baseline_visits", "n_events"])
    # calibration by tenth of predicted risk
    dec = []
    for m in ["A_population", "B_clinical", "D_personal_baseline", "E_forecast"]:
        q = pd.qcut(df[f"p_{m}"], 10, labels=False, duplicates="drop")
        g = df.groupby(q).agg(n=("incident_dm", "size"), observed=("incident_dm", "mean"), predicted=(f"p_{m}", "mean"))
        dec.append(g.reset_index(names="decile").assign(model=m))
    save(pd.concat(dec, ignore_index=True), "AUC_calibration_deciles", ["n"])
    return df, perf


ALERT_RULES = {"B_clinical": "B", "D_personal_baseline": "D", "E_forecast": "E"}
ALERT_DIFFS = [("D_personal_baseline", "B_clinical"), ("E_forecast", "B_clinical")]


def caught(y, score, k, jit):
    order = np.lexsort((jit, -np.nan_to_num(score, nan=-1e9)))
    return int(y[order[:k]].sum())


def alert_comparison(pred):
    """Cases among the 5% and 10% highest-risk baseline visits, with participant-bootstrap CIs of differences."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for label, s in [("all", pred), ("HbA1c 42-47.9", pred[pred["hba1c"] >= HBA1C_PREDIAB])]:
        s = s.reset_index(drop=True)
        y, n = s["incident_dm"].to_numpy(), len(s)
        scores = {m: s[f"p_{m}"].to_numpy() for m in ALERT_RULES}
        jit = rng.uniform(0, 1e-9, n)
        persons, idx = s["idauniq"].unique(), s.groupby("idauniq").indices
        boot = {(a, r): [] for a, _ in ALERT_DIFFS for r in [0.05, 0.10]}
        for _ in range(N_BOOT):
            ii = np.concatenate([idx[p] for p in rng.choice(persons, len(persons), replace=True)])
            jb = rng.uniform(0, 1e-9, len(ii))
            for r in [0.05, 0.10]:
                kb = int(round(r * len(ii)))
                c = {m: caught(y[ii], sc[ii], kb, jb) for m, sc in scores.items()}
                for a, b in ALERT_DIFFS:
                    boot[(a, r)].append(c[a] - c[b])
        for r in [0.05, 0.10]:
            k = int(round(r * n))
            c = {m: caught(y, sc, k, jit) for m, sc in scores.items()}
            row = dict(population=label, alert_rate_pct=100 * r, n_flagged=k, n_events_total=int(y.sum()),
                       **{f"caught_{ALERT_RULES[m]}": c[m] for m in ALERT_RULES})
            for a, b in ALERT_DIFFS:
                dd, tag = np.array(boot[(a, r)]), f"{ALERT_RULES[a]}-{ALERT_RULES[b]}"
                row.update({tag: c[a] - c[b], f"{tag}_lo": np.quantile(dd, .025), f"{tag}_hi": np.quantile(dd, .975)})
            rows.append(row)
    return save(pd.DataFrame(rows), "AUC_screening_counts")


# =====================================================================================================
# 5. Association analyses (equation 3) and the sensitivity analyses of Supplementary Table S1
# =====================================================================================================
def impute(df, cols):
    """Median-fill missing covariates and add a 'was missing' indicator."""
    X = pd.DataFrame(index=df.index)
    for c in cols:
        x = df[c].astype(float)
        if x.isna().any():
            X[f"{c}_miss"] = x.isna().astype(float)
        X[c] = x.fillna(x.median())
    return X


def spline_cols(x, prefix):
    """Natural cubic spline, 4 df (knots at the quartiles, boundary at min/max), centred."""
    s = dmatrix("cr(x, df=4, constraints='center') - 1", pd.DataFrame({"x": x}), return_type="dataframe")
    s.columns = [f"{prefix}{i}" for i in range(s.shape[1])]
    s.index = x.index
    return s


def clean_X(X, y):
    """Drop 'was missing' columns with separation, duplicated and constant columns."""
    sep = [c for c in X.columns if "_miss" in c and y[X[c] != 0].nunique() < 2]
    X = X.drop(columns=sep)
    X = X.loc[:, ~X.T.duplicated()]
    return X.loc[:, X.std() > 1e-8]


def glm_cluster(y, X, ids, weights=None):
    return sm.GLM(y, X.assign(const=1.0), family=sm.families.Binomial(), var_weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(ids)[0]})


def adjusted(d, expo, weights=None, adj=ADJ, level="hba1c", scale=None):
    """OR per SD of `expo`, adjusted for a natural spline in the HbA1c level, model-B covariates and
    ln(years to next visit); participant-clustered SEs. `scale` fixes the SD (common-scale analysis)."""
    d = d.dropna(subset=[expo, level])
    if d["incident_dm"].nunique() < 2:
        return None
    y = d["incident_dm"]
    spl = dmatrix(f"cr({level}, df=4, constraints='center') - 1", d, return_type="dataframe")
    spl.columns = [f"lev_s{i}" for i in range(spl.shape[1])]
    X = pd.concat([impute(d, adj), spl], axis=1)
    X = X.drop(columns=[c for c in X.columns if c.endswith("_miss") and y[X[c] == 1].nunique() < 2])
    X = X.loc[:, ~X.T.duplicated()]
    X["expo"] = (d[expo] - d[expo].mean()) / (d[expo].std() if scale is None else scale)
    X = X.loc[:, X.std() > 1e-8].assign(const=1.0)
    f = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=None if weights is None else weights.loc[d.index]
               ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["idauniq"])[0]})
    b, se = f.params["expo"], f.bse["expo"]
    return dict(n_baseline_visits=len(d), n_events=int(y.sum()), or_per_sd=np.exp(b),
                or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se), p_value=f.pvalues["expo"])


def pre(d):
    return d[d["hba1c"] >= HBA1C_PREDIAB]


def crude(d, expo, adj=()):
    """OR per SD of `expo` without adjustment (adj empty) or adjusted only for the columns in adj;
    standardised and clustered as in adjusted()."""
    d = d.dropna(subset=[expo, *adj])
    y = d["incident_dm"]
    X = pd.DataFrame({"expo": (d[expo] - d[expo].mean()) / d[expo].std()}, index=d.index)
    for c in adj:
        X[c] = d[c]
    f = glm_cluster(y, X, d["idauniq"])
    b, se = f.params["expo"], f.bse["expo"]
    return dict(n_baseline_visits=len(d), n_events=int(y.sum()), or_per_sd=np.exp(b),
                or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se), p_value=f.pvalues["expo"])


def slope_crude(df):
    """STROBE 16a: slope OR without adjustment and adjusted only for ln(years to next visit) (Table S1)."""
    rows = []
    for popl, d in [("all", df), ("HbA1c 42-47.9", pre(df))]:
        for label, adj in [("unadjusted", ()), ("follow-up duration only", ("log_years_next",))]:
            rows.append(dict(analysis=label, population=popl, exposure=EXPOSURES[SLOPE], **crude(d, SLOPE, adj)))
    return save(pd.DataFrame(rows), "S1_slope_crude", ["n_baseline_visits", "n_events"])


def slope_reversal_diagnostics(df):
    """Where does the sign change between the crude (OR < 1) and the adjusted slope OR (> 1) come from?

    S1_slope_reversal_ors: slope OR per SD by number of earlier HbA1c values, under four adjustment sets
      (none; ln(years to next visit); HbA1c spline only; HbA1c spline + ln(years) = level only;
      full adjustment as in the main analysis).
    S1_slope_correlations: correlation of the estimated slope with baseline HbA1c, the change since the last
      value and age, by number of earlier values (the shrunk slope's dependence on level)."""
    groups = [("all", df), ("1 earlier value", df[df["n_hist"] == 1]), (">= 2 earlier values", df[df["n_hist"] >= 2])]
    rows = []
    for popl, d in groups:
        for label, fn in [("unadjusted", lambda d: crude(d, SLOPE)),
                          ("follow-up duration only", lambda d: crude(d, SLOPE, ("log_years_next",))),
                          ("HbA1c spline only", lambda d: adjusted(d, SLOPE, adj=[])),
                          ("HbA1c spline + follow-up duration", lambda d: adjusted(d, SLOPE, adj=["log_years_next"])),
                          ("full (main analysis)", lambda d: adjusted(d, SLOPE))]:
            r = fn(d)
            if r:
                rows.append(dict(population=popl, adjustment=label, **r))
    ors = save(pd.DataFrame(rows), "S1_slope_reversal_ors", ["n_baseline_visits", "n_events"])
    rows = []
    for popl, d in groups + [("2 earlier values", df[df["n_hist"] == 2]), ("3 earlier values", df[df["n_hist"] == 3])]:
        if len(d) < MIN_CELL:
            continue
        r = dict(population=popl, n_baseline_visits=len(d), sd_slope=d[SLOPE].std())
        for c in ["hba1c", "delta_last", "age"]:
            x = d[[SLOPE, c]].dropna()
            r[f"pearson_slope_{c}"] = x[SLOPE].corr(x[c])
            r[f"spearman_slope_{c}"] = x[SLOPE].corr(x[c], method="spearman")
        rows.append(r)
    cor = save(pd.DataFrame(rows), "S1_slope_correlations", ["n_baseline_visits"])
    return ors, cor


def slope_covariate_steps(df):
    """Which clinical covariate moves the slope OR from the level-only to the fully adjusted estimate?

    S1_slope_covariate_steps: slope OR per SD with the HbA1c spline + ln(years) ('level only'), then
      (a) adding each covariate of ADJ_NO_HORIZON on its own to the level-only model, and
      (b) removing each covariate on its own from the full model.
    S1_slope_covariate_correlations: correlation of the slope with each covariate (observed values)."""
    covs = ADJ_NO_HORIZON
    groups = [("all", df), (">= 2 earlier values", df[df["n_hist"] >= 2]), ("1 earlier value", df[df["n_hist"] == 1])]
    rows = []
    for popl, d in groups:
        base = ["log_years_next"]
        r = adjusted(d, SLOPE, adj=base)
        rows.append(dict(population=popl, step="level only", covariate="", **r))
        for c in covs:
            r = adjusted(d, SLOPE, adj=base + [c])
            if r:
                rows.append(dict(population=popl, step="level only + one covariate", covariate=c, **r))
        r = adjusted(d, SLOPE)
        rows.append(dict(population=popl, step="full", covariate="", **r))
        for c in covs:
            r = adjusted(d, SLOPE, adj=[x for x in ADJ if x != c])
            if r:
                rows.append(dict(population=popl, step="full minus one covariate", covariate=c, **r))
    steps = save(pd.DataFrame(rows), "S1_slope_covariate_steps", ["n_baseline_visits", "n_events"])
    rows = []
    for popl, d in groups:
        r = dict(population=popl, n_baseline_visits=len(d))
        for c in covs:
            x = d[[SLOPE, c]].dropna()
            r[f"corr_slope_{c}"] = x[SLOPE].corr(x[c])
        rows.append(r)
    cor = save(pd.DataFrame(rows), "S1_slope_covariate_correlations", ["n_baseline_visits"])
    return steps, cor


def extended_set(features, iv):
    """Primary set + visits whose outcome is known from interviews.

    A visit without HbA1c at the next round is added if
      (a) the diabetes question was answered at an interview in the next health-visit round (that round's
          wave; for round 8/9, which spans waves 8 and 9, either wave), or
      (b) diabetes was reported at any interview after the baseline visit up to the next round, the window
          add_outcome() uses: such a report establishes a case even if the next round was missed.
    A "no" before the next round does not establish the status at the next round and does not qualify."""
    ans = iv[iv["dm_report"].notna()][["idauniq", "wave", "dm_report"]]
    m = features[["idauniq", "core_wave", "next_round", "next_core_wave"]].reset_index().merge(ans, on="idauniq")
    answered_next = (m["wave"] == m["next_core_wave"]) | ((m["next_round"] == 8) & m["wave"].isin([8, 9]))
    reported = (m["dm_report"] == 1) & (m["wave"] > m["core_wave"]) & (m["wave"] <= m["next_core_wave"])
    known = set(m.loc[answered_next | reported, "index"])
    return features[(features["primary"] | features.index.isin(known)) & features["z_personal"].notna()]


def ipw_weights(features):
    """Inverse probability of follow-up (ridge-penalised logistic model, probabilities bounded at 0.05)."""
    d = features[features["z_personal"].notna()]
    cols = ["age", "female", "hba1c", "bmi", "sysval", "hdl", "log_trig", "smoker", "vig_pa_weekly", "bp_med",
            "educ", "working", "n_hist"]
    X = pd.concat([impute(d, cols), pd.get_dummies(d["round"], prefix="r", dtype=float)], axis=1)
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    p = LogisticRegression(C=1.0, max_iter=3000).fit(Xs, d["primary"]).predict_proba(Xs)[:, 1]
    w = pd.Series(1 / np.clip(p, 0.05, 1), index=d.index)
    return w / w[d["primary"]].mean(), fast_auc(d["primary"].astype(int), p)


def step5_associations(features, df, iv):
    """Table 3 (four features by HbA1c range) and the main slope sensitivity analyses."""
    w, auc_followup = ipw_weights(features)
    ext = extended_set(features, iv)
    h4, h2 = df[df["round"].isin(HOMOGENEOUS_ROUNDS)], df[df["n_hist"] >= 2]
    ALL4 = ["z_personal", "delta_last", "logit_p_cross48", "slope_personal"]
    sets = [("primary", "all", df, ALL4, None, ADJ, "hba1c"),
            ("primary", "HbA1c < 42", df[df["hba1c"] < HBA1C_PREDIAB], ALL4, None, ADJ, "hba1c"),
            ("primary", "HbA1c 42-47.9", pre(df), ALL4, None, ADJ, "hba1c"),
            ("IPW for loss to follow-up", "all", df, [SLOPE], w, ADJ, "hba1c"),
            ("extended set", "all", ext, [SLOPE], None, ADJ, "hba1c"),
            ("~4-year interval only", "all", h4, [SLOPE], None, ADJ, "hba1c"),
            ("~4-year interval only", "HbA1c 42-47.9", pre(h4), [SLOPE], None, ADJ, "hba1c"),
            (">= 2 earlier values", "all", h2, [SLOPE], None, ADJ, "hba1c"),
            (">= 2 earlier values", "HbA1c 42-47.9", pre(h2), [SLOPE], None, ADJ, "hba1c"),
            ("adjusted for two-value mean", "all", df, [SLOPE], None, ADJ, "hba1c_mean2"),
            ("adjusted for two-value mean", "HbA1c 42-47.9", pre(df), [SLOPE], None, ADJ, "hba1c_mean2"),
            ("without follow-up-interval adjustment", "all", df, ["logit_p_cross48"], None, ADJ_NO_HORIZON, "hba1c")]
    rows = []
    for analysis, popl, d, expos, wt, adj, level in sets:
        for e in expos:
            r = adjusted(d, e, wt, adj, level)
            if r:
                rows.append(dict(analysis=analysis, population=popl, exposure=EXPOSURES[e], **r))
    assoc = save(pd.DataFrame(rows), "T3_S1_adjusted_associations", ["n_baseline_visits", "n_events"])
    # IPW diagnostics (Supplementary Methods S2)
    wp = w[features.loc[w.index, "primary"]]
    save(pd.DataFrame([dict(auc_followup_model=auc_followup, max_weight=wp.max(), p99_weight=wp.quantile(.99),
                            effective_sample_size=wp.sum() ** 2 / (wp ** 2).sum(), n_analysis=len(wp),
                            n_extended_set=len(ext), n_events_extended=int(ext["incident_dm"].sum()))]),
         "S1_ipw_and_extended_set_diagnostics")
    # common scale: per SD of the slope in all baseline visits
    sd_all = df[SLOPE].std()
    rows = []
    for popl, d in [("all", df), ("HbA1c 42-47.9", pre(df)), (">= 2 earlier values", h2)]:
        r = adjusted(d, SLOPE, scale=sd_all)
        rows.append(dict(population=popl, sd_slope_subgroup=d[SLOPE].std(), sd_slope_all=sd_all, **r))
    save(pd.DataFrame(rows), "S1_slope_common_scale", ["n_baseline_visits", "n_events"])
    return assoc


def adjusted_design(d, expo, adj=ADJ, level="hba1c"):
    d = d.dropna(subset=[expo, level])
    spl = dmatrix(f"cr({level}, df=4, constraints='center') - 1", d, return_type="dataframe")
    spl.columns = [f"lev_s{i}" for i in range(spl.shape[1])]
    X = pd.concat([impute(d, adj), spl], axis=1)
    y = d["incident_dm"]
    X = X.drop(columns=[c for c in X.columns if c.endswith("_miss") and y[X[c] == 1].nunique() < 2])
    X = X.loc[:, ~X.T.duplicated()]
    X["expo"] = (d[expo] - d[expo].mean()) / d[expo].std()
    return d, X, y


def interaction(d, expo=SLOPE):
    """Slope x HbA1c-range interaction in one model (ratio of slope ORs, 42-47.9 vs < 42)."""
    d, X, y = adjusted_design(d, expo)
    groups = pd.factorize(d["idauniq"])[0]
    Xb = X.copy()
    Xb["band"] = (d["hba1c"] >= HBA1C_PREDIAB).astype(float)
    Xb["expo_x_band"] = Xb["expo"] * Xb["band"]
    Xb = Xb.loc[:, Xb.std() > 1e-8].assign(const=1.0)
    f = sm.GLM(y, Xb, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    V = f.cov_params()
    b_lo, b_int = f.params["expo"], f.params["expo_x_band"]
    se_lo, se_int = np.sqrt(V.loc["expo", "expo"]), np.sqrt(V.loc["expo_x_band", "expo_x_band"])
    se_hi = np.sqrt(V.loc["expo", "expo"] + V.loc["expo_x_band", "expo_x_band"] + 2 * V.loc["expo", "expo_x_band"])
    rows = []
    for term, b, se in [("slope OR, HbA1c < 42", b_lo, se_lo), ("slope OR, HbA1c 42-47.9", b_lo + b_int, se_hi),
                        ("ratio of ORs (42-47.9 vs < 42)", b_int, se_int)]:
        rows.append(dict(term=term, estimate=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se),
                         p_value=2 * norm.sf(abs(b / se))))
    return pd.DataFrame(rows)


def interaction_by(d, flag, labels, expo=SLOPE):
    """Slope x subgroup interaction in one fully adjusted model (ratio of slope ORs, flag = 1 vs 0).
    The subgroup's main effect is included; the slope is standardised over all of d."""
    d, X, y = adjusted_design(d, expo)
    groups = pd.factorize(d["idauniq"])[0]
    Xb = X.copy()
    Xb["grp"] = flag.loc[d.index].astype(float)
    Xb["expo_x_grp"] = Xb["expo"] * Xb["grp"]
    Xb = Xb.loc[:, Xb.std() > 1e-8].assign(const=1.0)
    f = sm.GLM(y, Xb, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    V = f.cov_params()
    b0, bi = f.params["expo"], f.params["expo_x_grp"]
    se0, sei = np.sqrt(V.loc["expo", "expo"]), np.sqrt(V.loc["expo_x_grp", "expo_x_grp"])
    se1 = np.sqrt(V.loc["expo", "expo"] + V.loc["expo_x_grp", "expo_x_grp"] + 2 * V.loc["expo", "expo_x_grp"])
    rows = []
    for term, b, se in [(f"slope OR, {labels[0]}", b0, se0), (f"slope OR, {labels[1]}", b0 + bi, se1),
                        (f"ratio of ORs ({labels[1]} vs {labels[0]})", bi, sei)]:
        rows.append(dict(term=term, estimate=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se),
                         p_value=2 * norm.sf(abs(b / se))))
    return pd.DataFrame(rows)


def slope_history_and_age(df):
    """(a) Formal test: does the slope association differ by number of earlier values (>= 2 vs 1)?
    (b) Why does age matter? Partial correlation of the slope with age at the same baseline HbA1c
        (both residualised on the HbA1c spline and ln(years to next visit)), by number of earlier values."""
    t = interaction_by(df, df["n_hist"] >= 2, ("1 earlier value", ">= 2 earlier values"))
    save(t, "S1_slope_history_interaction")
    rows = []
    for popl, d in [("all", df), ("1 earlier value", df[df["n_hist"] == 1]), (">= 2 earlier values", df[df["n_hist"] >= 2])]:
        d = d.dropna(subset=[SLOPE, "age", "hba1c", "log_years_next"])
        Z = pd.concat([spline_cols(d["hba1c"], "s"), d[["log_years_next"]]], axis=1).assign(const=1.0)
        res = lambda v: v - Z.to_numpy() @ np.linalg.lstsq(Z.to_numpy(), v.to_numpy(), rcond=None)[0]
        rows.append(dict(population=popl, n_baseline_visits=len(d),
                         corr_slope_age=d[SLOPE].corr(d["age"]),
                         partial_corr_slope_age_given_hba1c=np.corrcoef(res(d[SLOPE]), res(d["age"]))[0, 1],
                         corr_slope_hba1c=d[SLOPE].corr(d["hba1c"]),
                         corr_age_hba1c=d["age"].corr(d["hba1c"])))
    save(pd.DataFrame(rows), "S1_slope_age_partial_correlation", ["n_baseline_visits"])
    return t


# ---- multiple imputation (Supplementary Methods S2)
MI_COVS = ["bmi", "sysval", "hdl", "log_trig", "smoker", "vig_pa_weekly", "bp_med"]
MI_BINARY = ["smoker", "vig_pa_weekly", "bp_med"]
MI_EXPOS = ["z_personal", "delta_last", "logit_p_cross48", "slope_personal"]


def impute_once(d, seed):
    cols = MI_COVS + ["hba1c", "age", "female", "log_years_next", "incident_dm"] + MI_EXPOS
    M = pd.concat([d[cols].astype(float), pd.get_dummies(d["round"], prefix="r", dtype=float)], axis=1)
    A = pd.DataFrame(IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed).fit_transform(M),
                     columns=M.columns, index=d.index)
    out = d.copy()
    for c in MI_COVS:
        out[c] = A[c].clip(0, 1).round() if c in MI_BINARY else A[c]
    return out


def fit_complete(d, expo, level="hba1c"):
    d = d.dropna(subset=[expo, level])
    spl = dmatrix(f"cr({level}, df=4, constraints='center') - 1", d, return_type="dataframe")
    spl.columns = [f"lev_s{i}" for i in range(spl.shape[1])]
    X = pd.concat([d[ADJ].astype(float), spl], axis=1)
    X["expo"] = (d[expo] - d[expo].mean()) / d[expo].std()
    X = X.loc[:, X.std() > 1e-8].assign(const=1.0)
    f = sm.GLM(d["incident_dm"], X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["idauniq"])[0]})
    return f.params["expo"], f.bse["expo"] ** 2


def rubin(qs, us):
    """Rubin's rules; t reference with Rubin's (1987) degrees of freedom."""
    qs, us = np.asarray(qs), np.asarray(us)
    m, qbar, ubar, b = len(qs), qs.mean(), us.mean(), qs.var(ddof=1)
    t = ubar + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ubar if ubar > 0 else np.inf
    nu = (m - 1) * (1 + 1 / r) ** 2 if b > 0 else np.inf
    tq = stats.t.ppf(0.975, nu) if np.isfinite(nu) else 1.96
    p = 2 * stats.t.sf(abs(qbar / np.sqrt(t)), nu) if np.isfinite(nu) else 2 * norm.sf(abs(qbar / np.sqrt(t)))
    return dict(or_per_sd=np.exp(qbar), or_lo=np.exp(qbar - tq * np.sqrt(t)), or_hi=np.exp(qbar + tq * np.sqrt(t)),
                p_value=p)


def multiple_imputation(df):
    save(pd.DataFrame([dict(covariate=c, n_missing=int(df[c].isna().sum()), pct_missing=100 * df[c].isna().mean())
                       for c in MI_COVS]), "S1_missingness", ["n_missing"])
    imputed = [impute_once(df, RANDOM_SEED + j) for j in range(N_IMP)]
    sets = [("all", lambda d: d, MI_EXPOS), ("HbA1c 42-47.9", pre, MI_EXPOS),
            (">= 2 earlier values", lambda d: d[d["n_hist"] >= 2], [SLOPE]),
            (">= 2 earlier values, HbA1c 42-47.9", lambda d: pre(d[d["n_hist"] >= 2]), [SLOPE])]
    rows = []
    for popl, sel, expos in sets:
        for e in expos:
            qs, us = zip(*[fit_complete(sel(d), e) for d in imputed])
            rows.append(dict(population=popl, exposure=EXPOSURES[e], n_baseline_visits=len(sel(df)),
                             n_events=int(sel(df)["incident_dm"].sum()), **rubin(qs, us)))
    return save(pd.DataFrame(rows), "S1_multiple_imputation", ["n_baseline_visits", "n_events"])


def slope_including_today(df, lmm_f, models):
    """Slope estimated from the earlier values AND the baseline value, with the fold models of step 3."""
    history_of = {pid: g.sort_values("t") for pid, g in lmm_f.groupby("idauniq")}
    out = pd.Series(np.nan, index=df.index)
    for k in range(N_FOLDS):
        m = models[k]
        for i, r in df[df["fold"] == k].iterrows():
            h = history_of.get(r["idauniq"])
            h = h[h["t"] < r["t"] - MIN_GAP_YEARS][["age", "female", "round", "hba1c"]]
            hh = pd.concat([h, pd.DataFrame([dict(age=r["age"], female=r["female"], round=r["round"],
                                                  hba1c=r["hba1c"])])], ignore_index=True)
            Xa, Za, _ = design(hh)
            mu, _ = posterior(m, Xa, Za, hh["hba1c"].to_numpy())
            out.loc[i] = m["beta"][1] + mu[1]
    return out


# ---- outcome by HbA1c alone (variant A of Table S1)
def adjusted_hb(d, expo=SLOPE):
    """Association model for the HbA1c-only outcome: one lab-missing indicator (HDL and triglycerides are
    missing together) and the spline flat below the lowest baseline HbA1c with a case (avoids separation)."""
    d = d.dropna(subset=[expo, "hba1c"])
    y = d["incident_dm"]
    Xc = impute(d, ADJ)
    if {"hdl_miss", "log_trig_miss"} <= set(Xc.columns):
        Xc["lab_miss"] = Xc[["hdl_miss", "log_trig_miss"]].max(axis=1)
        Xc = Xc.drop(columns=["hdl_miss", "log_trig_miss"])
    lo = d.loc[y == 1, "hba1c"].min()
    X = clean_X(pd.concat([Xc, spline_cols(d["hba1c"].clip(lower=lo), "spl_")], axis=1), y)
    X["expo"] = (d[expo] - d[expo].mean()) / d[expo].std()
    f = glm_cluster(y, X, d["idauniq"])
    b, se = f.params["expo"], f.bse["expo"]
    return dict(n_baseline_visits=len(d), n_events=int(y.sum()), or_per_sd=np.exp(b),
                or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se), p_value=f.pvalues["expo"])


def hba1c_only_outcome(pdf, nl):
    nxt = nl[["idauniq", "round", "hba1c", "fglu", "fasted"]].drop_duplicates(["idauniq", "round"]).rename(
        columns={"round": "next_round", "hba1c": "next_hba1c", "fglu": "next_fglu", "fasted": "next_fasted"})
    base = pdf.drop(columns=[c for c in ["next_round", "next_hba1c", "next_fglu", "next_fasted"] if c in pdf])
    d = base.assign(next_round=base["round"].map(NEXT_ROUND)).merge(nxt, on=["idauniq", "next_round"], how="left")
    d.index = pdf.index
    hb = d["next_hba1c"] >= HBA1C_DIABETES
    other = (d["incident_dm"] == 1) & ~hb                       # cases only through glucose or a report
    variants = {"A: HbA1c >= 48; other cases excluded": d[~other].assign(incident_dm=hb[~other].astype(float)),
                "B: HbA1c >= 48; other cases as non-cases": d.assign(incident_dm=hb.astype(float))}
    rows = []
    for name, dv in variants.items():
        for s, sub in [("all", dv), ("HbA1c 42-47.9", pre(dv))]:
            if sub["incident_dm"].sum() >= MIN_CELL:
                rows.append(dict(outcome=name, population=s, **adjusted_hb(sub)))
    return save(pd.DataFrame(rows), "S1_hba1c_only_outcome", ["n_baseline_visits", "n_events"])


# ---- deaths before the next health visit (scenario analysis)
def death_status(allv, iv):
    """Death wave per person from the harmonised interview status (5/6) and the wave-11 end-of-life file."""
    st = read_tab(DATA_DIR / HARMONIZED_FILE, ["idauniq"] + [f"r{w}iwstat" for w in W])
    death_wave = pd.Series(np.nan, index=st["idauniq"].values)
    for w in W:
        c = f"r{w}iwstat"
        if c in st:
            newly = death_wave.isna().values & st[c].isin([5, 6]).values
            death_wave[newly] = w
    eol = DATA_DIR / W11_EOL_FILE
    if eol.exists():
        ids = read_tab(eol, ["idauniq"])["idauniq"]
        add = ids[ids.isin(death_wave.index) & death_wave.reindex(ids).isna().values]
        death_wave.loc[add.values] = 11
    else:
        log("wave-11 end-of-life file not found: deaths after wave 10 are not counted")
    seen = set(map(tuple, iv.loc[iv["dm_report"].notna() | iv["t"].notna(), ["idauniq", "wave"]].to_numpy()))
    dw = allv["idauniq"].map(death_wave)

    def interviewed_next(p, nr, w):
        # round 8/9 spans waves 8 and 9: an interview at either counts, as in extended_set()
        return (p, w) in seen or (nr == 8 and ((p, 8) in seen or (p, 9) in seen))

    return pd.Series(np.select(
        [allv["primary"], dw.notna() & (dw <= allv["next_core_wave"]),
         [interviewed_next(p, nr, w) for p, nr, w in zip(allv["idauniq"], allv["next_round"], allv["next_core_wave"])]],
        ["HbA1c at next round (analysis set)", "died before next round", "interviewed at the next round, no HbA1c"],
        default="not interviewed at the next round"), index=allv.index)


def rubin_pool(qs, us):
    r = rubin(qs, us)
    return r["or_per_sd"], r["or_lo"], r["or_hi"], r["p_value"]


def deaths_scenarios(pdf, dead):
    """Deaths added with outcomes drawn from k x the risk predicted by a model with / without the slope."""
    rows = []
    both = pd.concat([pdf.assign(_dead=0), dead.assign(_dead=1)])
    fit_mask = both["_dead"].to_numpy() == 0
    slope_z = ((both[SLOPE] - pdf[SLOPE].mean()) / pdf[SLOPE].std()).rename("expo")
    for label, with_slope, seed in [("risk model with slope", True, RANDOM_SEED),
                                    ("risk model without slope", False, RANDOM_SEED + 1)]:
        parts = [impute(both, ADJ), spline_cols(both["hba1c"], "spl_")] + ([slope_z] if with_slope else [])
        Xb = pd.concat(parts, axis=1)
        Xb = Xb.loc[:, Xb.std() > 1e-8].assign(const=1.0)
        risk = sm.GLM(both.loc[fit_mask, "incident_dm"].to_numpy(), Xb[fit_mask].to_numpy(),
                      family=sm.families.Binomial()).fit()
        p_dead = risk.predict(Xb[~fit_mask].to_numpy())
        rng = np.random.default_rng(seed)
        for k in K_RISK:
            qs, us, ev = [], [], []
            pk = np.clip(k * p_dead, 0, 1)
            for _ in range(N_DRAW):
                y = (rng.random(len(dead)) < pk).astype(float)
                r = adjusted(pd.concat([pdf, dead.assign(incident_dm=y)]), SLOPE)
                qs.append(np.log(r["or_per_sd"])); us.append(((np.log(r["or_hi"]) - np.log(r["or_lo"])) / 3.92) ** 2)
                ev.append(y.sum())
            o, lo, hi, p = rubin_pool(qs, us)
            rows.append(dict(scenario=label, k=k, n_baseline_visits=len(pdf) + len(dead),
                             mean_imputed_cases_among_deaths=round(float(np.mean(ev))),
                             or_per_sd=o, or_lo=lo, or_hi=hi, p_value=p))
    return save(pd.DataFrame(rows), "S1_deaths_scenarios", ["n_baseline_visits", "mean_imputed_cases_among_deaths"])


def death_vs_slope(allv):
    d = allv[allv["status"].isin(["HbA1c at next round (analysis set)", "died before next round"])].copy()
    d["died"] = (d["status"] == "died before next round").astype(float)
    if d["died"].sum() < MIN_CELL:
        return None
    X = clean_X(pd.concat([impute(d, ADJ_NO_HORIZON), spline_cols(d["hba1c"], "spl_"),
                           pd.get_dummies(d["round"], prefix="r", drop_first=True, dtype=float)], axis=1), d["died"])
    X["expo"] = (d[SLOPE] - d[SLOPE].mean()) / d[SLOPE].std()
    f = glm_cluster(d["died"], X, d["idauniq"])
    return save(pd.DataFrame([dict(n_baseline_visits=len(d), n_deaths=int(d["died"].sum()),
                                   or_per_sd=np.exp(f.params["expo"]), or_lo=np.exp(f.conf_int().loc["expo", 0]),
                                   or_hi=np.exp(f.conf_int().loc["expo", 1]), p_value=f.pvalues["expo"])]),
                "S1_death_vs_slope", ["n_baseline_visits", "n_deaths"])


# ---- age, birth cohort and length of participation (>= 2 earlier values)
def adjusted_plus(d, expo=SLOPE, extra=(), age_spline=False):
    d = d.dropna(subset=[expo, "hba1c"] + list(extra))
    y = d["incident_dm"]
    adj = [c for c in ADJ if not (age_spline and c == "age")]
    parts = [impute(d, adj), spline_cols(d["hba1c"], "spl_")] + [d[[c]].astype(float) for c in extra]
    if age_spline:
        parts.append(spline_cols(d["age"], "age_s"))
    X = clean_X(pd.concat(parts, axis=1), y)
    X["expo"] = (d[expo] - d[expo].mean()) / d[expo].std()
    f = glm_cluster(y, X, d["idauniq"])
    b, se = f.params["expo"], f.bse["expo"]
    return dict(n_baseline_visits=len(d), n_events=int(y.sum()), or_per_sd=np.exp(b),
                or_lo=np.exp(b - 1.96 * se), or_hi=np.exp(b + 1.96 * se), p_value=f.pvalues["expo"])


def long_history_checks(pdf, lmm):
    fi = lmm.groupby("idauniq").agg(t_first=("t", "min"))
    d8 = pdf.join(fi, on="idauniq")
    d8["birth_year"] = d8["t"] - d8["age"]
    d8["years_since_first"] = d8["t"] - d8["t_first"]
    h2 = d8[d8["n_hist"] >= 2]
    rows = []
    for popl, d in [(">= 2 earlier values", h2), (">= 2 earlier values, HbA1c 42-47.9", pre(h2))]:
        for lab, kw, sub in [("main adjustment", {}, d), ("+ spline in age", dict(age_spline=True), d),
                             ("+ birth year", dict(extra=["birth_year"]), d),
                             ("+ years since first HbA1c", dict(extra=["years_since_first"]), d),
                             ("age < 70 only", {}, d[d["age"] < 70]), ("age >= 70 only", {}, d[d["age"] >= 70])]:
            n_ev = int(sub["incident_dm"].sum())
            if n_ev < MIN_CELL or n_ev > len(sub) - MIN_CELL:
                rows.append(dict(population=popl, analysis=lab, n_baseline_visits=len(sub), n_events=n_ev,
                                 note="fewer than 10 events: not estimated"))
                continue
            try:
                rows.append(dict(population=popl, analysis=lab, **adjusted_plus(sub, **kw)))
            except Exception as e:                       # e.g. a non-estimable stratum
                rows.append(dict(population=popl, analysis=lab, n_events=n_ev, note=str(e)[:80]))
    return save(pd.DataFrame(rows), "S1_age_cohort_participation", ["n_baseline_visits", "n_events"])


# =====================================================================================================
# Descriptive statistics: flow chart, Table 1, history structure
# =====================================================================================================
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
    by_outcome.rename_axis("characteristic").reset_index().to_csv(OUT_DIR / "T1_characteristics_by_outcome.csv", index=False)
    by_band.rename_axis("characteristic").reset_index().to_csv(OUT_DIR / "T1_characteristics_by_band.csv", index=False)

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


# =====================================================================================================
# 6. Drift model (equation 4): REML fits, profile likelihood for tau, censored ML fit, cap sweep
# =====================================================================================================
def add_crossing_visits(lmm, nl):
    """Non-diabetic visits + each person's next health visit after the last one, if HbA1c >= 48."""
    last = lmm.groupby("idauniq")["age"].max().rename("last_age")
    a = nl.merge(last, left_on="idauniq", right_index=True)
    a = a[(a["age"] > a["last_age"] + MIN_GAP_YEARS) & a["hba1c"].notna()].sort_values(["idauniq", "age"])
    nxt = a.groupby("idauniq").head(1)
    cross = nxt[(nxt["hba1c"] >= HBA1C_DIABETES) & (nxt["hba1c"] <= 200)]
    cols = ["idauniq", "round", "age", "female", "hba1c"]
    return pd.concat([lmm[cols], cross[cols]], ignore_index=True)


def design6(age, female, rnd, rounds):
    a = (np.asarray(age, float) - AGE_CENTRE) / 10
    R = [(np.asarray(rnd) == r).astype(float) for r in rounds[1:]]
    return np.column_stack([np.ones_like(a), a, a ** 2, np.asarray(female, float)] + R)


def group_by_size(d, rounds):
    """Persons grouped by number of visits (covariance matrices of one size handled together)."""
    d = d.sort_values(["idauniq", "age"])
    X = design6(d["age"], d["female"], d["round"], rounds)
    t, y, ids = d["age"].to_numpy(float), d["hba1c"].to_numpy(float), d["idauniq"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(d)]
    by_k = {}
    for s, e in zip(starts, ends):
        by_k.setdefault(e - s, []).append(np.arange(s, e))
    G = {k: dict(t=t[np.array(v)], a=(t[np.array(v)] - AGE_CENTRE) / 10, y=y[np.array(v)], X=X[np.array(v)])
         for k, v in by_k.items()}
    return G, X.shape[1], dict(persons=len(starts), visits=len(d))


def unpack(p, drift, tau_fixed=None):
    sd0, sd1, rho, se = np.exp(p[0]), np.exp(p[1]), np.tanh(p[2]), np.exp(p[3])
    if not drift:
        return sd0, sd1, rho, 0.0, se, None
    return sd0, sd1, rho, np.exp(p[4]), se, (tau_fixed if tau_fixed is not None else np.exp(p[5]))


def neg_reml(p, G, nX, drift, tau_fixed=None, return_beta=False):
    """Minus the REML log-likelihood (fixed effects by generalised least squares)."""
    sd0, sd1, rho, sw, se, tau = unpack(p, drift, tau_fixed)
    Gm = np.array([[sd0 ** 2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1 ** 2]])
    XtVX, XtVy, logdet, cache = np.zeros((nX, nX)), np.zeros(nX), 0.0, []
    for k, g in G.items():
        Z = np.stack([np.ones_like(g["a"]), g["a"]], axis=-1)
        V = Z @ Gm @ np.swapaxes(Z, 1, 2) + se ** 2 * np.eye(k)
        if drift and sw > 0:
            V = V + sw ** 2 * np.exp(-np.abs(g["t"][:, :, None] - g["t"][:, None, :]) / tau)
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return (1e12, None) if return_beta else 1e12
        logdet += 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum()
        Xs = np.linalg.solve(L, g["X"]); ys = np.linalg.solve(L, g["y"][..., None])[..., 0]
        XtVX += np.einsum("nkp,nkq->pq", Xs, Xs); XtVy += np.einsum("nkp,nk->p", Xs, ys)
        cache.append((Xs, ys))
    XtVX += 1e-10 * np.eye(nX)
    beta = np.linalg.solve(XtVX, XtVy)
    nll = 0.5 * (logdet + sum(((ys - Xs @ beta) ** 2).sum() for Xs, ys in cache) + np.linalg.slogdet(XtVX)[1])
    return (nll, beta) if return_beta else nll


def fit(G, nX, drift, start, tau_fixed=None):
    res = minimize(neg_reml, np.array(start, float), args=(G, nX, drift, tau_fixed), method="L-BFGS-B",
                   options=dict(maxiter=500))
    nll, beta = neg_reml(res.x, G, nX, drift, tau_fixed, return_beta=True)
    sd0, sd1, rho, sw, se, tau = unpack(res.x, drift, tau_fixed)
    return dict(nll=float(nll), x=res.x.tolist(), sd_level=float(sd0), sd_slope_per_decade=float(sd1),
                corr_level_slope=float(rho), sd_drift=float(sw), sd_noise=float(se),
                tau=None if tau is None else float(tau), drift_share=float(sw ** 2 / (sw ** 2 + se ** 2)),
                sd_within=float(np.sqrt(sw ** 2 + se ** 2)), beta=None if beta is None else beta.tolist())


def fit_drift_model(d, rounds, grid=TAU_GRID):
    """Independent-noise model, drift model on the tau grid (profile), then the drift model with tau free."""
    G, nX, info = group_by_size(d, rounds)
    sdy = float(d["hba1c"].std())
    white = fit(G, nX, False, [np.log(0.7 * sdy), np.log(0.25 * sdy), 0.0, np.log(0.4 * sdy)])
    s0 = white["x"][:3] + [np.log(max(0.6 * white["sd_noise"], 1e-3)), np.log(max(0.7 * white["sd_noise"], 1e-3))]
    profile, best = [], None
    for tau in grid:
        f = fit(G, nX, True, s0, tau_fixed=tau)
        profile.append(dict(tau=tau, nll=f["nll"]))
        if best is None or f["nll"] < best["nll"]:
            best = f
    drift = fit(G, nX, True, best["x"] + [np.log(best["tau"])])
    drift = best if drift["nll"] > best["nll"] else drift
    prof = pd.DataFrame(profile); prof["lr_drop"] = prof["nll"] - drift["nll"]
    inside = prof.loc[prof["lr_drop"] <= 1.92, "tau"]
    return dict(info=info, rounds=[int(r) for r in rounds], white=white, drift=drift,
                tau_ci=[float(min(inside.min(), drift["tau"])), float(max(inside.max(), drift["tau"]))],
                lr_drift_vs_white=float(2 * (white["nll"] - drift["nll"])),
                delta_aic=float((2 * drift["nll"] + 12) - (2 * white["nll"] + 8)), profile=prof.to_dict("list"))


def interp_ci(tau, drop, thr=1.92):
    """Profile-likelihood interval: where the drop crosses thr, linear interpolation on log(tau)."""
    tau, drop = np.asarray(tau, float), np.asarray(drop, float)
    o = np.argsort(tau); lt, dp = np.log(tau[o]), drop[o]
    k = int(np.argmin(dp)); lo = hi = np.nan
    for i in range(k, 0, -1):
        if dp[i - 1] > thr >= dp[i]:
            lo = np.exp(lt[i - 1] + (thr - dp[i - 1]) * (lt[i] - lt[i - 1]) / (dp[i] - dp[i - 1])); break
    for i in range(k, len(dp) - 1):
        if dp[i + 1] > thr >= dp[i]:
            hi = np.exp(lt[i] + (thr - dp[i]) * (lt[i + 1] - lt[i]) / (dp[i + 1] - dp[i])); break
    return lo, hi


def load_json(name):
    p = WORK_DIR / name
    return json.load(open(p)) if p.exists() and not REFIT else {}


def dump_json(obj, name):
    json.dump(obj, open(WORK_DIR / name, "w"), indent=1)


CAPPED = f"plus crossing visit, capped at {CROSSING_CAP:g}"


def drift_fits(datasets, rounds):
    """REML fits (non-diabetic, plus crossing value; the capped fit only provides starting values)."""
    FITS = load_json("06_drift_fits.json")
    for name, d in datasets.items():
        if name not in FITS:
            log(f"drift model: {name} ...")
            FITS[name] = fit_drift_model(d, rounds)
            dump_json(FITS, "06_drift_fits.json")
    # finer profile where the non-diabetic profile is flat, started at the best drift fit
    extra = load_json("08r_profile_extra.json") or {"censored": [], "nondiabetic": []}
    G, nX, _ = group_by_size(datasets["non-diabetic visits"], rounds)
    best = FITS["non-diabetic visits"]["drift"]
    done = {p["tau"] for p in extra["nondiabetic"]}
    for tau in [t for t in EXTRA_TAUS_NONDIABETIC if t not in done]:
        extra["nondiabetic"].append(dict(tau=tau, nll=fit(G, nX, True, best["x"][:5], tau_fixed=tau)["nll"]))
        dump_json(extra, "08r_profile_extra.json")
    return FITS, extra


def profile_frame(fitres, extra_points=()):
    p = pd.DataFrame(fitres["profile"])[["tau", "nll"]]
    if len(extra_points):
        p = pd.concat([p, pd.DataFrame(extra_points)[["tau", "nll"]]])
    p = p.drop_duplicates("tau", keep="last").sort_values("tau")
    p["lr_drop"] = p["nll"] - fitres["drift"]["nll"]
    return p


# ---- censored likelihood: the crossing value enters only as P(Y >= 48 | earlier values)
def groups_cens(d, rounds):
    d = d.sort_values(["idauniq", "age"])
    X = design6(d["age"], d["female"], d["round"], rounds)
    t, y, ids = d["age"].to_numpy(float), d["hba1c"].to_numpy(float), d["idauniq"].to_numpy()
    cens = d["cens"].to_numpy(bool)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(d)]
    by = {}
    for s, e in zip(starts, ends):
        c = bool(cens[e - 1])
        if c and e - s < 2:
            continue
        by.setdefault((e - s, c), []).append(np.arange(s, e))
    return {key: dict(t=t[np.array(v)], a=(t[np.array(v)] - AGE_CENTRE) / 10, y=y[np.array(v)], X=X[np.array(v)])
            for key, v in by.items()}, X.shape[1]


def neg_ml_cens(p, Gs, nX, tau_fixed=None, limit=HBA1C_DIABETES):
    """Minus the ML log-likelihood; for censored persons density of the observed values x P(last >= limit | them)."""
    sd0, sd1, rho, se = np.exp(p[0]), np.exp(p[1]), np.tanh(p[2]), np.exp(p[3])
    sw = np.exp(p[4]); tau = tau_fixed if tau_fixed is not None else np.exp(p[5])
    beta = np.asarray(p[5 if tau_fixed is not None else 6:])
    Gm = np.array([[sd0 ** 2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1 ** 2]])
    ll = 0.0
    for (k, cens), g in Gs.items():
        Z = np.stack([np.ones_like(g["a"]), g["a"]], axis=-1)
        V = Z @ Gm @ np.swapaxes(Z, 1, 2) + se ** 2 * np.eye(k) \
            + sw ** 2 * np.exp(-np.abs(g["t"][:, :, None] - g["t"][:, None, :]) / tau)
        mu = g["X"] @ beta
        o = k - 1 if cens else k
        try:
            L = np.linalg.cholesky(V[:, :o, :o])
        except np.linalg.LinAlgError:
            return 1e12
        r = (g["y"][:, :o] - mu[:, :o])[..., None]
        w = np.linalg.solve(L, r)[..., 0]
        ll += -0.5 * (2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum() + (w ** 2).sum() + r.size * np.log(2 * np.pi))
        if cens:
            a = np.linalg.solve(L, V[:, :o, o][..., None])[..., 0]
            m_c, v_c = mu[:, o] + (a * w).sum(1), V[:, o, o] - (a ** 2).sum(1)
            if np.any(v_c <= 0):
                return 1e12
            ll += norm.logcdf((m_c - limit) / np.sqrt(v_c)).sum()
    return -ll


def fit_ml(Gs, nX, start_cov, start_beta, tau_fixed=None):
    p0 = list(start_cov[:5]) + ([] if tau_fixed is not None else [start_cov[5]]) + list(start_beta)
    res = minimize(neg_ml_cens, np.array(p0, float), args=(Gs, nX, tau_fixed), method="L-BFGS-B",
                   options=dict(maxiter=2000, maxfun=60000))
    p = res.x
    sw, se = float(np.exp(p[4])), float(np.exp(p[3]))
    return dict(nll=float(res.fun), x=p.tolist(), converged=bool(res.success),
                tau=tau_fixed if tau_fixed is not None else float(np.exp(p[5])),
                sd_level=float(np.exp(p[0])), sd_slope_per_decade=float(np.exp(p[1])),
                corr_level_slope=float(np.tanh(p[2])), sd_drift=sw, sd_noise=se,
                drift_share=sw ** 2 / (sw ** 2 + se ** 2), sd_within=float(np.sqrt(sw ** 2 + se ** 2)))


NAME_C = "ML, crossing value censored at >= 48"
NAME_ML_OBS = "ML, crossing value as observed"


def censored_fits(base_cross, rounds, FITS, extra):
    """Censored drift fit (ML), its tau profile, the ML fit with crossing values as observed (REML check),
    and the independent-noise model with the same censored likelihood (several starts)."""
    CENS = load_json("08e_censored_drift.json")
    ref = FITS[CAPPED]["drift"]                                   # starting values
    start_cov, start_beta = ref["x"][:5] + [np.log(ref["tau"])], ref["beta"]
    variants = {NAME_ML_OBS: base_cross.assign(cens=False),
                NAME_C: base_cross.assign(cens=base_cross["hba1c"] >= HBA1C_DIABETES)}
    for name, d in variants.items():
        if name not in CENS:
            log(f"censored drift model: {name} ...")
            Gs, nX = groups_cens(d, rounds)
            CENS[name] = fit_ml(Gs, nX, start_cov, start_beta)
            dump_json(CENS, "08e_censored_drift.json")
    Gs_c, nX_c = groups_cens(variants[NAME_C], rounds)
    best = CENS[NAME_C]
    if "profile" not in CENS:
        CENS["profile"] = [dict(tau=t, nll=fit_ml(Gs_c, nX_c, best["x"][:5], best["x"][6:], tau_fixed=t)["nll"])
                           for t in CENSORED_PROFILE_TAUS]
        dump_json(CENS, "08e_censored_drift.json")
    done = {p["tau"] for p in extra["censored"]}
    for tau in [t for t in EXTRA_TAUS_CENSORED if t not in done]:
        f = fit_ml(Gs_c, nX_c, best["x"][:5], best["x"][6:], tau_fixed=tau)
        extra["censored"].append(dict(tau=tau, nll=f["nll"], converged=f["converged"]))
        dump_json(extra, "08r_profile_extra.json")
    if "white_censored" not in CENS:
        def neg_white_cens(q, Gs, nX):
            return neg_ml_cens(np.r_[q[:4], -30.0, q[4:]], Gs, nX, tau_fixed=1.0)     # drift switched off
        xc = best["x"]
        sd_within = np.sqrt(np.exp(2 * xc[3]) + np.exp(2 * xc[4]))
        bw = None
        for sd1_start in [0.3, 0.8, 1.2]:
            res = minimize(neg_white_cens, np.r_[xc[0], np.log(sd1_start), 0.0, np.log(sd_within), xc[6:]],
                           args=(Gs_c, nX_c), method="L-BFGS-B",
                           options=dict(maxiter=5000, maxfun=200000, ftol=1e-12, gtol=1e-6))
            if bw is None or res.fun < bw.fun:
                bw = res
        CENS["white_censored"] = dict(nll=float(bw.fun), converged=bool(bw.success),
                                      sd_slope_per_decade=float(np.exp(bw.x[1])),
                                      corr_level_slope=float(np.tanh(bw.x[2])), sd_noise=float(np.exp(bw.x[3])))
        dump_json(CENS, "08e_censored_drift.json")
    prof = pd.concat([pd.DataFrame(CENS["profile"])[["tau", "nll"]], pd.DataFrame(extra["censored"])[["tau", "nll"]],
                      pd.DataFrame([dict(tau=best["tau"], nll=best["nll"])])]).drop_duplicates("tau", keep="last")
    prof = prof.sort_values("tau")
    prof["lr_drop"] = prof["nll"] - prof["nll"].min()
    f_c = dict(best)
    f_c["beta"] = list(best["x"][6:])                                # fixed effects of the censored fit
    return CENS, prof, f_c, Gs_c, nX_c


def check_censored_likelihood(f_c, Gs_c, nX_c):
    """Independent check of the censored likelihood (full-covariance Gaussian conditioning, scipy)."""
    x = np.asarray(f_c["x"], float)
    sd0, sd1, rho, se, sw, tau = np.exp(x[0]), np.exp(x[1]), np.tanh(x[2]), np.exp(x[3]), np.exp(x[4]), np.exp(x[5])
    beta = x[6:]
    Gm = np.array([[sd0 ** 2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1 ** 2]])

    def brute(g, j, cens):
        t, a, y, X = g["t"][j], g["a"][j], g["y"][j], g["X"][j]
        Z = np.column_stack([np.ones_like(a), a])
        V = Z @ Gm @ Z.T + se ** 2 * np.eye(len(t)) + sw ** 2 * np.exp(-np.abs(t[:, None] - t[None, :]) / tau)
        mu = X @ beta
        if not cens:
            return stats.multivariate_normal(mu, V).logpdf(y)
        o = len(t) - 1
        k = np.linalg.solve(V[:o, :o], V[:o, o])
        return (stats.multivariate_normal(mu[:o], V[:o, :o]).logpdf(y[:o])
                + norm.logsf(HBA1C_DIABETES, loc=mu[o] + k @ (y[:o] - mu[:o]), scale=np.sqrt(V[o, o] - k @ V[:o, o])))

    diffs = []
    for key in [k for k in Gs_c if k[1]][:4] + [k for k in Gs_c if not k[1]][:2]:
        g = Gs_c[key]; n = min(20, len(g["y"]))
        mine = -neg_ml_cens(x, {key: {f: v[:n] for f, v in g.items()}}, nX_c)
        diffs.append(abs(mine - sum(brute(g, j, key[1]) for j in range(n))))
    log(f"censored likelihood check: max |difference| = {max(diffs):.1e}")
    return max(diffs)


def drift_table(FITS, extra, CENS, prof_c):
    """Table 4: independent-noise and drift models in three data sets."""
    rows = []
    for key, label in [("non-diabetic visits", "non-diabetic visits (REML)"),
                       ("plus crossing visit", "plus crossing value (REML)")]:
        r = FITS[key]
        p = profile_frame(r, extra["nondiabetic"] if key == "non-diabetic visits" else ())
        lo, hi = interp_ci(p["tau"], p["lr_drop"])
        rows.append(dict(data=label, persons=r["info"]["persons"], visits=r["info"]["visits"],
                         white_slope_sd=r["white"]["sd_slope_per_decade"], drift_slope_sd=r["drift"]["sd_slope_per_decade"],
                         drift_share=r["drift"]["drift_share"], corr_level_slope=r["drift"]["corr_level_slope"],
                         tau=r["drift"]["tau"], tau_lo=lo, tau_hi=hi, lr=r["lr_drift_vs_white"],
                         delta_aic=-r["lr_drift_vs_white"] + 4))          # AIC(drift) - AIC(independent noise)
    w, dr = CENS["white_censored"], CENS[NAME_C]
    lr = 2 * (w["nll"] - dr["nll"])
    lo, hi = interp_ci(prof_c["tau"], prof_c["lr_drop"])
    rows.append(dict(data="crossing value censored at 48 (ML)", persons=FITS["plus crossing visit"]["info"]["persons"],
                     visits=FITS["plus crossing visit"]["info"]["visits"], white_slope_sd=w["sd_slope_per_decade"],
                     drift_slope_sd=dr["sd_slope_per_decade"], drift_share=dr["drift_share"],
                     corr_level_slope=dr["corr_level_slope"], tau=dr["tau"], tau_lo=lo, tau_hi=hi, lr=lr,
                     delta_aic=-lr + 4))
    ml_obs = CENS[NAME_ML_OBS]
    rows.append(dict(data="check: plus crossing value, ML (compare with REML row)", drift_slope_sd=ml_obs["sd_slope_per_decade"],
                     drift_share=ml_obs["drift_share"], tau=ml_obs["tau"]))
    return save(pd.DataFrame(rows), "T4_drift_models")


def fit_drift_quick(d, rounds):
    """Cap sweep: as fit_drift_model on the coarse grid SWEEP_TAUS."""
    G, nX, info = group_by_size(d, rounds)
    sdy = float(d["hba1c"].std())
    white = fit(G, nX, False, [np.log(0.7 * sdy), np.log(0.25 * sdy), 0.0, np.log(0.4 * sdy)])
    s0 = white["x"][:3] + [np.log(max(0.6 * white["sd_noise"], 1e-3)), np.log(max(0.7 * white["sd_noise"], 1e-3))]
    prof, best = [], None
    for tau in SWEEP_TAUS:
        f = fit(G, nX, True, s0, tau_fixed=tau)
        prof.append(dict(tau=tau, nll=f["nll"]))
        if best is None or f["nll"] < best["nll"]:
            best = f
    drift = fit(G, nX, True, best["x"] + [np.log(best["tau"])])
    drift = best if drift["nll"] > best["nll"] else drift
    p = pd.DataFrame(prof); p["lr_drop"] = p["nll"] - drift["nll"]
    inside = p.loc[p["lr_drop"] <= 1.92, "tau"]
    return dict(info=info, white=white, drift=drift,
                tau_ci=[float(min(inside.min(), drift["tau"])), float(max(inside.max(), drift["tau"]))],
                lr_drift_vs_white=float(2 * (white["nll"] - drift["nll"])),
                delta_aic=float((2 * drift["nll"] + 12) - (2 * white["nll"] + 8)), profile=p.to_dict("list"))


def cap_sweep(base_cross, rounds, CENS, prof_c):
    """Supplement: the uncensored fit with the crossing values capped at 48-80 mmol/mol."""
    SWEEP = load_json("08d_cap_sweep.json")
    for cap in CAPS:
        key = "uncapped" if cap is None else f"{cap:g}"
        if key not in SWEEP:
            d = base_cross.copy()
            if cap is not None:
                d["hba1c"] = d["hba1c"].clip(upper=cap)
            log(f"cap sweep: {key} ...")
            SWEEP[key] = fit_drift_quick(d, rounds)
            dump_json(SWEEP, "08d_cap_sweep.json")
    n_cross = int((base_cross["hba1c"] >= HBA1C_DIABETES).sum())
    rows = []
    for cap in CAPS:
        key = "uncapped" if cap is None else f"{cap:g}"
        r, dr = SWEEP[key], SWEEP[key]["drift"]
        p = pd.DataFrame(r["profile"])
        lo, hi = interp_ci(p["tau"], p["lr_drop"])
        n_aff = 0 if cap is None else int((base_cross["hba1c"] > cap).sum())
        rows.append(dict(cap=key, crossing_values_above_cap=n_aff, pct_above_cap=100 * n_aff / n_cross,
                         tau=dr["tau"], tau_lo=lo, tau_hi=hi, slope_sd=dr["sd_slope_per_decade"],
                         drift_share=dr["drift_share"], corr_level_slope=dr["corr_level_slope"],
                         delta_aic=-r["lr_drift_vs_white"] + 4))
    lo, hi = interp_ci(prof_c["tau"], prof_c["lr_drop"])
    dc = CENS[NAME_C]
    rows.append(dict(cap="censored at 48", tau=dc["tau"], tau_lo=lo, tau_hi=hi, slope_sd=dc["sd_slope_per_decade"],
                     drift_share=dc["drift_share"], corr_level_slope=dc["corr_level_slope"],
                     delta_aic=-2 * (CENS["white_censored"]["nll"] - dc["nll"]) + 4))
    sweep = save(pd.DataFrame(rows), "S5_cap_sweep", ["crossing_values_above_cap"])

    plot_cap_sweep(sweep)
    return sweep


def crossing_value_numbers(base_cross, nl, lmm):
    """Crossing values: n, median, IQR, share above 55 and 80; raw HbA1c values > 200 mmol/mol."""
    raw = []
    for rnd, spec in NURSE_ROUNDS.items():
        n = read_tab(DATA_DIR / spec["file"], ["idauniq", "hba1c"])
        conv, _, _ = detect_units(n["hba1c"].where(n["hba1c"] >= 0).where(lambda s: s < 9990), "hba1c")
        raw.append(pd.DataFrame(dict(idauniq=n["idauniq"], round=rnd, hba1c_raw=conv)))
    raw = pd.concat(raw, ignore_index=True)
    cv = base_cross.loc[base_cross["hba1c"] >= HBA1C_DIABETES, "hba1c"]
    rows = [dict(item="HbA1c values > 200 mmol/mol, all rounds (raw files)", value=n_str((raw["hba1c_raw"] > 200).sum())),
            dict(item="crossing values, n", value=n_str(len(cv))),
            dict(item="crossing values, median (IQR)", value=med_iqr(cv, 0))]
    for thr in [55, 60, 65, 70, 80]:
        k = int((cv > thr).sum())
        rows.append(dict(item=f"crossing values > {thr}, n (%)", value=f"{n_str(k)} ({100 * k / len(cv):.1f})"))
    return save(pd.DataFrame(rows), "T4_crossing_values")


# =====================================================================================================
# 6b. Slope association adjusted for the drift-model estimate of the current level, and the self-test
# =====================================================================================================
def gmat(f):
    s0, s1, rho = f["sd_level"], f["sd_slope_per_decade"], f["corr_level_slope"]
    return np.array([[s0 ** 2, rho * s0 * s1], [rho * s0 * s1, s1 ** 2]])


def cov_hist(t, f):
    Z = np.column_stack([np.ones_like(t), (t - AGE_CENTRE) / 10])
    V = Z @ gmat(f) @ Z.T + f["sd_noise"] ** 2 * np.eye(len(t))
    if f["tau"]:
        V = V + f["sd_drift"] ** 2 * np.exp(-np.abs(t[:, None] - t[None, :]) / f["tau"])
    return Z, V


def level_blup(t_hist, r_hist, t0, f):
    """Conditional mean of the underlying level (trajectory + drift, no independent noise) at t0."""
    Z, V = cov_hist(t_hist, f)
    c = np.array([1.0, (t0 - AGE_CENTRE) / 10]) @ gmat(f) @ Z.T
    if f["tau"]:
        c = c + f["sd_drift"] ** 2 * np.exp(-np.abs(t0 - t_hist) / f["tau"])
    return float(c @ np.linalg.solve(V, r_hist))


def slope_blup(t_hist, r_hist, f):
    Z, V = cov_hist(t_hist, f)
    return float((gmat(f) @ Z.T @ np.linalg.solve(V, r_hist))[1])


def add_level(base, hist, f, rounds, col="level_drift", white=None):
    """Underlying-level estimate at each baseline visit from the values up to and including it."""
    H = {p: g.sort_values("age") for p, g in hist.groupby("idauniq")}
    beta, lev, slo = np.asarray(f["beta"]), [], []
    for _, row in base.iterrows():
        g = H.get(row["idauniq"])
        earlier = g[g["age"] < row["age"] - MIN_GAP_YEARS] if g is not None else None
        if earlier is None or earlier.empty:
            lev.append(np.nan); slo.append(np.nan); continue
        t = np.r_[earlier["age"].to_numpy(float), row["age"]]
        y = np.r_[earlier["hba1c"].to_numpy(float), row["hba1c"]]
        rnd = np.r_[earlier["round"].to_numpy(), row["round"]]
        mu = design6(t, np.full(len(t), row["female"]), rnd, rounds) @ beta
        lev.append(mu[-1] + level_blup(t, y - mu, row["age"], f))
        if white is not None:
            mw = design6(t, np.full(len(t), row["female"]), rnd, rounds) @ np.asarray(white["beta"])
            slo.append(slope_blup(t[:-1], (y - mw)[:-1], white))
    base = base.copy(); base[col] = lev
    if white is not None:
        base["slope_white"] = slo
    return base


LEVELS = {"hba1c": "baseline HbA1c", "level_drift": "drift-model level, crossing values observed",
          "level_drift_censored": "drift-model level, crossing values censored"}


def slope_vs_level(d, levels, expo=SLOPE, adj=ADJ):
    pops = [("all", d), ("HbA1c 42-47.9", pre(d))]
    if "n_hist" in d:
        h2 = d[d["n_hist"] >= 2]
        pops += [(">= 2 earlier values", h2), (">= 2 earlier values, HbA1c 42-47.9", pre(h2))]
    rows = []
    for popl, s in pops:
        s = s.dropna(subset=["level_drift"])
        for lev, lab in levels.items():
            r = adjusted(s, expo, adj=adj, level=lev) if lev in s else None
            if r:
                rows.append(dict(population=popl, adjusted_for=lab, **r))
    return pd.DataFrame(rows)


def simulate_selftest(n=12000, seed=5):
    """Simulated data in which risk depends only on the underlying current level (no ELSA data)."""
    rng = np.random.default_rng(seed)
    truth = dict(sd_level=3.0, sd_slope_per_decade=0.8, corr_level_slope=1.0, sd_drift=2.8, sd_noise=1.3, tau=5.6)
    white = dict(sd_level=3.4, sd_slope_per_decade=1.1, corr_level_slope=0.69, sd_drift=0.0, sd_noise=2.57, tau=None)
    truth["beta"] = white["beta"] = [38.6, 0.94, 0.0, 0.0, 0.0, 0.0, 0.0]
    hist, base = [], []
    for i in range(n):
        k = rng.integers(2, 5)
        t = 55 + rng.uniform(0, 15) + np.arange(k) * 4 + rng.normal(0, 0.4, k)
        u0 = rng.normal(0, truth["sd_level"]); u1 = u0 / truth["sd_level"] * truth["sd_slope_per_decade"]
        w = rng.multivariate_normal(np.zeros(k), truth["sd_drift"] ** 2 * np.exp(-np.abs(t[:, None] - t[None, :]) / truth["tau"]))
        a = (t - AGE_CENTRE) / 10
        L = 38.6 + 0.94 * a + u0 + u1 * a + w
        y = L + rng.normal(0, truth["sd_noise"], k)
        if (y >= HBA1C_DIABETES).any():
            continue
        hist += [dict(idauniq=i, age=t[j], hba1c=y[j], female=0, round=j % 4) for j in range(k)]
        p = 1 / (1 + np.exp(-(-3.2 + 0.45 * (L[-1] - 40))))
        base.append(dict(idauniq=i, age=t[-1], hba1c=y[-1], female=0, round=(k - 1) % 4, incident_dm=int(rng.random() < p)))
    st = add_level(pd.DataFrame(base), pd.DataFrame(hist), truth, [0, 1, 2, 3], "level_drift", white=white)
    return save(slope_vs_level(st, {"hba1c": "observed value", "level_drift": "estimated level"},
                               expo="slope_white", adj=[]), "S1_selftest_simulation")


# =====================================================================================================
# 7. Observability of personal slopes (equation 6)
# =====================================================================================================
def corr_slope(p, interval, length, rho=0.0, end_age=65):
    """r = corr(estimated, underlying slope) for measurements every `interval` years over `length` years,
    the last one at age 65; intercept-slope correlation rho (0 = reference scenario)."""
    t = np.arange(end_age - length, end_age + 1e-9, interval)
    Z = np.column_stack([np.ones_like(t), (t - AGE_CENTRE) / 10])
    c01 = rho * p["sd_level"] * p["sd_slope"]
    G = np.array([[p["sd_level"] ** 2, c01], [c01, p["sd_slope"] ** 2]])
    V0 = p["sd_noise"] ** 2 * np.eye(len(t))
    if p["tau"]:
        V0 = V0 + p["sd_drift"] ** 2 * np.exp(-np.abs(t[:, None] - t[None, :]) / p["tau"])
    post = G - G @ Z.T @ np.linalg.solve(Z @ G @ Z.T + V0, Z @ G)
    return float(np.sqrt(max(0.0, 1 - post[1, 1] / G[1, 1])))


def pars(f):
    return dict(sd_level=f["sd_level"], sd_slope=f["sd_slope_per_decade"], sd_noise=f["sd_noise"],
                sd_drift=f.get("sd_drift", 0.0) or 0.0, tau=f.get("tau"), rho=f.get("corr_level_slope", 0.0))


def plot_observability(scen):
    """Figure 6: correlation of the estimated with the true slope, for five visit intervals per model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    labels = {0.25: "3 months", 0.5: "6 months", 1: "1 year", 2: "2 years", 4: "4 years"}
    titles = {"Independent noise": "No drift (independent noise)",
          "Drift, with crossing values": "Drift, crossing values included",
          "Drift, crossing values censored": "Drift, crossing values censored"}
    
    for ax, (lab, (name, p)) in zip(axes, zip("ABC", scen.items())):
        for dlt in OBS_INTERVALS:
            ax.plot(OBS_LENGTHS, [corr_slope(p, dlt, L) for L in OBS_LENGTHS], marker="o", ms=3, label=labels[dlt])
        ax.set(title=f"({lab}) {titles[name]}", xlabel="Observation period (years)",
                xlim=(2, 31), ylim=(-0.05, 1.05), xticks=OBS_LENGTHS)
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        #ax.axvspan(4, 12, color="grey", alpha=0.12, lw=0); ax.spines[["top", "right"]].set_visible(False)
    
    axes[0].set_ylabel("Correlation $r$, estimated vs underlying slope")
    axes[0].legend(frameon=False, fontsize=10, title="Visit interval")
    fig.tight_layout(); fig.savefig(OUT_DIR / "FIG_fig_observability.png", dpi=300); plt.close(fig)


def observability(vc, FITS, f_c):
    scen = {"Independent noise": dict(sd_level=float(vc["sd_intercept"].mean()),
                                      sd_slope=float(vc["sd_slope_per_decade"].mean()),
                                      sd_noise=float(vc["sd_residual"].mean()), sd_drift=0.0, tau=None,
                                      rho=float(vc["corr_int_slope"].mean())),
            "Drift, with crossing values": pars(FITS["plus crossing visit"]["drift"]),
            "Drift, crossing values censored": pars(f_c)}
    rows = []
    for name, p in scen.items():
        for L in [4, 12, 30]:
            rows.append(dict(model=name, fitted_rho=p["rho"], history_years=L,
                             four_yearly_rho0=corr_slope(p, 4, L), four_yearly_fitted=corr_slope(p, 4, L, p["rho"]),
                             quarterly_rho0=corr_slope(p, 0.25, L), quarterly_fitted=corr_slope(p, 0.25, L, p["rho"])))
    save(pd.DataFrame(rows), "S2_observability_table")
    grid = [dict(model=n, tau=p["tau"], interval_years=i, history_years=L, r_rho0=corr_slope(p, i, L))
            for n, p in scen.items() for i in OBS_INTERVALS for L in OBS_LENGTHS]
    save(pd.DataFrame(grid), "S2_observability_grid")

    plot_observability(scen)


# =====================================================================================================
# 8. Model checks: semivariogram, mixed-model diagnostics, residual correlation across rounds
# =====================================================================================================
def variogram(d, fitset, rounds, rng):
    d = d.sort_values(["idauniq", "age"]).reset_index(drop=True)
    r = d["hba1c"].to_numpy() - design6(d["age"], d["female"], d["round"], rounds) @ np.asarray(fitset["white"]["beta"])
    ids, t = d["idauniq"].to_numpy(), d["age"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(d)]
    I, J, P = [], [], []
    for k, (s, e) in enumerate(zip(starts, ends)):
        if e - s > 1:
            ii, jj = np.triu_indices(e - s, 1)
            I.append(ii + s); J.append(jj + s); P.append(np.full(len(ii), k))
    I, J, P = np.concatenate(I), np.concatenate(J), np.concatenate(P)
    lag, v = np.abs(t[J] - t[I]), 0.5 * (r[J] - r[I]) ** 2
    b = np.digitize(lag, LAG_BINS) - 1
    ok = (b >= 0) & (b < len(LAG_BINS) - 1)
    lag, v, P, b = lag[ok], v[ok], P[ok], b[ok]
    nb, npers = len(LAG_BINS) - 1, len(starts)
    boot = np.empty((N_BOOT_VARIOGRAM, nb))
    for i in range(N_BOOT_VARIOGRAM):
        w = rng.multinomial(npers, np.full(npers, 1 / npers))[P]
        boot[i] = np.bincount(b, w * v, nb) / np.maximum(np.bincount(b, w, nb), 1e-12)
    rows = []
    for k in range(nb):
        m = b == k
        rows.append(dict(lag_from=LAG_BINS[k], lag_to=LAG_BINS[k + 1], pairs=int(m.sum()),
                         mean_lag=lag[m].mean() if m.any() else np.nan, semivariance=v[m].mean() if m.any() else np.nan,
                         lo=np.nanpercentile(boot[:, k], 2.5), hi=np.nanpercentile(boot[:, k], 97.5)))
    return pd.DataFrame(rows)


def semivariogram_figure(datasets, FITS, rounds):
    rng = np.random.default_rng(RANDOM_SEED)
    keys = list(FITS)                      # same order (and random stream) as notebook 08g
    vg = pd.concat([variogram(datasets[k], FITS[k], rounds, rng).assign(data=k) for k in keys], ignore_index=True)
    vg.loc[vg["pairs"] < MIN_CELL, ["semivariance", "lo", "hi"]] = np.nan
    save(vg, "S_variogram", ["pairs"])
    plot_variogram(vg, FITS)


def plot_variogram(vg, FITS):
    """Figure S2: empirical semivariogram (binned, with bootstrap intervals) and the fitted model curves."""
    u = np.linspace(0.01, 25, 200)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    for ax, k, letter in zip(axes, ["non-diabetic visits", "plus crossing visit"], "AB"):
        g = vg[vg["data"] == k]; ok = g["semivariance"].notna(); f = FITS[k]
        ax.errorbar(g.loc[ok, "mean_lag"], g.loc[ok, "semivariance"],
                    yerr=[g.loc[ok, "semivariance"] - g.loc[ok, "lo"], g.loc[ok, "hi"] - g.loc[ok, "semivariance"]],
                    fmt="o", color="black", capsize=3, label="empirical")
        for m, col, lab in [(f["white"], "#52514e", "independent noise"), (f["drift"], "#2a78d6", "drift")]:
            gam = m["sd_noise"] ** 2 + 0.5 * m["sd_slope_per_decade"] ** 2 * (u / 10) ** 2
            if m.get("tau"):
                gam = gam + m["sd_drift"] ** 2 * (1 - np.exp(-u / m["tau"]))
            ax.plot(u, gam, color=col, lw=1.5, label=lab + ((f" (τ = {m['tau']:.1f} y)" if m["tau"] < 10 else f" (τ = {m['tau']:.0f} y)") if m.get("tau") else ""))
        ax.set(title=f"{letter}  " + {"non-diabetic visits": "Non-diabetic visits", "plus crossing visit": "Plus crossing value"}[k],
               xlabel="years between the two values")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("semivariance (mmol/mol)²")
    fig.tight_layout(); fig.savefig(OUT_DIR / "FIG_fig_variogram.png", dpi=300); plt.close(fig)


def lmm_diagnostics(lmm, FITS):
    """QQ plots and residual correlation across rounds for the mixed model fitted once to all visits."""
    m3 = fit_population(lmm)
    L = lmm.sort_values(["idauniq", "t"]).reset_index(drop=True)
    X3, Z3, _ = design(L)
    marg = L["hba1c"].to_numpy() - X3 @ m3["beta"]
    cond, fitted, U = np.empty(len(L)), np.empty(len(L)), []
    for _, idx in L.groupby("idauniq").indices.items():
        mu, _ = posterior(m3, X3[idx], Z3[idx], L["hba1c"].to_numpy()[idx])
        cond[idx] = marg[idx] - Z3[idx] @ mu
        fitted[idx] = X3[idx] @ m3["beta"] + Z3[idx] @ mu
        U.append(mu)
    U = np.array(U)
    std_res = cond / np.sqrt(m3["s2"])
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6))
    axes = axes.ravel()
    for ax, x, title in [(axes[0], std_res, "A  standardised conditional residuals"),
                         (axes[1], U[:, 0], "B  personal level (EB estimate)"), (axes[2], U[:, 1], "C  personal slope (EB estimate)")]:
        sm.qqplot(x, line="s", ax=ax, markersize=2, alpha=0.4); ax.set_title(title, fontsize=10, loc="left")
    axes[3].scatter(fitted, std_res, s=2, alpha=0.2, color="#52514e")
    axes[3].axhline(0, color="#eb6834", lw=1)
    axes[3].set(xlabel="fitted HbA1c (mmol/mol)", ylabel="standardised residual")
    axes[3].set_title("D  residuals vs fitted", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT_DIR / "FIG_fig_lmm_diagnostics.png", dpi=300); plt.close(fig)
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    save(pd.DataFrame(dict(quantile=qs, standardised_residual=np.quantile(std_res, qs), normal=norm.ppf(qs))),
         "S_residual_quantiles")
    save(pd.DataFrame([dict(skewness=pd.Series(std_res).skew(), excess_kurtosis=pd.Series(std_res).kurt(),
                            sd_slope_estimates=U[:, 1].std())]), "S_residual_moments")

    Lr = L.assign(r=marg)
    W_r = Lr.pivot_table(index="idauniq", columns="round", values="r")
    W_a = Lr.pivot_table(index="idauniq", columns="round", values="age")

    def implied_corr(a1, a2, m):
        Z = np.array([[1, (a1 - AGE_CENTRE) / 10], [1, (a2 - AGE_CENTRE) / 10]])
        V = Z @ m["G"] @ Z.T + m["s2"] * np.eye(2)
        if m.get("tau"):
            V = V + m["sd_drift"] ** 2 * np.exp(-np.abs(np.subtract.outer([a1, a2], [a1, a2])) / m["tau"])
        return V[0, 1] / np.sqrt(V[0, 0] * V[1, 1])

    fd = FITS["non-diabetic visits"]["drift"]
    m_drift = dict(G=gmat(fd), s2=fd["sd_noise"] ** 2, sd_drift=fd["sd_drift"], tau=fd["tau"])
    m_white = dict(G=m3["G"], s2=m3["s2"])
    rows, rr = [], sorted(W_r.columns)
    for i, r1 in enumerate(rr):
        for r2 in rr[i + 1:]:
            ok = W_r[r1].notna() & W_r[r2].notna()
            n = int(ok.sum())
            a1, a2 = W_a.loc[ok, r1].mean(), W_a.loc[ok, r2].mean()
            rows.append(dict(round_1=r1, round_2=r2, persons=n, mean_years_apart=a2 - a1,
                             empirical=W_r.loc[ok, r1].corr(W_r.loc[ok, r2]) if n >= MIN_CELL else np.nan,
                             implied_independent_noise=implied_corr(a1, a2, m_white) if n >= MIN_CELL else np.nan,
                             implied_drift=implied_corr(a1, a2, m_drift) if n >= MIN_CELL else np.nan))
    cm = save(pd.DataFrame(rows).sort_values("mean_years_apart"), "S_residual_correlation", ["persons"])
    c = pd.DataFrame(rows).dropna(subset=["empirical"])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(c["mean_years_apart"], c["empirical"], color="black", label="empirical (pair of rounds)", zorder=3)
    ax.scatter(c["mean_years_apart"], c["implied_independent_noise"], marker="x", color="#52514e", label="independent noise")
    ax.scatter(c["mean_years_apart"], c["implied_drift"], marker="^", color="#2a78d6", label="drift")
    ax.set(xlabel="mean years between the two rounds", ylabel="correlation of residuals")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "FIG_fig_residual_correlation.png", dpi=300); plt.close(fig)
    return cm


# =====================================================================================================
# Figures 1 and 2
# =====================================================================================================
POPS = {"all": ("All baseline visits", "#0b0b0b", "o"), "HbA1c < 42": ("HbA1c < 42 mmol/mol", "#2a78d6", "s"),
        "HbA1c 42-47.9": ("HbA1c 42–47.9 mmol/mol", "#eb6834", "^")}
FEATURES_FIG = {EXPOSURES["z_personal"]: "Deviation from usual level", EXPOSURES["delta_last"]: "Change since last HbA1c",
                EXPOSURES["logit_p_cross48"]: "Forecast P(next HbA1c ≥ 48)", EXPOSURES["slope_personal"]: "Personal HbA1c slope"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def forest(ax, blocks, title):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    y, yticks, ylabels = 0, [], []
    for b_i, (block, rows) in enumerate(blocks):
        ax.text(-0.01, y, block, transform=trans, ha="right", va="center", fontweight="bold", color=INK)
        y -= 1
        for label, pop, r in rows:
            _, col, mk = POPS[pop]
            if label == "" and yticks:
                yticks[-1] = (yticks[-1] + y) / 2
            else:
                yticks.append(y); ylabels.append(label)
            if r is not None:
                ax.plot([r.or_lo, r.or_hi], [y, y], color=col, lw=1.4, solid_capstyle="butt")
                ax.plot(r.or_per_sd, y, marker=mk, color=col, ms=6, mec="white", mew=0.8, zorder=3)
                ax.text(1.02, y, f"{r.or_per_sd:.2f} ({r.or_lo:.2f}–{r.or_hi:.2f})", transform=trans, va="center",
                        color=INK, fontsize=8)
            y -= 1
        if b_i < len(blocks) - 1:
            y -= 0.6
    ax.axvline(1, color=INK2, lw=0.9, zorder=0)
    ax.set_xscale("log"); ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off(); ax.set_xlim(0.5, 3.0)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.4, 0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Odds ratio per SD (95% CI), log scale")
    ax.text(1.02, 1.0, "OR (95% CI)", transform=ax.transAxes, va="bottom", fontweight="bold", color=INK)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10, pad=14)


def figure1(assoc, sc):
    def get(population, exposure):
        r = assoc[(assoc.analysis == "primary") & (assoc.population == population) & (assoc.exposure == exposure)]
        return None if r.empty else r.iloc[0]

    def get_level(adjusted_for, population):
        r = sc[(sc.adjusted_for == adjusted_for) & (sc.population == population)]
        return None if r.empty else r.iloc[0]

    blocks_a = [(lab, [(POPS[p][0], p, get(p, f)) for p in POPS]) for f, lab in FEATURES_FIG.items()]
    rows_b = []
    for key, lab in [(LEVELS["hba1c"], "Baseline HbA1c (main analysis, as in A)"),
                     (LEVELS["level_drift"], "Drift-model level, crossing values observed"),
                     (LEVELS["level_drift_censored"], "Drift-model level, crossing values censored")]:
        for i, p in enumerate(["all", "HbA1c 42-47.9"]):
            rows_b.append((lab if i == 0 else "", p, get_level(key, p)))
    blocks_b = [("Personal HbA1c slope, adjusted for", rows_b)]
    n_a, n_b = sum(len(r) + 1 for _, r in blocks_a), sum(len(r) + 1 for _, r in blocks_b)
    with plt.rc_context({"font.size": 8.5, "pdf.fonttype": 42}):
        fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.2, 0.24 * (n_a + n_b) + 2.6),
                                       gridspec_kw={"height_ratios": [n_a + 1.5, n_b + 1.5]})
        forest(axA, blocks_a, "A  Personal features at the same HbA1c level")
        forest(axB, blocks_b, "B  Slope, adjusted for the estimated current level")
        handles = [Line2D([], [], color=c, marker=m, lw=1.4, ms=6, mec="white", label=l) for l, c, m in POPS.values()]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.035, 0.86, 1))
        fig.savefig(OUT_DIR / "FIG_fig1_odds_ratios.pdf"); fig.savefig(OUT_DIR / "FIG_fig1_odds_ratios.png", dpi=300)
        plt.close(fig)


def plot_cap_sweep(sweep):
    """Figure S1 from the cap-sweep table (S5_cap_sweep)."""
    # ---- Figure S1: estimates by cap on the crossing values; dashed line = censored fit for comparison
    sw = sweep[sweep["cap"] != "censored at 48"]
    cens = sweep[sweep["cap"] == "censored at 48"].iloc[0]
    x = np.arange(len(sw))
    lab = ["none" if c == "uncapped" else c for c in sw["cap"]]
    with plt.rc_context({"font.size": 9.5}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
        axes[0].errorbar(x, sw["tau"], yerr=[sw["tau"] - sw["tau_lo"], sw["tau_hi"] - sw["tau"]], fmt="o-",
                         color="#2a78d6", capsize=3)
        axes[0].set(yscale="log", ylabel="τ, years (log scale)")
        axes[0].set_yticks([2, 5, 10, 20, 50, 100]); axes[0].set_yticklabels(["2", "5", "10", "20", "50", "100"])
        axes[0].minorticks_off()
        axes[1].plot(x, sw["slope_sd"], "o-", color="#eb6834")
        axes[1].set(ylabel="slope SD, mmol/mol per decade", ylim=(0, None))
        axes[2].plot(x, sw["drift_share"], "o-", color="#1baf7a"); axes[2].set(ylabel="drift share π", ylim=(0, 1))
        titles = ["A  Drift time constant τ (95% CI)", "B  Between-person slope SD", "C  Drift share of within-person variance"]
        for ax, col, t in zip(axes, ["tau", "slope_sd", "drift_share"], titles):
            ax.axhline(cens[col], color="grey", lw=1, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(lab)
            ax.set_xlabel("cap on crossing values, mmol/mol")
            ax.set_title(t, loc="left", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        axes[2].text(len(x) - 1, cens["drift_share"] - 0.05, "censored fit", ha="right", va="top", fontsize=8, color="0.4")
        fig.tight_layout(); fig.savefig(OUT_DIR / "FIG_fig_cap_sweep.png", dpi=300); plt.close(fig)


def figure2_tau_profile(FITS, extra, CENS, prof_c):
    style = {"non-diabetic visits": ("#eb6834", "s", "Non-diabetic visits (REML)"),
             "plus crossing visit": ("#2a78d6", "o", "Plus crossing value (REML)"),
             "censored": ("#1baf7a", "^", "Crossing value censored at 48 (ML)")}
    prof = {"non-diabetic visits": profile_frame(FITS["non-diabetic visits"], extra["nondiabetic"]),
            "plus crossing visit": profile_frame(FITS["plus crossing visit"]), "censored": prof_c}
    best = {"non-diabetic visits": FITS["non-diabetic visits"]["drift"]["tau"],
            "plus crossing visit": FITS["plus crossing visit"]["drift"]["tau"], "censored": CENS[NAME_C]["tau"]}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rows = []
    for k, p in prof.items():
        col, mk, lab = style[k]
        lo, hi = interp_ci(p["tau"], p["lr_drop"])
        rows.append(dict(data=k, tau=best[k], tau_lo=lo, tau_hi=hi))
        ax.plot(p["tau"], p["lr_drop"], marker=mk, color=col, lw=1.6, ms=4,
                label=f"{lab}: τ = {best[k]:.1f} y ({lo:.1f}–{hi:.1f})".replace(f"{best[k]:.1f} y", f"{best[k]:.1f} y" if best[k] < 10 else f"{best[k]:.0f} y"))
    ax.axhline(1.92, color=INK2, lw=0.9, ls="--"); ax.text(0.55, 2.2, "95% interval", color=INK2, fontsize=8)
    ax.set(xscale="log", xlabel="drift time constant τ (years)", ylabel="log-likelihood below the best", ylim=(-0.3, 15))
    ax.set_xlim(2, 120)
    ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.minorticks_off(); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(); fig.savefig(OUT_DIR / "FIG_fig2_tau_profile.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    save(pd.DataFrame(rows), "T4_tau_intervals")


# =====================================================================================================
# Main
# =====================================================================================================
def figures_only():
    """Redraw Figures 4, 5, 6, S1 and S2 from the saved output tables (--out) and the cached drift fits (--work),
    without reading the ELSA data or refitting anything."""
    FITS, CENS = load_json("06_drift_fits.json"), load_json("08e_censored_drift.json")
    if not FITS or not CENS:
        raise SystemExit("cached drift fits not found in --work: run the full analysis first")
    extra = load_json("08r_profile_extra.json") or {"censored": [], "nondiabetic": []}
    best = CENS[NAME_C]
    prof_c = pd.concat([pd.DataFrame(CENS["profile"])[["tau", "nll"]], pd.DataFrame(extra["censored"], columns=["tau", "nll"]),
                        pd.DataFrame([dict(tau=best["tau"], nll=best["nll"])])]).drop_duplicates("tau", keep="last")
    prof_c = prof_c.sort_values("tau")
    prof_c["lr_drop"] = prof_c["nll"] - prof_c["nll"].min()
    f_c = dict(best)
    f_c["beta"] = list(best["x"][6:])
    rd = lambda name: pd.read_csv(OUT_DIR / f"{name}.csv")
    figure1(rd("T3_S1_adjusted_associations"), rd("S1_slope_vs_drift_level"))
    figure2_tau_profile(FITS, extra, CENS, prof_c)
    observability(rd("T2_variance_components_by_fold"), FITS, f_c)
    plot_cap_sweep(rd("S5_cap_sweep"))
    plot_variogram(rd("S_variogram"), FITS)
    log(f"figures redrawn in {OUT_DIR}")


def main():
    global DATA_DIR, WORK_DIR, OUT_DIR, N_BOOT, REFIT
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
    DATA_DIR, WORK_DIR, OUT_DIR = Path(a.data), Path(a.work), Path(a.out)
    WORK_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.figures_only:
        figures_only()
        return
    if a.from_features:
        features = derive(pd.read_pickle(WORK_DIR / "features.pkl"))
        iv = pd.read_pickle(WORK_DIR / "interviews.pkl")
        export_final_models(features)
        df = features[features["primary"] & features["z_personal"].notna()].copy()
        step5_associations(features, df, iv)
        slope_crude(df)
        slope_reversal_diagnostics(df)
        slope_covariate_steps(df)
        slope_history_and_age(df)
        log(f"done (from saved features): tables in {OUT_DIR}")
        return
    N_BOOT, REFIT = (100 if a.quick else 1000), a.refit

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
    log(f"done: tables and figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
