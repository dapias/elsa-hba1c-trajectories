"""Section 5. Association analyses (equation 4) and the sensitivity analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from . import config
from .config import (
    ADJ, ADJ_NO_HORIZON, EXPOSURES, HARMONIZED_FILE, HBA1C_DIABETES, HBA1C_PREDIAB, HOMOGENEOUS_ROUNDS,
    K_RISK, MIN_CELL, MIN_GAP_YEARS, NEXT_ROUND, N_DRAW, N_FOLDS, N_IMP, RANDOM_SEED, SLOPE, W,
    W11_EOL_FILE,
)
from .output import log, save
from .data import read_tab
from .features import design, posterior
from .prediction import fast_auc


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
    st = read_tab(config.DATA_DIR / HARMONIZED_FILE, ["idauniq"] + [f"r{w}iwstat" for w in W])
    death_wave = pd.Series(np.nan, index=st["idauniq"].values)
    for w in W:
        c = f"r{w}iwstat"
        if c in st:
            newly = death_wave.isna().values & st[c].isin([5, 6]).values
            death_wave[newly] = w
    eol = config.DATA_DIR / W11_EOL_FILE
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
