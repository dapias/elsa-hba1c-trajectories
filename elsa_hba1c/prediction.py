"""Section 4. Prediction: out-of-fold logistic models A-E, AUC, calibration, screening counts;
4b. final models A and B refitted on all baseline visits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import config
from .config import COMPARISONS, HBA1C_PREDIAB, HOMOGENEOUS_ROUNDS, MODELS, RANDOM_SEED
from .output import log, save
from .features import derive


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
    (config.OUT_DIR / "S8_final_models_rows.tex").write_text(s8_latex(t))
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
    boot = cluster_boot(df, [f"p_{m}" for m in MODELS], rng, config.N_BOOT)
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
    df.to_pickle(config.WORK_DIR / "predictions.pkl")
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
        for _ in range(config.N_BOOT):
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
