"""Section 3. Personal trajectories: cross-fitted linear mixed model and the four features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from . import config
from .config import HBA1C_DIABETES, MIN_GAP_YEARS, N_FOLDS, RANDOM_SEED, ROUNDS
from .output import log, save


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
    features.to_pickle(config.WORK_DIR / "features.pkl")
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
