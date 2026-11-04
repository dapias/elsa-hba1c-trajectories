"""Section 6. Drift model (equation 5): REML fits, profile likelihood for tau, censored ML fit, cap sweep."""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from . import config
from .config import (
    AGE_CENTRE, CAPS, CENSORED_PROFILE_TAUS, CROSSING_CAP, EXTRA_TAUS_CENSORED, EXTRA_TAUS_NONDIABETIC,
    HBA1C_DIABETES, MIN_GAP_YEARS, NURSE_ROUNDS, SWEEP_TAUS, TAU_GRID,
)
from .output import log, n_str, save
from .data import detect_units, read_tab
from .descriptives import med_iqr


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
    p = config.WORK_DIR / name
    return json.load(open(p)) if p.exists() and not config.REFIT else {}


def dump_json(obj, name):
    json.dump(obj, open(config.WORK_DIR / name, "w"), indent=1)


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
        n = read_tab(config.DATA_DIR / spec["file"], ["idauniq", "hba1c"])
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
        fig.tight_layout(); fig.savefig(config.OUT_DIR / "FIG_fig_cap_sweep.png", dpi=300); plt.close(fig)
