"""Section 6b. Slope association adjusted for the drift-model estimate of the current level,
and the simulation (self-test) of that adjustment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .config import ADJ, AGE_CENTRE, HBA1C_DIABETES, MIN_GAP_YEARS, SLOPE
from .output import save
from .association import adjusted, pre
from .drift import design6


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
