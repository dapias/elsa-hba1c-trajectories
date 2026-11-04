"""Section 8. Model checks: semivariogram, mixed-model diagnostics, residual correlation across rounds."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from . import config
from .config import AGE_CENTRE, LAG_BINS, MIN_CELL, N_BOOT_VARIOGRAM, RANDOM_SEED
from .output import save
from .features import design, fit_population, posterior
from .drift import design6
from .current_level import gmat


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
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "FIG_fig_variogram.png", dpi=300); plt.close(fig)


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
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "FIG_fig_lmm_diagnostics.png", dpi=300); plt.close(fig)
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
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "FIG_fig_residual_correlation.png", dpi=300); plt.close(fig)
    return cm
