"""Section 7. Observability of personal slopes (equation 8)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import config
from .config import AGE_CENTRE, OBS_INTERVALS, OBS_LENGTHS
from .output import save


def corr_slope(p, interval, length, rho=0.0, end_age=65):
    """r = corr(estimated, underlying slope) for measurements every `interval` years, starting at age
    65 - `length`. The last measurement is at age 65 when `length` is a multiple of `interval`, otherwise
    earlier (e.g. four-yearly over 30 years: ages 35, 39, ..., 63, a 28-year span).
    Intercept-slope correlation rho (0 = reference scenario)."""
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
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "FIG_fig_observability.png", dpi=300); plt.close(fig)


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
