"""Figures 4 and 5 and the redraw-only entry point."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import pandas as pd
from . import config
from .config import EXPOSURES
from .output import log, save
from .drift import NAME_C, interp_ci, load_json, plot_cap_sweep, profile_frame
from .current_level import LEVELS
from .observability import observability
from .checks import plot_variogram


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
        fig.savefig(config.OUT_DIR / "FIG_fig1_odds_ratios.pdf"); fig.savefig(config.OUT_DIR / "FIG_fig1_odds_ratios.png", dpi=300)
        plt.close(fig)


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
    fig.tight_layout(); fig.savefig(config.OUT_DIR / "FIG_fig2_tau_profile.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    save(pd.DataFrame(rows), "T4_tau_intervals")


# Main
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
    rd = lambda name: pd.read_csv(config.OUT_DIR / f"{name}.csv")
    figure1(rd("T3_S1_adjusted_associations"), rd("S1_slope_vs_drift_level"))
    figure2_tau_profile(FITS, extra, CENS, prof_c)
    observability(rd("T2_variance_components_by_fold"), FITS, f_c)
    plot_cap_sweep(rd("S5_cap_sweep"))
    plot_variogram(rd("S_variogram"), FITS)
    log(f"figures redrawn in {config.OUT_DIR}")
