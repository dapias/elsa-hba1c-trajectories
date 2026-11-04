"""Figure 1: personal baseline and the four features at the baseline visit.

Hypothetical participant with simulated values (no ELSA person-level data). Model parameters are the
ELSA estimates reported in the paper (trajectory model, mean over folds). Writes
figures/fig_personal_baseline.{png,pdf} next to this script.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):   # optional output folder: python fig_personal_baseline.py ../outputs
    OUT = Path(sys.argv[1])
else:
    try:
        OUT = Path(__file__).resolve().parent / "figures"
    except NameError:                               # running inside Jupyter
        OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "pdf.fonttype": 42})

# ---------------- ELSA trajectory model (mean over folds; men, round-corrected) ----------------
B0, B1, B2 = 35.83, 0.94, -0.125          # population curve: b0 + b1 a + b2 a^2, a = (age - 65)/10
SD_U0, SD_U1, CORR = 3.17, 0.84, -0.03    # personal intercept (mmol/mol) and slope (mmol/mol per decade)
SIGMA = 1.62                              # within-person SD (independent residuals)
G = np.array([[SD_U0**2, CORR * SD_U0 * SD_U1], [CORR * SD_U0 * SD_U1, SD_U1**2]])


def a_of(age):
    return (np.asarray(age, float) - 65) / 10


def mu(age):
    a = a_of(age)
    return B0 + B1 * a + B2 * a**2


def posterior(ages, y):
    """Empirical-Bayes posterior of (u0, u1) from earlier values (fixed effects known)."""
    Z = np.column_stack([np.ones(len(ages)), a_of(ages)])
    S = np.linalg.inv(np.linalg.inv(G) + Z.T @ Z / SIGMA**2)
    m = S @ Z.T @ (np.asarray(y) - mu(ages)) / SIGMA**2
    return m, S


def predict(age, m, S, noise=True):
    z = np.column_stack([np.ones(np.size(age)), a_of(age)])
    mean = mu(age) + z @ m
    var = np.einsum("ij,jk,ik->i", z, S, z) + (SIGMA**2 if noise else 0)
    return mean, np.sqrt(var)


# ================= Figure 1 =================
# One hypothetical participant measured every four years (true slope +3.0 mmol/mol per decade).
# The figure shows the four features at the baseline visit (age 65), computed from the three earlier values.
from scipy.stats import norm as _norm
P_AGES = np.array([53.0, 57.0, 61.0, 65.0])
TRUE_U = np.array([3.0, 2.06])                      # true personal deviation: level at 65, slope (true slope +3.0/decade)
TRUE_CURVE = lambda age: mu(age) + TRUE_U[0] + TRUE_U[1] * a_of(age)
P_VALS = np.round(TRUE_CURVE(P_AGES) + np.array([0.0, 0.5, 0.0, 1.0]), 1)   # simulated; the last value is the baseline visit
NEXT_AGE = 69.0                                     # next health visit, four years later
FS = 8.5

fig, ax = plt.subplots(figsize=(8.2, 4.9))
m3, S3 = posterior(P_AGES[:3], P_VALS[:3])          # personal baseline from the earlier values
grid = np.linspace(50, 72, 300)
g3 = grid[grid <= 65]
pc, sc = predict(g3, m3, S3, noise=False)
ax.fill_between(g3, pc - 1.96 * sc, pc + 1.96 * sc, color=BLUE, alpha=0.22, lw=0,
                label="95% credible band of the personal trajectory")
ax.plot(g3, pc, color=BLUE, lw=2.0, label="personal trajectory (from the earlier values)")
ax.plot(grid, mu(grid), color=GREY, lw=1.0, ls="--", label="population mean")
ax.axhline(48, color="#c0392b", lw=0.9, ls=(0, (4, 3)))
ax.text(50.3, 48.25, "diagnostic threshold, 48 mmol/mol", color="#c0392b", fontsize=7.5, va="bottom")
ax.scatter(P_AGES[:3], P_VALS[:3], color=INK, s=30, zorder=4, label="earlier values")
ax.scatter([65], [P_VALS[3]], marker="D", s=46, color=ORANGE, zorder=6, label="value at the baseline visit")

#ax.plot(grid, TRUE_CURVE(grid), color=INK, lw=0.9, ls=(0, (1, 2)), label="true trajectory ")

g_true = grid[grid <= 65]
ax.plot(g_true, TRUE_CURVE(g_true), color=AQUA, lw=1.4, ls=(0, (1, 1.5)), label="true personal trajectory (unknown in practice)")


def badge(x, y, k):
    ax.text(x, y, str(k), ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.25", facecolor=INK, edgecolor="none"), zorder=8)


# 1 deviation z: baseline value against the personal prediction and its 95% range for a new value
p0, s0 = predict(65.0, m3, S3)
z = (P_VALS[3] - p0[0]) / s0[0]
XA = 64.55                                          # deviation arrow, just left of the bar
ax.errorbar([65.0], p0, yerr=1.96 * s0, color=INK2, capsize=3, lw=1.1, zorder=5,
            label="95% range for a new value")
ax.plot([65.0], p0, marker="o", color=BLUE, mec="white", ms=6, zorder=6)
ax.plot([XA - 0.1, 65.0], [p0[0], p0[0]], color=INK, lw=0.6, ls=":")
ax.plot([XA - 0.1, 65.0], [P_VALS[3], P_VALS[3]], color=INK, lw=0.6, ls=":")
ax.annotate("", xy=(XA, P_VALS[3]), xytext=(XA, p0[0]),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkA=0, shrinkB=0))
#YZ = p0[0] + 1.96 * s0[0]                           # label level with the top of the bar
#ax.text(64.6, YZ, f"deviation z = {z:+.1f}", ha="right", va="center", fontsize=FS)
#badge(60.75, YZ, 1)
YZ = p0[0] + 1.96 * s0[0]                           # label level with the top of the bar
badge(61.0, YZ-0.4, 1)
ax.text(61.35, YZ-0.4, f"deviation z = {z:+.1f}", ha="left", va="center", fontsize=FS)

# 2 personal slope
slope = B1 + m3[1]
badge(50.9, 32.9, 2)
ax.text(51.35, 32.9, f"personal slope = {slope:+.1f} per decade (population: {B1:+.1f})",
        va="center", fontsize=FS)

# 3 change since last value
XC = 65.8
YC = 40.6                                           # label height: below the arrow, between forecast and population mean
ax.plot([61.2, XC + 0.15], [P_VALS[2], P_VALS[2]], color=INK2, lw=0.8, ls=":")
ax.plot([65.25, XC + 0.15], [P_VALS[3], P_VALS[3]], color=INK2, lw=0.8, ls=":")
ax.annotate("", xy=(XC, P_VALS[3]), xytext=(XC, P_VALS[2]),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1, shrinkA=0, shrinkB=0))
badge(65.95, YC, 3)
ax.text(66.3, YC, f"change = {P_VALS[3] - P_VALS[2]:+.1f}", va="center", fontsize=FS)

# 4 forecast: posterior updated with the baseline value, projected to the next visit
m4, S4 = posterior(P_AGES, P_VALS)
gf = np.linspace(65, NEXT_AGE, 60)
ax.plot(gf, predict(gf, m4, S4, noise=False)[0], color=BLUE, lw=1.4, ls="--", label="forecast to the next visit")
pn, sn = predict(NEXT_AGE, m4, S4)
pcross = 1 - _norm.cdf((48 - pn[0]) / sn[0])
yy = np.linspace(pn[0] - 3.5 * sn[0], 48, 400)
dens = _norm.pdf(yy, pn[0], sn[0]); dens = dens / dens.max() * 1.6
ax.fill_betweenx(yy, NEXT_AGE, NEXT_AGE + dens, color=BLUE, alpha=0.18, lw=0)
ax.plot(NEXT_AGE + dens, yy, color=BLUE, lw=1.1)
ax.plot([NEXT_AGE, NEXT_AGE], [32, 48], color=INK2, lw=0.6)
ptxt = "< 0.1%" if pcross < 0.001 else f"= {100 * pcross:.1f}%"
ax.text(68.8, 45.6, f"forecast: P(next value ≥ 48) {ptxt}", ha="right", va="center", fontsize=FS)
badge(62.15, 45.6, 4)
ax.text(NEXT_AGE - 0.15, 32.3, "next visit", fontsize=7.5, color=INK2, ha="right", va="bottom")

ax.set(xlim=(50, 71.5), ylim=(32, 49.5), xticks=[50, 55, 60, 65, 69], yticks=np.arange(33, 49, 3),
       xlabel="age (years)", ylabel="HbA1c (mmol/mol)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.27)
fig.savefig(OUT / "fig_personal_baseline.png", dpi=300)
fig.savefig(OUT / "fig_personal_baseline.pdf")
plt.close(fig)
print(f"written: {OUT / 'fig_personal_baseline.pdf'}")
print("values:", P_VALS, "features:", dict(z=round(z, 2), slope=round(slope, 2), change=round(P_VALS[3] - P_VALS[2], 1),
      pnext=round(float(pn[0]), 1), pcross=round(float(pcross), 4)))
