"""Figure: independent noise vs Ornstein-Uhlenbeck drift (hypothetical participant with simulated values).

Parameters are the ELSA estimates reported in the paper (within-person SD, drift share, time constants).
Writes figures/fig_drift.{png,pdf} next to this script.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    OUT = Path(__file__).resolve().parent / "figures"
except NameError:                                   # running inside Jupyter
    OUT = Path("figures")
OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "pdf.fonttype": 42})

SIGMA = 1.62                              # within-person SD of HbA1c (mmol/mol), ELSA trajectory model


# ================= Figure: drift (Ornstein-Uhlenbeck) =================
years = np.linspace(0, 24, 961)
dt = years[1] - years[0]
line = 38 + 0.12 * years                   # personal line: +1.2 mmol/mol per decade
sw, se = SIGMA * np.sqrt(0.8), SIGMA * np.sqrt(0.2)   # drift share 0.8 (drift model: 0.75-0.81)
visits = np.arange(0, 25, 4)
idx = np.searchsorted(years, visits)


def ou(tau, seed):
    r = np.random.default_rng(seed)
    w = np.zeros(len(years)); w[0] = r.normal(0, sw)
    a = np.exp(-dt / tau)
    for k in range(1, len(years)):
        w[k] = a * w[k - 1] + r.normal(0, sw * np.sqrt(1 - a * a))
    return w


rng = np.random.default_rng(3)
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [1.1, 1.1, 0.85]})

a = ax[0]                                   # A: independent noise around the line
meas = line[idx] + rng.normal(0, SIGMA, len(visits))
a.plot(years, line, color=BLUE, lw=2.2, label="personal trajectory")
for v, yv, l in zip(visits, meas, line[idx]):
    a.plot([v, v], [l, yv], color=ORANGE, lw=0.8, alpha=0.6)
a.scatter(visits, meas, color=ORANGE, s=34, zorder=3, label="measurement every 4 years")
a.set(title="A  Independent noise", xlabel="years", ylabel="HbA1c (mmol/mol)", ylim=(32, 46))
a.legend(loc="upper left", fontsize=8)
#a.text(0.98, 0.04, "each value scatters at random;\nthe next one knows nothing of the last",
#       transform=a.transAxes, ha="right", fontsize=8, color=INK2)

b = ax[1]                                   # B: slow drift around the line
w = ou(6.0, 11)
true = line + w
meas_d = true[idx] + np.random.default_rng(5).normal(0, se, len(visits))
b.plot(years, line, color=BLUE, lw=1.4, ls="--", label="personal trajectory")
b.plot(years, true, color=BLUE, lw=2.0, label="underlying level (trajectory + drift)")
b.fill_between(years, line, true, color=BLUE, alpha=0.10, lw=0)
b.scatter(visits, meas_d, color=ORANGE, s=34, zorder=3, label="measurement (+ small error)")
b.set(title="B  Slow drift (τ = 6 years)", xlabel="years", ylim=(32, 46))
b.legend(loc="upper left", fontsize=8)
#b.text(0.98, 0.04, "spells above or below the line that\nlast for years and then revert",
       #transform=b.transAxes, ha="right", fontsize=8, color=INK2)

c = ax[2]                                   # C: how much drift two visits share
u = np.linspace(0, 25, 300)
c.plot(u, np.exp(-u / 6), color=BLUE, lw=2.2, label="τ = 6 y")
c.plot(u, np.exp(-u / 18), color=AQUA, lw=2.2, label="τ = 18 y")
c.plot([0, 0, 25], [1, 0, 0], color=ORANGE, lw=1.8, ls=":", label="independent noise")
for v in [4, 8, 12]:
    c.axvline(v, color=INK2, lw=0.7, ls="--")
c.text(12.4, 0.62, "ELSA spacing:\n4, 8, 12 years", fontsize=8, color=INK2, va="top")
c.set(title="C  Drift shared by two visits", xlabel="time between the two visits, $\Delta$ (years)",
      ylabel="correlation of the drift, $e^{-\Delta/\\tau}$", ylim=(-0.03, 1.05), xlim=(0, 25))
c.legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig_drift.png", dpi=300)
fig.savefig(OUT / "fig_drift.pdf")
plt.close(fig)
