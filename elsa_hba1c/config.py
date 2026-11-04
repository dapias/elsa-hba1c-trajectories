"""Settings: paths (set by the entry scripts), constants, model and covariate definitions."""
from __future__ import annotations

from pathlib import Path


DATA_DIR = WORK_DIR = OUT_DIR = Path(".")         # set in main()
N_BOOT = 1000
REFIT = False
RANDOM_SEED = 20261008

HBA1C_DIABETES = 48.0          # mmol/mol
HBA1C_PREDIAB = 42.0           # mmol/mol
FGLU_DIABETES = 7.0            # mmol/l, fasting glucose
MIN_HISTORY = 1                # earlier non-diabetic HbA1c values required at a baseline visit
MIN_GAP_YEARS = 0.25           # "earlier" = at least 3 months before
AGE_CENTRE = 65                # age enters as (age - 65) / 10
ROUNDS = [2, 4, 6, 8, 11]      # health-visit rounds; 8 stands for 8/9
NEXT_ROUND = {2: 4, 4: 6, 6: 8, 8: 11}
HOMOGENEOUS_ROUNDS = [4, 6]    # baseline rounds with a ~4-year interval
N_FOLDS = 5
MIN_CELL = 10                  # disclosure control: counts below 10 are shown as "<10"

# drift model
TAU_GRID = [0.5, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12.5, 15, 17.5, 20, 25, 30, 50, 100]
EXTRA_TAUS_NONDIABETIC = [35, 40, 60, 70, 80, 90, 95]          # where the non-diabetic profile is flat
CENSORED_PROFILE_TAUS = [2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 35, 50]
EXTRA_TAUS_CENSORED = [13, 14, 17.5, 22.5, 30, 40, 45, 60]
CROSSING_CAP = 55.0            # only as starting values for the censored fit (and one point of the cap sweep)
CAPS = [48, 50, 52, 55, 58, 60, 65, 70, 80, None]               # cap sweep (supplement); None = uncapped
SWEEP_TAUS = [1, 2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20, 25, 35, 50, 100]
N_BOOT_VARIOGRAM = 200
LAG_BINS = [0, 3, 5, 7, 9.5, 12.5, 15, 25]

# sensitivity analyses
N_IMP = 20                     # multiple imputation
N_DRAW = 50                    # deaths: draws per scenario
K_RISK = [1, 2, 3]             # deaths: multiplier of the predicted risk

# observability
OBS_INTERVALS = [0.25, 0.5, 1, 2, 4]
OBS_LENGTHS = [4, 8, 12, 16, 20, 25, 30]

# ELSA files
HARMONIZED_FILE = "gh_elsa_h.tab"              # Gateway harmonised ELSA (waves 1-10)
W11_CORE_FILE = "wave_11_elsa_data_eul.tab"    # wave 11 is not in the harmonised file
W11_EOL_FILE = "wave_11_elsa_eol_eul.tab"      # end-of-life interviews after wave 10
NURSE_ROUNDS = {   # waves 8 and 9 are ONE health-visit round; the stand-alone wave-8 nurse file is not used
    2: dict(file="wave_2_nurse_data_v2.tab", fasting="fasteli", core_wave=2),
    4: dict(file="wave_4_nurse_data.tab", fasting="fastelig", core_wave=4),
    6: dict(file="wave_6_elsa_nurse_data_v2.tab", fasting="fastelig", core_wave=6),
    8: dict(file="elsa_nurse_w8w9_data_eul.tab", fasting="fastelig", core_wave="from_wave_column"),
    11: dict(file="wave_11_elsa_nurse_data_eul.tab", fasting="fastelig2", core_wave=11),
}
NURSE_VARS = ["hba1c", "fglu", "chol", "hdl", "ldl", "trig", "hscrp", "rtin", "hgb", "cfib", "wbc",
              "bmival", "wstval", "sysval", "diaval", "vismon", "visyear", "confage", "indager", "wave"]
PLAUSIBLE = {"hba1c": (15, 200), "fglu": (1.5, 40), "chol": (1, 20), "hdl": (0.1, 6), "ldl": (0.1, 15),
             "trig": (0.1, 40), "hscrp": (0.01, 300), "rtin": (1, 5000), "hgb": (4, 25), "cfib": (0.3, 12),
             "wbc": (0.5, 100), "bmival": (12, 80), "wstval": (40, 200), "sysval": (60, 260), "diaval": (30, 160)}
W = range(1, 11)               # interview waves in the harmonised file
HARM_STEMS = ["iwy", "iwm", "iwstat", "diabe", "rxdiab", "work", "jphysl", "jobsum", "nssec8", "smoken",
              "vgactx_e", "rxhibp", "mbmi", "mweight", "mheight"]

# models and covariates
HORIZON = ["log_years_next"]
BASE = ["hba1c", "age", "female"] + HORIZON
CLIN = BASE + ["bmi", "sysval", "hdl", "log_trig", "smoker", "vig_pa_weekly", "bp_med"]
MODELS = {"A_population": BASE, "B_clinical": CLIN, "C_naive_change": CLIN + ["delta_last"],
          "D_personal_baseline": CLIN + ["z_personal", "slope_personal"], "E_forecast": CLIN + ["logit_p_cross48"]}
COMPARISONS = [("D_personal_baseline", "B_clinical"), ("C_naive_change", "B_clinical"),
               ("D_personal_baseline", "C_naive_change"), ("E_forecast", "B_clinical"), ("B_clinical", "A_population")]
ADJ_NO_HORIZON = ["age", "female", "bmi", "sysval", "hdl", "log_trig", "smoker", "vig_pa_weekly", "bp_med"]
ADJ = ADJ_NO_HORIZON + ["log_years_next"]
EXPOSURES = {"z_personal": "deviation from personal baseline (z)",
             "delta_last": "change since last HbA1c, round-corrected",
             "logit_p_cross48": "forecast logit P(next HbA1c >= 48)",
             "slope_personal": "personal HbA1c slope (per decade)",
             "slope_incl_today": "personal slope incl. today's value (per decade)"}
SLOPE = "slope_personal"
