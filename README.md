# Observability of personal HbA1c trajectories in ELSA

Analysis code for

> Tapias D, Donati C, Völter-Mahlknecht S. *Observability of personal HbA1c trajectories: longitudinal dynamics and
> incident diabetes in the English Longitudinal Study of Ageing.* Preprint, 2026.

The code estimates personal HbA1c trajectories from repeated health-visit measurements in the English Longitudinal
Study of Ageing (ELSA), evaluates their association with incident diabetes and their incremental predictive value
(including a self-supervised time-aware Transformer), models within-person variation with an Ornstein–Uhlenbeck
drift process, and quantifies how reliably personal slopes can be recovered under different measurement schedules.

## Data

The ELSA data are **not** included and must never be committed to this repository. They are available from the
UK Data Service under an End User Licence (study SN 5050). Place the following `.tab` files in one folder
(passed as `--data`):

| File | Content |
|---|---|
| `gh_elsa_h.tab` | Gateway to Global Aging Data, harmonised ELSA (interview waves 1–10) |
| `wave_11_elsa_data_eul.tab` | wave-11 core interview |
| `wave_11_elsa_eol_eul.tab` | wave-11 end-of-life interviews |
| `wave_2_nurse_data_v2.tab`, `wave_4_nurse_data.tab`, `wave_6_elsa_nurse_data_v2.tab`, `elsa_nurse_w8w9_data_eul.tab`, `wave_11_elsa_nurse_data_eul.tab` | health-visit (nurse) data, rounds 2, 4, 6, 8/9 and 11 |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt          # PyTorch is needed only for the Transformer (scripts/05_transformer.py)
```

The scripts find the package without installation; `pip install -e .` also works.

## Running the analyses

All scripts take the same options: `--data` (UKDS files), `--work` (person-level work files and cached
likelihood fits; local only, never share) and `--out` (aggregate tables and figures).

```
python scripts/run_all.py --data ../tab --work ../work_local --out ../outputs/paper      # everything except the Transformer
python scripts/05_transformer.py --data ../tab --work ../work_local --out ../outputs/paper
```

`run_all.py` is the reference run (about 1–2 hours on a laptop the first time; the likelihood fits are cached in
`--work` and reused). The same analyses can be run section by section:

| Script | Paper | Needs |
|---|---|---|
| `01_prepare_data.py` | Sections 2.1–2.3; Table 2 | ELSA files |
| `02_prediction.py` | Section 2.4; Table 3, Tables S3–S4 | 01 |
| `03_association.py` | Sections 2.5–2.6; Table 1, Figure 3, Tables S1–S2, S7–S9, Figure 4A | 01 |
| `04_drift_observability.py` | Sections 2.8–2.9; Table 5, Figures 4B, 5, 6, S1–S4, Tables S10–S12 | 01, 03 |
| `05_transformer.py` | Section 2.7; Table 3 (row T), Tables S5–S6 | 01 |
| `06_figures.py` | redraws Figures 4–6, S3, S4 from saved outputs (seconds) | a full run |

Figures 1 and 2 are illustrations with simulated values (no ELSA data): `illustrations/`.

## Outputs and where they appear in the paper

All tables written to `--out` are aggregate; counts below 10 are shown as `<10`, and no table may contain the
person identifier (`elsa_hba1c/output.py`). File names keep the prefixes of the analysis versions; the mapping
to the published numbering is:

| Output files | Paper |
|---|---|
| `T1_flow` | Figure 3 |
| `T1_characteristics_by_outcome` | Table 1 |
| `T1_characteristics_by_band`, `T1_history_structure` | Tables S1, S2 |
| `T2_variance_components_by_fold`, `T2_individuality` | Table 2 |
| `AUC_model_performance`, `AUC_delta`, `AUC_screening_counts`, `AUC_calibration_deciles` | Table 3, Table S3, Section 3.2 |
| `S8_final_models_rows.tex` | Table S4 (`S8_final_model_parameters.csv`: full precision, local use only) |
| `07_strats_performance`, `07_strats_delta_auc`, `07_strats_pretraining` (Transformer) | Table 3 row T, Table S6 |
| `T3_S1_adjusted_associations`, `FIG_fig1_odds_ratios` | Table S7, Figure 4 |
| `S1_slope_crude`, `S1_slope_covariate_steps`, `S1_slope_common_scale`, `S1_age_cohort_participation`, `S1_slope_hba1c_interaction`, `S1_slope_history_interaction` and the other `S1_slope_*` diagnostics | Table S8, Supplementary Section S3 |
| `S1_ipw_and_extended_set_diagnostics`, `S1_deaths_scenarios`, `S1_death_vs_slope`, `S1_hba1c_only_outcome`, `S1_multiple_imputation`, `S1_missingness` | Table S9, Section 3.3 |
| `S1_slope_vs_drift_level`, `S1_slope_including_baseline_value` | Table S10, Figure 4B |
| `S1_selftest_simulation` | Section 3.3, Supplementary Section S4 |
| `T4_drift_models`, `T4_tau_intervals`, `T4_crossing_values`, `FIG_fig2_tau_profile` | Table 5, Figure 5 |
| `S2_observability_grid`, `S2_observability_table`, `FIG_fig_observability` | Figure 6, Table S12 |
| `S5_cap_sweep`, `FIG_fig_cap_sweep` | Table S11, Figure S4 |
| `S_residual_quantiles`, `S_residual_moments`, `FIG_fig_lmm_diagnostics` | Section 3.1, Figure S1 |
| `S_residual_correlation`, `FIG_fig_residual_correlation` | Figure S2 |
| `S_variogram`, `FIG_fig_variogram` | Figure S3 |
| `D_units` | Section 2.1 (HbA1c units by round) |

## Repository layout

```
elsa_hba1c/        functions only, one module per section of the paper
  config.py          settings, constants, model definitions
  output.py          writing aggregate tables (disclosure checks)
  data.py            1  health visits and interviews
  cohort.py          2  baseline visits and outcome
  features.py        3  cross-fitted mixed model and the four longitudinal features
  prediction.py      4  prediction models A-E, final models
  association.py     5  association and sensitivity analyses
  descriptives.py       Table 1, flow, histories
  drift.py           6  drift model, censored fit, cap sweep
  current_level.py   6b slope adjusted for the drift-model level; simulation
  observability.py   7  observability of personal slopes
  checks.py          8  semivariogram, diagnostics, residual correlation
  figures.py            Figures 4 and 5, redraw-only entry
  transformer.py        time-aware Transformer (PyTorch)
  cli.py                shared command-line options
scripts/           entry points (run_all.py and one script per part of the paper)
tools/             checks used during the analysis (flow counts, work-file check, target counts, refactoring check)
illustrations/     Figures 1 and 2 (simulated values)
```

## Code history

The analyses were developed as a single script (first commit, `elsa_hba1c_paper.py` and
`elsa_hba1c_transformer.py`), which produced the published results. It was then split into the package without
changing any function: `tools/verify_split.py` compares the abstract syntax tree of every function and setting in
the package with the original script (`git show <first commit>:elsa_hba1c_paper.py > original.py`), and a full run
of the package was compared with the output tables of the original run.
<!-- TODO: confirm the comparison before the release -->

## Citation

See `CITATION.cff`. If you use the code, please cite the paper.

## Licence

MIT (see `LICENSE`). The licence covers the code only; the ELSA data remain subject to the UK Data Service End User Licence.
