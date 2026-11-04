"""Personal HbA1c trajectories and incident diabetes in ELSA: analysis package.

The modules follow the sections of the paper (see README.md). They contain functions only; the scripts in
scripts/ set the paths in elsa_hba1c.config and run the analyses.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
