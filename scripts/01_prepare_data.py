"""Sections 1-3: read the ELSA files, build the baseline visits and outcome, and the cross-fitted personal features.

Writes nurse_long.pkl, interviews.pkl and features.pkl to --work (person-level; never share), and
D_units, T2_variance_components_by_fold and T2_individuality (Table 2) to --out. Run this first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

from elsa_hba1c.cli import setup
from elsa_hba1c.cohort import step2
from elsa_hba1c.data import step1
from elsa_hba1c.features import step3
from elsa_hba1c.output import log

if __name__ == "__main__":
    setup(__doc__)
    log("step 1: reading ELSA ..."); _, nl, iv = step1()
    nlm, lmm, cohort = step2(nl, iv)
    step3(cohort, lmm)
    log("done: work files written")
