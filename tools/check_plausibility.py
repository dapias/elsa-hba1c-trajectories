"""How many nurse-visit values does the plausibility range in config.PLAUSIBLE remove?

Repeats exactly the cleaning in elsa_hba1c.data.nurse_long (missing codes < 0 and >= 9990 removed, unit detection
per round) and counts, per analyte and round, how many converted values fall below or above the range.

Exact counts are printed to the screen (they stay on your machine). The CSV written to --out uses
elsa_hba1c.output.save, which suppresses counts < 10, so it can be shared.

    python tools/check_plausibility.py --data ../tab --work ../work_local --out ../outputs/check_plausibility
    python tools/check_plausibility.py ... --analytes hba1c fglu
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import pandas as pd

from elsa_hba1c import config
from elsa_hba1c.cli import setup
from elsa_hba1c.config import NURSE_ROUNDS, PLAUSIBLE
from elsa_hba1c.data import detect_units, read_tab
from elsa_hba1c.output import log, save


def count_out_of_range(analytes):
    rows = []
    for rnd, spec in NURSE_ROUNDS.items():
        n = read_tab(config.DATA_DIR / spec["file"], ["idauniq"] + [a for a in analytes])
        for a in analytes:
            if a not in n:
                continue
            raw = n[a].where(n[a] >= 0).where(lambda s: s < 9990)        # as in data.nurse_long
            conv, unit, _ = detect_units(raw, a)
            lo, hi = PLAUSIBLE[a]
            valid = conv.notna()
            rows.append(dict(analyte=a, round=rnd, unit_decision=unit, range=f"{lo}-{hi}",
                             n_nonmissing=int(valid.sum()),
                             n_below=int((conv < lo).sum()), n_above=int((conv > hi).sum()),
                             n_removed=int(((conv < lo) | (conv > hi)).sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--analytes", nargs="*", default=["hba1c"])
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    setup(__doc__)
    bad = [a for a in known.analytes if a not in PLAUSIBLE]
    if bad:
        raise SystemExit(f"not in config.PLAUSIBLE: {bad}")
    tab = count_out_of_range(known.analytes)
    tot = tab.groupby("analyte")[["n_nonmissing", "n_below", "n_above", "n_removed"]].sum().reset_index()
    print("\nPer round (exact counts, local only):\n" + tab.to_string(index=False))
    print("\nAll rounds:\n" + tot.to_string(index=False))
    save(tab, "plausibility_by_round", ["n_nonmissing", "n_below", "n_above", "n_removed"])
    save(tot, "plausibility_total", ["n_nonmissing", "n_below", "n_above", "n_removed"])
    log(f"done: tables in {config.OUT_DIR}")
