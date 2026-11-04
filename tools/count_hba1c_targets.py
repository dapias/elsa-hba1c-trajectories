"""Count the Transformer's pretraining validation forecast targets per fold (all variables and HbA1c), without
training: validation participants and cut times come from fixed seeds (footnote of Table S6). Requires PyTorch.

    python tools/count_hba1c_targets.py --data ../tab --work ../work_local
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

import argparse

import numpy as np
import pandas as pd

from elsa_hba1c import config
from elsa_hba1c import transformer as T

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../tab")
ap.add_argument("--work", default="../work_local")
a = ap.parse_args()
config.DATA_DIR, config.WORK_DIR = Path(a.data), Path(a.work)

feats = pd.read_pickle(config.WORK_DIR / "features.pkl")
tri, static = T.build_triplets()
store = T.Store(tri, static)
i_hb = store.var_id["hba1c"] - 1
fold_of = dict(zip(feats["idauniq"], feats["fold"]))
all_p = np.array(sorted(store.span.keys()))
extra = np.random.default_rng(config.RANDOM_SEED).integers(0, config.N_FOLDS, len(all_p))
fold_all = {p: fold_of.get(p, f) for p, f in zip(all_p, extra)}
rows = []
for k in range(config.N_FOLDS):
    tr_p = [p for p in all_p if fold_all[p] != k]
    store.normalise(tr_p)
    rng = np.random.default_rng(config.RANDOM_SEED + k)
    rng.shuffle(tr_p)
    val_p = list(set(tr_p[:int(T.HP["val_frac"] * len(tr_p))]))
    rv = np.random.default_rng(config.RANDOM_SEED)
    val = T.pretrain_samples(store, val_p, rv) + T.pretrain_samples(store, val_p, rv)
    m = np.stack([x[5] for x in val])
    rows.append(dict(fold=k, validation_participants=len(val_p), validation_samples=len(val),
                     targets_all_variables=int(m.sum()), hba1c_targets=int(m[:, i_hb].sum()),
                     hba1c_targets_with_earlier_hba1c=sum(1 for x in val if x[5][i_hb] and (x[1] == i_hb + 1).any())))
out = pd.DataFrame(rows)
out["hba1c_share_of_targets_%"] = (100 * out["hba1c_targets"] / out["targets_all_variables"]).round(1)
print(out.to_string(index=False))
