"""Time-aware Transformer (STraTS-style) as a contender to the personal-baseline models.

Run scripts/01_prepare_data.py (or run_all.py) first: this module reads the cross-fitted features
(features.pkl) and the health-visit file (nurse_long.pkl) saved in --work, and reuses the folds, models
B, D and E and the evaluation helpers of the main analysis. Requires PyTorch.

Reference architecture: Tipirneni & Reddy (2022), "Self-supervised Transformer for Sparse and
Irregularly Sampled Multivariate Clinical Time-Series", ACM TKDD.

Input per person = a set of TRIPLETS (time, variable, value) observed up to the index visit:
  * health-visit markers   (HbA1c, lipids, CRP, ferritin, Hb, fibrinogen, WBC, BMI, waist, BP)
  * interview-wave items   (self-rated health, smoking, physical activity, alcohol, BP medication,
                            hypertension / heart / stroke / high-cholesterol diagnosis, CES-D,
                            working, job physical demand, job stress, weight)
  + static (4): age at cut time, missing-birth-year flag, sex, and log(horizon): the interval to
    the next visit for the diabetes task (as given to models A-E in step 4; about 4 y from
    rounds 4/6, about 7 y from round 8/9), the fixed forecast horizon during pretraining.
    Education is NOT used (as in model B).
Lab markers are standardised WITHIN health-visit round (absorbs assay/calibration shifts);
everything else is standardised per variable. Normalisation statistics come from the training
persons of each fold only.

Training (per person-level fold, same folds as steps 3-5):
  1. self-supervised PRETRAINING on all training persons (not only the outcome cohort):
     random cut time T -> predict the values of every variable observed in [T, T + 2.5 y)
     from the triplets before T (masked MSE).  This is the STraTS forecasting task.
  2. FINE-TUNING for incident diabetes at the index visits of training persons.
  3. Ablation: the same network trained from scratch (no pretraining).
Evaluation: out-of-fold AUC vs models B (clinical), D (personal baseline), E (forecast), and
"B + Transformer score" (does the Transformer add information beyond clinical factors?).
Person-cluster bootstrap CIs.
Pretraining is summarised by validation forecast MSE over all variables and for HbA1c alone,
each against carrying the last value forward and against the population mean.

Outputs (aggregate, counts < 10 suppressed): --out/07_strats_performance{_tag}.csv, 07_strats_delta_auc{_tag}.csv,
                                             07_strats_pretraining{_tag}.csv
Local only: --work/strats_oof{_tag}.pkl

Checkpointing: each finished fold is saved to --work/strats_ckpt/ (LOCAL ONLY). If a run is
interrupted, start it again with the same options and finished folds are loaded instead of
recomputed. Checkpoints made with different settings or an older CODE_VERSION are ignored;
--fresh discards them.

Usage (same --data, --work and --out as for the other scripts):
        python scripts/05_transformer.py --quick --folds 1 --tag smoke   (smoke test, a few minutes)
        python scripts/05_transformer.py                                 (main run, Table 3)
        python scripts/05_transformer.py --d-model 32 --tag d32          (robustness runs, Table S6)
        python scripts/05_transformer.py --d-model 128 --tag d128
        python scripts/05_transformer.py --layers 3 --tag l3
        python scripts/05_transformer.py --horizon 4 --tag h4            (4-year pretraining window)
Device: --device auto (default) uses an NVIDIA GPU if PyTorch sees one, otherwise the CPU.
Threads: by default all CPU cores. When running several jobs at once on the CPU, give each
a share, e.g. --threads 8 on a 32-core machine for four parallel runs. Parallel runs are safe:
checkpoints are keyed by settings and outputs by --tag.
"""
import argparse
import hashlib
import json
import math
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from . import config                       # settings and paths of the main analysis
from .config import CLIN, MODELS
from .data import clean_negative, decimal_date, read_tab
from .features import derive as prep
from .output import save as safe_write
from .prediction import cluster_boot, fast_auc, oof_predict

warnings.filterwarnings("ignore")

CODE_VERSION = "2026-10-20"   # part of the checkpoint signature: bump after any code change

# ------------------------------------------------------------------ hyper-parameters
HP = dict(d_model=64, n_layers=2, n_heads=4, dropout=0.2, max_len=200,
          lr=5e-4, weight_decay=1e-4, batch=128,
          pre_epochs=15, ft_epochs=30, patience=5, forecast_horizon=2.5,
          n_seeds=2, val_frac=0.12)
NURSE_VARS = ["hba1c", "chol", "hdl", "ldl", "trig", "hscrp", "rtin", "hgb", "cfib", "wbc",
              "bmi", "wstval", "sysval", "diaval"]
LOG_VARS = {"trig", "hscrp", "rtin", "cfib", "wbc"}
IV_STEMS = ["shlt", "smoken", "smokef", "vgactx_e", "mdactx_e", "ltactx_e", "drinkwn_e", "rxhibp",
            "hibpe", "hearte", "stroke", "hchole", "cesd", "work", "jphysl", "jobsum", "mweight"]
N_STATIC = 4


# ================================================================== 1. triplet data
def build_triplets():
    nl = pd.read_pickle(config.WORK_DIR / "nurse_long.pkl")
    rows = []
    for v in NURSE_VARS:
        if v not in nl:
            continue
        d = nl[["idauniq", "t", "round", v]].dropna()
        x = np.log(d[v].clip(lower=1e-3)) if v in LOG_VARS else d[v]
        rows.append(pd.DataFrame({"idauniq": d["idauniq"], "t": d["t"], "var": v,
                                  "nkey": v + "_r" + d["round"].astype(int).astype(str), "raw": x}))
    # interview waves 1-10 from the harmonized file
    wanted = ["idauniq", "rabyear", "ragender"] + \
             [f"r{w}{s}" for w in range(1, 11) for s in IV_STEMS + ["iwy", "iwm", "iwstat"]]
    h = read_tab(config.DATA_DIR / config.HARMONIZED_FILE, wanted)
    for w in range(1, 11):
        if f"r{w}iwy" not in h:
            continue
        resp = h.get(f"r{w}iwstat", pd.Series(1, index=h.index)) == 1
        t = decimal_date(h[f"r{w}iwy"], h.get(f"r{w}iwm", pd.Series(np.nan, index=h.index)))
        for s in IV_STEMS:
            c = f"r{w}{s}"
            if c not in h:
                continue
            val = clean_negative(h[c])
            ok = resp & val.notna() & t.notna()
            rows.append(pd.DataFrame({"idauniq": h.loc[ok, "idauniq"], "t": t[ok], "var": s,
                                      "nkey": s, "raw": val[ok]}))
    tri = pd.concat(rows, ignore_index=True)
    tri = tri.sort_values(["idauniq", "t"]).reset_index(drop=True)
    static = pd.DataFrame({"idauniq": h["idauniq"], "rabyear": clean_negative(h["rabyear"]),
                           "female": (h["ragender"] == 2).astype(float).where(h["ragender"].isin([1, 2]))})
    return tri, static


class Store:
    """Flat arrays + per-person offsets for fast slicing."""

    def __init__(self, tri: pd.DataFrame, static: pd.DataFrame):
        self.vars = sorted(tri["var"].unique())
        self.var_id = {v: i + 1 for i, v in enumerate(self.vars)}          # 0 = padding
        self.nkeys = sorted(tri["nkey"].unique())
        nk_id = {k: i for i, k in enumerate(self.nkeys)}
        self.pid = tri["idauniq"].to_numpy()
        self.t = tri["t"].to_numpy(np.float32)
        self.v = tri["var"].map(self.var_id).to_numpy(np.int64)
        self.k = tri["nkey"].map(nk_id).to_numpy(np.int64)
        self.raw = tri["raw"].to_numpy(np.float32)
        self.val = np.zeros_like(self.raw)
        starts = np.flatnonzero(np.r_[True, self.pid[1:] != self.pid[:-1]])
        ends = np.r_[starts[1:], len(self.pid)]
        self.span = {p: (s, e) for p, s, e in zip(self.pid[starts], starts, ends)}
        st = static.set_index("idauniq")
        self.byear = st["rabyear"].to_dict()
        self.female = st["female"].to_dict()

    def normalise(self, train_ids):
        m = np.isin(self.pid, list(train_ids))
        df = pd.DataFrame({"k": self.k[m], "x": self.raw[m]})
        g = df.groupby("k")["x"].agg(["mean", "std"])
        mu = np.zeros(len(self.nkeys), np.float32); sd = np.ones(len(self.nkeys), np.float32)
        mu[g.index] = g["mean"].to_numpy(); sd[g.index] = g["std"].fillna(1).replace(0, 1).to_numpy()
        self.val = np.clip((self.raw - mu[self.k]) / sd[self.k], -6, 6).astype(np.float32)

    def static_vec(self, p, t_cut, horizon):
        """age (decades, centred at 65), missing-birth-year flag, female, ln(horizon / 4 y)."""
        by = self.byear.get(p, np.nan)
        age = (t_cut - by - 65) / 10 if by == by else 0.0
        f = self.female.get(p, np.nan)
        h = horizon if (horizon == horizon and horizon > 0) else 4.0
        return [age, 0.0 if by == by else 1.0,
                f if f == f else 0.5,
                float(np.log(h) - np.log(4.0))]

    def history(self, p, t_cut):
        s, e = self.span.get(p, (0, 0))
        tt = self.t[s:e]
        j = s + np.searchsorted(tt, t_cut, side="right") if e > s else s
        i = max(s, j - HP["max_len"])
        return (self.t[i:j] - t_cut) / 10.0, self.v[i:j], self.val[i:j]

    def future_targets(self, p, t_cut, horizon):
        s, e = self.span[p]
        tt = self.t[s:e]
        a = s + np.searchsorted(tt, t_cut, side="right")
        b = s + np.searchsorted(tt, t_cut + horizon, side="right")
        y = np.zeros(len(self.vars), np.float32); m = np.zeros(len(self.vars), np.float32)
        for vi, x in zip(self.v[a:b][::-1], self.val[a:b][::-1]):   # keep earliest obs per var
            y[vi - 1] = x; m[vi - 1] = 1
        return y, m


DEV = torch.device("cpu")


def collate(items):
    L = max(1, max(len(it[0]) for it in items))
    B = len(items)
    T = np.zeros((B, L), np.float32); V = np.zeros((B, L), np.int64); X = np.zeros((B, L), np.float32)
    M = np.zeros((B, L), bool)
    for b, it in enumerate(items):
        n = len(it[0])
        T[b, :n], V[b, :n], X[b, :n], M[b, :n] = it[0], it[1], it[2], True
    S = np.array([it[3] for it in items], np.float32)
    return [torch.from_numpy(a).to(DEV) for a in (T, V, X, M, S)]


# ================================================================== 2. model
class CVE(nn.Module):
    """Continuous value embedding (STraTS)."""

    def __init__(self, d):
        super().__init__()
        h = int(math.sqrt(d)) + 1
        self.net = nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, d, bias=False))

    def forward(self, x):
        return self.net(x.unsqueeze(-1))


class STraTS(nn.Module):
    def __init__(self, n_var, n_static):
        super().__init__()
        d = HP["d_model"]
        self.var_emb = nn.Embedding(n_var + 1, d, padding_idx=0)
        self.val_emb, self.time_emb = CVE(d), CVE(d)
        layer = nn.TransformerEncoderLayer(d, HP["n_heads"], 2 * d, HP["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, HP["n_layers"], enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(d, 2 * d), nn.Tanh(), nn.Linear(2 * d, 1))
        self.static = nn.Sequential(nn.Linear(n_static, d), nn.Tanh())
        self.forecast_head = nn.Linear(2 * d, n_var)
        self.cls_head = nn.Sequential(nn.Dropout(HP["dropout"]), nn.Linear(2 * d, 1))

    def represent(self, T, V, X, M, S):
        e = self.var_emb(V) + self.val_emb(X) + self.time_emb(T)
        h = self.encoder(e, src_key_padding_mask=~M)
        a = self.att(h).squeeze(-1).masked_fill(~M, -1e9).softmax(-1)
        z = (a.unsqueeze(-1) * h).sum(1)
        return torch.cat([z, self.static(S)], -1)

    def forward(self, T, V, X, M, S, task="cls"):
        r = self.represent(T, V, X, M, S)
        return self.forecast_head(r) if task == "forecast" else self.cls_head(r).squeeze(-1)


# ================================================================== 3. training loops
def pretrain_samples(store, ids, rng, min_hist=1):
    out = []
    for p in ids:
        if p not in store.span:
            continue
        s, e = store.span[p]
        cuts = np.unique(store.t[s:e])
        cuts = cuts[1:] if len(cuts) > 1 else cuts[:0]                 # need history before cut
        if not len(cuts):
            continue
        tc = float(rng.choice(cuts)) - 1e-4
        T, V, X = store.history(p, tc)
        if len(T) < min_hist:
            continue
        y, m = store.future_targets(p, tc, HP["forecast_horizon"])
        if m.sum() == 0:
            continue
        out.append((T, V, X, store.static_vec(p, tc, HP["forecast_horizon"]), y, m))
    return out


def run_pretrain(model, store, train_ids, val_ids, rng, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    # two different random cut times per validation person (one generator, drawn twice)
    rv = np.random.default_rng(config.RANDOM_SEED)
    val = pretrain_samples(store, val_ids, rv) + pretrain_samples(store, val_ids, rv)
    best, best_state, log = np.inf, None, []
    for ep in range(epochs):
        model.train()
        tr = pretrain_samples(store, train_ids, rng)
        rng.shuffle(tr)
        for i in range(0, len(tr), HP["batch"]):
            b = tr[i:i + HP["batch"]]
            T, V, X, M, S = collate(b)
            y = torch.from_numpy(np.stack([x[4] for x in b])).to(DEV)
            m = torch.from_numpy(np.stack([x[5] for x in b])).to(DEV)
            pred = model(T, V, X, M, S, task="forecast")
            loss = ((pred - y) ** 2 * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vl = eval_forecast(model, val)
        log.append(vl)
        if vl < best - 1e-4:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, val


@torch.no_grad()
def eval_forecast(model, samples, mode="model", var=None):
    """Masked MSE on standardised values.
    mode: 'model' (network forecast), 'last' (last observed value of the same variable,
    0 = population mean if none), 'mean' (population mean, i.e. 0).
    var: 0-based variable index to restrict to one variable (e.g. HbA1c); None = all variables."""
    if mode == "model":
        model.eval()
    se, n = 0.0, 0.0
    for i in range(0, len(samples), 512):
        b = samples[i:i + 512]
        y = np.stack([x[4] for x in b]); m = np.stack([x[5] for x in b])
        if var is not None:
            keep = np.zeros(m.shape[1], np.float32); keep[var] = 1
            m = m * keep
        if mode == "last":
            pred = np.zeros_like(y)
            for r, x in enumerate(b):
                for vi, val in zip(x[1], x[2]):
                    pred[r, vi - 1] = val
        elif mode == "mean":
            pred = np.zeros_like(y)
        else:
            T, V, X, M, S = collate(b)
            pred = model(T, V, X, M, S, task="forecast").cpu().numpy()
        se += float(((pred - y) ** 2 * m).sum()); n += float(m.sum())
    return se / max(n, 1)


def cls_samples(store, df):
    out = []
    for p, t, y, h in zip(df["idauniq"], df["t"], df["incident_dm"], df["years_to_next"]):
        tc = float(t) + 0.05                      # include the index visit itself
        T, V, X = store.history(p, tc)
        out.append((T, V, X, store.static_vec(p, tc, float(h)), float(y)))
    return out


@torch.no_grad()
def predict_cls(model, samples):
    model.eval()
    out = []
    for i in range(0, len(samples), 512):
        T, V, X, M, S = collate(samples[i:i + 512])
        out.append(torch.sigmoid(model(T, V, X, M, S)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def run_finetune(model, tr, va, rng, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"] / 2, weight_decay=HP["weight_decay"])
    ytr = np.array([x[4] for x in tr]); yva = np.array([x[4] for x in va])
    pw = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-3), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(tr))
        for i in range(0, len(idx), HP["batch"]):
            b = [tr[j] for j in idx[i:i + HP["batch"]]]
            T, V, X, M, S = collate(b)
            y = torch.tensor([x[4] for x in b], dtype=torch.float32, device=DEV)
            loss = lossf(model(T, V, X, M, S), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        auc = fast_auc(yva, predict_cls(model, va)) if yva.sum() > 0 else 0.5
        if auc > best + 1e-4:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)
    return best


# ================================================================== 4. main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../tab", help="folder with the UKDS .tab files")
    ap.add_argument("--work", default="../work_local", help="work folder of elsa_hba1c_paper.py (local only)")
    ap.add_argument("--out", default="../outputs/paper", help="aggregate tables")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu) | cpu | cuda | mps")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for PyTorch (default: all cores; split them when running jobs in parallel)")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--pre-epochs", type=int, default=None, help="override pretraining epochs")
    ap.add_argument("--ft-epochs", type=int, default=None, help="override fine-tuning epochs")
    ap.add_argument("--d-model", type=int, default=None, help="model dimension (divisible by 4 heads)")
    ap.add_argument("--layers", type=int, default=None, help="number of Transformer encoder layers")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. d32 (keeps runs apart)")
    ap.add_argument("--horizon", type=float, default=None,
                    help="pretraining forecast window in years (default 2.5)")
    ap.add_argument("--folds", type=int, default=config.N_FOLDS, help="run only the first N folds (testing)")
    ap.add_argument("--fresh", action="store_true", help="ignore and delete existing fold checkpoints")
    a = ap.parse_args()
    config.DATA_DIR, config.WORK_DIR, config.OUT_DIR = Path(a.data), Path(a.work), Path(a.out)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.quick:
        HP.update(n_seeds=1, pre_epochs=3, ft_epochs=6, patience=3)
    if a.seeds: HP["n_seeds"] = a.seeds
    if a.pre_epochs: HP["pre_epochs"] = a.pre_epochs
    if a.ft_epochs: HP["ft_epochs"] = a.ft_epochs
    if a.d_model: HP["d_model"] = a.d_model
    if a.layers: HP["n_layers"] = a.layers
    if a.horizon: HP["forecast_horizon"] = a.horizon
    if HP["d_model"] % HP["n_heads"]:
        raise SystemExit(f"--d-model must be divisible by n_heads={HP['n_heads']}")
    sfx = f"_{a.tag}" if a.tag else ""
    global DEV
    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEV = torch.device(dev)
    torch.set_num_threads(a.threads or (os.cpu_count() or 4))
    gpu = f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else ""
    print(f"  device {DEV}{gpu}, {torch.get_num_threads()} CPU threads | d_model {HP['d_model']}, "
          f"layers {HP['n_layers']}, seeds {HP['n_seeds']}, tag '{a.tag}'")
    t0 = time.time()

    feats = prep(pd.read_pickle(config.WORK_DIR / "features.pkl"))
    df = feats[feats["primary"] & feats["z_personal"].notna()].reset_index(drop=True)
    print(f"  evaluation set: {len(df):,} index visits, {int(df['incident_dm'].sum())} events"
          f"  (preprint: 5,828 / 234)")

    print("  building triplets ...")
    tri, static = build_triplets()
    store = Store(tri, static)
    print(f"  {len(tri):,} triplets, {len(store.vars)} variables, {tri['idauniq'].nunique():,} persons")
    i_hba1c = store.var_id["hba1c"] - 1

    # same person -> fold mapping as step 3 (persons outside the cohort get a random fold)
    fold_of = dict(zip(feats["idauniq"], feats["fold"]))
    all_p = np.array(sorted(store.span.keys()))
    rng0 = np.random.default_rng(config.RANDOM_SEED)
    extra = rng0.integers(0, config.N_FOLDS, len(all_p))
    fold_all = {p: fold_of.get(p, f) for p, f in zip(all_p, extra)}

    df["p_T_pre"] = np.nan; df["p_T_scratch"] = np.nan
    pre_rows = []
    # checkpoints are only reused if code version, settings and data are identical
    sig = hashlib.sha1(json.dumps(dict(code=CODE_VERSION, HP=HP, device=str(DEV), n_static=N_STATIC,
                                       n_cohort=len(df), n_triplets=len(tri), seed=config.RANDOM_SEED),
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]
    ckpt_dir = config.WORK_DIR / "strats_ckpt"
    if a.fresh and ckpt_dir.exists():
        for f in ckpt_dir.glob("fold_*.pkl"):
            f.unlink()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for k in range(a.folds):
        ck = ckpt_dir / f"fold_{k}_{sig}.pkl"
        if ck.exists():
            c = pd.read_pickle(ck)
            df.loc[c["index"], "p_T_pre"] = c["p_T_pre"]
            df.loc[c["index"], "p_T_scratch"] = c["p_T_scratch"]
            pre_rows.append(c["pre_row"])
            print(f"  fold {k}: loaded from checkpoint")
            continue
        t_fold = time.time()
        tr_p = [p for p in all_p if fold_all[p] != k]
        store.normalise(tr_p)
        rng = np.random.default_rng(config.RANDOM_SEED + k)
        rng.shuffle(tr_p)
        n_val = int(HP["val_frac"] * len(tr_p))
        val_p, fit_p = set(tr_p[:n_val]), tr_p[n_val:]

        d_tr = df[(df["fold"] != k) & ~df["idauniq"].isin(val_p)]
        d_va = df[(df["fold"] != k) & df["idauniq"].isin(val_p)]
        d_te = df[df["fold"] == k]
        tr, va, te = cls_samples(store, d_tr), cls_samples(store, d_va), cls_samples(store, d_te)

        preds_pre, preds_scr = [], []
        for seed in range(HP["n_seeds"]):
            s0 = config.RANDOM_SEED + 1000 * k + seed
            torch.manual_seed(s0)
            r = np.random.default_rng(s0)
            # --- pretrained
            m = STraTS(len(store.vars), N_STATIC).to(DEV)
            mse, val_samples = run_pretrain(m, store, fit_p, list(val_p), r, HP["pre_epochs"])
            if seed == 0:
                pre_rows.append(dict(
                    fold=k, n_pretrain_persons=len(fit_p), n_val_samples=len(val_samples),
                    forecast_mse_model=mse,
                    forecast_mse_last_value=eval_forecast(m, val_samples, mode="last"),
                    forecast_mse_mean=eval_forecast(m, val_samples, mode="mean"),
                    forecast_mse_hba1c_model=eval_forecast(m, val_samples, var=i_hba1c),
                    forecast_mse_hba1c_last_value=eval_forecast(m, val_samples, mode="last", var=i_hba1c),
                    forecast_mse_hba1c_mean=eval_forecast(m, val_samples, mode="mean", var=i_hba1c)))
            run_finetune(m, tr, va, r, HP["ft_epochs"])
            preds_pre.append(predict_cls(m, te))
            # --- from scratch
            torch.manual_seed(s0 + 500)
            m2 = STraTS(len(store.vars), N_STATIC).to(DEV)
            run_finetune(m2, tr, va, r, HP["ft_epochs"])
            preds_scr.append(predict_cls(m2, te))
        df.loc[d_te.index, "p_T_pre"] = np.mean(preds_pre, 0)
        df.loc[d_te.index, "p_T_scratch"] = np.mean(preds_scr, 0)
        pd.to_pickle(dict(index=d_te.index.to_numpy(), p_T_pre=np.mean(preds_pre, 0),
                          p_T_scratch=np.mean(preds_scr, 0), pre_row=pre_rows[-1]), ck)
        pr = pre_rows[-1]
        print(f"  fold {k}: AUC pretrained {fast_auc(d_te['incident_dm'], df.loc[d_te.index, 'p_T_pre']):.3f}"
              f" | scratch {fast_auc(d_te['incident_dm'], df.loc[d_te.index, 'p_T_scratch']):.3f}"
              f" | forecast MSE all {pr['forecast_mse_model']:.3f} vs last {pr['forecast_mse_last_value']:.3f}"
              f" | HbA1c {pr['forecast_mse_hba1c_model']:.3f} vs last {pr['forecast_mse_hba1c_last_value']:.3f}"
              f"  (fold {time.time() - t_fold:.0f}s, total {time.time() - t0:.0f}s)")

    # ------------------------------------------------------------ comparison
    for m_ in ["B_clinical", "D_personal_baseline", "E_forecast"]:
        df[f"p_{m_}"] = oof_predict(df, MODELS[m_])
    lt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    df["logit_T"] = lt(df["p_T_pre"])
    df["p_B_plus_T"] = oof_predict(df, CLIN + ["logit_T"])
    df = df.dropna(subset=["p_T_pre", "p_T_scratch", "p_B_clinical", "p_B_plus_T"]).reset_index(drop=True)
    df[["idauniq", "round", "p_T_pre", "p_T_scratch"]].to_pickle(config.WORK_DIR / f"strats_oof{sfx}.pkl")
    print(f"  AUC model B in this run: {fast_auc(df['incident_dm'], df['p_B_clinical']):.3f}  (preprint Table 3: 0.837)")

    cols = {"B_clinical": "p_B_clinical", "D_personal_baseline": "p_D_personal_baseline",
            "E_forecast": "p_E_forecast", "T_transformer_pretrained": "p_T_pre",
            "T_transformer_scratch": "p_T_scratch", "B_plus_transformer": "p_B_plus_T"}
    rng = np.random.default_rng(config.RANDOM_SEED)
    perf, comp = [], []
    for label, sub in [("all non-diabetic", df), ("HbA1c < 42", df[df["hba1c"] < config.HBA1C_PREDIAB])]:
        sub = sub.reset_index(drop=True)
        boot = cluster_boot(sub, list(cols.values()), rng, config.N_BOOT)
        y = sub["incident_dm"]
        for name, c in cols.items():
            perf.append(dict(population=label, model=name, n_index_visits=len(sub), n_events=int(y.sum()),
                             auc=fast_auc(y, sub[c]), auc_lo=boot[c].quantile(.025), auc_hi=boot[c].quantile(.975)))
        for a_, b_ in [("T_transformer_pretrained", "B_clinical"), ("T_transformer_pretrained", "E_forecast"),
                       ("T_transformer_pretrained", "T_transformer_scratch"), ("B_plus_transformer", "B_clinical")]:
            d = boot[cols[a_]] - boot[cols[b_]]
            comp.append(dict(population=label, comparison=f"{a_} - {b_}", n_index_visits=len(sub),
                             n_events=int(y.sum()), delta_auc=fast_auc(y, sub[cols[a_]]) - fast_auc(y, sub[cols[b_]]),
                             lo=d.quantile(.025), hi=d.quantile(.975)))
    safe_write(pd.DataFrame(perf), "07_strats_performance" + sfx, count_cols=["n_index_visits", "n_events"])
    safe_write(pd.DataFrame(comp), "07_strats_delta_auc" + sfx, count_cols=["n_index_visits", "n_events"])
    pre = pd.DataFrame(pre_rows).assign(n_variables=len(store.vars), n_static=N_STATIC,
                                        code_version=CODE_VERSION, tag=a.tag,
                                        **{f"hp_{k}": v for k, v in HP.items()})
    pre["mse_reduction_vs_last_all"] = 1 - pre["forecast_mse_model"] / pre["forecast_mse_last_value"]
    pre["mse_reduction_vs_last_hba1c"] = 1 - pre["forecast_mse_hba1c_model"] / pre["forecast_mse_hba1c_last_value"]
    safe_write(pre, "07_strats_pretraining" + sfx, count_cols=["n_pretrain_persons", "n_val_samples"])
    print(pd.DataFrame(perf).round(3).to_string(index=False))
    print(pd.DataFrame(comp).round(3).to_string(index=False))
    print(pre[["fold", "mse_reduction_vs_last_all", "mse_reduction_vs_last_hba1c"]].round(3).to_string(index=False))
    print(f"  total time {time.time() - t0:.0f}s")
