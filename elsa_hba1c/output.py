"""Output helpers: aggregate tables only, counts below MIN_CELL suppressed."""
from __future__ import annotations

import time
import pandas as pd
from . import config
from .config import MIN_CELL


def n_str(n):
    n = int(n)
    return f"<{MIN_CELL}" if 0 < n < MIN_CELL else f"{n:,}"


def save(t: pd.DataFrame, name: str, count_cols=()) -> pd.DataFrame:
    """Write an aggregate table to OUT_DIR/<name>.csv; counts in count_cols below MIN_CELL become '<10'."""
    if "idauniq" in t.columns:
        raise ValueError(f"{name}: refusing to write the person identifier")
    t = t.copy()
    for c in count_cols:
        if c in t:
            t[c] = t[c].map(lambda v: n_str(v) if pd.notna(v) and not isinstance(v, str) else v)
    t.to_csv(config.OUT_DIR / f"{name}.csv", index=False)
    return t


def or_text(r):
    return f"{r['or_per_sd']:.2f} ({r['or_lo']:.2f}-{r['or_hi']:.2f})"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
