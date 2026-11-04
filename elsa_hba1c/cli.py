"""Command-line options shared by the section scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config


def setup(description: str, extra=None) -> argparse.Namespace:
    """Parse --data, --work, --out, --quick and --refit, and set them in elsa_hba1c.config."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../tab", help="folder with the UKDS .tab files")
    ap.add_argument("--work", default="../work_local", help="person-level files and fit caches (local only)")
    ap.add_argument("--out", default="../outputs/paper", help="aggregate tables and figures")
    ap.add_argument("--quick", action="store_true", help="100 bootstrap repetitions instead of 1,000")
    ap.add_argument("--refit", action="store_true", help="ignore cached likelihood fits")
    if extra:
        extra(ap)
    a = ap.parse_args()
    config.DATA_DIR, config.WORK_DIR, config.OUT_DIR = Path(a.data), Path(a.work), Path(a.out)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.N_BOOT, config.REFIT = (100 if a.quick else 1000), a.refit
    return a
