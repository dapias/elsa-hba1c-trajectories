"""Section 1. Data: one row per health visit (nl) and per interview (iv)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .config import HARMONIZED_FILE, HARM_STEMS, NURSE_ROUNDS, NURSE_VARS, PLAUSIBLE, W, W11_CORE_FILE
from .output import save


def header(path: Path) -> list[str]:
    with open(path, "r", encoding="latin-1") as fh:
        return fh.readline().rstrip("\r\n").split("\t")


def read_tab(path: Path, wanted: list[str] | None = None) -> pd.DataFrame:
    """Read a UKDS .tab file, keeping `wanted` columns (case-insensitive); all values numeric."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - check --data")
    cols = header(path)
    keep = cols if wanted is None else [c for c in cols if c.lower() in {w.lower() for w in wanted}]
    df = pd.read_csv(path, sep="\t", usecols=keep, dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
    return df


def clean_negative(s):
    return s.where(s >= 0)                       # UKDS / harmonised special-missing codes are negative


def binary(s, yes=(1,), no=(0,)):
    out = pd.Series(np.nan, index=s.index)
    out[s.isin(yes)] = 1.0
    out[s.isin(no)] = 0.0
    return out


def decimal_date(year, month):
    """Decimal year; each date at the middle of its month, unknown month = mid-year."""
    y, m = clean_negative(year), clean_negative(month)
    m = m.where((m >= 1) & (m <= 12))
    return y + (m.fillna(6.5) - 0.5) / 12.0


def load_harmonized():
    wanted = ["idauniq", "rabyear", "ragender", "raeducl"] + [f"r{w}{s}" for w in W for s in HARM_STEMS]
    return read_tab(config.DATA_DIR / HARMONIZED_FILE, wanted)


def interviews_long(h):
    """One row per person and interview wave: diabetes reported (diagnosis or medication)."""
    rows = []
    for w in W:
        g = lambda s: h.get(f"r{w}{s}", pd.Series(np.nan, index=h.index))
        diag, med = binary(g("diabe")), binary(g("rxdiab"))
        dm = diag.where(med != 1, 1.0)
        d = pd.DataFrame({"idauniq": h["idauniq"], "wave": w, "t": decimal_date(g("iwy"), g("iwm")), "dm_report": dm})
        rows.append(d[(g("iwstat") == 1).values])
    c = read_tab(config.DATA_DIR / W11_CORE_FILE, ["idauniq", "iintdaty", "iintdatm", "hediaddi", "hedids12", "hehavedi"])
    dm = pd.Series(np.nan, index=c.index)
    dm[(c["hedids12"] == 0) | (c["hediaddi"] == 2)] = 0.0
    dm[(c["hediaddi"] == 1) | (c["hedids12"] == 1) | c["hehavedi"].isin([1, 2])] = 1.0
    rows.append(pd.DataFrame({"idauniq": c["idauniq"], "wave": 11,
                              "t": decimal_date(c["iintdaty"], c["iintdatm"]), "dm_report": dm}))
    return pd.concat(rows, ignore_index=True).sort_values(["idauniq", "wave"])


def detect_units(x, analyte):
    """Unit of a whole round from its median: HbA1c % (DCCT) -> mmol/mol (IFCC); haemoglobin g/l -> g/dl."""
    v = x[x > 0]
    med = float(v.median()) if len(v) else np.nan
    if analyte == "hba1c" and med < 15:
        return (x - 2.15) * 10.929, "% (DCCT) -> mmol/mol", med
    if analyte == "hgb" and med > 30:
        return x / 10.0, "g/l -> g/dl", med
    return x, "as recorded", med


def nurse_long(h):
    """One row per person and health visit, all rounds, units harmonised, interview covariates attached."""
    out, unit_rows = [], []
    for rnd, spec in NURSE_ROUNDS.items():
        n = read_tab(config.DATA_DIR / spec["file"], ["idauniq", spec["fasting"], "sex", "indsex"] + NURSE_VARS)
        d = pd.DataFrame({"idauniq": n["idauniq"], "round": rnd})
        d["t"] = decimal_date(n["visyear"], n.get("vismon", pd.Series(np.nan, index=n.index)))
        if spec["core_wave"] == "from_wave_column":
            cw = clean_negative(n["wave"]) if "wave" in n else pd.Series(np.nan, index=n.index)
            cw = cw.where(cw.isin([8, 9])).fillna(pd.Series(np.where(d["t"] < 2018, 8, 9), index=n.index))
            d["core_wave"] = cw.astype(int)
        else:
            d["core_wave"] = spec["core_wave"]
        d["fasted"] = binary(n.get(spec["fasting"], pd.Series(np.nan, index=n.index)), yes=(1,), no=(2,))
        for a in PLAUSIBLE:
            if a not in n:
                continue
            raw = n[a].where(n[a] >= 0).where(lambda s: s < 9990)
            conv, unit, med = detect_units(raw, a)
            lo, hi = PLAUSIBLE[a]
            d[a] = conv.where((conv >= lo) & (conv <= hi))
            if a in ("hba1c", "hgb", "fglu"):
                unit_rows.append(dict(round=rnd, analyte=a, unit_decision=unit, raw_median=med,
                                      n_valid=int(d[a].notna().sum())))
        if "fglu" in d:
            d.loc[d["fasted"] != 1, "fglu"] = np.nan          # fasting glucose only if fasted
        for c in ("confage", "indager"):
            if c in n:
                d[f"_{c}"] = n[c].where(n[c] >= 0)
        sx = n.get("sex", n.get("indsex"))
        d["_nsex"] = sx.where(sx.isin([1, 2])) if sx is not None else np.nan
        out.append(d)
    nl = pd.concat(out, ignore_index=True)

    p = h[["idauniq", "rabyear", "ragender", "raeducl"]].copy()
    p["rabyear"] = clean_negative(p["rabyear"])
    nl = nl.merge(p, on="idauniq", how="left")
    nl["age"] = nl["t"] - nl["rabyear"] - 0.5
    fb = nl["_confage"] if "_confage" in nl else pd.Series(np.nan, index=nl.index)
    if "_indager" in nl:
        fb = fb.fillna(nl["_indager"])
    nl["age"] = nl["age"].fillna(fb)
    nl["female"] = np.where(nl["ragender"].isin([1, 2]), nl["ragender"] == 2,
                            np.where(nl["_nsex"].isin([1, 2]), nl["_nsex"] == 2, np.nan)).astype(float)
    nl["educ"] = clean_negative(nl["raeducl"])

    cov_rows = []
    for w in W:
        g = lambda s: h.get(f"r{w}{s}", pd.Series(np.nan, index=h.index))
        cov_rows.append(pd.DataFrame({
            "idauniq": h["idauniq"], "core_wave": w,
            "smoker": binary(g("smoken")),
            "vig_pa_weekly": clean_negative(g("vgactx_e")).map({2: 1, 3: 1, 4: 0, 5: 0}),
            "bp_med": binary(g("rxhibp")).where(g("rxhibp") != -24, 0.0),
            "working": binary(g("work")), "job_phys": clean_negative(g("jphysl")),
            "job_stress": clean_negative(g("jobsum")), "nssec8": clean_negative(g("nssec8")),
            "h_bmi": clean_negative(g("mbmi")), "h_weight": clean_negative(g("mweight")),
            "h_height": clean_negative(g("mheight"))}))
    cov = pd.concat(cov_rows, ignore_index=True).sort_values(["idauniq", "core_wave"])
    cov["h_height_last"] = cov.groupby("idauniq")["h_height"].ffill()
    nl = nl.merge(cov, on=["idauniq", "core_wave"], how="left")
    bmi_calc = nl["h_weight"] / nl["h_height_last"] ** 2
    nl["bmi"] = nl["bmival"] if "bmival" in nl else np.nan
    nl["bmi"] = nl["bmi"].fillna(nl["h_bmi"]).fillna(bmi_calc.where((bmi_calc > 12) & (bmi_calc < 80)))
    nl = nl.drop(columns=[c for c in nl.columns if c.startswith("_")] +
                 ["ragender", "raeducl", "h_bmi", "h_weight", "h_height", "h_height_last"])
    return nl, pd.DataFrame(unit_rows)


def one_visit_per_round(nl):
    """One health visit per person and round: valid HbA1c first, then the earlier visit (waves 8/9 only)."""
    return (nl.assign(_no=nl["hba1c"].isna()).sort_values(["idauniq", "round", "_no", "t"])
              .drop_duplicates(["idauniq", "round"]).drop(columns="_no")
              .sort_values(["round", "idauniq"]).reset_index(drop=True))


def step1():
    h = load_harmonized()
    iv = interviews_long(h)
    nl, units = nurse_long(h)
    nl = one_visit_per_round(nl)
    nl.to_pickle(config.WORK_DIR / "nurse_long.pkl"); iv.to_pickle(config.WORK_DIR / "interviews.pkl")
    save(units, "D_units")
    return h, nl, iv
