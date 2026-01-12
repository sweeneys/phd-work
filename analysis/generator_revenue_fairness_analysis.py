#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fairness + System-Value (STRICT separation: LMP vs AMM)
Updated: editor-safe defaults, hard validation, better diagnostics.

What this script does
---------------------
A) Annual fairness view (simple + robust):
   - Load gens_static and costs.
   - Load LMP per-generator summary (from LMP run’s per_generator_summary.csv).
   - Load AMM per-generator summary (from availabilityPayments per_generator_summary.csv).
   - Merge into a unified table and compute fairness metrics/plots.

B) System-value view (kept FULLY SEPARATE for LMP vs AMM):
   LMP path:
     - Loads LMP nodal prices, dispatch, demand, flows from LMP run directory.
     - Computes system price (load-weighted).
     - Tightness: file if provided, else a price-based proxy (LMP-only).
     - SAOI, NVF, LAA (price-premium fallback), congestion, CRI, value-duration.

   AMM path:
     - Loads ONLY AMM dispatch/demand/flows from AMM run directory (NO LMP use).
     - Tightness: file if provided; else a demand-based proxy.
     - SAOI only (NVF only if your AMM tightness file includes 'system_price').
     - LAA from AMM import/local split if available.
     - Congestion, CRI, value-duration from AMM data alone.

Outputs
-------
- Fairness (annual):         <out_dir>/
- LMP system-value:          <out_dir>/LMP_system_value/
- AMM system-value:          <out_dir>/AMM_system_value/
- Extended fairness:         <out_dir>/extended_fairness/
- Debug:                     <out_dir>/fairness_inputs_audit.txt, unified_per_generator_DEBUG.csv
"""

import os, glob, json, argparse, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# ============================================================
# Small utilities
# ============================================================

def LOG(msg: str):
    print(msg, file=sys.stderr)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _read_csv(path: str, **kw) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"[MISSING FILE] {path}")
    return pd.read_csv(path, **kw)

def _maybe_read_csv_or_empty_with_genid(path: str) -> pd.DataFrame:
    """Read CSV if given & exists; else return an empty frame with 'gen_id' column to keep merges safe."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["gen_id"])
    df = pd.read_csv(path)
    if "gen_id" not in df.columns:
        cand = None
        for c in df.columns:
            if str(c).strip().lower() in {"generator","id","name"}:
                cand = c; break
        if cand is None:
            return pd.DataFrame(columns=["gen_id"])
        df = df.rename(columns={cand: "gen_id"})
    return df

def _norm_gen_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def _col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _pick_contains(df: pd.DataFrame, needles: List[str]) -> Optional[str]:
    for c in df.columns:
        lc = c.lower()
        if all(n in lc for n in needles):
            return c
    return None

def _savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _even_step_minutes(ts: pd.DatetimeIndex) -> float:
    if len(ts) < 2:
        raise ValueError("Need >=2 timestamps.")
    d = np.diff(ts.view("i8"))
    if not np.all(d == d[0]):
        raise ValueError("Timestamps not evenly spaced.")
    return float(d[0]/1e9/60.0)

def _step_hours_from_ts(ts: pd.Series) -> float:
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(ts).unique()))
    return _even_step_minutes(idx)/60.0

def _assert_nonempty(df: pd.DataFrame, name: str, path_hint: str = ""):
    if df is None or df.empty:
        hint = f" at path: {path_hint}" if path_hint else ""
        raise RuntimeError(f"[EMPTY INPUT] {name} appears empty{hint}. "
                           f"Please verify the path and that the file has rows.")

# ============================================================
# Unified per-generator (fairness) loaders
# ============================================================

def load_gens_static(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    gid  = _col(df, ["gen_id","generator","id","name"])
    if gid is None:
        raise RuntimeError("[BAD gens_static] Could not find a gen_id/generator/id/name column.")
    tech = _col(df, ["tech","technology","fuel","fuel_type"])
    pmax = _col(df, ["pmax_cap_for_model","pmax","capacity_mw","nameplate_mw","Pmax"])
    out = pd.DataFrame({"gen_id": _norm_gen_id(df[gid])})
    out["tech"] = df[tech].astype(str) if tech else "unknown"
    out["Pmax_MW"] = _safe_num(df[pmax]) if pmax else np.nan
    if "node" in df.columns:
        out["node"] = df["node"].astype(str)
    return out.drop_duplicates("gen_id")

def load_costs(path: str) -> pd.DataFrame:
    df = _read_csv(path)
    gid = _col(df, ["gen_id","generator","id","name"])
    if gid is None:
        raise RuntimeError("[BAD costs CSV] Could not find a gen_id/generator/id/name column.")
    out = pd.DataFrame({"gen_id": _norm_gen_id(df[gid])})
    col_opex = _col(df, ["OpEx_nonfuel_total","OpEx_non_fuel_total","opex_nonfuel_total"])
    col_capy = _col(df, ["CapEx_per_year","capex_per_year","capex_annual"])
    col_payb = _col(df, ["payback_years","expected_payback_years","payback"])
    out["OpEx_nonfuel_total"] = _safe_num(df[col_opex]) if col_opex else np.nan
    out["CapEx_per_year"]     = _safe_num(df[col_capy]) if col_capy else np.nan
    out["payback_years_expected"] = _safe_num(df[col_payb]) if col_payb else np.nan
    return out

def load_lmp_pergen(path: str) -> pd.DataFrame:
    """
    Expected columns (from LMP audit script):
      gen_id, R_total_annual, Energy_revenue_gross_per_year, Charge_Cost_per_year, Reserve_revenue_per_year
    We tolerate partial data and compute lmp_total / lmp_energy_net when possible.
    """
    df = _maybe_read_csv_or_empty_with_genid(path)
    if df.empty:
        return df
    out = pd.DataFrame({"gen_id": _norm_gen_id(df["gen_id"])})
    if "R_total_annual" in df.columns: out["lmp_total"] = _safe_num(df["R_total_annual"])
    if "Energy_revenue_gross_per_year" in df.columns: out["lmp_energy_gross"] = _safe_num(df["Energy_revenue_gross_per_year"])
    if "Charge_Cost_per_year" in df.columns: out["lmp_charge_cost"] = _safe_num(df["Charge_Cost_per_year"])
    if "Reserve_revenue_per_year" in df.columns: out["lmp_reserve"] = _safe_num(df["Reserve_revenue_per_year"])
    # derive energy net if possible
    if "lmp_energy_gross" in out and "lmp_charge_cost" in out:
        out["lmp_energy_net"] = out["lmp_energy_gross"] - out["lmp_charge_cost"]
    elif "lmp_total" in out and "lmp_reserve" in out:
        out["lmp_energy_net"] = out["lmp_total"] - out["lmp_reserve"]
    if "lmp_total" not in out.columns:
        out["lmp_total"] = out.get("lmp_energy_net", 0.0) + out.get("lmp_reserve", 0.0)
    out["lmp_energy_net"] = out.get("lmp_energy_net", 0.0)
    out["lmp_reserve"]    = out.get("lmp_reserve", 0.0)
    return out

def load_amm_pergen(path: str) -> pd.DataFrame:
    """
    Reads your AMM availabilityPayments per_generator_summary.csv.

    Interprets:
      - Pot 1 (availability / cost-recovery minimum): R_BASE
        (plus aliases: Availability_Payments, BasePot, R_cost_recovery_min, etc.)
      - Pot 2 (LMP equalisation / top-up): R_DELTA
        (plus aliases: Equalisation_Payments, TopUpPot, R_lmp_equalisation, etc.)
      - Optional energy/reserve components if present.

    Fails loudly if gen_id missing or file empty.
    """
    df = _maybe_read_csv_or_empty_with_genid(path)
    if df.empty:
        return df  # handled & validated later

    # Auto-map common aliases (non-destructive if not present)
    RENAME = {
        # Old names → new canonical names
        "Availability_Payments": "R_BASE",
        "Equalisation_Payments": "R_DELTA",
        "BasePot": "R_BASE",
        "TopUpPot": "R_DELTA",
    }
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    if "gen_id" not in df.columns:
        raise RuntimeError("[BAD AMM pergen] gen_id column missing even after fallback detection.")
    df["gen_id"] = _norm_gen_id(df["gen_id"])

    out = pd.DataFrame({"gen_id": df["gen_id"]})

    # -------- Pot 1 (availability / cost recovery) --------
    p1 = (
        _col(df, ["R_BASE", "R_cost_recovery_min"]) or
        _pick_contains(df, ["availability", "payment"]) or
        _pick_contains(df, ["availability", "pot"]) or
        _col(df, ["amm_pot1_total", "pot1", "pot_1"])
    )
    if p1:
        out["amm_pot1_total"] = _safe_num(df[p1])

    # -------- Pot 2 (LMP equalisation / top-up) --------
    p2 = (
        _col(df, ["R_DELTA", "R_lmp_equalisation"]) or
        _pick_contains(df, ["equalisation"]) or
        _col(df, ["amm_pot2_total", "pot2", "pot_2"])
    )
    if p2:
        out["amm_pot2_total"] = _safe_num(df[p2])

    # Optional energy/reserve totals if present
    e_col = (_pick_contains(df, ["energy", "revenue"]) or
             _pick_contains(df, ["energy", "total"]))
    r_col = (_pick_contains(df, ["reserve", "revenue"]) or
             _pick_contains(df, ["reserve", "total"]))
    if e_col:
        out["amm_energy_total"] = _safe_num(df[e_col])
    if r_col:
        out["amm_reserve_total"] = _safe_num(df[r_col])

    # Fill missing with zeros so downstream math doesn't explode
    for c in ["amm_pot1_total", "amm_pot2_total", "amm_energy_total", "amm_reserve_total"]:
        if c not in out.columns:
            out[c] = 0.0

    return out


# ============================================================
# Build unified per-gen + fairness metrics (annual)
# ============================================================

def build_unified(gens_static, costs_csv, lmp_sum, amm_sum):
    df = gens_static.merge(costs_csv, on="gen_id", how="outer")
    df = df.merge(lmp_sum, on="gen_id", how="outer")
    df = df.merge(amm_sum, on="gen_id", how="outer")

    for col in ["OpEx_nonfuel_total","CapEx_per_year","payback_years_expected",
                "lmp_total","lmp_energy_net","lmp_reserve",
                "amm_pot1_total","amm_pot2_total","amm_energy_total","amm_reserve_total"]:
        if col not in df.columns:
            df[col] = np.nan if col in ("OpEx_nonfuel_total","CapEx_per_year","payback_years_expected") else 0.0

    df["modelled_costs"] = df["OpEx_nonfuel_total"].fillna(0.0) + df["CapEx_per_year"].fillna(0.0)
    df["Capacity_GW"] = _safe_num(df.get("Pmax_MW", np.nan)) / 1000.0
    eps = 1e-12

    # Totals per approach
    df["LMP_total"]  = df["lmp_energy_net"].fillna(0.0) + df["lmp_reserve"].fillna(0.0)
    df["AMM1_total"] = df["amm_pot1_total"].fillna(0.0) + df["amm_energy_total"].fillna(0.0) + df["amm_reserve_total"].fillna(0.0)
    df["AMM2_total"] = df["amm_pot2_total"].fillna(0.0) + df["amm_energy_total"].fillna(0.0) + df["amm_reserve_total"].fillna(0.0)

    # Nets vs OpEx_nonfuel
    df["net_LMP_abs"]  = df["LMP_total"]  - df["OpEx_nonfuel_total"].fillna(0.0)
    df["net_AMM1_abs"] = df["AMM1_total"] - df["OpEx_nonfuel_total"].fillna(0.0)
    df["net_AMM2_abs"] = df["AMM2_total"] - df["OpEx_nonfuel_total"].fillna(0.0)

    df["net_LMP_perGW"] = np.where(df["Capacity_GW"] > eps, df["net_LMP_abs"]  / df["Capacity_GW"], np.nan)
    df["net_A1_perGW"]  = np.where(df["Capacity_GW"] > eps, df["net_AMM1_abs"] / df["Capacity_GW"], np.nan)
    df["net_A2_perGW"]  = np.where(df["Capacity_GW"] > eps, df["net_AMM2_abs"] / df["Capacity_GW"], np.nan)

    df["CapEx_total"] = np.where(df["payback_years_expected"].notna(),
                                 df["CapEx_per_year"] * df["payback_years_expected"], np.nan)

    def _pb(net_abs_col):
        net = _safe_num(df[net_abs_col]).fillna(0.0)
        ok = df["CapEx_total"].notna() & (net > eps)
        return np.where(ok, df["CapEx_total"] / net, np.inf)

    df["PB_LMP"]  = _pb("net_LMP_abs")
    df["PB_AMM1"] = _pb("net_AMM1_abs")
    df["PB_AMM2"] = _pb("net_AMM2_abs")

    exp_pb = _safe_num(df["payback_years_expected"])
    df["PBdiff_LMP"]  = df["PB_LMP"]  - exp_pb
    df["PBdiff_AMM1"] = df["PB_AMM1"] - exp_pb
    df["PBdiff_AMM2"] = df["PB_AMM2"] - exp_pb

    def _ratio(total_col):
        return _safe_num(df[total_col]) / df["modelled_costs"].replace(0.0, np.nan)
    df["Adeq_LMP"] = _ratio("LMP_total")
    df["Adeq_A1"]  = _ratio("AMM1_total")
    df["Adeq_A2"]  = _ratio("AMM2_total")

    tb = df.get("tech", pd.Series(["unknown"]*len(df))).astype(str).str.lower().replace({"bess":"battery","ps":"battery"})
    tb = tb.where(tb.isin(["wind","gas","nuclear","battery"]), "other")
    df["tech_bucket"] = tb

    return df

# ---- Fairness helpers/plots ----

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    if np.allclose(x, 0): return 0.0
    x = np.sort(x); n = x.size; cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

def atkinson(x: np.ndarray, epsilon: float = 1.0) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0: return np.nan
    if np.allclose(x, 0): return 0.0
    mu = np.mean(x)
    if epsilon == 1.0:
        xp = x[x > 0]; 
        if xp.size == 0: return 1.0
        return 1.0 - np.exp(np.mean(np.log(xp))) / mu
    return 1.0 - (np.mean(x**(1 - epsilon)) ** (1 / (1 - epsilon))) / mu

def theil_T(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0: return np.nan
    mu = np.mean(x); return np.mean((x / mu) * np.log(x / mu))

def headcount_cost_recovery(net_abs: np.ndarray) -> Tuple[int, float]:
    ok = np.isfinite(net_abs); n = ok.sum()
    if n == 0: return 0, np.nan
    count = np.sum(net_abs[ok] >= 0.0); return int(count), count / n

def share_rapid_payback(pb: np.ndarray, years: float) -> float:
    ok = np.isfinite(pb); n = ok.sum()
    if n == 0: return np.nan
    return float(np.sum(pb[ok] <= years)) / n

def lorenz_points(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0: return np.array([0,1]), np.array([0,1])
    x = np.sort(np.clip(x, 0, None)); cum = np.cumsum(x); total = cum[-1]
    L = np.concatenate([[0.0], cum / total if total > 0 else np.zeros_like(cum)])
    P = np.linspace(0, 1, L.size); return P, L

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    approaches = {"LMP":("net_LMP_perGW","net_LMP_abs","PB_LMP","Adeq_LMP"),
                  "AMM1":("net_A1_perGW","net_AMM1_abs","PB_AMM1","Adeq_A1"),
                  "AMM2":("net_A2_perGW","net_AMM2_abs","PB_AMM2","Adeq_A2")}
    for name,(pergw_col, net_col, pb_col, adeq_col) in approaches.items():
        pergw = _safe_num(df[pergw_col]).values
        netab = _safe_num(df[net_col]).values
        pb    = _safe_num(df[pb_col]).values
        adeq  = _safe_num(df[adeq_col]).values
        pergw_nn = np.clip(pergw, 0, None)
        row = dict(approach=name)
        row["gini_net_perGW"] = gini(pergw_nn)
        row["atkinson_e05"]   = atkinson(pergw_nn, 0.5)
        row["atkinson_e10"]   = atkinson(pergw_nn, 1.0)
        row["theil_T"]        = theil_T(pergw_nn)
        row["mean_adequacy_ratio"] = np.nanmean(adeq)
        row["p25_adequacy_ratio"]  = np.nanpercentile(adeq, 25)
        row["p75_adequacy_ratio"]  = np.nanpercentile(adeq, 75)
        cnt, share = headcount_cost_recovery(netab)
        row["n_cost_recovery"]     = cnt
        row["share_cost_recovery"] = share
        row["median_payback_years"] = np.nanmedian(pb)
        row["p90_payback_years"]    = np.nanpercentile(pb[np.isfinite(pb)], 90) if np.isfinite(pb).any() else np.nan
        row["share_pb_le_1y"]       = share_rapid_payback(pb, 1.0)
        row["share_pb_le_0p2y_approx60d"] = share_rapid_payback(pb, 0.2)
        rows.append(row)
    met = pd.DataFrame(rows)

    # Transparent composite score
    def _norm01(s):
        v = s.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.ones_like(v) * 0.5
        return (v - lo) / (hi - lo)

    comp = pd.DataFrame()
    comp["ineq"]   = 1.0 - met["gini_net_perGW"].clip(0, 1)
    comp["adequ"]  = met["share_cost_recovery"].clip(0, 1)
    comp["pb_med"] = (1.0 / (1.0 + met["median_payback_years"].clip(lower=0))).clip(0, 1)
    comp["anti_jackpot"] = 1.0 - met["share_pb_le_0p2y_approx60d"].clip(0, 1)
    met["fairness_score_composite"] = comp.apply(_norm01).mean(axis=1)
    return met.sort_values("fairness_score_composite", ascending=False)

# ---- Fairness plots ----

def plot_lorenz(df, out_png):
    plt.figure(figsize=(9,7))
    for lab,col in [("LMP","net_LMP_perGW"),("AMM 1","net_A1_perGW"),("AMM 2","net_A2_perGW")]:
        x = np.clip(_safe_num(df[col]).values, 0, None); P,L = lorenz_points(x)
        plt.plot(P, L, label=lab, linewidth=2)
    plt.plot([0,1],[0,1], linestyle="--", color="#666", label="Equality line")
    plt.xlabel("Share of generators (sorted, per-GW net ascending)")
    plt.ylabel("Share of total per-GW net")
    plt.title("Lorenz curves of annual net (£/GW)")
    plt.legend(); _savefig(out_png)

def plot_ecdf_payback_diff(df, out_png):
    plt.figure(figsize=(9,7))
    for lab,col in [("LMP","PBdiff_LMP"),("AMM 1","PBdiff_AMM1"),("AMM 2","PBdiff_AMM2")]:
        x = _safe_num(df[col]).values; x = x[np.isfinite(x)]; x = np.sort(x)
        y = np.linspace(0, 1, len(x), endpoint=True); plt.plot(x,y,label=lab,linewidth=2)
    plt.axvline(0, color="#444", linestyle=":")
    plt.xlabel("Actual − expected payback (years)  [negative = faster]")
    plt.ylabel("Cumulative fraction of generators")
    plt.title("ECDF of payback differentials"); plt.legend(loc="lower right")
    _savefig(out_png)

def plot_box_violin_by_tech(df, out_png):
    techs = ["wind","nuclear","gas","battery","other"]; data=[]; labels=[]
    for col_lab,col in [("LMP","PBdiff_LMP"),("AMM 1","PBdiff_AMM1"),("AMM 2","PBdiff_AMM2")]:
        for t in techs:
            vals = _safe_num(df.loc[df["tech_bucket"]==t, col]).values
            vals = vals[np.isfinite(vals)]
            if vals.size: data.append(vals); labels.append(f"{t} — {col_lab}")
    if not data: return
    plt.figure(figsize=(12,7))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts['bodies']: pc.set_alpha(0.6)
    plt.boxplot(data, widths=0.06, positions=np.arange(1, len(data)+1))
    plt.xticks(np.arange(1, len(data)+1), labels, rotation=60, ha="right")
    plt.ylabel("Actual − expected payback (years)")
    plt.title("Payback differential by tech and approach"); _savefig(out_png)

def plot_cost_recovery_headcount(df, out_png):
    rows=[]
    for lab,col in [("LMP","net_LMP_abs"),("AMM 1","net_AMM1_abs"),("AMM 2","net_AMM2_abs")]:
        net = _safe_num(df[col]).values; n, share = headcount_cost_recovery(net)
        rows.append((lab, share))
    lab, val = zip(*rows)
    plt.figure(figsize=(7,5)); plt.bar(lab, val); plt.ylim(0,1)
    plt.ylabel("Share with net ≥ 0"); plt.title("Adequacy headcount by approach")
    for i,v in enumerate(val): plt.text(i, v+0.02, f"{v*100:.0f}%", ha="center", va="bottom")
    _savefig(out_png)

def plot_rapid_payback_share(df, out_png):
    rows1, rows02 = [], []
    for lab,col in [("LMP","PB_LMP"),("AMM 1","PB_AMM1"),("AMM 2","PB_AMM2")]:
        x = _safe_num(df[col]).values
        rows1.append((lab, share_rapid_payback(x, 1.0)))
        rows02.append((lab, share_rapid_payback(x, 0.2)))
    labs = [r[0] for r in rows1]; v1=[r[1] for r in rows1]; v2=[r[1] for r in rows02]
    a = np.arange(len(labs)); w=0.35
    plt.figure(figsize=(8,5))
    plt.bar(a-w/2, v1, width=w, label="≤ 1 year"); plt.bar(a+w/2, v2, width=w, label="≤ ~60 days")
    plt.xticks(a, labs); plt.ylim(0,1); plt.ylabel("Share of generators")
    plt.title("Ultra-rapid paybacks (jackpots)"); plt.legend()
    for i,(A,B) in enumerate(zip(v1,v2)):
        plt.text(i-w/2, A+0.02, f"{A*100:.0f}%", ha="center", va="bottom")
        plt.text(i+w/2, B+0.02, f"{B*100:.0f}%", ha="center", va="bottom")
    _savefig(out_png)

def plot_net_perGW_distributions(df, out_png):
    """
    Boxplot version of per-GW net distributions.
    Makes narrow distributions (e.g. AMM1) visible alongside heavy-tailed ones (e.g. LMP).
    """
    series = [
        ("LMP",   "net_LMP_perGW"),
        ("AMM 1", "net_A1_perGW"),
        ("AMM 2", "net_A2_perGW"),
    ]

    data, labels = [], []
    for lab, col in series:
        x = _safe_num(df[col]).values
        x = x[np.isfinite(x)]
        if x.size:
            data.append(x)
            labels.append(lab)

    if not data:
        return

    plt.figure(figsize=(9, 5))

    bp = plt.boxplot(
        data,
        labels=labels,
        vert=False,          # horizontal = easier with long tails
        showfliers=True,     # keep jackpots visible
        whis=1.5,
        patch_artist=True
    )

    # Light styling
    for patch in bp["boxes"]:
        patch.set_alpha(0.5)
    for k in ["medians", "whiskers", "caps"]:
        for obj in bp[k]:
            obj.set_linewidth(1.5)

    plt.xlabel("Annual net per GW ( £ / GW / year )")
    plt.ylabel("Approach")
    plt.title("Distribution of per-GW net by approach (boxplot)")
    _savefig(out_png)


def plot_composite_score(metrics, out_png):
    plt.figure(figsize=(7,5))
    plt.bar(metrics["approach"], metrics["fairness_score_composite"])
    for i,v in enumerate(metrics["fairness_score_composite"]): plt.text(i, v+0.02, f"{v:.2f}", ha="center", va="bottom")
    plt.ylim(0,1.1); plt.ylabel("Composite fairness score (0–1)"); plt.title("Composite fairness score")
    _savefig(out_png)

def top_bottom_tables(df: pd.DataFrame) -> pd.DataFrame:
    pieces=[]
    for lab,col in [("LMP","PB_LMP"),("AMM1","PB_AMM1"),("AMM2","PB_AMM2")]:
        d = df[["gen_id","tech_bucket", col]].rename(columns={col:"payback_years"}).copy()
        d["approach"]=lab; d = d[np.isfinite(d["payback_years"])]
        top = d.nsmallest(5, "payback_years").assign(rank="fastest")
        bot = d.nlargest(5, "payback_years").assign(rank="slowest")
        pieces += [top, bot]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

# ============================================================
# Time-series loaders (LMP & AMM)
# ============================================================

def load_lmp_timeseries(lmp_run_root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not lmp_run_root or not os.path.exists(lmp_run_root):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    days = sorted([d for d in glob.glob(os.path.join(lmp_run_root, "*")) if os.path.isdir(d)])
    rows_p, rows_d, rows_dem = [], [], []
    for day in days:
        # LMPs
        for fn in ["settlement_RT.csv", "lmps_RT.csv"]:
            fp = os.path.join(day, fn)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                if "timestamp" in df and "node" in df:
                    price_col = "settlement_lmp" if "settlement_lmp" in df.columns else ("lmp" if "lmp" in df.columns else None)
                    if price_col:
                        df["timestamp"]=pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
                        rows_p.append(df[["timestamp","node",price_col]].rename(columns={price_col:"lmp"}))
                break
        # Dispatch
        fpd = os.path.join(day, "dispatch.csv")
        if os.path.exists(fpd):
            dd = pd.read_csv(fpd)
            req = {"timestamp","gen_id","node","p_MW"}
            if req.issubset(dd.columns):
                dd["timestamp"]=pd.to_datetime(dd["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
                keep = ["timestamp","gen_id","node","p_MW","p_dis_MW","p_ch_MW","cost_MWh"]
                rows_d.append(dd[[c for c in keep if c in dd.columns]])
        # Demand
        fps = os.path.join(day, "served_unserved.csv")
        if os.path.exists(fps):
            ds = pd.read_csv(fps)
            if {"timestamp","node","demand_MW"}.issubset(ds.columns):
                ds["timestamp"]=pd.to_datetime(ds["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
                keep = ["timestamp","node","demand_MW","served_MW","unserved_MW","local_serve_MW","import_MW"]
                rows_dem.append(ds[[c for c in keep if c in ds.columns]])

    lmps    = pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame()
    dispatch= pd.concat(rows_d, ignore_index=True) if rows_d else pd.DataFrame()
    demand  = pd.concat(rows_dem, ignore_index=True) if rows_dem else pd.DataFrame()
    return lmps, dispatch, demand

def load_amm_timeseries(amm_run_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dispatch, demand) from AMM run tree (NO prices)."""
    if not amm_run_root or not os.path.exists(amm_run_root):
        return pd.DataFrame(), pd.DataFrame()
    days = sorted([d for d in glob.glob(os.path.join(amm_run_root, "*")) if os.path.isdir(d)])
    rows_d, rows_dem = [], []
    for day in days:
        fpd = os.path.join(day, "dispatch.csv")
        if os.path.exists(fpd):
            dd = pd.read_csv(fpd)
            if {"timestamp","gen_id","node","p_MW"}.issubset(dd.columns):
                dd["timestamp"]=pd.to_datetime(dd["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
                keep = ["timestamp","gen_id","node","p_MW","p_dis_MW","p_ch_MW","cost_MWh","reserve_up_MW"]
                rows_d.append(dd[[c for c in keep if c in dd.columns]])
        fps = os.path.join(day, "served_unserved.csv")
        if os.path.exists(fps):
            ds = pd.read_csv(fps)
            if {"timestamp","node","demand_MW"}.issubset(ds.columns):
                ds["timestamp"]=pd.to_datetime(ds["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
                keep = ["timestamp","node","demand_MW","served_MW","unserved_MW","local_serve_MW","import_MW"]
                rows_dem.append(ds[[c for c in keep if c in ds.columns]])
    dispatch= pd.concat(rows_d, ignore_index=True) if rows_d else pd.DataFrame()
    demand  = pd.concat(rows_dem, ignore_index=True) if rows_dem else pd.DataFrame()
    return dispatch, demand

def load_flows_and_caps(run_root: str, network_json: str) -> tuple[pd.DataFrame, Dict[str,float], Dict[str,tuple]]:
    if not run_root or not os.path.exists(run_root) or not os.path.exists(network_json):
        return pd.DataFrame(), {}, {}
    with open(network_json, "r") as f:
        net = json.load(f)
    edges = [(str(a),str(b)) for a,b in net["edges"]]
    capmap = {f"L{i+1}": float(net["edge_capacity"].get(f"{a},{b}", net["edge_capacity"].get(f"{b},{a}", np.nan))) for i,(a,b) in enumerate(edges)}
    endpoints = {f"L{i+1}": (a,b) for i,(a,b) in enumerate(edges)}

    days = sorted([d for d in glob.glob(os.path.join(run_root, "*")) if os.path.isdir(d)])
    rows=[]
    for day in days:
        fp = os.path.join(day, "flows.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if {"timestamp","line_id","flow_MW"}.issubset(df.columns):
                df["timestamp"]=pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
                keep = ["timestamp","line_id","flow_MW","from_node","to_node"]
                rows.append(df[[c for c in keep if c in df.columns]])
    flows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return flows, capmap, endpoints

# ============================================================
# Tightness & derived metrics (separate logic for LMP vs AMM)
# ============================================================

def compute_system_price_from_lmps(lmps: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    if lmps.empty: return pd.DataFrame()
    prices = lmps.copy()
    prices["timestamp"]=pd.to_datetime(prices["timestamp"])
    price_col = "lmp"
    if demand is not None and not demand.empty:
        dmd = demand.copy(); dmd["timestamp"]=pd.to_datetime(dmd["timestamp"])
        w = dmd.groupby(["timestamp","node"], as_index=False)["demand_MW"].sum()
        df = prices.merge(w, on=["timestamp","node"], how="left")
        df["w"] = df["demand_MW"].fillna(0.0)
        grp = df.groupby("timestamp", as_index=False).apply(lambda g: pd.Series({
            "sys_price": np.average(g[price_col].fillna(0.0), weights=(g["w"].fillna(0.0)+1e-12)),
            "sys_demand": g["w"].sum()
        })).reset_index(drop=True)
        return grp
    else:
        grp = prices.groupby("timestamp", as_index=False)["lmp"].mean().rename(columns={"lmp":"sys_price"})
        grp["sys_demand"]=np.nan
        return grp

def compute_tightness_lmp(sys_price_df: pd.DataFrame, lmp_tightness_csv: str | None) -> pd.DataFrame:
    if lmp_tightness_csv and os.path.exists(lmp_tightness_csv):
        tt = pd.read_csv(lmp_tightness_csv)
        if "timestamp" not in tt.columns:
            raise ValueError("LMP tightness file must include 'timestamp'.")
        tt["timestamp"] = pd.to_datetime(tt["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        if "tightness" in tt.columns and "node" in tt.columns:
            out = (tt.groupby("timestamp", as_index=False)["tightness"].mean()
                    .rename(columns={"tightness":"tightness_sys"}))
        elif "system_tightness" in tt.columns:
            out = tt[["timestamp","system_tightness"]].rename(columns={"system_tightness":"tightness_sys"})
        elif "tightness" in tt.columns:
            out = tt[["timestamp","tightness"]].rename(columns={"tightness":"tightness_sys"})
        else:
            raise ValueError("Tightness file needs 'tightness' or 'system_tightness'.")
        out["tightness_sys"] = _safe_num(out["tightness_sys"]).clip(lower=0.0).fillna(0.0)
        return out.sort_values("timestamp", ignore_index=True)
    if sys_price_df is None or sys_price_df.empty:
        return pd.DataFrame(columns=["timestamp","tightness_sys"])
    df = sys_price_df.copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    sp = _safe_num(df["sys_price"])
    p25 = float(np.nanpercentile(sp.values, 25))
    dev = np.clip(sp - p25, 0, None)
    mean_dev = float(np.nanmean(dev))
    tight = np.zeros_like(dev, dtype=float) if (not np.isfinite(mean_dev) or mean_dev <= 0) else (dev / mean_dev)
    out = df[["timestamp"]].copy(); out["tightness_sys"] = tight
    return out

def compute_tightness_amm(demand_ts: pd.DataFrame, amm_tightness_csv: str | None) -> pd.DataFrame:
    if amm_tightness_csv and os.path.exists(amm_tightness_csv):
        tt = pd.read_csv(amm_tightness_csv)
        if "timestamp" not in tt.columns:
            raise ValueError("AMM tightness file must include 'timestamp'.")
        tt["timestamp"] = pd.to_datetime(tt["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        if "node" in tt.columns and "tightness" in tt.columns:
            out = (tt.groupby("timestamp", as_index=False)["tightness"].mean()
                     .rename(columns={"tightness":"tightness_sys"}))
        elif "system_tightness" in tt.columns:
            out = tt[["timestamp","system_tightness"]].rename(columns={"system_tightness":"tightness_sys"})
        elif "tightness" in tt.columns:
            out = tt[["timestamp","tightness"]].rename(columns={"tightness":"tightness_sys"})
        else:
            raise ValueError("AMM tightness file needs 'tightness' or 'system_tightness'.")
        out["tightness_sys"] = _safe_num(out["tightness_sys"]).clip(lower=0.0).fillna(0.0)
        if "system_price" in tt.columns:
            sp = tt[["timestamp","system_price"]].copy()
            sp["timestamp"]=pd.to_datetime(sp["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
            out = out.merge(sp, on="timestamp", how="left")
        return out.sort_values("timestamp", ignore_index=True)

    if demand_ts is None or demand_ts.empty:
        return pd.DataFrame(columns=["timestamp","tightness_sys"])

    d = demand_ts.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    sysd = d.groupby("timestamp", as_index=False)["demand_MW"].sum().rename(columns={"demand_MW":"sys_demand"})
    sd = _safe_num(sysd["sys_demand"]).values
    p25 = float(np.nanpercentile(sd, 25))
    dev = np.clip(sd - p25, 0, None)
    mean_dev = float(np.nanmean(dev))
    tight = np.zeros_like(dev, dtype=float) if (not np.isfinite(mean_dev) or mean_dev<=0) else (dev/mean_dev)
    out = sysd[["timestamp"]].copy(); out["tightness_sys"] = tight
    return out

# ============================================================
# System-value computations
# ============================================================

def compute_dispatch_mwh(dispatch: pd.DataFrame) -> pd.DataFrame:
    if dispatch.empty: return pd.DataFrame()
    d = dispatch.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    step_h = _step_hours_from_ts(d["timestamp"])
    d["MWh"] = _safe_num(d["p_MW"]).fillna(0.0) * step_h
    cols = ["timestamp","gen_id"]
    if "node" in d.columns: cols.append("node")
    return d[cols + ["MWh"]].groupby(cols, as_index=False)["MWh"].sum()

def attach_nodes_to_gens(gen_df: pd.DataFrame, gens_static: pd.DataFrame) -> pd.DataFrame:
    if gen_df.empty: return gen_df
    if "node" not in gen_df.columns:
        gs = gens_static[["gen_id","node"]] if "node" in gens_static.columns else gens_static.assign(node=np.nan)[["gen_id","node"]]
        gen_df = gen_df.merge(gs, on="gen_id", how="left")
    return gen_df

def compute_saoi(gen_dispatch_mwh: pd.DataFrame, tightness_df: pd.DataFrame) -> pd.DataFrame:
    if gen_dispatch_mwh.empty or tightness_df.empty:
        return pd.DataFrame()
    disp = gen_dispatch_mwh.copy(); disp["timestamp"]=pd.to_datetime(disp["timestamp"])
    ti   = tightness_df.copy();     ti["timestamp"]=pd.to_datetime(ti["timestamp"])
    df = disp.merge(ti[["timestamp","tightness_sys"]], on="timestamp", how="left")
    df["tightness_sys"] = df["tightness_sys"].fillna(0.0)
    by_g = df.groupby("gen_id", as_index=False).agg(MWh=("MWh","sum"))
    wsum = df.assign(MWh_w=lambda r: r["MWh"] * r["tightness_sys"]) \
             .groupby("gen_id", as_index=False)["MWh_w"].sum() \
             .rename(columns={"MWh_w":"tight_MWh"})
    by_g = by_g.merge(wsum, on="gen_id", how="left")
    by_g["SAOI"] = np.where(by_g["MWh"]>0, by_g["tight_MWh"]/by_g["MWh"], np.nan)
    return by_g[["gen_id","SAOI"]]

def compute_nvf(gen_dispatch_mwh: pd.DataFrame, sys_price_df: pd.DataFrame) -> pd.DataFrame:
    if gen_dispatch_mwh.empty or sys_price_df.empty or "sys_price" not in sys_price_df.columns:
        return pd.DataFrame()
    disp = gen_dispatch_mwh.copy(); disp["timestamp"]=pd.to_datetime(disp["timestamp"])
    sp   = sys_price_df.copy();     sp["timestamp"]=pd.to_datetime(sp["timestamp"])
    df = disp.merge(sp[["timestamp","sys_price"]], on="timestamp", how="left")
    overall_mean_price = float(np.nanmean(sp["sys_price"]))
    price_w = (df.assign(MWhP=lambda r: r["MWh"] * r["sys_price"])
                 .groupby("gen_id", as_index=False)[["MWhP","MWh"]].sum())
    price_w["NVF"] = np.where(
        (price_w["MWh"]>0) & np.isfinite(overall_mean_price) & (overall_mean_price>0),
        (price_w["MWhP"]/price_w["MWh"]) / overall_mean_price,
        np.nan
    )
    return price_w[["gen_id","NVF"]]

def compute_node_laa_from_import_split(demand_ts: pd.DataFrame, tightness_df: pd.DataFrame) -> pd.DataFrame:
    if demand_ts.empty or tightness_df.empty:
        return pd.DataFrame()
    d = demand_ts.copy(); d["timestamp"]=pd.to_datetime(d["timestamp"])
    ti= tightness_df.copy(); ti["timestamp"]=pd.to_datetime(ti["timestamp"])
    if not {"local_serve_MW","import_MW"}.issubset(d.columns):
        return pd.DataFrame()
    df = d.merge(ti, on="timestamp", how="left").fillna({"tightness_sys":0.0})
    denom = df["demand_MW"].replace(0.0, np.nan)
    df["import_share"] = df["import_MW"] / denom
    laa = (df.assign(prod=lambda r: r["import_share"] * r["tightness_sys"])
             .groupby("node", as_index=False)
             .agg(LAA=("prod","sum"), w=("tightness_sys","sum")))
    laa["LAA"] = np.where(laa["w"]>0, laa["LAA"]/laa["w"], np.nan)
    return laa[["node","LAA"]]

def compute_node_laa_price_premium(lmps_node_ts: pd.DataFrame, sys_price_df: pd.DataFrame, tightness_df: pd.DataFrame) -> pd.DataFrame:
    if lmps_node_ts.empty or sys_price_df.empty or tightness_df.empty:
        return pd.DataFrame()
    lp = lmps_node_ts.copy(); lp["timestamp"]=pd.to_datetime(lp["timestamp"])
    sp = sys_price_df.copy();  sp["timestamp"]=pd.to_datetime(sp["timestamp"])
    ti = tightness_df.copy();  ti["timestamp"]=pd.to_datetime(ti["timestamp"])
    df = lp.merge(sp, on="timestamp", how="left").merge(ti, on="timestamp", how="left")
    df["tightness_sys"] = df["tightness_sys"].fillna(0.0)
    df["prem"] = df["lmp"] - df["sys_price"]
    laa = (df.assign(prod=lambda r: r["prem"] * r["tightness_sys"])
             .groupby("node", as_index=False)
             .agg(LAA=("prod","sum"), w=("tightness_sys","sum")))
    laa["LAA"] = np.where(laa["w"]>0, laa["LAA"]/laa["w"], np.nan)
    return laa[["node","LAA"]]

def compute_congestion_and_cri(flows: pd.DataFrame, cap_by_line: Dict[str,float], endpoints_by_line: Dict[str,tuple],
                               gen_dispatch_mwh: pd.DataFrame, gens_static: pd.DataFrame,
                               tightness_df: pd.DataFrame, thresh: float = 0.98) -> tuple[pd.DataFrame, pd.DataFrame]:
    if flows.empty or not cap_by_line or tightness_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    fl = flows.copy(); fl["timestamp"]=pd.to_datetime(fl["timestamp"])
    ti = tightness_df.copy(); ti["timestamp"]=pd.to_datetime(ti["timestamp"])

    fl["cap"] = fl["line_id"].map(cap_by_line).astype(float)
    fl["is_cong"] = np.where(
        (fl["cap"].notna()) & (np.abs(_safe_num(fl["flow_MW"])) >= thresh * fl["cap"]), 1, 0
    )
    cong_by_line = (fl.groupby("line_id", as_index=False)
                      .agg(n=("is_cong","sum"), T=("timestamp","nunique")))
    cong_by_line["freq"] = np.where(cong_by_line["T"]>0, cong_by_line["n"]/cong_by_line["T"], np.nan)
    ep_df = pd.DataFrame([{"line_id": lid, "from_node": ab[0], "to_node": ab[1]} for lid,ab in endpoints_by_line.items()])
    cong_by_line = cong_by_line.merge(ep_df, on="line_id", how="left")

    # Node congestion flags by time
    if {"from_node","to_node"}.issubset(fl.columns):
        fn = fl.copy()
        fn["cong_at_from"] = fn["is_cong"]; fn["cong_at_to"] = fn["is_cong"]
        cong_from = fn.groupby(["timestamp","from_node"], as_index=False)["cong_at_from"].max() \
                      .rename(columns={"from_node":"node","cong_at_from":"cong_any"})
        cong_to   = fn.groupby(["timestamp","to_node"], as_index=False)["cong_at_to"].max() \
                      .rename(columns={"to_node":"node","cong_at_to":"cong_any"})
        node_cong = pd.concat([cong_from, cong_to], ignore_index=True) \
                     .groupby(["timestamp","node"], as_index=False)["cong_any"].max()
    else:
        node_rows=[]
        for (ts, lid), grp in fl.groupby(["timestamp","line_id"]):
            is_c = int(grp["is_cong"].max()); a,b = endpoints_by_line.get(lid, (None,None))
            if a is not None: node_rows.append({"timestamp": ts, "node": a, "cong_any": is_c})
            if b is not None: node_rows.append({"timestamp": ts, "node": b, "cong_any": is_c})
        node_cong = pd.DataFrame(node_rows).groupby(["timestamp","node"], as_index=False)["cong_any"].max()

    node_cong["timestamp"]=pd.to_datetime(node_cong["timestamp"])
    node_cong = node_cong.merge(ti, on="timestamp", how="left").fillna({"tightness_sys":0.0})

    # CRI per generator
    if gen_dispatch_mwh.empty or gens_static.empty:
        return cong_by_line, pd.DataFrame()

    gd = gen_dispatch_mwh.copy()
    gd["timestamp"] = pd.to_datetime(gd["timestamp"])

    gs = (
        gens_static[["gen_id", "node"]].copy()
        if "node" in gens_static.columns
        else gens_static.assign(node=np.nan)[["gen_id", "node"]]
    )

    gd = gd.merge(gs, on="gen_id", how="left")

    # Normalise node column so we have a plain 'node' for the congestion merge
    if "node_x" in gd.columns or "node_y" in gd.columns:
        if "node_x" in gd.columns and "node_y" in gd.columns:
            gd["node"] = gd["node_x"].fillna(gd["node_y"])
        elif "node_x" in gd.columns:
            gd["node"] = gd["node_x"]
        else:  # only node_y present
            gd["node"] = gd["node_y"]
        gd = gd.drop(columns=[c for c in ("node_x", "node_y") if c in gd.columns])

    gd = gd.merge(
        node_cong,
        on=["timestamp", "node"],
        how="left",
    ).fillna({"cong_any": 0, "tightness_sys": 0.0})

    agg = (
        gd.assign(MWh_w=lambda r: r["MWh"] * r["cong_any"] * r["tightness_sys"])
          .groupby("gen_id", as_index=False)
          .agg(MWh=("MWh", "sum"), MWh_cong_w=("MWh_w", "sum"))
    )
    agg["CRI"] = np.where(agg["MWh"] > 0, agg["MWh_cong_w"] / agg["MWh"], np.nan)

    return cong_by_line, agg[["gen_id", "CRI"]]


def value_duration_points(gen_dispatch_mwh: pd.DataFrame, tightness_df: pd.DataFrame) -> pd.DataFrame:
    if gen_dispatch_mwh.empty or tightness_df.empty:
        return pd.DataFrame()
    disp = gen_dispatch_mwh.copy(); disp["timestamp"]=pd.to_datetime(disp["timestamp"])
    ti   = tightness_df.copy();     ti["timestamp"]=pd.to_datetime(ti["timestamp"])
    ts_rank = ti.sort_values("tightness_sys", ascending=False).reset_index(drop=True)
    ts_rank["t_rank"] = np.arange(1, len(ts_rank)+1)
    ts_rank["t_frac"] = ts_rank["t_rank"] / len(ts_rank)
    d = disp.merge(ts_rank[["timestamp","t_rank","t_frac"]], on="timestamp", how="inner")
    d = d.sort_values(["gen_id","t_rank"])
    d["cum_MWh"] = d.groupby("gen_id")["MWh"].cumsum()
    total = d.groupby("gen_id", as_index=False)["MWh"].sum().rename(columns={"MWh":"MWh_tot"})
    d = d.merge(total, on="gen_id", how="left")
    d["cum_frac"] = np.where(d["MWh_tot"]>0, d["cum_MWh"]/d["MWh_tot"], np.nan)
    return d[["gen_id","t_frac","cum_frac"]]

# ---------------- Plots for system-value ----------------

def plot_saoi_vs_nvf(per_gen_sv: pd.DataFrame, out_png: str, title_prefix: str):
    if per_gen_sv.empty: return
    plt.figure(figsize=(8,6))
    if "tech_bucket" in per_gen_sv.columns:
        for tech in sorted(per_gen_sv["tech_bucket"].dropna().unique()):
            sub = per_gen_sv[per_gen_sv["tech_bucket"]==tech]
            plt.scatter(sub["NVF"], sub["SAOI"], alpha=0.6, label=str(tech))
        plt.legend()
    else:
        plt.scatter(per_gen_sv["NVF"], per_gen_sv["SAOI"], alpha=0.7)
    plt.xlabel("NVF — price alignment (≥1 = above-average when producing)")
    plt.ylabel("SAOI — system-aligned output (0–∞, typically 0–2)")
    plt.title(f"{title_prefix}: SAOI vs NVF by generator")
    _savefig(out_png)

def plot_node_laa_map(laa_df: pd.DataFrame, out_png: str, title_prefix: str):
    if laa_df.empty: return
    dd = laa_df.copy().dropna(subset=["LAA"]).sort_values("LAA", ascending=False)
    plt.figure(figsize=(10,6))
    plt.bar(dd["node"].astype(str), dd["LAA"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("LAA (tightness-weighted import share or price premium)")
    plt.title(f"{title_prefix}: Local Adequacy Alignment (by node)")
    _savefig(out_png)

def plot_congestion_map(cong_by_line: pd.DataFrame, out_png: str, title_prefix: str):
    if cong_by_line.empty: return
    cong = cong_by_line.copy().sort_values("freq", ascending=False)
    labels = cong.apply(lambda r: f'{r["line_id"]} ({str(r.get("from_node","?"))}-{str(r.get("to_node","?"))})', axis=1)
    plt.figure(figsize=(12,6))
    plt.bar(labels, cong["freq"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Congestion frequency (|flow| ≥ 98% cap)")
    plt.title(f"{title_prefix}: Congestion hotspots (lines)")
    _savefig(out_png)

def plot_value_duration(vd: pd.DataFrame, out_png: str, title_prefix: str):
    if vd.empty: return
    plt.figure(figsize=(9,6))
    plt.plot(vd.groupby("t_frac", as_index=False)["cum_frac"].median().set_index("t_frac"), linewidth=2)
    plt.xlabel("Cumulative share of time (tightness-ranked)")
    plt.ylabel("Cumulative share of generator output")
    plt.title(f"{title_prefix}: Value-duration (median across generators)")
    _savefig(out_png)

# ============================================================
# NEW: Extended fairness helpers (totals, shares, HHI/CRn)
# ============================================================

def compute_revenue_concentration(unified: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for appr, col in [("LMP","LMP_total"),("AMM1","AMM1_total"),("AMM2","AMM2_total")]:
        r = _safe_num(unified[col]).clip(lower=0).fillna(0.0).values
        S = r.sum()
        if S <= 0:
            rows.append({"approach":appr, "HHI":np.nan, "CR4":np.nan, "CR10":np.nan})
            continue
        share = np.sort(r / S)[::-1]
        rows.append({
            "approach": appr,
            "HHI": float(np.sum(share**2)),
            "CR4": float(np.sum(share[:4])) if share.size>=4 else float(np.sum(share)),
            "CR10": float(np.sum(share[:10])) if share.size>=10 else float(np.sum(share))
        })
    return pd.DataFrame(rows)

def build_extended_fairness_outputs(unified: pd.DataFrame, out_dir: str,
                                    saoi_lmp_csv: str | None = None,
                                    saoi_amm_csv: str | None = None):
    os.makedirs(out_dir, exist_ok=True)
    # 1) System totals by approach
    totals = pd.DataFrame({
        "approach": ["LMP","AMM1","AMM2"],
        "total_revenue": [
            _safe_num(unified["LMP_total"]).sum(),
            _safe_num(unified["AMM1_total"]).sum(),
            _safe_num(unified["AMM2_total"]).sum()
        ],
        "total_modelled_costs": [_safe_num(unified["modelled_costs"]).sum()]*3
    })
    totals["rev_minus_costs"] = totals["total_revenue"] - totals["total_modelled_costs"]
    totals.to_csv(os.path.join(out_dir, "extended_totals_by_approach.csv"), index=False)

    # 2) Per-generator revenue table with shares and ranks
    pergen = unified.copy()
    for appr, col in [("LMP","LMP_total"),("AMM1","AMM1_total"),("AMM2","AMM2_total")]:
        S = _safe_num(pergen[col]).sum()
        pergen[f"{appr}_rev_share"] = np.where(S>0, _safe_num(pergen[col]) / S, np.nan)
        pergen[f"{appr}_rank"] = _safe_num(pergen[col]).rank(ascending=False, method="min")
    keep_cols = ["gen_id","tech_bucket","Capacity_GW",
                 "LMP_total","AMM1_total","AMM2_total",
                 "net_LMP_abs","net_AMM1_abs","net_AMM2_abs",
                 "PB_LMP","PB_AMM1","PB_AMM2","Adeq_LMP","Adeq_A1","Adeq_A2",
                 "LMP_rev_share","AMM1_rev_share","AMM2_rev_share",
                 "LMP_rank","AMM1_rank","AMM2_rank"]
    pergen[keep_cols].to_csv(os.path.join(out_dir, "per_generator_revenues_all_approaches.csv"), index=False)

    # 3) Concentration metrics (HHI, CR4, CR10)
    conc = compute_revenue_concentration(unified)
    conc.to_csv(os.path.join(out_dir, "revenue_concentration.csv"), index=False)

# ============================================================
# VALIDATION & DEBUG
# ============================================================

def _assert_amm_has_revenue(amm_sum: pd.DataFrame, src_path: str):
    if amm_sum is None or amm_sum.empty:
        raise RuntimeError(
            f"[AMM ERROR] AMM per-generator summary is empty from: {src_path}\n"
            f"Ensure the path is correct and the file has rows."
        )
    if not (("amm_pot1_total" in amm_sum.columns) or ("amm_pot2_total" in amm_sum.columns) or
            ("R_cost_recovery_min" in amm_sum.columns) or ("R_lmp_equalisation" in amm_sum.columns)):
        # load_amm_pergen normalizes to amm_pot1_total/amm_pot2_total; but double-check just in case
        pass
    if float(_safe_num(amm_sum.get("amm_pot1_total", 0)).fillna(0).sum()) == 0 and \
       float(_safe_num(amm_sum.get("amm_pot2_total", 0)).fillna(0).sum()) == 0:
        LOG(f"[WARN] AMM totals sum to zero from: {src_path}. "
            f"If this is unexpected, re-check column names or file contents.")

def _write_inputs_audit(out_dir: str, gens, costs, lmp_sum, amm_sum, paths: Dict[str,str]):
    lines = []
    lines.append("=== INPUTS AUDIT ===")
    for name, df in [("gens_static", gens), ("costs_csv", costs), ("lmp_pergen", lmp_sum), ("amm_pergen", amm_sum)]:
        lines.append(f"\n[{name}] rows={len(df)}")
        lines.append(f"columns: {list(df.columns)}")
        if name in {"lmp_pergen","amm_pergen"}:
            totals = {c: float(_safe_num(df[c]).fillna(0).sum())
                      for c in df.columns if any(k in c.lower() for k in ["total","revenue","reserve","pot","lmp","energy","cost_recovery","equalisation"])}
            lines.append(f"quick_totals: {totals}")
    lines.append("\n=== PATHS ===")
    for k,v in paths.items():
        lines.append(f"{k}: {v}")
    p = os.path.join(out_dir, "fairness_inputs_audit.txt")
    with open(p, "w") as f:
        f.write("\n".join(lines))

# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    # Annual fairness inputs
    ap.add_argument("--gens-static", default="./gens_static.csv")
    ap.add_argument("--costs-csv",   default="./generator_costs_per_generator.csv")
    ap.add_argument("--lmp-pergen-summary", default="./marketExecution_Actual/generator_revenue_audit_lmp/per_generator_summary.csv")
    # UPDATED default for editor use:
    ap.add_argument("--amm-pergen-summary", default="./marketExecution_AMM/availabilityPayments/analysis/run_20251205_152214/per_generator_summary.csv")
    # Time-series roots
    ap.add_argument("--lmp-run-root", default="./marketExecution_Actual/outputs/uc_da_rt_sced_bat_reserves/")
    ap.add_argument("--amm-run-root", default="./marketExecution_AMM/outputs/run_20250914_194030")
    # Network
    ap.add_argument("--network-json", default="./marketExecution_Actual/data/network_uk.json")
    # Optional tightness files
    ap.add_argument("--lmp-tightness", default="")
    ap.add_argument("--amm-tightness", default="")
    # Output
    ap.add_argument("--out-dir", default="./fairness_outputs")
    # Editor-friendly: allow ENV override without CLI
    args = ap.parse_args([])  # <- ensures editor “Run” uses defaults; CLI users can comment this out

    out_dir = _ensure_dir(args.out_dir)

    # ---- Load unified (annual fairness) ----
    gens    = load_gens_static(args.gens_static)
    _assert_nonempty(gens, "gens_static", args.gens_static)

    costs   = load_costs(args.costs_csv)
    _assert_nonempty(costs, "costs_csv", args.costs_csv)

    lmp_sum = load_lmp_pergen(args.lmp_pergen_summary)
    _assert_nonempty(lmp_sum, "LMP per_generator_summary", args.lmp_pergen_summary)

    amm_sum = load_amm_pergen(args.amm_pergen_summary)
    _assert_amm_has_revenue(amm_sum, args.amm_pergen_summary)

    unified = build_unified(gens, costs, lmp_sum, amm_sum)

    # Debug drops
    unified.to_csv(os.path.join(out_dir, "unified_per_generator_DEBUG.csv"), index=False)
    _write_inputs_audit(out_dir, gens, costs, lmp_sum, amm_sum, {
        "gens_static": args.gens_static,
        "costs_csv": args.costs_csv,
        "lmp_pergen_summary": args.lmp_pergen_summary,
        "amm_pergen_summary": args.amm_pergen_summary,
        "lmp_run_root": args.lmp_run_root,
        "amm_run_root": args.amm_run_root,
        "network_json": args.network_json,
        "lmp_tightness": args.lmp_tightness or "(none)",
        "amm_tightness": args.amm_tightness or "(none)",
    })

    # ---- Fairness metrics + plots ----
    metrics = compute_metrics(unified)
    metrics.to_csv(os.path.join(out_dir, "fairness_metrics_summary.csv"), index=False)
    tb = top_bottom_tables(unified)
    if not tb.empty:
        tb.to_csv(os.path.join(out_dir, "top_bottom_paybacks_by_approach.csv"), index=False)

    plot_lorenz(unified, os.path.join(out_dir, "fig_lorenz_perGW_net.pdf"))
    plot_ecdf_payback_diff(unified, os.path.join(out_dir, "fig_ecdf_payback_diff.pdf"))
    plot_box_violin_by_tech(unified, os.path.join(out_dir, "fig_box_violin_paybak_diff_by_tech.pdf"))
    plot_cost_recovery_headcount(unified, os.path.join(out_dir, "fig_cost_recovery_headcount.pdf"))
    plot_rapid_payback_share(unified, os.path.join(out_dir, "fig_rapid_payback_share.pdf"))
    plot_net_perGW_distributions(unified, os.path.join(out_dir, "fig_net_perGW_distributions.pdf"))
    plot_composite_score(metrics, os.path.join(out_dir, "fig_composite_fairness_score.pdf"))
    LOG(f"[OK] Fairness outputs → {os.path.abspath(out_dir)}")

    # ========================================================
    # LMP system-value (prices allowed here)
    # ========================================================
    lmp_out = _ensure_dir(os.path.join(out_dir, "LMP_system_value"))
    lmps_ts, disp_ts_lmp, demand_ts_lmp = load_lmp_timeseries(args.lmp_run_root)
    flows_lmp, cap_by_line, endpoints = load_flows_and_caps(args.lmp_run_root, args.network_json)

    sys_price_df = compute_system_price_from_lmps(lmps_ts, demand_ts_lmp)
    tight_lmp    = compute_tightness_lmp(sys_price_df, args.lmp_tightness)

    gen_MWh_lmp = compute_dispatch_mwh(disp_ts_lmp)
    gen_MWh_lmp = attach_nodes_to_gens(gen_MWh_lmp, unified)

    # SAOI & NVF
    saoi_lmp = compute_saoi(gen_MWh_lmp, tight_lmp)
    nvf_lmp  = compute_nvf(gen_MWh_lmp, sys_price_df)
    if not saoi_lmp.empty:
        per_gen = saoi_lmp.merge(nvf_lmp, on="gen_id", how="left")
        if "tech_bucket" in unified.columns:
            per_gen = per_gen.merge(unified[["gen_id","tech_bucket"]], on="gen_id", how="left")
        per_gen.to_csv(os.path.join(lmp_out, "per_generator_SAOI_NVF.csv"), index=False)
        if "NVF" in per_gen.columns and per_gen["NVF"].notna().any():
            plot_saoi_vs_nvf(per_gen, os.path.join(lmp_out, "fig_saoi_vs_nvf.pdf"), "LMP")

    # LAA (price-premium fallback)
    laa_lmp = compute_node_laa_price_premium(lmps_ts, sys_price_df, tight_lmp)
    if not laa_lmp.empty:
        laa_lmp.to_csv(os.path.join(lmp_out, "node_LAA.csv"), index=False)
        plot_node_laa_map(laa_lmp, os.path.join(lmp_out, "fig_node_laa_map.pdf"), "LMP")

    # Congestion + CRI
    cong_by_line_lmp, cri_lmp = compute_congestion_and_cri(flows_lmp, cap_by_line, endpoints, gen_MWh_lmp, unified, tight_lmp)
    if not cong_by_line_lmp.empty:
        cong_by_line_lmp.to_csv(os.path.join(lmp_out, "congestion_frequency_by_line.csv"), index=False)
        plot_congestion_map(cong_by_line_lmp, os.path.join(lmp_out, "fig_congestion_map.pdf"), "LMP")
    if not cri_lmp.empty:
        cri_lmp.to_csv(os.path.join(lmp_out, "per_generator_CRI.csv"), index=False)

    # Value-duration
    vd_lmp = value_duration_points(gen_MWh_lmp, tight_lmp)
    if not vd_lmp.empty:
        vd_lmp.to_csv(os.path.join(lmp_out, "value_duration_points.csv"), index=False)
        plot_value_duration(vd_lmp, os.path.join(lmp_out, "fig_value_duration.pdf"), "LMP")

    LOG(f"[OK] LMP system-value outputs → {os.path.abspath(lmp_out)}")

    # ========================================================
    # AMM system-value (NO prices used unless provided in AMM file)
    # ========================================================
    amm_out = _ensure_dir(os.path.join(out_dir, "AMM_system_value"))
    disp_ts_amm, demand_ts_amm = load_amm_timeseries(args.amm_run_root)
    flows_amm, cap_by_line_A, endpoints_A = load_flows_and_caps(args.amm_run_root, args.network_json)

    tight_amm = compute_tightness_amm(demand_ts_amm, args.amm_tightness)

    gen_MWh_amm = compute_dispatch_mwh(disp_ts_amm)
    gen_MWh_amm = attach_nodes_to_gens(gen_MWh_amm, unified)

    # SAOI (always), NVF (only if 'system_price' provided in AMM tightness file)
    saoi_amm = compute_saoi(gen_MWh_amm, tight_amm)
    nvf_amm  = pd.DataFrame()
    if "system_price" in tight_amm.columns:
        sys_price_amm = tight_amm[["timestamp","system_price"]].copy()
        nvf_amm = compute_nvf(gen_MWh_amm, sys_price_amm.rename(columns={"system_price":"sys_price"}))

    if not saoi_amm.empty:
        per_gen_A = saoi_amm.copy()
        if not nvf_amm.empty:
            per_gen_A = per_gen_A.merge(nvf_amm, on="gen_id", how="left")
        if "tech_bucket" in unified.columns:
            per_gen_A = per_gen_A.merge(unified[["gen_id","tech_bucket"]], on="gen_id", how="left")
        per_gen_A.to_csv(os.path.join(amm_out, "per_generator_SAOI_NVF.csv"), index=False)
        if "NVF" in per_gen_A.columns and per_gen_A["NVF"].notna().any():
            plot_saoi_vs_nvf(per_gen_A, os.path.join(amm_out, "fig_saoi_vs_nvf.pdf"), "AMM")

    # LAA (preferred: import/local split present in AMM demand)
    laa_amm = compute_node_laa_from_import_split(demand_ts_amm, tight_amm)
    if not laa_amm.empty:
        laa_amm.to_csv(os.path.join(amm_out, "node_LAA.csv"), index=False)
        plot_node_laa_map(laa_amm, os.path.join(amm_out, "fig_node_laa_map.pdf"), "AMM")

    # Congestion + CRI
    cong_by_line_A, cri_A = compute_congestion_and_cri(flows_amm, cap_by_line_A, endpoints_A, gen_MWh_amm, unified, tight_amm)
    if not cong_by_line_A.empty:
        cong_by_line_A.to_csv(os.path.join(amm_out, "congestion_frequency_by_line.csv"), index=False)
        plot_congestion_map(cong_by_line_A, os.path.join(amm_out, "fig_congestion_map.pdf"), "AMM")
    if not cri_A.empty:
        cri_A.to_csv(os.path.join(amm_out, "per_generator_CRI.csv"), index=False)

    # Value-duration
    vd_amm = value_duration_points(gen_MWh_amm, tight_amm)
    if not vd_amm.empty:
        vd_amm.to_csv(os.path.join(amm_out, "value_duration_points.csv"), index=False)
        plot_value_duration(vd_amm, os.path.join(amm_out, "fig_value_duration.pdf"), "AMM")

    LOG(f"[OK] AMM system-value outputs → {os.path.abspath(amm_out)}")

    # ========================================================
    # Extended comparisons (totals, shares, HHI/CRn)
    # ========================================================
    ext_dir = _ensure_dir(os.path.join(out_dir, "extended_fairness"))
    build_extended_fairness_outputs(
        unified, ext_dir,
        saoi_lmp_csv=os.path.join(lmp_out, "per_generator_SAOI_NVF.csv"),
        saoi_amm_csv=os.path.join(amm_out, "per_generator_SAOI_NVF.csv"),
    )

    LOG(f"[DONE] All outputs → {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 120)
    main()
