#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allocate revenue pots using precomputed Shapley norms (HALF-HOURLY).

Interpretation (availability payments layer)
--------------------------------------------
- Wind and Nuclear get GUARANTEED fixed annual payments equal to their
  annualised non-fuel OpEx + CapEx_per_year (CapEx_total/payback if
  CapEx_per_year missing).
- Gas/Battery (and other controllable flex) share Shapley-based pots.

Changes vs older version
------------------------
1) Wind & Nuclear fixed annual payments are still the SAME in total, but
   are now distributed over time using each generator's OWN Shapley
   profile (rather than a generic timing weight). When their Shapley norm
   is low, they get less of their fixed pot in that period; when high,
   they get more. If a fixed-class generator never has positive Shapley,
   its fixed pot is spread uniformly across all timestamps.

2) The timing weights for the pots (BASE/DELTA/TARGET) are now based on
   total Shapley mass per timestamp:
       w_t ∝ sum_norm_all(t)
   so high-scarcity (high Shapley) hours carry a larger slice of the
   annual pot than slack hours.

Outputs (under analysis run dir)
--------------------------------
- class_pots_timeseries.csv
- generator_revenue_timeseries_BASE.csv
- generator_revenue_timeseries_DELTA.csv
- generator_revenue_timeseries_TARGET.csv (optional)
- generator_revenue_timeseries_ALL.csv
- per_generator_summary.csv

STRICT: reads EXACTLY this Shapley file (no scanning/fallbacks):
    marketExecution_AMM/availabilityPayments/outputs/long/shapley_allocations_generators.csv
"""

import os, json
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

# ---------- CONSTANT: read THIS file only ----------
LONG_SHAPLEY_DEFAULT = "marketExecution_AMM/availabilityPayments/outputs/long/shapley_allocations_generators.csv"

# ---------- Other defaults ----------
OUTPUTS_ROOT_DEFAULT   = os.path.join("marketExecution_AMM", "availabilityPayments", "outputs")
ANALYSIS_ROOT_DEFAULT  = os.path.join("marketExecution_AMM", "availabilityPayments", "analysis")
NETWORK_PATH_DEFAULT   = os.path.join("availabilityPayments", "data", "network_uk.json")
COSTS_CSV_DEFAULT      = "./generator_costs_per_generator.csv"
LMP_TOTALS_DEFAULT     = os.path.join("marketExecution_Actual", "outputs", "analysis", "totals_overall.csv")
AMM_SETTLE_DEFAULT     = os.path.join("marketExecution_AMM", "outputs", "analysis", "settlement_fuel_costs_per_generator.csv")
GENS_STATIC_DEFAULT    = "./gens_static.csv"

def log(s): print(s, flush=True)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- Robust timestamp parsing ----------
def parse_ts_mixed(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # normalise whitespace and ISO "T"
    s = (s.str.replace("\u00A0", " ", regex=False)
           .str.replace("\u2007", " ", regex=False)
           .str.replace("\u202F", " ", regex=False)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.replace("T", " ", regex=False))
    ts = pd.to_datetime(s, errors="coerce", utc=True)

    # second pass: explicit formats where needed (align by index labels)
    need = ts.isna()
    if need.any():
        s_need = s[need]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            parsed = pd.to_datetime(s_need, format=fmt, errors="coerce")
            ok = parsed.notna()
            if ok.any():
                idx_ok = s_need.index[ok]
                ts.loc[idx_ok] = parsed[ok].dt.tz_localize("UTC")
                s_need = s_need.loc[~s_need.index.isin(idx_ok)]
            if s_need.empty:
                break

    if ts.isna().any():
        s_need = s[ts.isna()]
        parsed = pd.to_datetime(s_need, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            idx_ok = s_need.index[ok]
            ts.loc[idx_ok] = parsed[ok].dt.tz_localize("UTC")

    if ts.isna().any():
        bad = s[ts.isna()].head(10).tolist()
        raise RuntimeError(f"Unparsed timestamps remain after robust parsing. Samples: {bad}")

    return ts.dt.tz_convert("UTC").dt.tz_localize(None)

# ---------- Helpers ----------
def _coalesce_numeric(df, cols, default=0.0):
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in cols:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    return out.fillna(default)

def _smart_pick(df, prefer, tokens, label):
    for c in prefer:
        if c in df.columns: 
            return c
    low = [t.lower() for t in tokens]
    for c in df.columns:
        cl = c.lower()
        if all(t in cl for t in low):
            return c
    raise KeyError(f"Missing {label} column (tried {prefer} or contains {tokens}).")

def is_wind_str(x: str) -> bool:
    t = str(x).strip().lower()
    return "wind" in t

def is_nuclear_str(x: str) -> bool:
    t = str(x).strip().lower()
    return "nuclear" in t

# ---------- Metadata ----------
def read_gens_static(path):
    df = pd.read_csv(path)
    cand_id = ["gen_id","generator","generator_id","id","name","GenID"]
    cand_t  = ["tech","technology","fuel","fuel_type","Fuel","Tech"]
    gid = next((c for c in cand_id if c in df.columns), None)
    gtech = next((c for c in cand_t  if c in df.columns), None)
    if gid is None or gtech is None:
        raise ValueError(f"{path}: need gen_id + tech columns.")
    out = df[[gid, gtech]].copy()
    out.columns = ["gen_id","tech"]
    out["gen_id"] = out["gen_id"].astype(str)
    out["tech"] = out["tech"].astype(str).str.strip().str.lower()
    out["tech"] = out["tech"].map(lambda t: {"bess":"battery","ps":"battery"}.get(t, t))
    return out

def read_network_generators(network_path: str) -> pd.DataFrame:
    with open(network_path, "r") as f:
        net = json.load(f)
    rows = [{"gen_id": str(g),
             "tech": str(info.get("tech", info.get("technology",""))).strip().lower()}
            for g, info in net["generators"].items()]
    df = pd.DataFrame(rows)
    if "tech" in df:
        df["tech"] = df["tech"].map(lambda t: {"bess":"battery","ps":"battery"}.get(t, t))
    return df

# ---------- INPUT: READ SPECIFIC SHAPLEY FILE ----------
def read_shapley_allocations_exact(file_path: str) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Shapley file not found: {file_path}")
    raw = pd.read_csv(file_path)

    # Normalise columns
    if "shap_norm" not in raw.columns and "norm" in raw.columns:
        raw = raw.rename(columns={"norm":"shap_norm"})
    need = {"timestamp","gen_id","shap_norm"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Shapley CSV missing columns: {missing}")

    df = raw[["timestamp","gen_id","shap_norm"]].copy()
    df["timestamp"] = parse_ts_mixed(df["timestamp"])
    df["gen_id"]    = df["gen_id"].astype(str)
    df["shap_norm"] = pd.to_numeric(df["shap_norm"], errors="coerce").fillna(0.0)
    df = df[df["timestamp"].notna() & df["gen_id"].notna()]

    # Diagnostics (OPTIONAL)
    tsu = pd.DatetimeIndex(df["timestamp"].drop_duplicates()).sort_values()
    if len(tsu) >= 2:
        dt_med = (tsu.to_series().diff().median())
        log(f"[IN] unique stamps={len(tsu):,}, median step={dt_med}")
    return df.drop_duplicates(subset=["timestamp","gen_id"], keep="last")

# ---------- Costs & fixed (wind + nuclear) ----------
def read_costs_and_fixed(tech_df: pd.DataFrame,
                         costs_path: str,
                         wind_payback_years: float,
                         nuclear_payback_years: float):
    """
    Returns:
      fixed_per (gen_id, fixed_payment) for wind + nuclear,
      eligible_fixed_total (sum Annual_required_cost for eligible),
      costs_enriched (with Annual_required_cost, payback, CapEx_total),
      excluded_ids (set of gen_ids for wind+nuclear)
    """
    df = pd.read_csv(costs_path)
    # Coalesce key fields
    opex = _coalesce_numeric(df, ["OpEx_nonfuel_total","opex_nonfuel_total","OpEx_non_fuel_total","opex_non_fuel"])
    cap_per_year = _coalesce_numeric(df, ["CapEx_per_year","capex_per_year","CapExPerYear"])
    cap_total    = _coalesce_numeric(df, ["CapEx_total","capex_total","CapEx"], default=np.nan)
    pby          = _coalesce_numeric(df, ["payback_years","Payback_Years","payback","payback_year"], default=np.nan)

    cand_id = ["gen_id","generator","generator_id","id","name","GenID"]
    gid = next((c for c in cand_id if c in df.columns), None)
    if gid is None:
        raise ValueError(f"{costs_path}: need gen_id column.")
    base = pd.DataFrame({
        "gen_id": df[gid].astype(str),
        "OpEx_nonfuel_total": opex,
        "CapEx_per_year_raw": cap_per_year.replace({0: np.nan}),
        "CapEx_total": cap_total.replace({0: np.nan}),
        "payback_years_raw": pby.replace({0: np.nan})
    })

    # Attach tech (from tech_df only; avoid tech_x/tech_y later)
    tech_map = tech_df[["gen_id","tech"]].drop_duplicates()
    tmp = base.merge(tech_map, on="gen_id", how="left")
    tmp["tech"] = tmp["tech"].astype(str)

    # Identify excluded classes
    is_wind    = tmp["tech"].map(is_wind_str)
    is_nuclear = tmp["tech"].map(is_nuclear_str)
    is_excl    = is_wind | is_nuclear

    # Derive CapEx_per_year for wind/nuclear if missing using payback years
    CapEx_per_year = tmp["CapEx_per_year_raw"].copy()
    need_wind = is_wind & CapEx_per_year.isna() & tmp["CapEx_total"].notna()
    need_nuc  = is_nuclear & CapEx_per_year.isna() & tmp["CapEx_total"].notna()
    if need_wind.any():
        CapEx_per_year.loc[need_wind] = tmp.loc[need_wind, "CapEx_total"] / max(wind_payback_years, 1e-9)
    if need_nuc.any():
        CapEx_per_year.loc[need_nuc] = tmp.loc[need_nuc, "CapEx_total"] / max(nuclear_payback_years, 1e-9)
    tmp["CapEx_per_year"] = CapEx_per_year

    # Annual requirement and CapEx_total fallback (for payback reporting)
    tmp["Annual_required_cost"] = tmp["OpEx_nonfuel_total"].fillna(0.0) + tmp["CapEx_per_year"].fillna(0.0)
    tmp["CapEx_total"] = tmp["CapEx_total"].where(
        tmp["CapEx_total"].notna(),
        tmp["CapEx_per_year"] * tmp["payback_years_raw"]
    )

    # Fixed per-generator for excluded classes (wind + nuclear)
    fixed_per = tmp.loc[is_excl, ["gen_id","Annual_required_cost"]].rename(
        columns={"Annual_required_cost":"fixed_payment"}
    ).copy()

    # Eligible fixed total = sum Annual_required_cost for eligible (gas + battery etc.)
    eligible_fixed_total = float(tmp.loc[~is_excl, "Annual_required_cost"].fillna(0.0).sum())

    # Payback column for reporting (use provided; fill defaults by class if missing)
    tmp["payback_years"] = tmp["payback_years_raw"]
    tmp.loc[is_wind & tmp["payback_years"].isna(),    "payback_years"] = wind_payback_years
    tmp.loc[is_nuclear & tmp["payback_years"].isna(), "payback_years"] = nuclear_payback_years

    excluded_ids = set(tmp.loc[is_excl, "gen_id"].astype(str))
    # drop tech to avoid tech_x/tech_y later
    return fixed_per, eligible_fixed_total, tmp.drop(columns=["tech"]), excluded_ids

# ---------- Shapley timing & allocation ----------
def build_timing_weights(df_shap: pd.DataFrame,
                         tech_df: pd.DataFrame,
                         excluded_ids: set):
    """
    Build:
      - 'timing' per timestamp with:
           sum_norm_all, sum_norm_eligible, f_t, w_t
        where w_t is a scarcity-based time weight:
           w_t ∝ sum_norm_all(t)
      - elig_weights: Shapley norms for ELIGIBLE generators
      - excl_weights: Shapley norms for EXCLUDED generators (wind + nuclear)
    """
    tech_map = tech_df[["gen_id","tech"]].drop_duplicates()
    df = df_shap.merge(tech_map, on="gen_id", how="left")
    df["gen_id"] = df["gen_id"].astype(str)
    df["is_excluded"] = df["gen_id"].isin(excluded_ids)
    df["is_eligible"] = ~df["is_excluded"]

    total_norm = (
        df.groupby("timestamp", as_index=False)["shap_norm"]
          .sum()
          .rename(columns={"shap_norm":"sum_norm_all"})
    )
    elig_norm = (
        df[df["is_eligible"]]
          .groupby("timestamp", as_index=False)["shap_norm"]
          .sum()
          .rename(columns={"shap_norm":"sum_norm_eligible"})
    )

    timing = total_norm.merge(elig_norm, on="timestamp", how="left").fillna(
        {"sum_norm_eligible": 0.0}
    )

    # Fraction of total Shapley mass that is eligible (for diagnostics)
    timing["f_t"] = np.where(
        timing["sum_norm_all"] > 0.0,
        timing["sum_norm_eligible"] / timing["sum_norm_all"],
        0.0
    )

    # Scarcity-based time weights: w_t ∝ sum_norm_all(t)
    w_raw = timing["sum_norm_all"].clip(lower=0.0)
    denom = float(w_raw.sum())
    if denom <= 0.0:
        # fallback: uniform over timestamps
        timing["w_t"] = 1.0 / max(len(timing), 1)
    else:
        timing["w_t"] = w_raw / denom

    elig_weights = df[df["is_eligible"]][["timestamp","gen_id","shap_norm"]].copy()
    excl_weights = df[df["is_excluded"]][["timestamp","gen_id","shap_norm"]].copy()
    return timing.sort_values("timestamp"), elig_weights, excl_weights

def allocate_pot_per_timestamp(elig_weights: pd.DataFrame,
                               timing: pd.DataFrame,
                               pot_col: str) -> pd.DataFrame:
    """
    For a given pot timeseries pot_t_* (BASE/DELTA/TARGET), allocate among
    ELIGIBLE generators proportional to their Shapley norm at each timestamp.
    """
    df = elig_weights.merge(timing[["timestamp", pot_col]], on="timestamp", how="inner")
    denom = df.groupby("timestamp")["shap_norm"].transform(lambda s: s.clip(lower=0.0).sum())
    w = np.where(denom > 0, df["shap_norm"].clip(lower=0.0) / denom, 0.0)
    out = df[["timestamp","gen_id"]].copy()
    out[f"alloc_{pot_col}"] = df[pot_col].values * w
    return out

def build_fixed_timeseries_from_shapley(excl_weights: pd.DataFrame,
                                        fixed_per_excl: pd.DataFrame,
                                        all_timestamps) -> pd.DataFrame:
    """
    Distribute FIXED annual payments for wind + nuclear over time using
    their OWN Shapley profiles:

        For each excluded generator g:
            w_g(t) ∝ max(shap_norm(g,t), 0)
            Sum_t w_g(t) = 1

    If a generator never shows positive Shapley mass, its fixed_payment
    is spread UNIFORMLY over all timestamps.
    """
    if excl_weights.empty:
        return pd.DataFrame(columns=["timestamp","gen_id","revenue_fixed"])

    df = excl_weights.copy()
    df["gen_id"] = df["gen_id"].astype(str)
    df["shap_norm"] = pd.to_numeric(df["shap_norm"], errors="coerce").fillna(0.0)
    df["shap_pos"] = df["shap_norm"].clip(lower=0.0)

    gen_sum = (
        df.groupby("gen_id", as_index=False)["shap_pos"]
          .sum()
          .rename(columns={"shap_pos":"sum_shap"})
    )
    df = df.merge(gen_sum, on="gen_id", how="left")

    # Non-zero Shapley gens: weights from their profile
    df_nonzero = df[df["sum_shap"] > 0].copy()
    df_nonzero["w_g_t"] = df_nonzero["shap_pos"] / df_nonzero["sum_shap"]

    # Zero-Shapley gens: uniform across all timestamps
    zero_gens = gen_sum.loc[gen_sum["sum_shap"] <= 0, "gen_id"].astype(str).tolist()
    ts_sorted = pd.DatetimeIndex(sorted(pd.to_datetime(all_timestamps)))
    n_ts = len(ts_sorted)
    rows_uniform = []
    if zero_gens and n_ts > 0:
        w_uniform = 1.0 / n_ts
        for g in zero_gens:
            for ts in ts_sorted:
                rows_uniform.append({"gen_id": g, "timestamp": ts, "w_g_t": w_uniform})
    df_uniform = pd.DataFrame(rows_uniform) if rows_uniform else pd.DataFrame(columns=["gen_id","timestamp","w_g_t"])

    fixed_w = pd.concat(
        [
            df_nonzero[["timestamp","gen_id","w_g_t"]],
            df_uniform[["timestamp","gen_id","w_g_t"]]
        ],
        ignore_index=True
    )

    # Attach fixed annual payments
    fixed_df = fixed_per_excl.copy()
    fixed_df["gen_id"] = fixed_df["gen_id"].astype(str)
    fixed_ts = fixed_w.merge(fixed_df, on="gen_id", how="left")
    fixed_ts["fixed_payment"] = pd.to_numeric(fixed_ts["fixed_payment"], errors="coerce").fillna(0.0)

    fixed_ts["revenue_fixed"] = fixed_ts["fixed_payment"] * fixed_ts["w_g_t"]
    fixed_ts = fixed_ts[["timestamp","gen_id","revenue_fixed"]].sort_values(["timestamp","gen_id"])
    return fixed_ts

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # READ
    ap.add_argument("--long-shapley", default=LONG_SHAPLEY_DEFAULT,
                    help="Exact path to shapley_allocations_generators.csv (half-hourly). No directory scanning.")
    ap.add_argument("--outputs-root", default=OUTPUTS_ROOT_DEFAULT)
    ap.add_argument("--network", default=NETWORK_PATH_DEFAULT)
    ap.add_argument("--gens-static", default=GENS_STATIC_DEFAULT)
    ap.add_argument("--costs-csv", default=COSTS_CSV_DEFAULT)
    ap.add_argument("--lmp-totals", default=LMP_TOTALS_DEFAULT)
    ap.add_argument("--amm-settlement", default=AMM_SETTLE_DEFAULT)
    # WRITE
    ap.add_argument("--analysis-root", default=ANALYSIS_ROOT_DEFAULT)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--allow-negative-delta", action="store_true")
    # Payback defaults
    ap.add_argument("--default-wind-payback", type=float, default=25.0)
    ap.add_argument("--nuclear-payback-years", type=float, default=40.0,
                    help="Payback horizon (years) for nuclear when deriving CapEx_per_year if missing (default 40).")
    # Optional TARGET pot sizing
    ap.add_argument("--target-pot-multiplier", type=float, default=2.0,
                    help="If >0, TARGET pot = multiplier * BASE pot (default 2.0). If 0, skip TARGET.")
    args = ap.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_ts}" + (f"_{args.run_tag}" if args.run_tag else "")
    RUN_DIR = os.path.join(args.analysis_root, run_name)
    ensure_dir(RUN_DIR)
    log(f"Writing outputs to: {RUN_DIR}")

    # Tech labels
    try:
        tech_df = read_gens_static(args.gens_static)
        log(f"Read tech from gens_static: {args.gens_static} (rows={len(tech_df)})")
    except Exception as e:
        log(f"[WARN] gens_static unavailable/invalid ({e}); falling back to network.")
        tech_df = read_network_generators(args.network)
        log(f"Read tech from network: {args.network} (rows={len(tech_df)})")

    # Shapley norms — exact file
    df_shap = read_shapley_allocations_exact(args.long_shapley)
    log(f"Shapley rows: {len(df_shap):,}")

    # Costs & fixed classes (wind + nuclear excluded from Shapley pots)
    fixed_per_excl, eligible_fixed_total, costs_enriched, excluded_ids = read_costs_and_fixed(
        tech_df, args.costs_csv, args.default_wind_payback, args.nuclear_payback_years
    )
    log(f"Excluded (fixed) generators: {len(excluded_ids)}")

    # Build timing weights & eligible/excluded weights
    timing, elig, excl = build_timing_weights(df_shap, tech_df, excluded_ids)
    log(f"Timestamps with weights: {len(timing):,}")
    log(f"Eligible rows (gas + battery etc.): {len(elig):,}")
    log(f"Excluded rows (wind + nuclear Shapley rows): {len(excl):,}")

    # ------------ Pots ------------
    # BASE pot = sum of (OpEx_nonfuel + CapEx_per_year) for eligible (NOT wind/nuclear)
    BASE_POT = float(eligible_fixed_total)

    # DELTA pot from LMP/AMM (as before)
    def _read_lmp_totals(path: str):
        df = pd.read_csv(path)
        e_col = _smart_pick(df, ["total_revenue_generators_energy"], ["revenue","generators","energy"], "LMP energy total")
        r_col = _smart_pick(df, ["total_revenue_generators_reserve"], ["revenue","generators","reserve"], "LMP reserve total")
        e = float(pd.to_numeric(df[e_col], errors="coerce").fillna(0).sum())
        r = float(pd.to_numeric(df[r_col], errors="coerce").fillna(0).sum())
        return e, r

    def _read_amm_totals(path: str):
        df = pd.read_csv(path)
        e_col = _smart_pick(df, ["settlement_energy_cost"],  ["energy","cost"],  "AMM energy cost")
        r_col = _smart_pick(df, ["settlement_reserve_cost"], ["reserve","cost"], "AMM reserve cost")
        e = float(pd.to_numeric(df[e_col], errors="coerce").fillna(0).sum())
        r = float(pd.to_numeric(df[r_col], errors="coerce").fillna(0).sum())
        return e, r

    lmp_e, lmp_r = _read_lmp_totals(args.lmp_totals)
    amm_e, amm_r = _read_amm_totals(args.amm_settlement)
    excl_fixed_total = float(fixed_per_excl["fixed_payment"].sum())
    DELTA_RAW  = float(lmp_e + lmp_r - amm_e - amm_r - excl_fixed_total)
    DELTA_POT  = DELTA_RAW if args.allow_negative_delta else max(0.0, DELTA_RAW)

    # TARGET pot
    target_multiplier = float(max(args.target_pot_multiplier, 0.0))
    TARGET_POT = float(BASE_POT * target_multiplier) if target_multiplier > 0 else 0.0

    log(f"BASE_POT  (eligible only) = {BASE_POT:,.2f}")
    log(f"DELTA_POT (used)          = {DELTA_POT:,.2f}  (raw={DELTA_RAW:,.2f}; excl fixed={excl_fixed_total:,.2f})")
    if TARGET_POT > 0:
        log(f"TARGET_POT                = {TARGET_POT:,.2f}  (multiplier={target_multiplier})")

    # Spread pots over time (scarcity-shaped)
    timing = timing.copy()
    timing["pot_t_BASE"]   = timing["w_t"] * BASE_POT
    timing["pot_t_DELTA"]  = timing["w_t"] * DELTA_POT
    timing["pot_t_TARGET"] = timing["w_t"] * TARGET_POT

    # Save pot timeseries (audit)
    timing[["timestamp","w_t","f_t","sum_norm_all","sum_norm_eligible",
            "pot_t_BASE","pot_t_DELTA","pot_t_TARGET"]].to_csv(
        os.path.join(RUN_DIR, "class_pots_timeseries.csv"), index=False
    )

    # Allocate pots to eligible (Shapley)
    alloc_base   = allocate_pot_per_timestamp(elig, timing, "pot_t_BASE")
    alloc_delta  = allocate_pot_per_timestamp(elig, timing, "pot_t_DELTA")
    alloc_target = allocate_pot_per_timestamp(elig, timing, "pot_t_TARGET") if TARGET_POT > 0 else None

    # Fixed (wind + nuclear) per timestamp using their OWN Shapley profiles
    fixed_ts = build_fixed_timeseries_from_shapley(
        excl_weights=excl,
        fixed_per_excl=fixed_per_excl,
        all_timestamps=timing["timestamp"].unique()
    )

    def build_ts(alloc_df):
        if alloc_df is None:
            return None
        ts = alloc_df.rename(columns={alloc_df.columns[-1]:"revenue_gbp"})[["timestamp","gen_id","revenue_gbp"]]
        ts = (
            pd.concat(
                [ts, fixed_ts.rename(columns={"revenue_fixed":"revenue_gbp"})],
                ignore_index=True
            )
            .groupby(["timestamp","gen_id"], as_index=False)["revenue_gbp"].sum()
            .sort_values(["timestamp","gen_id"])
        )
        return ts

    ts_base   = build_ts(alloc_base)
    ts_delta  = build_ts(alloc_delta)
    ts_target = build_ts(alloc_target) if TARGET_POT > 0 else None

    # Write per-timestamp series (half-hourly)
    if ts_base is not None:
        ts_base.to_csv(os.path.join(RUN_DIR, "generator_revenue_timeseries_BASE.csv"),  index=False)
    if ts_delta is not None:
        ts_delta.to_csv(os.path.join(RUN_DIR, "generator_revenue_timeseries_DELTA.csv"), index=False)
    if ts_target is not None:
        ts_target.to_csv(os.path.join(RUN_DIR, "generator_revenue_timeseries_TARGET.csv"), index=False)

    both_list = []
    if ts_base is not None:   both_list.append(ts_base.assign(approach="BASE"))
    if ts_delta is not None:  both_list.append(ts_delta.assign(approach="DELTA"))
    if ts_target is not None: both_list.append(ts_target.assign(approach="TARGET"))
    if both_list:
        both = (
            pd.concat(both_list, ignore_index=True)
              .sort_values(["timestamp","gen_id","approach"])
        )
        both.to_csv(os.path.join(RUN_DIR, "generator_revenue_timeseries_ALL.csv"), index=False)

    # Annual totals per generator
    totals = {}
    if ts_base is not None:
        totals["BASE"]  = ts_base.groupby("gen_id", as_index=False)["revenue_gbp"].sum().rename(columns={"revenue_gbp":"R_BASE"})
    if ts_delta is not None:
        totals["DELTA"] = ts_delta.groupby("gen_id", as_index=False)["revenue_gbp"].sum().rename(columns={"revenue_gbp":"R_DELTA"})
    if ts_target is not None:
        totals["TARGET"]= ts_target.groupby("gen_id", as_index=False)["revenue_gbp"].sum().rename(columns={"revenue_gbp":"R_TARGET"})

    # Build per-generator summary with a SINGLE 'tech' column
    per = tech_df[["gen_id","tech"]].drop_duplicates()
    per = per.merge(costs_enriched, on="gen_id", how="outer")
    for k, df in totals.items():
        per = per.merge(df, on="gen_id", how="left")

    # Numeric coercion
    num_cols = ["R_BASE","R_DELTA","R_TARGET","Annual_required_cost",
                "OpEx_nonfuel_total","CapEx_per_year","payback_years","CapEx_total"]
    for c in num_cols:
        if c in per.columns:
            per[c] = pd.to_numeric(per[c], errors="coerce")

    # Paybacks vs CapEx_total
    eps = 1e-9
    if "CapEx_total" not in per.columns or per["CapEx_total"].isna().all():
        per["CapEx_total"] = per["CapEx_per_year"] * per["payback_years"]

    for tag, col in [("A","R_BASE"), ("B","R_DELTA"), ("T","R_TARGET")]:
        if col in per.columns:
            per[f"Net_after_OpEx_{tag}"] = per[col] - per["OpEx_nonfuel_total"]
            per[f"Actual_payback_{tag}"] = np.where(
                per["CapEx_total"].notna() & (per[f"Net_after_OpEx_{tag}"] > eps),
                per["CapEx_total"] / per[f"Net_after_OpEx_{tag}"],
                np.inf
            )

    # Tripwire: excluded gens must have R_BASE == Annual_required_cost (within tolerance)
    if "R_BASE" in per.columns:
        per["is_excluded"] = per["gen_id"].astype(str).isin(excluded_ids)
        chk = per.loc[per["is_excluded"], ["gen_id","tech","Annual_required_cost","R_BASE"]].copy()
        chk["diff"] = (chk["R_BASE"] - chk["Annual_required_cost"]).abs()
        bad = chk[chk["diff"] > 1e-6]
        if len(bad):
            log("[ERROR] Fixed-class revenue mismatch (should match annual required cost exactly):")
            log(bad.head(10).to_string(index=False))
            raise RuntimeError(
                "Fixed payments not equal to annual requirement. "
                "Check exclusion mask / Shapley-based timing for fixed classes."
            )

    # Write summary
    sort_cols = [c for c in ["tech", "gen_id"] if c in per.columns]
    if not sort_cols:
        sort_cols = ["gen_id"]
    per.sort_values(sort_cols).to_csv(os.path.join(RUN_DIR, "per_generator_summary.csv"), index=False)

    # Console totals
    msg_tot = []
    if "Annual_required_cost" in per:
        msg_tot.append("Modelled costs="+f"{float(per['Annual_required_cost'].sum()):,.2f}")
    if "R_BASE"  in per:
        msg_tot.append("BASE="+f"{float(per['R_BASE'].sum()):,.2f}")
    if "R_DELTA" in per:
        msg_tot.append("DELTA="+f"{float(per['R_DELTA'].sum()):,.2f}")
    if "R_TARGET"in per:
        msg_tot.append("TARGET="+f"{float(per['R_TARGET'].sum()):,.2f}")

        # ---------------- High-level summary: LMP vs AMM1 (BASE) vs AMM2 (DELTA) ----------------
    summary_rows = []

    # LMP: no explicit capacity pot here
    summary_rows.append({
        "case": "LMP",
        "energy_total_gbp": lmp_e,
        "reserve_total_gbp": lmp_r,
        "capacity_total_gbp": 0.0,
    })

    # AMM1: BASE pot scenario
    summary_rows.append({
        "case": "AMM1_BASE",
        "energy_total_gbp": amm_e,
        "reserve_total_gbp": amm_r,
        "capacity_total_gbp": BASE_POT + excl_fixed_total,  # or BASE_POT only, if you prefer
    })

    # AMM2: DELTA pot scenario
    summary_rows.append({
        "case": "AMM2_DELTA",
        "energy_total_gbp": amm_e,
        "reserve_total_gbp": amm_r,
        "capacity_total_gbp": DELTA_POT + excl_fixed_total,  # or DELTA_POT only
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RUN_DIR, "generator_payment_totals_LMP_AMM1_AMM2.csv")
    summary_df.to_csv(summary_path, index=False)
    log(f"[OK] Wrote payment summary to {summary_path}")

    log("[OK] Totals — " + "  ".join(msg_tot))

if __name__ == "__main__":
    main()
