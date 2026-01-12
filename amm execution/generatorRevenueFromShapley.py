#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Allocate two revenue pots using precomputed Shapley norms (HALF-HOURLY),
output per-timestamp generator revenue time series, and make audit figures.

STRICT: Reads EXACTLY this Shapley file (no scanning/fallbacks):
    marketExecution_AMM/availabilityPayments/outputs/long/shapley_allocations_generators.csv

Final rules (per user requirements):

AMM-1 capacity distribution:
- BASE_POT = sum over all generators of (CapEx_per_year + OpEx_nonfuel_total).
- Wind & Nuclear receive revenue equal to their annual fixed costs (CapEx_per_year + OpEx_nonfuel_total),
  spread over time using w_t (timing based on sum of ALL Shapley norms).
- The REMAINING BASE_POT (after subtracting wind & nuclear totals) is allocated to GAS + BATTERY
  by half-hour using Shapley weights AMONG ELIGIBLE UNITS ONLY (gas+battery).

AMM-2 capacity (identity closure):
- DELTA_POT = (LMP energy + LMP reserves) − (AMM energy settlements + AMM reserve settlements) − (Wind+Nuclear fixed)
  (clamped at zero unless --allow-negative-delta).
- DELTA pot (timeseries/recon) is allocated ONLY to gas+battery so Allocated_DELTA_timeseries_sum == DELTA_POT.

Per-generator Option B totals for payback:
    * Wind & Nuclear: B = fixed costs (CapEx_per_year + OpEx_nonfuel_total).
    * Gas & Battery:  B = DELTA allocation (from the timeseries).

Identity checked & written to reconciliation_totals.csv:
    (W+N fixed + DELTA_POT) + AMM_energy + AMM_reserve  ==  LMP_energy + LMP_reserve

'payback_years' is read from the costs CSV and retained in per_generator_summary.csv.
"""

import os, json
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Eligibility: capacity pots go ONLY to these techs
ELIGIBLE_TECHS = {"gas", "battery"}
WIND_TECH = "wind"
NUCLEAR_TECH = "nuclear"

def log(s): print(s, flush=True)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- Robust timestamp parsing ----------
def parse_ts_mixed(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = (s.str.replace("\u00A0", " ", regex=False)
           .str.replace("\u2007", " ", regex=False)
           .str.replace("\u202F", " ", regex=False)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.replace("T", " ", regex=False))
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    mask = ts.isna()
    if mask.any():
        s2 = s[mask]
        try_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
        filled = ts.copy()
        for fmt in try_formats:
            m = filled.isna()
            if not m.any(): break
            parsed = pd.to_datetime(s2[m[mask]], format=fmt, errors="coerce")
            ok = parsed.notna()
            if ok.any():
                parsed_loc = parsed[ok].dt.tz_localize("UTC")
                filled.loc[mask[mask].index[ok]] = parsed_loc
        ts = filled
    if ts.isna().any():
        alt = pd.to_datetime(s[ts.isna()], errors="coerce")
        ok = alt.notna()
        if ok.any():
            ts.loc[ts.isna().index[ok]] = alt[ok].dt.tz_localize("UTC")
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
        if c in df.columns: return c
    low = [t.lower() for t in tokens]
    for c in df.columns:
        cl = c.lower()
        if all(t in cl for t in low): return c
    raise KeyError(f"Missing {label} column (tried {prefer} or contains {tokens}).")

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
    tsu = pd.DatetimeIndex(df["timestamp"].drop_duplicates()).sort_values()
    in_counts = tsu.to_series().resample("D").count()
    dt_med    = tsu.to_series().diff().median()
    log(f"[IN] First day count={int(in_counts.iloc[0]) if len(in_counts) else 0}, median step={dt_med}")
    if (len(in_counts) > 0 and int(in_counts.iloc[0]) < 10) or (pd.notna(dt_med) and dt_med >= pd.Timedelta("1D")):
        d0 = tsu[0].normalize()
        sample = tsu[(tsu>=d0) & (tsu<d0+pd.Timedelta("1D"))].to_pydatetime()
        raise RuntimeError(f"Input looks coarser than half-hourly. First-day stamps: {sample[:48]}")
    return df.drop_duplicates(subset=["timestamp","gen_id"], keep="last")

# ---------- Costs & totals ----------
def read_fixed_costs_with_payback(costs_path: str, tech_df: pd.DataFrame):
    """Return per-gen fixed, tech & payback; plus wind/nuclear totals and per-gen tables."""
    df = pd.read_csv(costs_path)
    opex = _coalesce_numeric(df, ["OpEx_nonfuel_total","opex_nonfuel_total","OpEx_non_fuel_total","opex_non_fuel"])
    capy = _coalesce_numeric(df, ["CapEx_per_year","capex_per_year","CapExPerYear"])
    payc = next((c for c in ["payback_years","Payback_Years","payback","payback_year"] if c in df.columns), None)

    cand_id = ["gen_id","generator","generator_id","id","name","GenID"]
    gid = next((c for c in cand_id if c in df.columns), None)
    if gid is None:
        raise ValueError(f"{costs_path}: need gen_id column.")

    costs = pd.DataFrame({
        "gen_id": df[gid].astype(str),
        "OpEx_nonfuel_total": opex,
        "CapEx_per_year": capy
    })
    if payc:
        costs["payback_years"] = pd.to_numeric(df[payc], errors="coerce")

    # sum duplicates if any
    costs = costs.groupby("gen_id", as_index=False).agg({
        "OpEx_nonfuel_total":"sum",
        "CapEx_per_year":"sum",
        **({"payback_years":"max"} if "payback_years" in costs.columns else {})
    })

    # attach tech
    tmp = costs.merge(tech_df, on="gen_id", how="left")
    tmp["tech"] = tmp["tech"].astype(str).str.lower()
    tmp["fixed"] = tmp["OpEx_nonfuel_total"].fillna(0.0) + tmp["CapEx_per_year"].fillna(0.0)

    wind_total    = float(tmp.loc[tmp["tech"] == WIND_TECH,    "fixed"].sum())
    nuclear_total = float(tmp.loc[tmp["tech"] == NUCLEAR_TECH, "fixed"].sum())
    base_pot_all  = float(tmp["fixed"].sum())

    # per-gen wind / nuclear frames (for time-series spreading)
    wind_per    = tmp.loc[tmp["tech"] == WIND_TECH,    ["gen_id","fixed"]].rename(columns={"fixed":"wind_fixed_payment"})
    nuclear_per = tmp.loc[tmp["tech"] == NUCLEAR_TECH, ["gen_id","fixed"]].rename(columns={"fixed":"nuclear_fixed_payment"})

    log(f"[COSTS] wind_fixed={wind_total:,.2f}  nuclear_fixed={nuclear_total:,.2f}  base_pot_all_fixed={base_pot_all:,.2f}")
    # return costs with tech/payback for merging later
    return costs.merge(tech_df, on="gen_id", how="left"), wind_total, nuclear_total, base_pot_all, wind_per, nuclear_per

def read_lmp_totals(path: str):
    df = pd.read_csv(path)
    e_col = _smart_pick(df, ["total_revenue_generators_energy"], ["revenue","generators","energy"], "LMP energy total")
    r_col = _smart_pick(df, ["total_revenue_generators_reserve"], ["revenue","generators","reserve"], "LMP reserve total")
    e = float(pd.to_numeric(df[e_col], errors="coerce").fillna(0).sum())
    r = float(pd.to_numeric(df[r_col], errors="coerce").fillna(0).sum())
    log(f"[LMP] energy={e:,.2f}, reserve={r:,.2f}  total={e+r:,.2f}")
    return e, r

def read_amm_settlement_totals(path: str):
    df = pd.read_csv(path)
    e_col = _smart_pick(df, ["settlement_energy_cost"],  ["energy","cost"],  "AMM energy cost")
    r_col = _smart_pick(df, ["settlement_reserve_cost"], ["reserve","cost"], "AMM reserve cost")
    e_vals = pd.to_numeric(df[e_col], errors="coerce").fillna(0.0)
    r_vals = pd.to_numeric(df[r_col], errors="coerce").fillna(0.0)
    # Use magnitudes to be robust to sign conventions in the settlements file
    e = float(e_vals.abs().sum())
    r = float(r_vals.abs().sum())
    log(f"[AMM] energy={e:,.2f}, reserve={r:,.2f}  total={e+r:,.2f}")
    return e, r

# ---------- Shapley timing & allocation ----------
def build_timing_and_weights(df_shap: pd.DataFrame, tech_df: pd.DataFrame):
    """
    Build:
      - timing['w_t']: normalised sum of ALL shap_norm per timestamp (all techs influence timing)
      - elig_weights:  shapley weights for eligible techs (gas+battery) for BASE remainder
      - all_weights:   shapley weights for ALL units (only used if needed elsewhere)
    """
    df = df_shap.merge(tech_df, on="gen_id", how="left")
    df["tech"] = df["tech"].astype(str).str.lower()
    df["is_eligible"] = df["tech"].isin(ELIGIBLE_TECHS)

    sum_all = df.groupby("timestamp", as_index=False)["shap_norm"].sum().rename(columns={"shap_norm":"sum_norm_all"})
    # Timing weights from ALL shapley, normalised over horizon
    timing = sum_all.copy()
    w = timing["sum_norm_all"].clip(lower=0.0)
    denom = w.sum()
    timing["w_t"] = (1.0/len(timing)) if denom <= 0 else (w / denom)

    elig_weights = df[df["is_eligible"]][["timestamp","gen_id","shap_norm"]].copy()
    all_weights  = df[["timestamp","gen_id","shap_norm"]].copy()
    return timing.sort_values("timestamp"), elig_weights, all_weights

def allocate_pot_per_timestamp(weights_df: pd.DataFrame, timing: pd.DataFrame, pot_col: str) -> pd.DataFrame:
    df = weights_df.merge(timing[["timestamp", pot_col]], on="timestamp", how="inner")
    denom = df.groupby("timestamp")["shap_norm"].transform(lambda s: s.clip(lower=0.0).sum())
    w = np.where(denom > 0, df["shap_norm"].clip(lower=0.0)/denom, 0.0)
    out = df[["timestamp","gen_id"]].copy()
    out[f"alloc_{pot_col}"] = df[pot_col].values * w
    return out

# ---------- Plot helpers ----------
def _titlecase_tech(t: str) -> str:
    t = str(t).strip().lower()
    return {"bess":"Battery","ps":"Battery"}.get(t, t.capitalize())

def _label_with_tech(gen_ids, techs):
    return [f"{gid} ({_titlecase_tech(t)})" for gid, t in zip(map(str, gen_ids), techs)]

def _ordered_by_tech(per_gen: pd.DataFrame):
    g = per_gen.copy()
    tech_col = "tech" if "tech" in g.columns else ( "tech_x" if "tech_x" in g.columns else ("tech_y" if "tech_y" in g.columns else None))
    if tech_col is None:
        g["__tech__"] = "unknown"; tech_col = "__tech__"
    g[tech_col] = g[tech_col].fillna("unknown").astype(str).str.lower()
    g = g.sort_values([tech_col, "gen_id"], kind="mergesort").reset_index(drop=True)
    bounds = []
    if g.empty: return g, bounds
    s = 0; cur = g.loc[0, tech_col]
    for i in range(1, len(g)):
        if g.loc[i, tech_col] != cur:
            bounds.append((s, i-1, _titlecase_tech(cur))); s = i; cur = g.loc[i, tech_col]
    bounds.append((s, len(g)-1, _titlecase_tech(cur)))
    return g, bounds

def _draw_group_shading(ax, bounds, ymax):
    for k, (s,e,tech_label) in enumerate(bounds):
        if k % 2 == 0:
            ax.axvspan(s-0.5, e+0.5, color="#f2f2f2", alpha=0.5, zorder=0)
        xmid = (s+e)/2.0
        ax.text(xmid, ymax*1.02, tech_label, ha="center", va="bottom", fontsize=10, fontweight="bold")

# ---------- Plots ----------
def plot_revenue_multiseries(per: pd.DataFrame, out_png: str):
    SERIES = [
        ("Modelled costs",           "Annual_required_cost", dict(alpha=0.65, hatch="//", edgecolor="k", linewidth=0.4)),
        ("Cost recovery pot minimum","R_cost_recovery_min",  dict(alpha=0.95, edgecolor="k", linewidth=0.4)),
        ("LMP pot equalisation",     "R_lmp_equalisation",   dict(alpha=0.85, edgecolor="k", linewidth=0.4)),
    ]
    series_colors = {"Modelled costs":"#9c9c9c","Cost recovery pot minimum":"#1f77b4","LMP pot equalisation":"#2ca02c"}

    df, bounds = _ordered_by_tech(per)
    if df.empty: return
    tech_col = "tech" if "tech" in df.columns else ("tech_x" if "tech_x" in df.columns else ("tech_y" if "tech_y" in df.columns else None))
    labels = _label_with_tech(df["gen_id"].values, df[tech_col].values if tech_col else ["unknown"] * len(df))

    x = np.arange(len(labels)); width = 0.26
    figw = max(14, 0.36*len(labels) + 4)
    plt.figure(figsize=(figw, 8)); ax = plt.gca()

    for (name, col, style), dx in zip(SERIES, [-width, 0.0, width]):
        ax.bar(x + dx, df[col].values, width, label=name, color=series_colors[name], **style)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("£ per year"); ax.set_title("Modelled costs vs two allocation approaches — all generators")
    ymax = float(np.nanmax(df[[c for _,c,_ in SERIES]].values))
    _draw_group_shading(ax, bounds, ymax if np.isfinite(ymax) else 1.0)

    patches = [
        mpatches.Patch(facecolor=series_colors["Modelled costs"], alpha=0.65, hatch="//", edgecolor="k", label="Modelled costs"),
        mpatches.Patch(facecolor=series_colors["Cost recovery pot minimum"], alpha=0.95, edgecolor="k", label="Cost recovery pot minimum"),
        mpatches.Patch(facecolor=series_colors["LMP pot equalisation"], alpha=0.85, edgecolor="k", label="LMP pot equalisation"),
    ]
    ax.legend(handles=patches, loc="upper left")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_scatter_two(per: pd.DataFrame, out_png: str):
    plt.figure(figsize=(10, 8))
    x  = pd.to_numeric(per["Annual_required_cost"], errors="coerce").to_numpy()
    yA = pd.to_numeric(per["R_cost_recovery_min"],   errors="coerce").to_numpy()
    yB = pd.to_numeric(per["R_lmp_equalisation"],    errors="coerce").to_numpy()
    if len(per) == 0:
        lim = 1.0
    else:
        stack = np.column_stack([x, yA, yB])
        lim = 1.0 if np.all(np.isnan(stack)) else float(np.nanmax(stack)) * 1.05
    plt.scatter(x, yA, alpha=0.7, label="Cost recovery pot minimum", s=18)
    plt.scatter(x, yB, alpha=0.7, label="LMP pot equalisation", marker="^", s=22)
    plt.plot([0, lim], [0, lim], linewidth=1.2, color="#666666")
    plt.xlabel("Modelled costs (CapEx_per_year + OpEx_nonfuel_total)")
    plt.ylabel("Allocated annual revenue")
    plt.title("Allocated revenue vs modelled costs")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_top_surplus_deficit_two(per: pd.DataFrame, out_png: str, topN=20):
    per = per.copy()
    per["Def_A"] = per["R_cost_recovery_min"] - per["Annual_required_cost"]
    per["Def_B"] = per["R_lmp_equalisation"]  - per["Annual_required_cost"]
    per["Worst"] = per[["Def_A","Def_B"]].min(axis=1)
    head = pd.concat([per.sort_values("Worst").head(topN),
                      per.sort_values("Worst").tail(topN)], axis=0)
    labels = head["gen_id"].astype(str).values
    idx = np.arange(len(labels))
    plt.figure(figsize=(12,8))
    plt.barh(idx-0.15, head["Def_A"].values, height=0.3, label="Cost recovery pot minimum")
    plt.barh(idx+0.15, head["Def_B"].values, height=0.3, label="LMP pot equalisation")
    plt.yticks(idx, labels); plt.xlabel("Surplus / Deficit (allocated - modelled costs)")
    plt.title(f"Top ±{topN} Surplus/Deficit — two approaches"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_payback_three(per: pd.DataFrame, out_png: str, cap_years: float = 50.0):
    df, bounds = _ordered_by_tech(per)
    if df.empty: return
    tech_col = "tech" if "tech" in df.columns else ("tech_x" if "tech_x" in df.columns else "tech_y")
    labels = _label_with_tech(df["gen_id"].values, df[tech_col].values)
    x = np.arange(len(labels)); width = 0.25
    exp = df["payback_years"].values
    a   = df["Actual_payback_A"].values
    b   = df["Actual_payback_B"].values
    a_fin = np.isfinite(a); b_fin = np.isfinite(b)
    a_plot = np.where(a_fin, np.minimum(a, cap_years), cap_years)
    b_plot = np.where(b_fin, np.minimum(b, cap_years), cap_years)
    plt.figure(figsize=(max(14, 0.36*len(labels)+4), 8)); ax = plt.gca()
    c_expected, c_costrec, c_lmp = "#9c9c9c", "#1f77b4", "#2ca02c"
    ax.bar(x - width, exp,    width, label="Assumed expected payback period", color=c_expected, alpha=0.65, hatch="//", edgecolor="k", linewidth=0.3)
    barsA = ax.bar(x,         a_plot, width, label="Actual (Cost recovery pot minimum)", color=c_costrec, alpha=0.95, edgecolor="k", linewidth=0.3)
    barsB = ax.bar(x + width, b_plot, width, label="Actual (LMP pot equalisation)",      color=c_lmp,     alpha=0.85, edgecolor="k", linewidth=0.3)
    for i, ok in enumerate(a_fin):
        if not ok: barsA[i].set_hatch("xx"); ax.text(x[i], a_plot[i]+0.6, "∞", ha="center", va="bottom", fontsize=9)
    for i, ok in enumerate(b_fin):
        if not ok: barsB[i].set_hatch("xx"); ax.text(x[i]+width, b_plot[i]+0.6, "∞", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Years"); ax.set_title("Assumed vs actual payback — two allocation approaches")
    ymax = np.nanmax([np.nanmax(exp[np.isfinite(exp)]) if np.isfinite(exp).any() else 0, np.nanmax(a_plot), np.nanmax(b_plot)])
    ax.set_ylim(0, max(cap_years*1.1, (ymax if np.isfinite(ymax) else cap_years)*1.12))
    _draw_group_shading(ax, bounds, ymax if np.isfinite(ymax) else cap_years)
    ax.legend(loc="upper left"); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

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
    # Plots
    ap.add_argument("--make-plots", action="store_true", default=True)
    ap.add_argument("--topN", type=int, default=20)
    ap.add_argument("--payback-cap", type=float, default=50.0)
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

    # Shapley norms (HALF-HOURLY) — read EXACT file
    df_shap = read_shapley_allocations_exact(args.long_shapley)
    log(f"Shapley rows: {len(df_shap):,}")

    # Timing weights & allocation weights
    timing, elig_weights, _all_weights = build_timing_and_weights(df_shap, tech_df)
    log(f"Timestamps with weights: {len(timing):,}")
    log(f"Eligible rows (gas+battery): {len(elig_weights):,}")

    # Costs & fixed buckets (preserve payback_years!)
    costs_enriched, wind_fixed_total, nuclear_fixed_total, BASE_POT, wind_per, nuclear_per = \
        read_fixed_costs_with_payback(args.costs_csv, tech_df)

    # LMP & AMM settlements (magnitudes)
    lmp_e, lmp_r = read_lmp_totals(args.lmp_totals)
    amm_e, amm_r = read_amm_settlement_totals(args.amm_settlement)

    # ---- Identity capacity definition ----
    # Known (guaranteed) capacity payments to W+N:
    WN_FIXED = float(wind_fixed_total + nuclear_fixed_total)

    # Make the identity hold: (WN_FIXED + DELTA) + AMM_E + AMM_R == LMP_E + LMP_R
    DELTA_RAW = float((lmp_e + lmp_r) - (amm_e + amm_r) - WN_FIXED)
    DELTA_POT = DELTA_RAW if args.allow_negative_delta else max(0.0, DELTA_RAW)

    log("[RECON] --- Pot reconciliation (per identity) ---")
    log(f"[RECON] LMP totals (E+R):                 £{(lmp_e + lmp_r):,.2f}")
    log(f"[RECON] AMM settlements (E+R):            £{(amm_e + amm_r):,.2f}")
    log(f"[RECON] W+N fixed (capacity guaranteed):  £{WN_FIXED:,.2f}")
    log(f"[RECON] DELTA_POT (after clamp):          £{DELTA_POT:,.2f}")

    # -------- Spread pots over time (HALF-HOURLY) --------
    timing = timing.copy()

    # (1) BASE (for audit): show full BASE over time (AMM-1)
    timing["pot_t_BASE_total"] = timing["w_t"] * BASE_POT

    # (2) Wind & Nuclear fixed paid inside BASE (time-spread by w_t)
    wind_ts = pd.DataFrame(columns=["timestamp","gen_id","revenue_gbp"])
    if not wind_per.empty:
        wind_ts = (timing[["timestamp","w_t"]].assign(key=1)
                   .merge(wind_per.assign(key=1), on="key", how="outer")
                   .drop("key", axis=1))
        wind_ts["revenue_gbp"] = wind_ts["w_t"] * wind_ts["wind_fixed_payment"]
        wind_ts = wind_ts[["timestamp","gen_id","revenue_gbp"]]

    nuclear_ts = pd.DataFrame(columns=["timestamp","gen_id","revenue_gbp"])
    if not nuclear_per.empty:
        nuclear_ts = (timing[["timestamp","w_t"]].assign(key=1)
                      .merge(nuclear_per.assign(key=1), on="key", how="outer")
                      .drop("key", axis=1))
        nuclear_ts["revenue_gbp"] = nuclear_ts["w_t"] * nuclear_ts["nuclear_fixed_payment"]
        nuclear_ts = nuclear_ts[["timestamp","gen_id","revenue_gbp"]]

    # (3) Remaining BASE for eligible (gas+battery) after paying wind+nuclear
    base_remaining_total = max(0.0, BASE_POT - WN_FIXED)
    timing["pot_t_BASE_remaining"] = timing["w_t"] * base_remaining_total

    # (4) DELTA: timeseries allocated ONLY to eligible (gas+battery)
    timing["pot_t_DELTA"] = timing["w_t"] * DELTA_POT

    # Allocate remaining BASE and DELTA to eligible using Shapley weights
    alloc_base_rem = allocate_pot_per_timestamp(elig_weights, timing, "pot_t_BASE_remaining")
    alloc_delta    = allocate_pot_per_timestamp(elig_weights, timing, "pot_t_DELTA")

    # -------- Merge allocations → strictly half-hourly outputs --------
    ts_base = pd.concat(
        [
            alloc_base_rem.rename(columns={"alloc_pot_t_BASE_remaining":"revenue_gbp"})[["timestamp","gen_id","revenue_gbp"]],
            wind_ts,
            nuclear_ts
        ],
        ignore_index=True
    ).groupby(["timestamp","gen_id"], as_index=False)["revenue_gbp"].sum().sort_values(["timestamp","gen_id"])

    ts_delta = (alloc_delta
                .rename(columns={"alloc_pot_t_DELTA":"revenue_gbp"})
                [["timestamp","gen_id","revenue_gbp"]]
               ).groupby(["timestamp","gen_id"], as_index=False)["revenue_gbp"].sum().sort_values(["timestamp","gen_id"])

    # -------- Persist per-timestamp time series --------
    pA   = os.path.join(RUN_DIR, "generator_revenue_timeseries_BASE.csv")
    pB   = os.path.join(RUN_DIR, "generator_revenue_timeseries_DELTA.csv")
    pAll = os.path.join(RUN_DIR, "generator_revenue_timeseries_ALL.csv")
    ts_base.to_csv(pA,  index=False)
    ts_delta.to_csv(pB,  index=False)
    pd.concat(
        [ts_base.assign(approach="BASE"), ts_delta.assign(approach="DELTA")],
        ignore_index=True
    ).sort_values(["timestamp","gen_id","approach"]).to_csv(pAll, index=False)

    # --- Tripwire: keep original half-hour grid ---
    in_tsu    = pd.DatetimeIndex(df_shap["timestamp"].drop_duplicates()).sort_values()
    out_tsu_A = pd.DatetimeIndex(ts_base["timestamp"].drop_duplicates()).sort_values()
    in_counts  = in_tsu.to_series().resample("D").count()
    out_counts = out_tsu_A.to_series().resample("D").count()
    if len(in_counts) and len(out_counts):
        log(f"[CHECK] First day input steps:  {int(in_counts.iloc[0])}")
        log(f"[CHECK] First day output steps: {int(out_counts.iloc[0])}")
        if int(out_counts.iloc[0]) < int(in_counts.iloc[0]):
            first_day = out_tsu_A[0].normalize()
            sample_out = out_tsu_A[(out_tsu_A>=first_day) & (out_tsu_A<first_day+pd.Timedelta('1D'))].to_pydatetime()
            sample_in  = in_tsu[(in_tsu>=first_day) & (in_tsu<first_day+pd.Timedelta('1D'))].to_pydatetime()
            raise RuntimeError(
                f"Output lost granularity. IN ({len(sample_in)}): {sample_in[:48]} ... "
                f"OUT ({len(sample_out)}): {sample_out[:48]} ..."
            )

    # -------- Annual totals per generator (from time series) --------
    gen_base_tot  = ts_base.groupby("gen_id", as_index=False)["revenue_gbp"].sum() \
                          .rename(columns={"revenue_gbp":"R_cost_recovery_min"})
    gen_delta_tot_timeseries = ts_delta.groupby("gen_id", as_index=False)["revenue_gbp"].sum() \
                                      .rename(columns={"revenue_gbp":"R_lmp_equalisation_timeseries"})

    # -------- Build per-generator summary for payback --------
    costs = costs_enriched.copy()
    costs["Annual_required_cost"] = costs["OpEx_nonfuel_total"].fillna(0.0) + costs["CapEx_per_year"].fillna(0.0)

    # Keep a single 'tech' (prefer from gens_static)
    costs_for_merge = costs.drop(columns=["tech"], errors="ignore")
    left_gens = read_gens_static(GENS_STATIC_DEFAULT) if os.path.isfile(GENS_STATIC_DEFAULT) else tech_df

    per = (left_gens
        .merge(costs_for_merge, on="gen_id", how="outer")
        .merge(gen_base_tot, on="gen_id", how="left")
        .merge(gen_delta_tot_timeseries, on="gen_id", how="left"))

    # Final Option B totals for payback:
    #   - wind & nuclear:   B = fixed = Annual_required_cost
    #   - gas & battery:    B = DELTA allocation (from timeseries)
    tech_series = per["tech"].astype(str).str.lower()
    is_wind_nuc = tech_series.isin({WIND_TECH, NUCLEAR_TECH})
    per["R_lmp_equalisation"] = np.where(
        is_wind_nuc,
        per["Annual_required_cost"].fillna(0.0),
        per["R_lmp_equalisation_timeseries"].fillna(0.0)
    )

    # Clean numerics
    for c in ["R_cost_recovery_min","R_lmp_equalisation","Annual_required_cost",
              "OpEx_nonfuel_total","CapEx_per_year","payback_years"]:
        if c in per.columns:
            per[c] = pd.to_numeric(per[c], errors="coerce")

    # Payback calcs
    if "payback_years" not in per.columns:
        per["payback_years"] = np.nan
    per["CapEx_total"] = np.where(per["payback_years"].notna(),
                                  per["CapEx_per_year"] * per["payback_years"], np.nan)
    eps = 1e-9
    per["Net_after_OpEx_A"] = per["R_cost_recovery_min"].fillna(0.0) - per["OpEx_nonfuel_total"].fillna(0.0)
    per["Net_after_OpEx_B"] = per["R_lmp_equalisation"].fillna(0.0)  - per["OpEx_nonfuel_total"].fillna(0.0)
    per["Actual_payback_A"] = np.where((per["CapEx_total"].notna()) & (per["Net_after_OpEx_A"]>eps),
                                       per["CapEx_total"]/per["Net_after_OpEx_A"], np.inf)
    per["Actual_payback_B"] = np.where((per["CapEx_total"].notna()) & (per["Net_after_OpEx_B"]>eps),
                                       per["CapEx_total"]/per["Net_after_OpEx_B"], np.inf)

    # Persist per-generator summary
    per.sort_values(["tech","gen_id"] if "tech" in per.columns else ["gen_id"]) \
       .to_csv(os.path.join(RUN_DIR, "per_generator_summary.csv"), index=False)

    # -------- Figures --------
    if args.make_plots:
        plot_scatter_two(per, os.path.join(RUN_DIR, "fig_revenue_vs_required_scatter_two.png"))
        plot_top_surplus_deficit_two(per, os.path.join(RUN_DIR, "fig_surplus_deficit_topN_two.png"), topN=args.topN)
        plot_revenue_multiseries(per, os.path.join(RUN_DIR, "fig_all_generators_revenue_expected_vs_two.png"))
        plot_payback_three(per, os.path.join(RUN_DIR, "fig_all_generators_payback_expected_vs_two.png"), cap_years=args.payback_cap)

    # -------- Console totals & cross-checks --------
    totA = float(per["R_cost_recovery_min"].sum())
    # For B totals: sum of (W+N fixed) + (gas+battery delta timeseries)
    wn_fixed_sum = float(per.loc[is_wind_nuc, "Annual_required_cost"].fillna(0.0).sum())
    gb_delta_sum = float(per.loc[~is_wind_nuc, "R_lmp_equalisation_timeseries"].fillna(0.0).sum())
    totB = wn_fixed_sum + gb_delta_sum
    log(f"[OK] Totals — Cost-recovery (BASE) = £{totA:,.2f}  LMP-equalisation (Option B) = £{totB:,.2f}")

    # Expected sums from construction (time series checks)
    exp_base  = BASE_POT
    exp_delta = DELTA_POT
    sum_ts_base  = float(ts_base["revenue_gbp"].sum())
    sum_ts_delta = float(ts_delta["revenue_gbp"].sum())
    log(f"[CHK] Expected BASE total:  £{exp_base:,.2f}  vs BASE timeseries:  £{sum_ts_base:,.2f}")
    log(f"[CHK] Expected DELTA total: £{exp_delta:,.2f}  vs DELTA timeseries: £{sum_ts_delta:,.2f}")

    # -------- Reconciliation CSV (explicit identity) --------
    cap_for_identity = wn_fixed_sum  # equals WN_FIXED numerically; using per table for transparency
    lmp_side  = (lmp_e + lmp_r)
    amm_side  = cap_for_identity + DELTA_POT + (amm_e + amm_r)
    identity_diff = amm_side - lmp_side

    recon_rows = [
        {"item": "LMP_energy_total",                      "value_gbp": lmp_e},
        {"item": "LMP_reserve_total",                     "value_gbp": lmp_r},
        {"item": "AMM_energy_settlement_total",           "value_gbp": amm_e},
        {"item": "AMM_reserve_settlement_total",          "value_gbp": amm_r},
        {"item": "Wind_fixed_total",                      "value_gbp": wn_fixed_sum},
        {"item": "Nuclear_fixed_total",                   "value_gbp": float(per.loc[per['tech'].astype(str).str.lower()==NUCLEAR_TECH, 'Annual_required_cost'].fillna(0.0).sum())},
        {"item": "BASE_POT_all_fixed",                    "value_gbp": BASE_POT},
        {"item": "DELTA_raw_before_clamp",                "value_gbp": DELTA_RAW},
        {"item": "DELTA_POT_used",                        "value_gbp": DELTA_POT},
        {"item": "Allocated_BASE_timeseries_sum",         "value_gbp": sum_ts_base},
        {"item": "Allocated_DELTA_timeseries_sum",        "value_gbp": sum_ts_delta},
        {"item": "OptionB_total_for_payback",             "value_gbp": totB},

        # Identity sides:
        {"item": "IDENTITY_AMM_SIDE_capacityEplusR",      "value_gbp": amm_side},
        {"item": "IDENTITY_LMP_SIDE_EplusR",              "value_gbp": lmp_side},
        {"item": "IDENTITY_DIFF_AMM_minus_LMP",           "value_gbp": identity_diff},
    ]
    pd.DataFrame(recon_rows).to_csv(os.path.join(RUN_DIR, "reconciliation_totals.csv"), index=False)

    log(f"[OK] Identity check — AMM side: £{amm_side:,.2f}  vs LMP side: £{lmp_side:,.2f}  (diff=£{identity_diff:,.2f})")
    log(f"[OK] Wrote outputs to: {RUN_DIR}")

if __name__ == "__main__":
    main()
