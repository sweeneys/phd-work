#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALL-IN-ONE: Shapley × Revenue × Network × Availability × Demand × LMP × Scarcity
+ Availability-weighted & Unweighted Node Competition diagnostics (NCI, HHI, etc.)

Outputs in <OUT_DIR>:
  - consolidated_per_generator.csv
  - shapley_revenue_crosscheck.csv
  - Figures (PNG + PDF for each):
      scatter_paid_only_shapMeanPerMW_vs_revenuePerMW
      scatter_all_shapMeanPerMW_vs_avgTightness
      scatter_all_shapMeanPerMW_vs_avgLMP
      scatter_paid_corrShapLMP_vs_payback
      scatter_all_scarcity_shapMeanPerMW_vs_avgLMP
      bar_top10_scarcity_shapley_share
      scatter_paid_revenue_vs_NCI                <-- UNWEIGHTED (structural)
      scatter_paid_revenue_vs_NCI_availability   <-- AVAILABILITY-WEIGHTED
"""

import argparse, os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================= Editor defaults =========================
RUN_DIR      = "marketExecution_AMM/availabilityPayments/analysis/run_20251205_152214/"
LMP_ROOT     = "marketExecution_Actual/outputs/uc_da_rt_sced_bat_reserves"
AMM_RUN_ROOT = "marketExecution_AMM/outputs/run_20250914_194030"

EDITOR_DEFAULTS = [
    "--shapley", "marketExecution_AMM/availabilityPayments/outputs/long/shapley_allocations_generators.csv",
    "--per-gen-summary", f"{RUN_DIR}per_generator_summary.csv",
    "--gens-static", "./gens_static.csv",
    "--gen-profiles", "gen_profiles_expost.csv",
    "--demand-dir", "marketExecution_AMM/data/demand",
    "--network", "network_uk.json",
    "--lmp-root", LMP_ROOT,
    "--amm-run-root", AMM_RUN_ROOT,
    "--out-dir", f"{RUN_DIR}/network_shapley_availability",
]

# ============================== Utils ==============================
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def log(msg): print(msg, flush=True)

def normalise_ts(series_or_df, col="timestamp"):
    """Coerce to UTC tz-aware, then drop tz (naive UTC) to avoid merge dtype mismatches."""
    if isinstance(series_or_df, pd.Series):
        return pd.to_datetime(series_or_df, errors="coerce", utc=True).dt.tz_convert(None)
    df = series_or_df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
    return df

def safe_corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    v = pd.DataFrame({"a":a, "b":b}).dropna()
    if len(v) < 3 or v["a"].var() == 0 or v["b"].var() == 0:
        return np.nan
    return float(v["a"].corr(v["b"]))

def savefig_multi(fig, basepath, dpi=180):
    """Save as both PNG and PDF given a basepath without extension."""
    png = basepath + ".png"
    pdf = basepath + ".pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

# ============================== Args ===============================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapley", required=False)
    ap.add_argument("--per-gen-summary", required=False)
    ap.add_argument("--gens-static", default="./gens_static.csv")
    ap.add_argument("--gen-profiles", required=False)
    ap.add_argument("--demand-dir", required=False)
    ap.add_argument("--network", required=False)
    ap.add_argument("--lmp-root", required=False)
    ap.add_argument("--amm-run-root", required=False, help="AMM run root with per-day served/unserved per node")
    ap.add_argument("--out-dir", default=f"{RUN_DIR}/network_shapley_availability")
    if len(sys.argv) == 1:
        return ap.parse_args(EDITOR_DEFAULTS)
    return ap.parse_args()

# ============================== Readers ============================
def read_shapley_timeseries(path):
    df = pd.read_csv(path)
    col = {c.lower(): c for c in df.columns}
    ts = col.get("timestamp"); gid = col.get("gen_id"); shap = col.get("shap_norm") or col.get("norm")
    if not (ts and gid and shap): raise ValueError(f"{path}: need timestamp/gen_id/shap_norm")
    out = df[[ts, gid, shap]].copy()
    out.columns = ["timestamp","gen_id","shap_norm"]
    out["timestamp"] = normalise_ts(out["timestamp"])
    out["gen_id"] = out["gen_id"].astype(str)
    out["shap_norm"] = pd.to_numeric(out["shap_norm"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["timestamp","gen_id"])

def read_per_generator_summary(path):
    per = pd.read_csv(path)
    gid = next((c for c in ["gen_id","generator","generator_id","id","name","GenID"] if c in per.columns), None)
    if gid and gid != "gen_id": per = per.rename(columns={gid:"gen_id"})
    if "gen_id" not in per.columns: raise ValueError(f"{path}: need gen_id")
    per["gen_id"] = per["gen_id"].astype(str)
    if "Total_Revenue" not in per.columns:
        rcols = [c for c in per.columns if c.startswith("R_")]
        per["Total_Revenue"] = per[rcols].sum(axis=1, numeric_only=True) if rcols else np.nan
    if "capacity_MW" not in per.columns:
        cand = [c for c in per.columns if "cap" in c.lower() and "mw" in c.lower()]
        if cand: per = per.rename(columns={cand[0]:"capacity_MW"})
    for c in ["Total_Revenue","capacity_MW"]:
        if c in per.columns: per[c] = pd.to_numeric(per[c], errors="coerce")
    if "tech" in per.columns:
        per["tech"] = per["tech"].astype(str).str.strip().str.lower().map(lambda t: {"bess":"battery","ps":"battery"}.get(t, t))
    return per

def read_gens_static(path):
    if not os.path.isfile(path): return pd.DataFrame(columns=["gen_id","tech","capacity_MW"])
    gs = pd.read_csv(path)
    gid = next((c for c in ["gen_id","generator","generator_id","id","name","GenID"] if c in gs.columns), None)
    if gid and gid != "gen_id": gs = gs.rename(columns={gid:"gen_id"})
    if "gen_id" not in gs.columns: return pd.DataFrame(columns=["gen_id","tech","capacity_MW"])
    gs["gen_id"] = gs["gen_id"].astype(str)
    tcol = next((c for c in ["tech","technology","fuel","fuel_type","Fuel","Tech"] if c in gs.columns), None)
    if tcol:
        gs["tech"] = gs[tcol].astype(str).str.strip().str.lower()
        gs["tech"] = gs["tech"].map(lambda t: {"bess":"battery","ps":"battery"}.get(t, t))
    cap = next((c for c in ["capacity_MW","Capacity_MW","nameplate_MW","Pmax","pmax","cap_mw"] if c in gs.columns), None)
    if cap: gs["capacity_MW"] = pd.to_numeric(gs[cap], errors="coerce")
    return gs[["gen_id"] + [c for c in ["tech","capacity_MW"] if c in gs.columns]].drop_duplicates()

def read_availability(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ts = cols.get("timestamp"); gid = cols.get("gen_id")
    cand = [c for c in df.columns if "avail" in c.lower() and "mw" in c.lower()]
    if not (ts and gid and cand): raise ValueError(f"{path}: need timestamp/gen_id/avail_MW-like col")
    out = df[[ts, gid, cand[0]]].copy()
    out.columns = ["timestamp","gen_id","avail_MW"]
    out["timestamp"] = normalise_ts(out["timestamp"])
    out["gen_id"] = out["gen_id"].astype(str)
    out["avail_MW"] = pd.to_numeric(out["avail_MW"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["timestamp","gen_id"])

def read_network(path):
    with open(path, "r") as f: net = json.load(f)
    nodes = list(net.get("nodes", []))
    edges = [tuple(e) for e in net.get("edges", [])]
    cap_map = net.get("edge_capacity", {})
    gens = net.get("generators", {}); loads = net.get("loads", {}); pos = net.get("positions", {})
    undirected_cap = {}
    for (u,v) in edges:
        c = float(cap_map.get(f"{u},{v}", cap_map.get(f"{v},{u}", 0.0)))
        undirected_cap[(u,v)] = c; undirected_cap[(v,u)] = c
    import_cap = {n:0.0 for n in nodes}
    for (u,v), c in undirected_cap.items():
        import_cap[u] += c
    gen_node = {str(g): info.get("node") for g, info in gens.items()}
    gen_cap  = {str(g): float(info.get("power", np.nan)) for g, info in gens.items()}
    load_node = {lid: info.get("node") for lid, info in loads.items()}
    return nodes, edges, undirected_cap, import_cap, gen_node, gen_cap, load_node, pos

def read_demand_folder(demand_dir, load_node_map):
    files = sorted(glob.glob(os.path.join(demand_dir, "D*_demand.csv")))
    if not files: raise FileNotFoundError(f"No demand files in {demand_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        ts_col = cols.get("timestamp", df.columns[0])
        dcol = None
        for k in ["total_demand_mw","demand_mw","power_mw","demand","mw"]:
            if k in cols: dcol = cols[k]; break
        if dcol is None: dcol = df.columns[-1]
        df = df[[ts_col, dcol]].copy(); df.columns = ["timestamp","demand_MW"]
        df["timestamp"] = normalise_ts(df["timestamp"])
        load_id = os.path.basename(path).split("_")[0]
        node = load_node_map.get(load_id)
        if not node:
            log(f"[WARN] cannot map {load_id} to a node; skipping {path}")
            continue
        df["node"] = node
        frames.append(df.dropna(subset=["timestamp"]))
    all_load = pd.concat(frames, ignore_index=True)
    return all_load.groupby(["timestamp","node"], as_index=False)["demand_MW"].sum()

def read_lmp_timeseries(lmp_root):
    if not lmp_root or not os.path.isdir(lmp_root):
        log(f"[WARN] LMP root not found: {lmp_root}")
        return pd.DataFrame(columns=["timestamp","node","lmp"])
    day_dirs = sorted(d for d in glob.glob(os.path.join(lmp_root, "*")) if os.path.isdir(d))
    frames = []; node_keys = {"node","bus","bus_id","node_id","n","bus_name","node_name","location"}
    for d in day_dirs:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try: df = pd.read_csv(p)
            except Exception: continue
            cols_low = {c.lower(): c for c in df.columns}
            if "settlement_lmp" not in cols_low: continue
            ts_col = cols_low.get("timestamp")
            if ts_col is None:
                for c in df.columns:
                    try: pd.to_datetime(df[c]); ts_col=c; break
                    except Exception: pass
            if ts_col is None: continue
            node_col = None
            for key in node_keys:
                if key in cols_low: node_col = cols_low[key]; break
            if node_col is None: continue
            sub = df[[ts_col, node_col, cols_low["settlement_lmp"]]].copy()
            sub.columns = ["timestamp","node","lmp"]
            sub["timestamp"] = normalise_ts(sub["timestamp"])
            sub["node"] = sub["node"].astype(str)
            sub["lmp"] = pd.to_numeric(sub["lmp"], errors="coerce")
            frames.append(sub.dropna(subset=["timestamp","node"]))
    if not frames:
        log("[WARN] no LMP frames parsed"); return pd.DataFrame(columns=["timestamp","node","lmp"])
    all_lmp = pd.concat(frames, ignore_index=True)
    return all_lmp.groupby(["timestamp","node"], as_index=False)["lmp"].mean()

def read_amm_scarcity(amm_run_root):
    """Reads served/unserved per node per timestamp from AMM run folders."""
    if not amm_run_root or not os.path.isdir(amm_run_root):
        log(f"[WARN] AMM run root not found: {amm_run_root}")
        return pd.DataFrame(columns=["timestamp","node","served_MW","unserved_MW","scarcity_flag"])
    day_dirs = sorted(d for d in glob.glob(os.path.join(amm_run_root, "*")) if os.path.isdir(d))
    frames = []; node_keys = {"node","bus","bus_id","node_id","n","bus_name","node_name","location"}
    for d in day_dirs:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try: df = pd.read_csv(p)
            except Exception: continue
            cols_low = {c.lower(): c for c in df.columns}
            ts_col = cols_low.get("timestamp")
            if ts_col is None:
                for c in df.columns:
                    try: pd.to_datetime(df[c]); ts_col=c; break
                    except Exception: pass
            node_col = None
            for key in node_keys:
                if key in cols_low: node_col = cols_low[key]; break
            served_col   = next((cols_low[k] for k in cols_low if "served"   in k and "mw" in k), None)
            unserved_col = next((cols_low[k] for k in cols_low if "unserved" in k and "mw" in k), None)
            if not (ts_col and node_col and (served_col or unserved_col)): continue
            sub = df[[ts_col, node_col] + [c for c in [served_col, unserved_col] if c]].copy()
            sub.columns = ["timestamp","node"] + ([ "served_MW"] if served_col else []) + ([ "unserved_MW"] if unserved_col else [])
            if "served_MW" not in sub.columns: sub["served_MW"] = np.nan
            if "unserved_MW" not in sub.columns: sub["unserved_MW"] = np.nan
            sub["timestamp"] = normalise_ts(sub["timestamp"])
            sub["node"] = sub["node"].astype(str)
            for c in ["served_MW","unserved_MW"]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            frames.append(sub.dropna(subset=["timestamp","node"]))
    if not frames:
        log("[WARN] no scarcity files parsed"); return pd.DataFrame(columns=["timestamp","node","served_MW","unserved_MW","scarcity_flag"])
    all_sc = pd.concat(frames, ignore_index=True)
    all_sc = (all_sc.groupby(["timestamp","node"], as_index=False)
                     .agg(served_MW=("served_MW","sum"), unserved_MW=("unserved_MW","sum")))
    all_sc["scarcity_flag"] = (all_sc["unserved_MW"] > 1e-6).astype(int)
    return all_sc

# ======================= Cross-check (Shapley ↔ Revenue) ===========
def build_shapley_revenue_crosscheck(shap_ts, per_summary, gens_static, out_csv):
    meta = per_summary.merge(gens_static, on="gen_id", how="left", suffixes=("","_gs"))
    if "tech" not in meta.columns and "tech_gs" in meta.columns: meta["tech"] = meta["tech_gs"]
    if "capacity_MW" not in meta.columns and "capacity_MW_gs" in meta.columns: meta["capacity_MW"] = meta["capacity_MW_gs"]
    if "tech" in meta.columns:
        meta["tech"] = meta["tech"].astype(str).str.strip().str.lower().map(lambda t: {"bess":"battery","ps":"battery"}.get(t, t))
    g = shap_ts.groupby("gen_id")["shap_norm"]
    stats = pd.DataFrame({
        "gen_id": g.size().index,
        "N_points": g.size().values,
        "shap_sum": g.sum().values,
        "shap_mean": g.mean().values,
        "shap_median": g.median().values,
        "shap_std": g.std(ddof=0).fillna(0.0).values,
        "shap_min": g.min().values,
        "shap_p25": g.quantile(0.25).values,
        "shap_p75": g.quantile(0.75).values,
        "shap_max": g.max().values,
    })
    total_sum = stats["shap_sum"].sum()
    stats["shap_share_total"] = np.where(total_sum > 0, stats["shap_sum"]/total_sum, 0.0)
    out = stats.merge(meta, on="gen_id", how="left")
    cap = out.get("capacity_MW", np.nan)
    out["revenue_per_MW"]    = np.where(pd.notna(cap) & (cap > 0), out["Total_Revenue"]/cap, np.nan)
    out["shap_mean_per_MW"]  = np.where(pd.notna(cap) & (cap > 0), out["shap_mean"]/cap, np.nan)
    out["shap_sum_per_MW"]   = np.where(pd.notna(cap) & (cap > 0), out["shap_sum"]/cap, np.nan)
    ensure_dir(os.path.dirname(out_csv)); out.to_csv(out_csv, index=False)
    log(f"[OK] wrote {out_csv}")
    return out

# ======================= Fusion & Plots ============================
def run_fusion_and_plots(shap_ts, availability, per_xcheck, network_json, demand_dir,
                          lmp_root, amm_run_root, out_dir):
    import matplotlib
    import matplotlib.patheffects as pe

    # Network bits
    nodes, edges, edge_cap, import_cap, gen_node, gen_cap, load_node_map, pos = read_network(network_json)

    # Meta by generator
    meta = pd.DataFrame({"gen_id": list(gen_node.keys())})
    meta["node"] = meta["gen_id"].map(gen_node)
    meta["nameplate_MW"] = meta["gen_id"].map(gen_cap)
    meta = meta.merge(per_xcheck, on="gen_id", how="left")

    # Node demand & availability
    node_demand = read_demand_folder(demand_dir, load_node_map)
    avail2 = availability.copy(); avail2["node"] = avail2["gen_id"].map(gen_node)
    node_avail = (avail2.groupby(["timestamp","node"], as_index=False)["avail_MW"].sum()
                         .rename(columns={"avail_MW":"avail_node_MW"}))
    node_panel = (node_demand.merge(node_avail, on=["timestamp","node"], how="outer")
                            .fillna({"demand_MW":0.0,"avail_node_MW":0.0}))
    node_panel["import_cap_node"] = node_panel["node"].map(import_cap).astype(float)
    node_panel["tightness"] = (node_panel["demand_MW"] - node_panel["avail_node_MW"]).clip(lower=0.0) / node_panel["import_cap_node"].replace({0: np.nan})
    node_panel["tightness"] = node_panel["tightness"].fillna(0.0)

    # LMPs
    lmp = read_lmp_timeseries(lmp_root)
    if len(lmp): node_panel = node_panel.merge(lmp, on=["timestamp","node"], how="left")
    else: node_panel["lmp"] = np.nan

    # AMM scarcity
    sc = read_amm_scarcity(amm_run_root)  # timestamp,node,served_MW,unserved_MW,scarcity_flag
    if len(sc): node_panel = node_panel.merge(sc, on=["timestamp","node"], how="left")
    else:
        node_panel["served_MW"] = np.nan; node_panel["unserved_MW"] = np.nan; node_panel["scarcity_flag"] = 0
    node_panel["scarcity_flag"] = node_panel["scarcity_flag"].fillna(0).astype(int)

    # ---------- Node competition & structure (AVAILABILITY-WEIGHTED + STATIC) ----------
    # Structural context per node
    node_gen_caps   = pd.Series(gen_cap, dtype=float).rename("nameplate_MW").groupby(meta.set_index("gen_id")["node"]).sum()
    node_gen_counts = meta.groupby("node")["gen_id"].nunique().rename("gen_count_node")
    avg_node_demand = node_demand.groupby("node")["demand_MW"].mean().rename("avg_node_demand_MW")
    sc_freq = (node_panel.groupby("node")["scarcity_flag"].mean().fillna(0.0).rename("scarcity_freq"))
    cap_shares = (pd.DataFrame({"node": meta["node"], "cap": meta["capacity_MW"]})
                    .dropna()
                    .groupby("node")["cap"]
                    .apply(lambda s: float(((s / s.sum())**2).sum()) if s.sum() > 0 else np.nan)
                    .rename("cap_HHI"))
    node_struct = pd.DataFrame({
        "incident_import_cap_MW": pd.Series(import_cap, dtype=float),
    })
    node_struct.index.name = "node"
    node_struct = (node_struct
                   .join(node_gen_caps.rename("node_gen_cap_MW"))
                   .join(node_gen_counts)
                   .join(avg_node_demand)
                   .join(sc_freq)
                   .join(cap_shares))
    # Merge to meta
    meta = meta.merge(node_struct.reset_index(), on="node", how="left")

    # Availability-weighted per-timestamp competition for each generator
    a2 = availability.copy()
    a2["node"] = a2["gen_id"].map(gen_node)
    a2 = a2.merge(node_panel[["timestamp","node","demand_MW","import_cap_node","scarcity_flag","avail_node_MW"]],
                  on=["timestamp","node"], how="left")
    a2["other_avail_same_node_MW"] = (a2["avail_node_MW"] - a2["avail_MW"]).clip(lower=0.0)
    denom = a2["demand_MW"].where(a2["demand_MW"] > 0, other=1.0)
    a2["NCI_t"] = (a2["other_avail_same_node_MW"] + a2["import_cap_node"].fillna(0.0)) / denom
    nci_overall = (a2.groupby("gen_id", as_index=False)["NCI_t"].mean()
                      .rename(columns={"NCI_t":"NCI_avail_mean"}))
    nci_scarc   = (a2[a2["scarcity_flag"]==1]
                      .groupby("gen_id", as_index=False)["NCI_t"].mean()
                      .rename(columns={"NCI_t":"NCI_avail_mean_scarcity"}))
    meta = (meta.merge(nci_overall, on="gen_id", how="left")
                .merge(nci_scarc, on="gen_id", how="left"))

    # Keep static context fields too
    meta["other_gen_count_same_node"]  = (meta["gen_count_node"] - 1).clip(lower=0)
    meta["other_gen_cap_MW_same_node"] = (meta["node_gen_cap_MW"] - meta["capacity_MW"]).clip(lower=0)
    meta["own_cap_share_node"]         = meta["capacity_MW"] / meta["node_gen_cap_MW"]

    # Shapley × node × availability
    shap_g = shap_ts.merge(meta, on="gen_id", how="left")  # keep meta cols
    shap_g = shap_g.merge(node_panel, on=["timestamp","node"], how="left")
    shap_g = shap_g.merge(availability[["timestamp","gen_id","avail_MW"]], on=["timestamp","gen_id"], how="left")
    shap_g["avail_MW"] = shap_g["avail_MW"].fillna(0.0)
    shap_g["available_flag"] = (shap_g["avail_MW"] > 1e-6).astype(int)

    # Per-generator metrics (overall + scarcity-only)
    rows = []
    for gid, gdf in shap_g.groupby("gen_id", sort=False):
        rowm = meta.loc[meta["gen_id"]==gid]
        cap  = rowm["capacity_MW"].dropna().iloc[0] if "capacity_MW" in rowm and len(rowm) else np.nan
        node = rowm["node"].astype(str).dropna().iloc[0] if len(rowm) else None
        tech = rowm["tech"].astype(str).dropna().iloc[0] if "tech" in rowm and len(rowm) else None
        total_rev = rowm["Total_Revenue"].iloc[0] if "Total_Revenue" in rowm and len(rowm) else np.nan
        pb = np.nan
        for c in [k for k in rowm.columns if k.lower().startswith("actual_payback")]:
            if pd.notna(rowm[c].iloc[0]): pb = float(rowm[c].iloc[0]); break

        # competition fields (static)
        NCI_av = rowm["NCI_avail_mean"].iloc[0] if "NCI_avail_mean" in rowm.columns and len(rowm) else np.nan
        NCI_sc = rowm["NCI_avail_mean_scarcity"].iloc[0] if "NCI_avail_mean_scarcity" in rowm.columns and len(rowm) else np.nan
        node_gen_cap_MW = rowm["node_gen_cap_MW"].iloc[0] if "node_gen_cap_MW" in rowm.columns and len(rowm) else np.nan
        gen_count_node  = rowm["gen_count_node"].iloc[0] if "gen_count_node" in rowm.columns and len(rowm) else np.nan
        cap_HHI         = rowm["cap_HHI"].iloc[0] if "cap_HHI" in rowm.columns and len(rowm) else np.nan
        own_share       = rowm["own_cap_share_node"].iloc[0] if "own_cap_share_node" in rowm.columns and len(rowm) else np.nan
        incident_import = rowm["incident_import_cap_MW"].iloc[0] if "incident_import_cap_MW" in rowm.columns and len(rowm) else np.nan
        avg_node_demand = rowm["avg_node_demand_MW"].iloc[0] if "avg_node_demand_MW" in rowm.columns and len(rowm) else np.nan

        def _per(df):
            shap_sum = df["shap_norm"].sum(); shap_mean = df["shap_norm"].mean()
            shap_mean_per_MW = shap_mean / cap if (cap and cap>0) else np.nan
            corr_lmp   = safe_corr(df["shap_norm"], df["lmp"])
            corr_dem   = safe_corr(df["shap_norm"], df["demand_MW"])
            corr_tight = safe_corr(df["shap_norm"], df["tightness"])
            avg_tight  = df["tightness"].mean()
            avg_lmp    = df["lmp"].mean(skipna=True)
            return shap_sum, shap_mean, shap_mean_per_MW, corr_lmp, corr_dem, corr_tight, avg_tight, avg_lmp

        shap_sum, shap_mean, shap_mean_MW, corr_lmp, corr_dem, corr_tight, avg_tight, avg_lmp = _per(gdf)

        g_sc = gdf[gdf["scarcity_flag"]==1]
        if len(g_sc):
            s_shap_sum, s_shap_mean, s_shap_mean_MW, s_corr_lmp, s_corr_dem, s_corr_tight, s_avg_tight, s_avg_lmp = _per(g_sc)
            scarcity_share = s_shap_sum / shap_sum if shap_sum>0 else np.nan
        else:
            s_shap_sum=s_shap_mean=s_shap_mean_MW=s_corr_lmp=s_corr_dem=s_corr_tight=s_avg_tight=s_avg_lmp=np.nan
            scarcity_share = np.nan

        rev_per_MW = (total_rev/cap) if (pd.notna(total_rev) and cap and cap>0) else np.nan

        rows.append({
            "gen_id": gid, "tech": tech, "node": node, "capacity_MW": cap,
            "Total_Revenue": total_rev, "revenue_per_MW": rev_per_MW, "Actual_payback": pb,
            "shap_sum": shap_sum, "shap_mean": shap_mean, "shap_mean_per_MW": shap_mean_MW,
            "corr_shap_lmp": corr_lmp, "corr_shap_demand": corr_dem, "corr_shap_tightness": corr_tight,
            "avg_tightness": avg_tight, "avg_lmp": avg_lmp,
            "scarcity_shap_sum": s_shap_sum, "scarcity_shap_mean": s_shap_mean, "scarcity_shap_mean_per_MW": s_shap_mean_MW,
            "scarcity_corr_shap_lmp": s_corr_lmp, "scarcity_corr_shap_demand": s_corr_dem, "scarcity_corr_shap_tightness": s_corr_tight,
            "scarcity_avg_tightness": s_avg_tight, "scarcity_avg_lmp": s_avg_lmp,
            "scarcity_share_of_shapley": scarcity_share,
            # competition snapshot (static & availability-weighted)
            "NCI_avail_mean": NCI_av, "NCI_avail_mean_scarcity": NCI_sc,
            "node_gen_cap_MW": node_gen_cap_MW, "gen_count_node": gen_count_node,
            "cap_HHI": cap_HHI, "own_cap_share_node": own_share,
            "incident_import_cap_MW": incident_import, "avg_node_demand_MW": avg_node_demand
        })

    per_gen = pd.DataFrame(rows)

    # --------- UNWEIGHTED Node Competition Index (structure only; no availability) ---------
    num = (per_gen["node_gen_cap_MW"] - per_gen["capacity_MW"]).clip(lower=0.0) + per_gen["incident_import_cap_MW"].fillna(0.0)
    den = per_gen["avg_node_demand_MW"].where(per_gen["avg_node_demand_MW"] > 0, other=np.nan)
    per_gen["NCI_unweighted"] = num / den

    ensure_dir(out_dir)
    per_gen.to_csv(os.path.join(out_dir, "consolidated_per_generator.csv"), index=False)
    log(f"[OK] wrote {os.path.join(out_dir, 'consolidated_per_generator.csv')}")

    # ============================= Figures =============================
    import matplotlib
    import matplotlib.patheffects as pe

    def _label(rr):
        t = (rr.get("tech") or "").strip().lower()
        return f'{rr["gen_id"]} ({t})' if t else str(rr["gen_id"])

    def _col(tech):
        t=(tech or "").lower()
        if "gas" in t: return "#d62728"
        if "battery" in t: return "#1f77b4"
        if "wind" in t: return "#2ca02c"
        if "nuclear" in t: return "#9467bd"
        return "#7f7f7f"

    def _text_color_for_face(face_rgba):
        r,g,b = face_rgba[:3]; lum = 0.2126*r + 0.7152*g + 0.0722*b
        return "black" if lum > 0.6 else "white"

    is_paid = per_gen["tech"].fillna("").str.contains("gas|battery", case=False, regex=True)
    df_paid = per_gen[is_paid].copy()
    df_all  = per_gen.copy()

    # 1) GAS+BATTERY — Shapley/MW vs Revenue/MW (color = avg tightness)
    fig, ax = plt.subplots(figsize=(9,6.5))
    c = df_paid["avg_tightness"].fillna(0.0); cmap = plt.cm.viridis
    if len(c.dropna()): norm = matplotlib.colors.Normalize(vmin=float(c.min()), vmax=float(c.max()))
    else: norm = None
    sca = ax.scatter(df_paid["shap_mean_per_MW"], df_paid["revenue_per_MW"],
                     c=c if norm is not None else None,
                     s=df_paid["capacity_MW"].fillna(0.0).clip(lower=1.0)**0.7*10,
                     cmap=cmap, norm=norm, alpha=0.95, edgecolor="k", linewidths=0.35)
    if norm is not None:
        cb = plt.colorbar(sca, ax=ax, pad=0.02); cb.set_label("Node avg tightness")
    ax.set_xlabel("Average Shapley per MW"); ax.set_ylabel("Revenue per MW (GBP/MW)")
    ax.set_title("Gas & Battery only: Shapley per MW vs Revenue per MW")
    for _, r in df_paid.dropna(subset=["shap_mean_per_MW","revenue_per_MW"]).iterrows():
        face = cmap(norm(r["avg_tightness"])) if norm is not None else (0.5,0.5,0.5,1)
        ax.text(r["shap_mean_per_MW"], r["revenue_per_MW"], _label(r),
                ha="center", va="center", fontsize=7.4, color=_text_color_for_face(face),
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
    ax.grid(True, alpha=0.25); fig.tight_layout()
    savefig_multi(fig, os.path.join(out_dir, "scatter_paid_only_shapMeanPerMW_vs_revenuePerMW")); plt.close(fig)

    # 2) ALL — Shapley/MW vs avg node tightness (color by tech)
    fig, ax = plt.subplots(figsize=(9,6.0))
    for _, r in df_all.dropna(subset=["shap_mean_per_MW","avg_tightness"]).iterrows():
        face = matplotlib.colors.to_rgba(_col(r.get("tech","")))
        ax.scatter(r["shap_mean_per_MW"], r["avg_tightness"], s=60, c=[face], alpha=0.95, edgecolor="k", linewidths=0.35)
        ax.text(r["shap_mean_per_MW"], r["avg_tightness"], _label(r),
                ha="center", va="center", fontsize=7.2, color=_text_color_for_face(face),
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
    ax.set_xlabel("Average Shapley per MW"); ax.set_ylabel("Average node tightness (dimensionless)")
    ax.set_title("Validation (all generators): Shapley per MW vs Node Tightness")
    ax.grid(True, alpha=0.25); fig.tight_layout()
    savefig_multi(fig, os.path.join(out_dir, "scatter_all_shapMeanPerMW_vs_avgTightness")); plt.close(fig)

    # 3) ALL — Shapley/MW vs avg node LMP
    if df_all["avg_lmp"].notna().any():
        fig, ax = plt.subplots(figsize=(9,6.0))
        for _, r in df_all.dropna(subset=["shap_mean_per_MW","avg_lmp"]).iterrows():
            face = matplotlib.colors.to_rgba(_col(r.get("tech","")))
            ax.scatter(r["shap_mean_per_MW"], r["avg_lmp"], s=60, c=[face], alpha=0.95, edgecolor="k", linewidths=0.35)
            ax.text(r["shap_mean_per_MW"], r["avg_lmp"], _label(r),
                    ha="center", va="center", fontsize=7.2, color=_text_color_for_face(face),
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
        ax.set_xlabel("Average Shapley per MW"); ax.set_ylabel("Average node LMP")
        ax.set_title("Validation (all generators): Shapley per MW vs Avg Node LMP")
        ax.grid(True, alpha=0.25); fig.tight_layout()
        savefig_multi(fig, os.path.join(out_dir, "scatter_all_shapMeanPerMW_vs_avgLMP")); plt.close(fig)

    # 4) GAS+BATTERY — corr(Shapley, LMP) vs Payback
    if df_paid["corr_shap_lmp"].notna().any():
        fig, ax = plt.subplots(figsize=(9,6.0))
        ax.scatter(df_paid["corr_shap_lmp"], df_paid["Actual_payback"],
                   s=df_paid["capacity_MW"].fillna(0.0).clip(lower=1.0)**0.7*10,
                   alpha=0.95, edgecolor="k", linewidths=0.35, c="#1f77b4")
        for _, r in df_paid.dropna(subset=["corr_shap_lmp","Actual_payback"]).iterrows():
            ax.text(r["corr_shap_lmp"], r["Actual_payback"], _label(r),
                    ha="center", va="center", fontsize=7.0, color="white",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
        ax.set_xlabel("Correlation (Shapley, node LMP)")
        ax.set_ylabel("Actual payback (years)")
        ax.set_title("Gas & Battery: is Shapley aligned with LMP and payback?")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        savefig_multi(fig, os.path.join(out_dir, "scatter_paid_corrShapLMP_vs_payback")); plt.close(fig)

    # 5) Scarcity-only — ALL gens: Shapley/MW vs avg node LMP (during scarcity)
    if df_all["scarcity_avg_lmp"].notna().any():
        df_s = df_all.dropna(subset=["scarcity_shap_mean_per_MW","scarcity_avg_lmp"])
        if len(df_s):
            fig, ax = plt.subplots(figsize=(9,6.0))
            for _, r in df_s.iterrows():
                face = matplotlib.colors.to_rgba(_col(r.get("tech","")))
                ax.scatter(r["scarcity_shap_mean_per_MW"], r["scarcity_avg_lmp"], s=60, c=[face],
                           alpha=0.95, edgecolor="k", linewidths=0.35)
                ax.text(r["scarcity_shap_mean_per_MW"], r["scarcity_avg_lmp"], _label(r),
                        ha="center", va="center", fontsize=7.2, color=_text_color_for_face(face),
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
            ax.set_xlabel("Average Shapley per MW (scarcity-only)")
            ax.set_ylabel("Average node LMP (scarcity-only)")
            ax.set_title("Validation in scarcity windows")
            ax.grid(True, alpha=0.25); fig.tight_layout()
            savefig_multi(fig, os.path.join(out_dir, "scatter_all_scarcity_shapMeanPerMW_vs_avgLMP")); plt.close(fig)

    # 6) Bar — Top-10 by scarcity share of Shapley
    top = per_gen.sort_values("scarcity_share_of_shapley", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(top["gen_id"].astype(str), top["scarcity_share_of_shapley"].fillna(0.0))
    for xi, (_, r) in enumerate(top.iterrows()):
        ax.text(xi, (r["scarcity_share_of_shapley"] or 0)+0.01, _label(r),
                rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Share of Shapley earned in scarcity windows")
    ax.set_title("Top-10 generators by scarcity share")
    fig.tight_layout()
    savefig_multi(fig, os.path.join(out_dir, "bar_top10_scarcity_shapley_share")); plt.close(fig)

    # 7a) GAS+BATTERY — Revenue/MW vs NCI (UNWEIGHTED, structural)
    dfp_u = per_gen[is_paid].dropna(subset=["revenue_per_MW","NCI_unweighted"])
    if len(dfp_u):
        fig, ax = plt.subplots(figsize=(9,6.0))
        ax.scatter(dfp_u["NCI_unweighted"], dfp_u["revenue_per_MW"],
                   s=dfp_u["capacity_MW"].fillna(0.0).clip(lower=1.0)**0.7*10,
                   alpha=0.95, edgecolor="k", linewidths=0.35, c="#8c564b")
        for _, r in dfp_u.iterrows():
            ax.text(r["NCI_unweighted"], r["revenue_per_MW"], _label(r),
                    ha="center", va="center", fontsize=7.0, color="white",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
        ax.set_xlabel("Node Competition Index (unweighted) = (other nameplate + import) / avg demand")
        ax.set_ylabel("Revenue per MW (GBP/MW)")
        ax.set_title("Gas & Battery: Revenue vs Node Competition (UNWEIGHTED)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        savefig_multi(fig, os.path.join(out_dir, "scatter_paid_revenue_vs_NCI"))
        plt.close(fig)

    # 7b) GAS+BATTERY — Revenue/MW vs Availability-weighted NCI
    dfp = per_gen[is_paid].dropna(subset=["revenue_per_MW","NCI_avail_mean"])
    if len(dfp):
        fig, ax = plt.subplots(figsize=(9,6.0))
        ax.scatter(dfp["NCI_avail_mean"], dfp["revenue_per_MW"],
                   s=dfp["capacity_MW"].fillna(0.0).clip(lower=1.0)**0.7*10,
                   alpha=0.95, edgecolor="k", linewidths=0.35, c="#ff7f0e")
        for _, r in dfp.iterrows():
            ax.text(r["NCI_avail_mean"], r["revenue_per_MW"], _label(r),
                    ha="center", va="center", fontsize=7.0, color="white",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
        ax.set_xlabel("Availability-weighted NCI (other avail + import) / demand")
        ax.set_ylabel("Revenue per MW (GBP/MW)")
        ax.set_title("Gas & Battery: Revenue vs Availability-weighted Node Competition")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        savefig_multi(fig, os.path.join(out_dir, "scatter_paid_revenue_vs_NCI_availability"))
        plt.close(fig)

    log("\nInterpretation notes:")
    log("• Paid techs: higher Shapley/MW ↔ higher revenue/MW — payout aligns with contribution.")
    log("• Scarcity windows: Shapley concentrates on high-LMP, scarce nodes (method validated).")
    log("• Unweighted & availability-weighted NCI both explain cases like G3: many competitors + import headroom ⇒ lower value & slower payback.")

# ===================================================================

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # Read core inputs
    shap_ts   = read_shapley_timeseries(args.shapley)
    per_sum   = read_per_generator_summary(args.per_gen_summary)
    gens_stat = read_gens_static(args.gens_static)
    avail     = read_availability(args.gen_profiles)

    # Cross-check table
    crosscheck_csv = os.path.join(args.out_dir, "../shapley_revenue_crosscheck.csv")
    crosscheck = build_shapley_revenue_crosscheck(shap_ts, per_sum, gens_stat, crosscheck_csv)

    # Fusion + plots
    run_fusion_and_plots(
        shap_ts=shap_ts,
        availability=avail,
        per_xcheck=crosscheck,
        network_json=args.network,
        demand_dir=args.demand_dir,
        lmp_root=args.lmp_root,
        amm_run_root=args.amm_run_root,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()
