#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, datetime as _dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Defaults =====================
AMM_RUN_ROOT_DEFAULT = "marketExecution_AMM/outputs/run_20250914_194030"   # per-day folders with dispatch.csv
# Directory that contains served_breakdown_D*.csv produced by compute_served_by_product.py
SERVED_DIR_DEFAULT   = "marketExecution_AMM/analysis/served_products"
GENS_STATIC_DEFAULT  = "./gens_static.csv"
GENREV_ALL_DEFAULT   = "marketExecution_AMM/availabilityPayments/analysis/run_20251205_152214/generator_revenue_timeseries_ALL.csv"
OUT_ROOT_DEFAULT     = os.path.join("marketExecution_AMM","availabilityPayments","analysis")

# Household totals (override as requested)
HOUSEHOLDS = {"P1": 19_000_000, "P2": 6_000_000, "P3": 2_500_000, "P4": 1_500_000}
# ====================================================

# ---------------- helpers ----------------
def _ensure_ts(df, col="timestamp"):
    if col not in df.columns:
        df = df.rename(columns={df.columns[0]: col})
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(None)
    return df

def _round_to_step(df: pd.DataFrame, col: str, step_minutes: int) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(None)
    df[col] = df[col].dt.round(f"{int(step_minutes)}min")
    return df

def _tech_to_class(tech: str) -> str:
    t = str(tech).strip().lower()
    return "U" if "wind" in t or "wnd" in t else "C"

def _to_gwh(mwh_val: float) -> float:
    try:
        return float(mwh_val) / 1e3
    except Exception:
        return 0.0

# -------------- served-demand loaders --------------
def load_served_breakdowns(served_dir: str, step_minutes: int):
    """
    Reads served_breakdown_D*.csv files produced by compute_served_by_product.py and returns:

    ppl_res_served: per-load residential per-product served power:
        [timestamp, demand_id, P1, P2, P3, P4, total_res_served_MW]

    res_share_served: system totals per timestamp:
        [timestamp, D_total_served_MW, D_nonres_served_MW, D_res_served_MW, f_res_served]
    """
    files = sorted(glob.glob(os.path.join(served_dir, "served_breakdown_D*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No served_breakdown_D*.csv found in {served_dir}. "
            f"Please run compute_served_by_product.py first."
        )

    rows_ppl = []
    rows_tot = []

    for fp in files:
        did = os.path.basename(fp).split(".")[0].split("_")[-1]  # Dk
        df = _ensure_ts(pd.read_csv(fp))
        df = _round_to_step(df, "timestamp", step_minutes)

        # Required columns as produced by helper script:
        need = {
            "timestamp","served_frac","served_MW","unserved_MW",
            "P1_served_MW","P2_served_MW","P3_served_MW","P4_served_MW",
            "nonres_served_MW","total_requested_MW","total_served_MW_calc"
        }
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{fp} is missing columns: {sorted(miss)}")

        # per-load residential served
        out_ppl = pd.DataFrame({
            "timestamp": df["timestamp"],
            "demand_id": did,
            "P1": pd.to_numeric(df["P1_served_MW"], errors="coerce").fillna(0.0),
            "P2": pd.to_numeric(df["P2_served_MW"], errors="coerce").fillna(0.0),
            "P3": pd.to_numeric(df["P3_served_MW"], errors="coerce").fillna(0.0),
            "P4": pd.to_numeric(df["P4_served_MW"], errors="coerce").fillna(0.0),
        })
        out_ppl["total_res_served_MW"] = out_ppl[["P1","P2","P3","P4"]].sum(axis=1)
        rows_ppl.append(out_ppl)

        # system totals contribution from this load
        out_tot = pd.DataFrame({
            "timestamp": df["timestamp"],
            "D_total_served_MW": pd.to_numeric(df["served_MW"], errors="coerce").fillna(0.0),
            "D_nonres_served_MW": pd.to_numeric(df["nonres_served_MW"], errors="coerce").fillna(0.0),
            "D_res_served_MW": pd.to_numeric(
                df["P1_served_MW"] + df["P2_served_MW"] + df["P3_served_MW"] + df["P4_served_MW"],
                errors="coerce"
            ).fillna(0.0),
        })
        out_tot["demand_id"] = did
        rows_tot.append(out_tot)

    ppl_res_served = (
        pd.concat(rows_ppl, ignore_index=True)
          .sort_values(["timestamp","demand_id"])
          .reset_index(drop=True)
    )

    tot_all = pd.concat(rows_tot, ignore_index=True)
    res_share_served = (
        tot_all.groupby("timestamp", as_index=False)
               .agg(
                    D_total_served_MW=("D_total_served_MW","sum"),
                    D_nonres_served_MW=("D_nonres_served_MW","sum"),
                    D_res_served_MW=("D_res_served_MW","sum"),
               )
    )
    # f_res based on *served* split
    res_share_served["f_res_served"] = np.where(
        res_share_served["D_total_served_MW"] > 0,
        res_share_served["D_res_served_MW"] / res_share_served["D_total_served_MW"],
        0.0
    )

    return ppl_res_served, res_share_served

def aggregate_res_products_from_per_load_served(ppl_res_served: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    d = _round_to_step(ppl_res_served.copy(), "timestamp", step_minutes=step_minutes)
    agg = (
        d.groupby("timestamp", as_index=False)
         .agg(P1=("P1","sum"), P2=("P2","sum"), P3=("P3","sum"), P4=("P4","sum"))
    )
    agg["D_res_MW"] = agg[["P1","P2","P3","P4"]].sum(axis=1)
    return agg.sort_values("timestamp")

# ---------- Tech / dispatch / U vs C ----------
def load_tech_map(gens_static_path: str) -> pd.DataFrame:
    df = pd.read_csv(gens_static_path)
    gid  = next((c for c in ["gen_id","generator","generator_id","id","name","GenID"] if c in df.columns), None)
    tcol = next((c for c in ["tech","technology","fuel","fuel_type","Fuel","Tech"] if c in df.columns), None)
    if gid is None or tcol is None:
        raise ValueError(f"{gens_static_path}: need gen_id + tech columns")
    m = df[[gid,tcol]].rename(columns={gid:"gen_id", tcol:"tech"})
    m["gen_id"] = m["gen_id"].astype(str)
    m["tech"]   = m["tech"].astype(str).str.strip().str.lower()
    m["class"]  = m["tech"].apply(_tech_to_class)
    return m

def read_dispatch_tree(run_root: str, step_minutes: int) -> pd.DataFrame:
    if "outputsrun_" in run_root.replace("\\","/"):
        run_root = run_root.replace("outputsrun_", "outputs/run_")

    cand = []
    cand += glob.glob(os.path.join(run_root, "*", "dispatch.csv"))
    cand += glob.glob(os.path.join(run_root, "**", "dispatch.csv"), recursive=True)
    files = sorted(set(cand))
    if not files:
        raise FileNotFoundError(f"No dispatch.csv under {run_root}")

    dfs=[]
    for fp in files:
        d = _ensure_ts(pd.read_csv(fp))
        gid = "gen_id" if "gen_id" in d.columns else next(
            (c for c in ["generator","generator_id","id","name","GenID"] if c in d.columns), None
        )
        if gid is None:
            raise ValueError(f"{fp}: no gen_id column")
        d = d.rename(columns={gid:"gen_id"})
        d = _round_to_step(d, "timestamp", step_minutes=step_minutes)
        for c in ["p_MW","cost_MWh","energy_cost_gbp"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        d["gen_id"] = d["gen_id"].astype(str)
        dfs.append(d)

    return pd.concat(dfs, ignore_index=True).sort_values(["timestamp","gen_id"])

def uc_timeseries_from_dispatch(dispatch: pd.DataFrame, tech_map: pd.DataFrame,
                                cap_total_df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    s = dispatch.merge(tech_map[["gen_id","class"]], on="gen_id", how="left")
    p = pd.to_numeric(s.get("p_MW", 0.0), errors="coerce").fillna(0.0)
    s["eff_gen_MW"] = p.clip(lower=0.0)

    uc = (
        s.groupby(["timestamp","class"], as_index=False)["eff_gen_MW"].sum()
         .pivot(index="timestamp", columns="class", values="eff_gen_MW")
         .fillna(0.0).rename(columns={"U":"U_MW","C":"C_MW"}).reset_index()
    )
    uc = _round_to_step(uc, "timestamp", step_minutes=step_minutes)

    cap = _round_to_step(cap_total_df.copy(), "timestamp", step_minutes=step_minutes)
    uc = uc.merge(cap, on="timestamp", how="left")
    cap_MW = pd.to_numeric(uc.get("cap_total_MW", np.nan), errors="coerce").fillna(np.inf)
    total  = uc["U_MW"] + uc["C_MW"]
    scale  = np.where(total > 0, np.minimum(1.0, cap_MW / total), 1.0)

    uc["U_MW"] = uc["U_MW"] * scale
    uc["C_MW"] = uc["C_MW"] * scale
    return uc[["timestamp","U_MW","C_MW"]]

# ------ per product splits of UC (RES only) — use global U/C shares per timestamp ------
def uc_per_product_from_global_uc_share(prod_res_served: pd.DataFrame,
                                        uc_totals: pd.DataFrame,
                                        dt_h: float,
                                        step_minutes: int):
    """
    For each timestamp t:
        share_U(t) = U_MW_total(t) / (U_MW_total(t) + C_MW_total(t))
        share_C(t) = 1 - share_U(t)

    Then for EACH residential product p in {P1..P4}:
        u_MW_p(t) = share_U(t) * P_p_served(t)
        c_MW_p(t) = share_C(t) * P_p_served(t)

    Returns two DataFrames:
        - per-product MW by timestamp (u_MW, c_MW)
        - per-product MWh by timestamp (u_served_MWh, c_served_MWh)
    """
    df_prod = _round_to_step(prod_res_served.copy(), "timestamp", step_minutes=step_minutes)
    df_uc   = _round_to_step(uc_totals[["timestamp","U_MW","C_MW"]].copy(), "timestamp", step_minutes=step_minutes)

    # ensure numeric
    for p in ["P1","P2","P3","P4","D_res_MW"]:
        if p in df_prod.columns:
            df_prod[p] = pd.to_numeric(df_prod[p], errors="coerce").fillna(0.0)
    df_uc[["U_MW","C_MW"]] = df_uc[["U_MW","C_MW"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    m = df_prod.merge(df_uc, on="timestamp", how="inner")
    total_uc = (m["U_MW"] + m["C_MW"]).replace(0.0, np.nan)
    m["share_U"] = (m["U_MW"] / total_uc).fillna(0.0).clip(0.0, 1.0)
    m["share_C"] = (m["C_MW"] / total_uc).fillna(0.0).clip(0.0, 1.0)

    rows_MW=[]; rows_MWh=[]
    for _, r in m.iterrows():
        sU = float(r["share_U"]); sC = float(r["share_C"])
        for p in ["P1","P2","P3","P4"]:
            Pp   = float(r.get(p, 0.0))
            u_mw = sU * Pp
            c_mw = sC * Pp
            rows_MW.append({
                "timestamp": r["timestamp"],
                "product": p,
                "u_MW": u_mw,
                "c_MW": c_mw
            })
            rows_MWh.append({
                "timestamp": r["timestamp"],
                "product": p,
                "u_served_MWh": u_mw * dt_h,
                "c_served_MWh": c_mw * dt_h
            })

    return pd.DataFrame(rows_MW), pd.DataFrame(rows_MWh)

# ---- fuel cost from dispatch (controllable only) ----
def fuel_cost_timeseries_from_dispatch(dispatch: pd.DataFrame,
                                       tech_map: pd.DataFrame,
                                       dt_h: float,
                                       step_minutes: int) -> pd.DataFrame:
    s = dispatch.merge(tech_map[["gen_id","class","tech"]], on="gen_id", how="left")
    s = _round_to_step(s, "timestamp", step_minutes=step_minutes)

    if "energy_cost_gbp" in s.columns and pd.to_numeric(
        s["energy_cost_gbp"], errors="coerce"
    ).fillna(0.0).abs().sum() > 0:
        s["_fuel_cost"] = pd.to_numeric(s["energy_cost_gbp"], errors="coerce").fillna(0.0)
    else:
        p     = pd.to_numeric(s.get("p_MW", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        price = pd.to_numeric(s.get("cost_MWh", 0.0), errors="coerce").fillna(0.0)
        s["_fuel_cost"] = p * dt_h * price

    sc = s[s["class"]=="C"].copy()
    ts = (
        sc.groupby("timestamp", as_index=False)
          .agg(total_fuel_cost_gbp=("_fuel_cost","sum"))
    )
    ts = _round_to_step(ts, "timestamp", step_minutes=step_minutes)
    return ts

# ---- revenue → class pots ----
def read_gen_rev_all(fp: str, approach_prefix: str, step_minutes: int) -> pd.DataFrame:
    """
    Read generator revenue timeseries and keep ALL rows whose 'approach'
    starts with the given prefix (e.g. BASE, BASE_RESERVE, BASE_*, etc.).
    This ensures that reserve-related revenue rows (e.g. BASE_RESERVE,
    DELTA_RESERVE) are fully included in the class pots.
    """
    df = _ensure_ts(pd.read_csv(fp))
    df = _round_to_step(df, "timestamp", step_minutes=step_minutes)

    need = {"timestamp","gen_id","approach","revenue_gbp"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{fp} missing {sorted(miss)}")

    df["gen_id"]      = df["gen_id"].astype(str)
    df["approach"]    = df["approach"].astype(str).str.upper()
    df["revenue_gbp"] = pd.to_numeric(df["revenue_gbp"], errors="coerce").fillna(0.0)

    # IMPORTANT: prefix match, to include reserves and any other sub-services
    mask = df["approach"].str.startswith(approach_prefix.upper())
    sel  = df.loc[mask, ["timestamp","gen_id","revenue_gbp"]]
    return sel.groupby(["timestamp","gen_id"], as_index=False)["revenue_gbp"].sum()

def class_pots_from_gen_revenue_ts(gen_rev: pd.DataFrame,
                                   tech_map: pd.DataFrame,
                                   step_minutes: int) -> pd.DataFrame:
    g = gen_rev.merge(tech_map[["gen_id","class"]], on="gen_id", how="left")
    g = _round_to_step(g, "timestamp", step_minutes=step_minutes)
    pots = (
        g.groupby(["timestamp","class"], as_index=False)["revenue_gbp"].sum()
         .pivot(index="timestamp", columns="class", values="revenue_gbp")
         .fillna(0.0).reindex(columns=["U","C"],fill_value=0.0)
    ).reset_index()
    return pots.rename(columns={"U":"pot_U_gbp","C":"pot_C_gbp"})

# ---- per-timestamp non-fuel (IndividualTS) ----
def build_costs_per_ts(
    uc_per_prod_MWh_res: pd.DataFrame,
    uc_per_prod_MW_res: pd.DataFrame,
    prod_res_df: pd.DataFrame,
    pots_ts_total: pd.DataFrame,
    fuel_ts_total: pd.DataFrame,
    uc_totals_MW: pd.DataFrame,
    f_res_df: pd.DataFrame,
    out_dir: str,
    tag: str,
    step_minutes: int,
    perHH_reserves_gbp_per_month: float = 0.0,
):
    """
    Non-fuel (CapEx/OpEx) allocated per TIMESTAMP using capacity-weighted controllable MW.
    Fuel allocated per TIMESTAMP using controllable energy shares.

    perHH_reserves_gbp_per_month:
        uniform per-household reserves charge, added on top of the base
        subscription (fuel + CapEx/OpEx) for each product.
    """
    dt_h = max(step_minutes/60.0, 1e-9)

    # Scale pots & fuel by residential share
    pots = _round_to_step(pots_ts_total.copy(), "timestamp", step_minutes)
    fuel = _round_to_step(fuel_ts_total.copy(), "timestamp", step_minutes)
    fdf  = _round_to_step(f_res_df.copy(), "timestamp", step_minutes)

    pots = pots.merge(fdf[["timestamp","f_res_served"]], on="timestamp", how="left").fillna({"f_res_served":0.0})
    pots["potU_res_gbp"] = pots["pot_U_gbp"] * pots["f_res_served"]
    pots["potC_res_gbp"] = pots["pot_C_gbp"] * pots["f_res_served"]

    fuel = fuel.merge(fdf[["timestamp","f_res_served"]], on="timestamp", how="left").fillna({"f_res_served":0.0})
    fuel["fuel_res_gbp"] = fuel["total_fuel_cost_gbp"] * fuel["f_res_served"]

    # Build controllable energy by product (RES) to split FUEL (no capacity weighting)
    up_res_MWh = _round_to_step(uc_per_prod_MWh_res.copy(), "timestamp", step_minutes)
    Cw = (
        up_res_MWh.pivot_table(index="timestamp", columns="product", values="c_served_MWh", aggfunc="sum")
                  .reindex(columns=["P1","P2","P3","P4"], fill_value=0.0)
    ).reset_index()
    Cw.columns = ["timestamp","P1C","P2C","P3C","P4C"]
    Cw["Ctot"] = Cw[["P1C","P2C","P3C","P4C"]].sum(axis=1)

    # Residential U/C MW totals and RES/OTHER splits for debug
    uc_tot = _round_to_step(uc_totals_MW.copy(), "timestamp", step_minutes)
    uc_tot = uc_tot.merge(fdf[["timestamp","f_res_served"]], on="timestamp", how="left").fillna({"f_res_served":0.0})
    uc_tot["U_res_MW"]   = uc_tot["U_MW"] * uc_tot["f_res_served"]
    uc_tot["C_res_MW"]   = uc_tot["C_MW"] * uc_tot["f_res_served"]
    uc_tot["U_other_MW"] = uc_tot["U_MW"] - uc_tot["U_res_MW"]
    uc_tot["C_other_MW"] = uc_tot["C_MW"] - uc_tot["C_res_MW"]

    # Capacity weights per product for non-fuel pots
    cap_w = {"P1":1.0, "P2":2.0, "P3":1.0, "P4":2.0}

    up_res_MW = _round_to_step(uc_per_prod_MW_res.copy(), "timestamp", step_minutes)

    rows=[]; debug=[]

    up_res_MWh_agg = (
        up_res_MWh.groupby(["timestamp","product"], as_index=False)
                  .agg(u_served_MWh=("u_served_MWh","sum"),
                       c_served_MWh=("c_served_MWh","sum"))
    )
    up_res_MW_agg = (
        up_res_MW.groupby(["timestamp","product"], as_index=False)
                 .agg(u_MW=("u_MW","sum"),
                      c_MW=("c_MW","sum"))
    )

    pot_join    = pots[["timestamp","potU_res_gbp","potC_res_gbp"]]
    fuel_join   = fuel[["timestamp","fuel_res_gbp"]]
    Cw_join     = Cw[["timestamp","P1C","P2C","P3C","P4C","Ctot"]]
    prod_res_power = prod_res_df[["timestamp","P1","P2","P3","P4","D_res_MW"]]
    uc_debug    = uc_tot[["timestamp","U_MW","C_MW","U_res_MW","C_res_MW","U_other_MW","C_other_MW"]]

    ts_list = prod_res_df["timestamp"].unique()
    for ts in ts_list:
        pot_row  = pot_join.loc[pot_join["timestamp"].eq(ts)]
        potU_res = float(pot_row["potU_res_gbp"].sum()) if len(pot_row) else 0.0
        potC_res = float(pot_row["potC_res_gbp"].sum()) if len(pot_row) else 0.0

        fuel_row = fuel_join.loc[fuel_join["timestamp"].eq(ts)]
        fuel_res = float(fuel_row["fuel_res_gbp"].sum()) if len(fuel_row) else 0.0

        Cw_row = Cw_join.loc[Cw_join["timestamp"].eq(ts)]
        C1 = float(Cw_row["P1C"].sum()) if len(Cw_row) else 0.0
        C2 = float(Cw_row["P2C"].sum()) if len(Cw_row) else 0.0
        C3 = float(Cw_row["P3C"].sum()) if len(Cw_row) else 0.0
        C4 = float(Cw_row["P4C"].sum()) if len(Cw_row) else 0.0
        Ctot = float(Cw_row["Ctot"].sum()) if len(Cw_row) else 0.0

        MW_row = up_res_MW_agg.loc[up_res_MW_agg["timestamp"].eq(ts)]
        cMW = {p: float(MW_row.loc[MW_row["product"].eq(p), "c_MW"].sum()) for p in ["P1","P2","P3","P4"]}

        denom_cap = sum(cap_w[p]*cMW[p] for p in ["P1","P2","P3","P4"])

        if denom_cap <= 0 and (potU_res>0 or potC_res>0):
            pr = prod_res_power.loc[prod_res_power["timestamp"].eq(ts)]
            P  = {p: float(pr[p].sum()) if len(pr) else 0.0 for p in ["P1","P2","P3","P4"]}
            denom_cap = sum((cap_w[p] if P[p]>0 else 0.0) for p in ["P1","P2","P3","P4"])

        if Ctot > 0:
            fuel_w = {"P1": C1/Ctot, "P2": C2/Ctot, "P3": C3/Ctot, "P4": C4/Ctot}
        else:
            pr = prod_res_power.loc[prod_res_power["timestamp"].eq(ts)]
            D = float(pr["D_res_MW"].sum()) if len(pr) else 0.0
            fuel_w = (
                {p: float(pr[p].sum())/D for p in ["P1","P2","P3","P4"]}
                if D>0 else {"P1":0,"P2":0,"P3":0,"P4":0}
            )

        for p in ["P1","P2","P3","P4"]:
            row_e = up_res_MWh_agg.loc[
                (up_res_MWh_agg["timestamp"].eq(ts)) &
                (up_res_MWh_agg["product"].eq(p))
            ]
            u_MWh = float(row_e["u_served_MWh"].sum()) if len(row_e) else 0.0
            c_MWh = float(row_e["c_served_MWh"].sum()) if len(row_e) else 0.0

            if denom_cap > 0:
                cap_share = (cap_w[p]*cMW[p]) / denom_cap
            else:
                pr  = prod_res_power.loc[prod_res_power["timestamp"].eq(ts)]
                act = [q for q in ["P1","P2","P3","P4"] if (len(pr) and float(pr[q].sum())>0)]
                cap_share = (1.0/len(act)) if act and p in act else 0.0

            potU_p = potU_res * cap_share
            potC_p = potC_res * cap_share
            fuel_p = fuel_res * fuel_w[p]

            rows.append({
                "timestamp": ts,
                "product": p,
                "u_served_MWh": u_MWh,
                "c_served_MWh": c_MWh,
                "capex_opex_from_U": potU_p,
                "capex_opex_from_C": potC_p,
                "fuel_cost_gbp": fuel_p,
                "total_cost_gbp": potU_p + potC_p + fuel_p
            })

        dbg = uc_debug.loc[uc_debug["timestamp"].eq(ts)]
        if len(dbg):
            dbg = dbg.iloc[0]
            debug.append({
                "timestamp": ts,
                "U_total_MW": float(dbg["U_MW"]),
                "C_total_MW": float(dbg["C_MW"]),
                "U_res_MW": float(dbg["U_res_MW"]),
                "C_res_MW": float(dbg["C_res_MW"]),
                "U_other_MW": float(dbg["U_other_MW"]),
                "C_other_MW": float(dbg["C_other_MW"]),
                "potU_res_gbp": potU_res,
                "potC_res_gbp": potC_res,
                "fuel_res_gbp": fuel_res,
                "denom_cap_weighted_MW": denom_cap,
                "cMW_P1_res": cMW["P1"],
                "cMW_P2_res": cMW["P2"],
                "cMW_P3_res": cMW["P3"],
                "cMW_P4_res": cMW["P4"],
                "Ctot_MWh_res": Ctot
            })

    costs_ts = pd.DataFrame(rows).sort_values(["timestamp","product"])
    costs_ts.to_csv(
        os.path.join(out_dir, f"per_product_timeseries_costs_fuel_capex_opex_NonFuelOpexCapExIndividualTS_{tag}.csv"),
        index=False
    )

    # Per-household subscription (monthly) — base + reserves
    annual = (
        costs_ts.groupby("product", as_index=False)
                .agg(
                    total_gbp=("total_cost_gbp","sum"),
                    capex_opex_U=("capex_opex_from_U","sum"),
                    capex_opex_C=("capex_opex_from_C","sum"),
                    fuel_gbp=("fuel_cost_gbp","sum")
                )
    )

    annual["households"] = annual["product"].map(HOUSEHOLDS).astype(float).clip(lower=1.0)

    # Base subscription (without reserves)
    annual["flat_subscription_base_gbp_per_HH_per_month"] = (
        annual["total_gbp"] / annual["households"] / 12.0
    )
    annual["perHH_perMonth_capex_opex_U"] = annual["capex_opex_U"] / annual["households"] / 12.0
    annual["perHH_perMonth_capex_opex_C"] = annual["capex_opex_C"] / annual["households"] / 12.0
    annual["perHH_perMonth_fuel"]         = annual["fuel_gbp"]        / annual["households"] / 12.0

    # Reserves: uniform per HH per month across all residential customers
    annual["perHH_perMonth_reserves"] = float(perHH_reserves_gbp_per_month)

    # Total subscription including reserves
    annual["flat_subscription_gbp_per_HH_per_month"] = (
        annual["flat_subscription_base_gbp_per_HH_per_month"] +
        annual["perHH_perMonth_reserves"]
    )

    # Record reserves contribution at product level (for total_gbp consistency)
    annual["total_gbp_base"]     = annual["total_gbp"]
    annual["reserves_total_gbp"] = (
        annual["perHH_perMonth_reserves"] * annual["households"] * 12.0
    )
    annual["total_gbp"] = annual["total_gbp_base"] + annual["reserves_total_gbp"]

    annual["approach"] = f"NonFuelOpexCapExIndividualTS_{tag}"
    annual.to_csv(
        os.path.join(out_dir, f"per_product_subscription_flat_NonFuelOpexCapExIndividualTS_{tag}.csv"),
        index=False
    )

    # Return also scaled pots/fuel for verification
    pots_res  = pots[["timestamp","potU_res_gbp","potC_res_gbp"]].copy()
    fuel_res  = fuel[["timestamp","fuel_res_gbp"]].copy()
    return annual, costs_ts, pots_res, fuel_res

# ---- AGGREGATE non-fuel per period (fuel still per-timestamp) ----
def build_costs_aggregate_nonfuel_per_period(
    uc_per_prod_MWh_res: pd.DataFrame,
    prod_res_df: pd.DataFrame,
    pots_ts_total: pd.DataFrame,
    fuel_ts_total: pd.DataFrame,
    f_res_df: pd.DataFrame,
    out_dir: str,
    tag: str,
    step_minutes: int,
    perHH_reserves_gbp_per_month: float = 0.0,
):
    """
    Non-fuel (CapEx/OpEx):
        - Sum residential-scaled pots over the WHOLE period.
        - Allocate U pot by aggregate U energy shares per product (Σ u_served_MWh_p).
        - Allocate C pot by aggregate C energy shares per product (Σ c_served_MWh_p).

    Fuel:
        - Still allocated PER TIMESTAMP via controllable energy shares.

    perHH_reserves_gbp_per_month:
        uniform per-household reserves charge added on top of base subscription.
    """
    dt_h = max(step_minutes/60.0, 1e-9)

    # Residential-scaled pots over time
    pots = _round_to_step(pots_ts_total.copy(), "timestamp", step_minutes)
    fdf  = _round_to_step(f_res_df.copy(), "timestamp", step_minutes)
    pots = pots.merge(fdf[["timestamp","f_res_served"]], on="timestamp", how="left").fillna({"f_res_served":0.0})
    pots["potU_res_gbp"] = pots["pot_U_gbp"] * pots["f_res_served"]
    pots["potC_res_gbp"] = pots["pot_C_gbp"] * pots["f_res_served"]

    # Totals over the whole period
    potU_total = float(pots["potU_res_gbp"].sum())
    potC_total = float(pots["potC_res_gbp"].sum())

    # Aggregate U/C energy by product (already using global U/C shares via per-timestamp split)
    up_res_MWh = _round_to_step(uc_per_prod_MWh_res.copy(), "timestamp", step_minutes)
    agg_energy = (
        up_res_MWh.groupby("product", as_index=False)
                  .agg(U_MWh=("u_served_MWh","sum"),
                       C_MWh=("c_served_MWh","sum"))
    )
    U_sum = float(agg_energy["U_MWh"].sum())
    C_sum = float(agg_energy["C_MWh"].sum())

    # Per-timestamp fuel allocation (unchanged)
    fuel = _round_to_step(fuel_ts_total.copy(), "timestamp", step_minutes)
    fuel = fuel.merge(fdf[["timestamp","f_res_served"]], on="timestamp", how="left").fillna({"f_res_served":0.0})
    fuel["fuel_res_gbp"] = fuel["total_fuel_cost_gbp"] * fuel["f_res_served"]

    rows=[]

    # 1) Fuel per timestamp: controllable energy shares per ts
    per_ts = (
        up_res_MWh.groupby(["timestamp","product"], as_index=False)
                  .agg(u_served_MWh=("u_served_MWh","sum"),
                       c_served_MWh=("c_served_MWh","sum"))
    )
    for ts, g in per_ts.groupby("timestamp"):
        fuel_ts_val = float(fuel.loc[fuel["timestamp"].eq(ts), "fuel_res_gbp"].sum())
        C_ts = float(g["c_served_MWh"].sum())
        for _, r in g.iterrows():
            p     = r["product"]
            u_MWh = float(r["u_served_MWh"])
            c_MWh = float(r["c_served_MWh"])
            fuel_p = (fuel_ts_val * (c_MWh / C_ts)) if C_ts > 0 else 0.0
            rows.append({
                "timestamp": ts,
                "product": p,
                "u_served_MWh": u_MWh,
                "c_served_MWh": c_MWh,
                "capex_opex_from_U": 0.0,
                "capex_opex_from_C": 0.0,
                "fuel_cost_gbp": fuel_p,
                "total_cost_gbp": fuel_p,
            })

    costs_ts = pd.DataFrame(rows)

    # 2) Period-level non-fuel allocation by aggregate energy shares
    nonfuel_rows=[]
    for _, r in agg_energy.iterrows():
        p = r["product"]
        u_share = (float(r["U_MWh"]) / U_sum) if U_sum > 0 else 0.0
        c_share = (float(r["C_MWh"]) / C_sum) if C_sum > 0 else 0.0
        capU    = potU_total * u_share
        capC    = potC_total * c_share
        nonfuel_rows.append({
            "timestamp": pd.NaT,
            "product": p,
            "u_served_MWh": 0.0,
            "c_served_MWh": 0.0,
            "capex_opex_from_U": capU,
            "capex_opex_from_C": capC,
            "fuel_cost_gbp": 0.0,
            "total_cost_gbp": capU + capC,
        })

    nonfuel_df = pd.DataFrame(nonfuel_rows)
    costs_ts_all = pd.concat([costs_ts, nonfuel_df], ignore_index=True)
    costs_ts_all = costs_ts_all.sort_values(["product","timestamp"], na_position="last")
    costs_ts_all.to_csv(
        os.path.join(out_dir, f"per_product_timeseries_costs_fuel_capex_opex_NonFuelOpexCapExAggregate_{tag}.csv"),
        index=False
    )

    # Annual (base + reserves)
    annual = (
        costs_ts_all.groupby("product", as_index=False)
                    .agg(
                        total_gbp=("total_cost_gbp","sum"),
                        capex_opex_U=("capex_opex_from_U","sum"),
                        capex_opex_C=("capex_opex_from_C","sum"),
                        fuel_gbp=("fuel_cost_gbp","sum")
                    )
    )

    annual["households"] = annual["product"].map(HOUSEHOLDS).astype(float).clip(lower=1.0)

    # Base subscription without reserves
    annual["flat_subscription_base_gbp_per_HH_per_month"] = (
        annual["total_gbp"] / annual["households"] / 12.0
    )
    annual["perHH_perMonth_capex_opex_U"] = annual["capex_opex_U"] / annual["households"] / 12.0
    annual["perHH_perMonth_capex_opex_C"] = annual["capex_opex_C"] / annual["households"] / 12.0
    annual["perHH_perMonth_fuel"]         = annual["fuel_gbp"]        / annual["households"] / 12.0

    # Reserves per HH per month (uniform)
    annual["perHH_perMonth_reserves"] = float(perHH_reserves_gbp_per_month)

    # Total subscription including reserves
    annual["flat_subscription_gbp_per_HH_per_month"] = (
        annual["flat_subscription_base_gbp_per_HH_per_month"] +
        annual["perHH_perMonth_reserves"]
    )

    # Track reserves in total_gbp
    annual["total_gbp_base"]     = annual["total_gbp"]
    annual["reserves_total_gbp"] = (
        annual["perHH_perMonth_reserves"] * annual["households"] * 12.0
    )
    annual["total_gbp"] = annual["total_gbp_base"] + annual["reserves_total_gbp"]

    annual["approach"] = f"NonFuelOpexCapExAggregate_{tag}"
    annual.to_csv(
        os.path.join(out_dir, f"per_product_subscription_flat_NonFuelOpexCapExAggregate_{tag}.csv"),
        index=False
    )

    # Return also scaled pots/fuel for verification
    pots_res = pots[["timestamp","potU_res_gbp","potC_res_gbp"]].copy()
    fuel_res = fuel[["timestamp","fuel_res_gbp"]].copy()
    return annual, costs_ts_all, pots_res, fuel_res

# -------------- plots --------------
def plot_uc_stacked(uc_per_prod_MWh_res: pd.DataFrame,
                    out_tot_png: str,
                    out_perhh_png: str):
    prods = ["P1","P2","P3","P4"]; x = np.arange(len(prods))
    agg = (
        uc_per_prod_MWh_res.groupby("product", as_index=False)
                           .agg(U_MWh=("u_served_MWh","sum"),
                                C_MWh=("c_served_MWh","sum"))
    )
    U_vals = [float(agg.loc[agg["product"]==p, "U_MWh"].sum())/1e6 for p in prods]
    C_vals = [float(agg.loc[agg["product"]==p, "C_MWh"].sum())/1e6 for p in prods]

    plt.figure(figsize=(10,5))
    plt.bar(x, U_vals, label="Uncontrollable (U)")
    plt.bar(x, C_vals, bottom=U_vals, label="Controllable (C)")
    plt.xticks(x, prods)
    plt.ylabel("TWh")
    plt.title("Residential energy by product (stacked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_tot_png, dpi=150)
    plt.close()

    # per-HH (MWh/HH)
    HH   = HOUSEHOLDS
    U_hh = [(float(agg.loc[agg["product"]==p,"U_MWh"].sum()) / HH[p]) if HH[p]>0 else 0.0 for p in prods]
    C_hh = [(float(agg.loc[agg["product"]==p,"C_MWh"].sum()) / HH[p]) if HH[p]>0 else 0.0 for p in prods]

    plt.figure(figsize=(10,5))
    plt.bar(x, U_hh, label="U per HH (MWh)")
    plt.bar(x, C_hh, bottom=U_hh, label="C per HH (MWh)")
    plt.xticks(x, prods)
    plt.ylabel("MWh / HH")
    plt.title("Residential energy per household (stacked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_perhh_png, dpi=150)
    plt.close()

def plot_flat_subscriptions_all_variants(annual_ts_BASE: pd.DataFrame,
                                         annual_ts_DELTA: pd.DataFrame,
                                         annual_ag_BASE: pd.DataFrame,
                                         annual_ag_DELTA: pd.DataFrame,
                                         out_png: str):
    prods = ["P1","P2","P3","P4"]; x = np.arange(len(prods)); w = 0.18
    def vals(df):
        return [float(df.loc[df["product"]==p, "flat_subscription_gbp_per_HH_per_month"].values[0]) for p in prods]
    y1 = vals(annual_ts_BASE); y2 = vals(annual_ts_DELTA)
    y3 = vals(annual_ag_BASE); y4 = vals(annual_ag_DELTA)

    plt.figure(figsize=(12,6))
    b1 = plt.bar(x-1.5*w, y1, width=w, label="IndividualTS_Base")
    b2 = plt.bar(x-0.5*w, y2, width=w, label="IndividualTS_Delta")
    b3 = plt.bar(x+0.5*w, y3, width=w, label="Aggregate_Base")
    b4 = plt.bar(x+1.5*w, y4, width=w, label="Aggregate_Delta")

    for bars in (b1,b2,b3,b4):
        for b in bars:
            plt.text(
                b.get_x()+b.get_width()/2,
                b.get_height()*1.01,
                f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8
            )

    plt.xticks(x, prods)
    plt.ylabel("£ / HH / month")
    plt.title("Flat subscription per household per month — four variants (incl. reserves)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_flat_subscription_components_all_variants(annual_ts_BASE: pd.DataFrame,
                                                   annual_ts_DELTA: pd.DataFrame,
                                                   annual_ag_BASE: pd.DataFrame,
                                                   annual_ag_DELTA: pd.DataFrame,
                                                   out_png: str):
    prods = ["P1","P2","P3","P4"]; x = np.arange(len(prods)); w = 0.16
    def comp(df, col):
        return [float(df.loc[df["product"]==p, col].values[0]) for p in prods]

    # We now have: fuel, CapEx/OpEx_U, CapEx/OpEx_C, and reserves as separate per-HH component.
    layers = [
        ("Fuel",                "perHH_perMonth_fuel"),
        ("CapEx/OpEx from U",   "perHH_perMonth_capex_opex_U"),
        ("CapEx/OpEx from C",   "perHH_perMonth_capex_opex_C"),
        ("Reserves",            "perHH_perMonth_reserves"),
    ]
    variants = [
        ("IndTS_Base", annual_ts_BASE,  -1.5*w),
        ("IndTS_Delta", annual_ts_DELTA, -0.5*w),
        ("Agg_Base",   annual_ag_BASE,   0.5*w),
        ("Agg_Delta",  annual_ag_DELTA,  1.5*w),
    ]

    plt.figure(figsize=(14,7))
    for label, df, shift in variants:
        bottoms = np.zeros(len(prods))
        for lname, col in layers:
            vals = np.array(comp(df, col))
            plt.bar(x+shift, vals, width=w, bottom=bottoms, label=f"{lname} ({label})")
            bottoms += vals

    plt.xticks(x, prods)
    plt.ylabel("£ / HH / month")
    plt.title("Subscription components per household — four variants (incl. reserves)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------- verification & summaries --------------
def verify_and_write(df_timeseries: pd.DataFrame,
                     flat_df: pd.DataFrame,
                     pots_res_ts: pd.DataFrame,
                     fuel_res_ts: pd.DataFrame,
                     out_csv: str):
    pots_total_res = float(pots_res_ts["potU_res_gbp"].sum() + pots_res_ts["potC_res_gbp"].sum())
    alloc_capex_opex = float((df_timeseries["capex_opex_from_U"] + df_timeseries["capex_opex_from_C"]).sum())
    fuel_total_res = float(fuel_res_ts["fuel_res_gbp"].sum()) if not fuel_res_ts.empty else 0.0
    fuel_alloc     = float(df_timeseries["fuel_cost_gbp"].sum())
    grand_alloc    = float(df_timeseries["total_cost_gbp"].sum())

    # NOTE: subscription_revenue now includes reserves, so this will exceed grand_alloc
    # by the total residential reserves amount.
    subs_revenue = float(
        (flat_df["flat_subscription_gbp_per_HH_per_month"] * flat_df["households"] * 12.0).sum()
    )

    rows = [
        {"check":"pots_res_total_gbp",           "value": pots_total_res},
        {"check":"allocated_capex_opex_gbp",     "value": alloc_capex_opex},
        {"check":"fuel_res_total_gbp",           "value": fuel_total_res},
        {"check":"fuel_allocated_gbp",           "value": fuel_alloc},
        {"check":"grand_total_allocated_gbp",    "value": grand_alloc},
        {"check":"subscription_revenue_gbp",     "value": subs_revenue},
        {"check":"diff_pots_vs_alloc_gbp",       "value": alloc_capex_opex - pots_total_res},
        {"check":"diff_fuel_vs_alloc_gbp",       "value": fuel_alloc - fuel_total_res},
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def verify_energy_balance_per_product_ts(uc_per_prod_MW_res: pd.DataFrame,
                                         prod_res_served_ts: pd.DataFrame,
                                         out_csv: str,
                                         tol_MW: float = 1e-6):
    """
    Check for each timestamp×product: (allocated U + C) - served_product == 0 (within tol)
    Writes a CSV with columns: timestamp, product, allocated_MW, served_MW, diff_MW
    """
    # allocated wide
    alloc = (
        uc_per_prod_MW_res.groupby(["timestamp","product"], as_index=False)
                          .agg(alloc_U_MW=("u_MW","sum"),
                               alloc_C_MW=("c_MW","sum"))
    )
    alloc["allocated_MW"] = alloc["alloc_U_MW"] + alloc["alloc_C_MW"]

    # served wide from product timeseries (P1..P4)
    served = prod_res_served_ts.copy()
    served = served.melt(
        id_vars=["timestamp"],
        value_vars=["P1","P2","P3","P4"],
        var_name="product",
        value_name="served_MW"
    )

    # join
    chk = alloc.merge(served, on=["timestamp","product"], how="left").fillna({"served_MW":0.0})
    chk["diff_MW"] = chk["allocated_MW"] - chk["served_MW"]
    chk["ok"]      = np.isclose(chk["diff_MW"], 0.0, atol=tol_MW)
    chk.sort_values(["timestamp","product"]).to_csv(out_csv, index=False)

def write_variant_summary_csv(
    variant_name: str,
    out_dir: str,
    annual_per_product: pd.DataFrame,  # columns: product, fuel_gbp, capex_opex_U, capex_opex_C, total_gbp, households
    uc_per_prod_MWh_res: pd.DataFrame, # rows per-timestamp per-product: u_served_MWh, c_served_MWh
    pots_res_ts: pd.DataFrame,         # columns: timestamp, potU_res_gbp, potC_res_gbp
    fuel_res_ts: pd.DataFrame,         # columns: timestamp, fuel_res_gbp
    pots_total_ts: pd.DataFrame,       # columns: timestamp, pot_U_gbp, pot_C_gbp (UNSCALED)
    fuel_total_ts: pd.DataFrame        # columns: timestamp, total_fuel_cost_gbp (UNSCALED)
):
    prods = ["P1","P2","P3","P4"]

    # Energy by product (RES)
    aggE = (
        uc_per_prod_MWh_res.groupby("product", as_index=False)
                           .agg(U_MWh=("u_served_MWh","sum"),
                                C_MWh=("c_served_MWh","sum"))
    )
    C_total_MWh = float(aggE["C_MWh"].sum())

    # Non-res totals from UNSCALED minus RES-scaled
    potU_total_all = float(pots_total_ts.get("pot_U_gbp", 0.0).sum())
    potC_total_all = float(pots_total_ts.get("pot_C_gbp", 0.0).sum())
    fuel_total_all = float(fuel_total_ts.get("total_fuel_cost_gbp", 0.0).sum())

    potU_total_res = float(pots_res_ts.get("potU_res_gbp", 0.0).sum()) if not pots_res_ts.empty else 0.0
    potC_total_res = float(pots_res_ts.get("potC_res_gbp", 0.0).sum()) if not pots_res_ts.empty else 0.0
    fuel_total_res = float(fuel_res_ts.get("fuel_res_gbp", 0.0).sum()) if not fuel_res_ts.empty else 0.0

    nonres_capex_opex_U = potU_total_all - potU_total_res
    nonres_capex_opex_C = potC_total_all - potC_total_res
    nonres_fuel_gbp     = fuel_total_all - fuel_total_res
    nonres_total_gbp    = nonres_capex_opex_U + nonres_capex_opex_C + nonres_fuel_gbp

    rows = []

    # Energy rows (U/C GWh and C% share)
    for p in prods:
        U_MWh = float(aggE.loc[aggE["product"]==p, "U_MWh"].sum()) if (aggE["product"]==p).any() else 0.0
        C_MWh = float(aggE.loc[aggE["product"]==p, "C_MWh"].sum()) if (aggE["product"]==p).any() else 0.0
        C_share_pct = (100.0 * C_MWh / C_total_MWh) if C_total_MWh > 0 else 0.0
        rows += [
            {
                "variant": variant_name,
                "section":"energy",
                "metric":"U_energy",
                "product":p,
                "value": _to_gwh(U_MWh),
                "units":"GWh"
            },
            {
                "variant": variant_name,
                "section":"energy",
                "metric":"C_energy",
                "product":p,
                "value": _to_gwh(C_MWh),
                "units":"GWh"
            },
            {
                "variant": variant_name,
                "section":"energy",
                "metric":"C_share_of_total_controllable",
                "product":p,
                "value": C_share_pct,
                "units":"%"
            },
        ]

    # Residential costs by product (+ households)
    for _, r in annual_per_product.iterrows():
        p = r["product"]
        rows += [
            {
                "variant": variant_name,
                "section":"residential_costs",
                "metric":"fuel_cost_gbp",
                "product":p,
                "value": float(r.get("fuel_gbp", 0.0)),
                "units":"GBP"
            },
            {
                "variant": variant_name,
                "section":"residential_costs",
                "metric":"capex_opex_U_gbp",
                "product":p,
                "value": float(r.get("capex_opex_U", 0.0)),
                "units":"GBP"
            },
            {
                "variant": variant_name,
                "section":"residential_costs",
                "metric":"capex_opex_C_gbp",
                "product":p,
                "value": float(r.get("capex_opex_C", 0.0)),
                "units":"GBP"
            },
            {
                "variant": variant_name,
                "section":"residential_costs",
                "metric":"total_cost_gbp",
                "product":p,
                "value": float(r.get("total_gbp", 0.0)),
                "units":"GBP"
            },
            {
                "variant": variant_name,
                "section":"households",
                "metric":"households_count",
                "product":p,
                "value": float(r.get("households", 0.0)),
                "units":"count"
            },
            {
                "variant": variant_name,
                "section":"subscriptions",
                "metric":"flat_sub_gbp_per_HH_month",
                "product":p,
                "value": float(r.get("flat_subscription_gbp_per_HH_per_month", 0.0)),
                "units":"GBP/HH/month"
            },
        ]

    # Non-residential totals (no product)
    rows += [
        {
            "variant": variant_name,
            "section":"nonresidential_totals",
            "metric":"capex_opex_U_gbp",
            "product":"NONRES",
            "value": nonres_capex_opex_U,
            "units":"GBP"
        },
        {
            "variant": variant_name,
            "section":"nonresidential_totals",
            "metric":"capex_opex_C_gbp",
            "product":"NONRES",
            "value": nonres_capex_opex_C,
            "units":"GBP"
        },
        {
            "variant": variant_name,
            "section":"nonresidential_totals",
            "metric":"fuel_cost_gbp",
            "product":"NONRES",
            "value": nonres_fuel_gbp,
            "units":"GBP"
        },
        {
            "variant": variant_name,
            "section":"nonresidential_totals",
            "metric":"total_cost_gbp",
            "product":"NONRES",
            "value": nonres_total_gbp,
            "units":"GBP"
        },
    ]

    df_out = pd.DataFrame(rows)
    out_fp = os.path.join(out_dir, f"summary_{variant_name}.csv")
    df_out.to_csv(out_fp, index=False)
    return out_fp

# ---- Optional: Non-residential U/C split (global shares) writer ----
def write_nonres_uc_timeseries(res_share_served: pd.DataFrame,
                               uc_totals: pd.DataFrame,
                               out_csv: str,
                               step_minutes: int):
    """
    Apply the same global U/C shares to NON-RESIDENTIAL served MW per timestamp:

        U_nonres = share_U * D_nonres_served_MW
        C_nonres = share_C * D_nonres_served_MW
    """
    r = _round_to_step(res_share_served.copy(), "timestamp", step_minutes=step_minutes)
    u = _round_to_step(uc_totals[["timestamp","U_MW","C_MW"]].copy(), "timestamp", step_minutes=step_minutes)
    m = r.merge(u, on="timestamp", how="inner")

    total = (m["U_MW"] + m["C_MW"]).replace(0.0, np.nan)
    m["share_U"] = (m["U_MW"]/total).fillna(0.0).clip(0.0,1.0)
    m["share_C"] = (m["C_MW"]/total).fillna(0.0).clip(0.0,1.0)

    m["U_nonres_MW"] = m["share_U"] * pd.to_numeric(m.get("D_nonres_served_MW", 0.0), errors="coerce").fillna(0.0)
    m["C_nonres_MW"] = m["share_C"] * pd.to_numeric(m.get("D_nonres_served_MW", 0.0), errors="coerce").fillna(0.0)

    out = m[["timestamp","D_nonres_served_MW","U_nonres_MW","C_nonres_MW","share_U","share_C"]].copy()
    out.sort_values("timestamp").to_csv(out_csv, index=False)

# ---- Reserves allocation: global RES/NONRES split and per-HH charge ----
def compute_reserves_allocation(
    res_share_served: pd.DataFrame,
    step_minutes: int,
    reserves_fp: str,
    out_dir: str
):
    """
    1) Read total reserves paid to generators from reserves_fp.
       We try to find a sensible revenue column:
         - 'reserve_revenue_gbp'
         - 'reserves_gbp'
         - 'reserve_gbp'
         - 'revenue_gbp'
         - otherwise first numeric column.
    2) Compute total RES and NONRES energy served over time (MWh).
    3) Split total reserves between RES and NONRES in proportion to these totals.
    4) Allocate residential reserves equally across all residential households.

    Writes:
      - reserves_allocation_summary.csv in out_dir

    Returns:
      perHH_reserves_gbp_per_month (float)
    """
    # ---- 1) Read reserves-by-generator and infer the revenue column ----
    if not os.path.exists(reserves_fp):
        raise FileNotFoundError(f"Reserves file not found: {reserves_fp}")

    dfR = pd.read_csv(reserves_fp)

    # Try to guess the reserves revenue column
    cand_cols = []
    for c in dfR.columns:
        lc = str(c).lower()
        if lc in ("reserve_revenue_gbp", "reserves_gbp", "reserve_gbp", "revenue_gbp"):
            cand_cols.append(c)

    if cand_cols:
        rev_col = cand_cols[0]
    else:
        # Fallback: first numeric column
        num_cols = dfR.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError(
                f"Could not find a numeric reserves revenue column in {reserves_fp}."
            )
        rev_col = num_cols[0]

    total_reserves_gbp = float(pd.to_numeric(dfR[rev_col], errors="coerce").fillna(0.0).sum())

    # ---- 2) Compute total RES / NONRES energy served (MWh) over the whole period ----
    dt_h = max(step_minutes / 60.0, 1e-9)
    r    = res_share_served.copy()
    r    = _round_to_step(r, "timestamp", step_minutes=step_minutes)

    # They already contain D_res_served_MW and D_nonres_served_MW
    r["res_MWh"]    = pd.to_numeric(r["D_res_served_MW"],    errors="coerce").fillna(0.0) * dt_h
    r["nonres_MWh"] = pd.to_numeric(r["D_nonres_served_MW"], errors="coerce").fillna(0.0) * dt_h

    total_res_MWh    = float(r["res_MWh"].sum())
    total_nonres_MWh = float(r["nonres_MWh"].sum())
    total_MWh        = total_res_MWh + total_nonres_MWh

    if total_MWh > 0:
        f_res_global = total_res_MWh / total_MWh
    else:
        f_res_global = 0.0  # degenerate case

    # ---- 3) Split reserves between RES and NONRES ----
    reserves_res_gbp    = total_reserves_gbp * f_res_global
    reserves_nonres_gbp = total_reserves_gbp - reserves_res_gbp

    # ---- 4) Allocate RES reserves equally across all residential households ----
    total_households = float(sum(HOUSEHOLDS.values()))
    if total_households <= 0:
        raise ValueError("HOUSEHOLDS sum must be > 0 to allocate reserves per HH.")

    perHH_reserves_gbp_per_year  = reserves_res_gbp / total_households
    perHH_reserves_gbp_per_month = perHH_reserves_gbp_per_year / 12.0

    # ---- 5) Write a small summary CSV ----
    rows = [
        {"metric": "total_reserves_gbp",             "value": total_reserves_gbp},
        {"metric": "residential_reserves_gbp",       "value": reserves_res_gbp},
        {"metric": "nonresidential_reserves_gbp",    "value": reserves_nonres_gbp},
        {"metric": "total_res_served_MWh",           "value": total_res_MWh},
        {"metric": "total_nonres_served_MWh",        "value": total_nonres_MWh},
        {"metric": "f_res_global",                   "value": f_res_global},
        {"metric": "perHH_reserves_gbp_per_year",    "value": perHH_reserves_gbp_per_year},
        {"metric": "perHH_reserves_gbp_per_month",   "value": perHH_reserves_gbp_per_month},
    ]
    df_out = pd.DataFrame(rows)
    out_fp = os.path.join(out_dir, "reserves_allocation_summary.csv")
    df_out.to_csv(out_fp, index=False)

    return perHH_reserves_gbp_per_month

# ===================== main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amm-run-root", default=AMM_RUN_ROOT_DEFAULT)
    ap.add_argument("--served-dir", default=SERVED_DIR_DEFAULT,
                    help="Directory with served_breakdown_D*.csv")
    ap.add_argument("--gens-static", default=GENS_STATIC_DEFAULT)
    ap.add_argument(
        "--gen-rev-all",
        default=GENS_STATIC_DEFAULT.replace(
            "gens_static.csv",
            "marketExecution_AMM/availabilityPayments/analysis/run_20251205_152214/generator_revenue_timeseries_ALL.csv"
        ) if GENS_STATIC_DEFAULT.endswith("gens_static.csv") else GENREV_ALL_DEFAULT
    )
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--amm-dt-min", type=int, default=30)
    ap.add_argument("--run-tag", default="")
    ap.add_argument(
        "--reserves-by-generator",
        default="marketExecution_AMM/outputs/analysis/reserves_by_generator.csv",
        help="CSV with reserves paid per generator"
    )
    args = ap.parse_args()

    step_minutes = int(args.amm_dt_min)
    dt_h         = max(step_minutes/60.0, 1e-9)

    run_ts  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = os.path.join(
        args.out_root,
        f"run_{run_ts}" + (f"_{args.run_tag}" if args.run_tag else "")
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load SERVED residential per-load and system totals from served_breakdown_* inputs
    ppl_res_served, res_share_served = load_served_breakdowns(args.served_dir, step_minutes=step_minutes)
    res_share_served.to_csv(os.path.join(OUT_DIR, "res_vs_nonres_served_timeseries.csv"), index=False)
    ppl_res_served.to_csv(os.path.join(OUT_DIR, "per_load_residential_products_served_timeseries.csv"), index=False)

    # 1b) Reserves allocation per household — compute EARLY so it feeds subscription logic
    perHH_reserves_gbp_per_month = compute_reserves_allocation(
        res_share_served=res_share_served,
        step_minutes=step_minutes,
        reserves_fp=args.reserves_by_generator,
        out_dir=OUT_DIR,
    )

    # 2) Systemwide residential served products (P1..P4, D_res)
    prod_res_served = aggregate_res_products_from_per_load_served(ppl_res_served, step_minutes=step_minutes)
    prod_res_served.to_csv(os.path.join(OUT_DIR, "system_residential_products_served_timeseries.csv"), index=False)

    # 3) Tech + dispatch
    tech_map = load_tech_map(args.gens_static)
    dispatch = read_dispatch_tree(args.amm_run_root, step_minutes=step_minutes)

    # 4) U/C totals from dispatch, capping by TOTAL *served* demand (res + non-res)
    cap_total = res_share_served[["timestamp","D_total_served_MW"]].copy().rename(
        columns={"D_total_served_MW":"cap_total_MW"}
    )
    uc_tot = uc_timeseries_from_dispatch(dispatch, tech_map, cap_total, step_minutes=step_minutes)
    uc_tot.to_csv(os.path.join(OUT_DIR, "uc_totals_timeseries.csv"), index=False)

    # 5) Align timestamps
    ts = set(prod_res_served["timestamp"]) & set(uc_tot["timestamp"]) & set(dispatch["timestamp"])
    if not ts:
        raise RuntimeError("No overlapping timestamps across served demand/dispatch/UC")

    prod_res_served = prod_res_served[prod_res_served["timestamp"].isin(ts)].sort_values("timestamp")
    res_share_served = res_share_served[res_share_served["timestamp"].isin(ts)].sort_values("timestamp")
    uc_tot = uc_tot[uc_tot["timestamp"].isin(ts)].sort_values("timestamp")
    dispatch = dispatch[dispatch["timestamp"].isin(ts)].sort_values("timestamp")

    # 6) Split U/C totals into RES vs OTHER using f_res_served (kept for scaling pots/fuel & debug)
    uc_res_other = uc_tot.merge(res_share_served[["timestamp","f_res_served"]],
                                on="timestamp", how="left").fillna({"f_res_served":0.0})
    uc_res_other["U_res_MW"]   = uc_res_other["U_MW"] * uc_res_other["f_res_served"]
    uc_res_other["C_res_MW"]   = uc_res_other["C_MW"] * uc_res_other["f_res_served"]
    uc_res_other["U_other_MW"] = uc_res_other["U_MW"] - uc_res_other["U_res_MW"]
    uc_res_other["C_other_MW"] = uc_res_other["C_MW"] - uc_res_other["C_res_MW"]
    uc_res_other.to_csv(os.path.join(OUT_DIR, "uc_res_vs_other_served_timeseries.csv"), index=False)

    # 7) Per-product U/C (RES ONLY, served) — MW and MWh — using GLOBAL U/C shares
    uc_per_prod_MW_res, uc_per_prod_MWh_res = uc_per_product_from_global_uc_share(
        prod_res_served,                       # P1..P4 served MW
        uc_tot[["timestamp","U_MW","C_MW"]],   # GLOBAL totals to form shares
        dt_h,
        step_minutes=step_minutes
    )
    uc_per_prod_MW_res.to_csv(os.path.join(OUT_DIR,"per_product_UC_power_timeseries_RES_served.csv"), index=False)
    uc_per_prod_MWh_res.to_csv(os.path.join(OUT_DIR,"per_product_UC_energy_timeseries_RES_served.csv"), index=False)

    # 7b) Optional: non-residential U/C timeseries using the same global shares
    write_nonres_uc_timeseries(
        res_share_served=res_share_served,
        uc_totals=uc_tot,
        out_csv=os.path.join(OUT_DIR, "nonresidential_UC_timeseries_global_share.csv"),
        step_minutes=step_minutes
    )

    # 8) Fuel timeseries (controllable only) — total; will be scaled by f_res_served later
    ts_fuel_total = fuel_cost_timeseries_from_dispatch(dispatch, tech_map, dt_h, step_minutes=step_minutes)

    # 9) Pots BASE & DELTA (total); later scaled by f_res_served
    def class_pots(approach_prefix):
        gr = read_gen_rev_all(args.gen_rev_all, approach_prefix, step_minutes=step_minutes)
        return class_pots_from_gen_revenue_ts(gr, tech_map, step_minutes=step_minutes)

    pots_BASE_total  = class_pots("BASE")
    pots_DELTA_total = class_pots("DELTA")

    pots_BASE_total_ts  = pots_BASE_total.copy()
    pots_DELTA_total_ts = pots_DELTA_total.copy()
    fuel_total_ts        = ts_fuel_total.copy()

    # 10a) Per-timestamp non-fuel allocation (IndividualTS) — using served splits
    flat_ts_BASE, ts_BASE, pots_BASE_res, fuel_res_ts = build_costs_per_ts(
        uc_per_prod_MWh_res,
        uc_per_prod_MW_res,
        prod_res_served,
        pots_BASE_total,
        ts_fuel_total,
        uc_tot,
        res_share_served[["timestamp","f_res_served"]],
        OUT_DIR,
        "Base",
        step_minutes,
        perHH_reserves_gbp_per_month=perHH_reserves_gbp_per_month,
    )
    flat_ts_DELTA, ts_DELTA, pots_DELTA_res, fuel_res_ts2 = build_costs_per_ts(
        uc_per_prod_MWh_res,
        uc_per_prod_MW_res,
        prod_res_served,
        pots_DELTA_total,
        ts_fuel_total,
        uc_tot,
        res_share_served[["timestamp","f_res_served"]],
        OUT_DIR,
        "Delta",
        step_minutes,
        perHH_reserves_gbp_per_month=perHH_reserves_gbp_per_month,
    )

    # 10b) Aggregate non-fuel over the period (fuel still per-timestamp) — served splits
    flat_ag_BASE, ts_ag_BASE, pots_BASE_res2, fuel_res_ts3 = build_costs_aggregate_nonfuel_per_period(
        uc_per_prod_MWh_res,
        prod_res_served,
        pots_BASE_total,
        ts_fuel_total,
        res_share_served[["timestamp","f_res_served"]],
        OUT_DIR,
        "Base",
        step_minutes,
        perHH_reserves_gbp_per_month=perHH_reserves_gbp_per_month,
    )
    flat_ag_DELTA, ts_ag_DELTA, pots_DELTA_res2, fuel_res_ts4 = build_costs_aggregate_nonfuel_per_period(
        uc_per_prod_MWh_res,
        prod_res_served,
        pots_DELTA_total,
        ts_fuel_total,
        res_share_served[["timestamp","f_res_served"]],
        OUT_DIR,
        "Delta",
        step_minutes,
        perHH_reserves_gbp_per_month=perHH_reserves_gbp_per_month,
    )

    # 11) Plots
    plot_uc_stacked(
        uc_per_prod_MWh_res,
        os.path.join(OUT_DIR,"fig_UC_split_by_product_TWh_RES_served.pdf"),
        os.path.join(OUT_DIR,"fig_UC_per_HH_by_product_MWh_per_HH_RES_served.pdf")
    )
    plot_flat_subscriptions_all_variants(
        flat_ts_BASE, flat_ts_DELTA, flat_ag_BASE, flat_ag_DELTA,
        os.path.join(OUT_DIR,"fig_subscription_per_HH_by_product_bar_RES_ALL_VARIANTS_served.pdf")
    )
    plot_flat_subscription_components_all_variants(
        flat_ts_BASE, flat_ts_DELTA, flat_ag_BASE, flat_ag_DELTA,
        os.path.join(OUT_DIR,"fig_subscription_components_per_HH_by_product_bar_RES_ALL_VARIANTS_served.pdf")
    )

    # 12) Verification CSVs — allocation balances
    verify_and_write(
        ts_BASE, flat_ts_BASE, pots_BASE_res,  fuel_res_ts,
        os.path.join(OUT_DIR, "verify_balance_NonFuelOpexCapExIndividualTS_Base.csv")
    )
    verify_and_write(
        ts_DELTA, flat_ts_DELTA, pots_DELTA_res, fuel_res_ts2,
        os.path.join(OUT_DIR, "verify_balance_NonFuelOpexCapExIndividualTS_Delta.csv")
    )
    verify_and_write(
        ts_ag_BASE, flat_ag_BASE, pots_BASE_res2, fuel_res_ts3,
        os.path.join(OUT_DIR, "verify_balance_NonFuelOpexCapExAggregate_Base.csv")
    )
    verify_and_write(
        ts_ag_DELTA, flat_ag_DELTA, pots_DELTA_res2, fuel_res_ts4,
        os.path.join(OUT_DIR, "verify_balance_NonFuelOpexCapExAggregate_Delta.csv")
    )

    # 13) Verification — ENERGY equality per ts×product: (U+C) vs served demand
    verify_energy_balance_per_product_ts(
        uc_per_prod_MW_res=uc_per_prod_MW_res,
        prod_res_served_ts=prod_res_served[["timestamp","P1","P2","P3","P4"]],
        out_csv=os.path.join(OUT_DIR, "verify_energy_balance_allocated_vs_served_per_product_ts.csv"),
        tol_MW=1e-6
    )

    # 14) Summary CSVs for all 4 variants
    fp_base_ind = write_variant_summary_csv(
        "NonFuelOpexCapExIndividualTS_Base",
        OUT_DIR,
        flat_ts_BASE,
        uc_per_prod_MWh_res,
        pots_BASE_res,
        fuel_res_ts,
        pots_BASE_total_ts,
        fuel_total_ts
    )
    fp_delta_ind = write_variant_summary_csv(
        "NonFuelOpexCapExIndividualTS_Delta",
        OUT_DIR,
        flat_ts_DELTA,
        uc_per_prod_MWh_res,
        pots_DELTA_res,
        fuel_res_ts2,
        pots_DELTA_total_ts,
        fuel_total_ts
    )
    fp_base_ag = write_variant_summary_csv(
        "NonFuelOpexCapExAggregate_Base",
        OUT_DIR,
        flat_ag_BASE,
        uc_per_prod_MWh_res,
        pots_BASE_res2,
        fuel_res_ts3,
        pots_BASE_total_ts,
        fuel_total_ts
    )
    fp_delta_ag = write_variant_summary_csv(
        "NonFuelOpexCapExAggregate_Delta",
        OUT_DIR,
        flat_ag_DELTA,
        uc_per_prod_MWh_res,
        pots_DELTA_res2,
        fuel_res_ts4,
        pots_DELTA_total_ts,
        fuel_total_ts
    )

    # 15) High-level AMM payment summary (treating everything as services; reserves added via subs)
    def _extract_services_totals(summary_csv: str):
        df = pd.read_csv(summary_csv)
        # Residential total cost = fuel + capex/opex (+ reserves, since total_gbp now includes reserves)
        res_total = float(
            df[
                (df["section"] == "residential_costs") &
                (df["metric"]  == "total_cost_gbp")
            ]["value"].sum()
        )
        # Non-residential total cost = fuel + capex/opex (no reserves added here)
        nonres_total = float(
            df[
                (df["section"] == "nonresidential_totals") &
                (df["metric"]  == "total_cost_gbp")
            ]["value"].sum()
        )
        return res_total, nonres_total

    amm1_res, amm1_nonres = _extract_services_totals(fp_base_ind)
    amm2_res, amm2_nonres = _extract_services_totals(fp_delta_ind)

    csv_rows = [
        {"case": "AMM1_BASE",  "customer_group": "Residential",     "total_services_gbp": amm1_res},
        {"case": "AMM1_BASE",  "customer_group": "Non-residential", "total_services_gbp": amm1_nonres},
        {"case": "AMM2_DELTA", "customer_group": "Residential",     "total_services_gbp": amm2_res},
        {"case": "AMM2_DELTA", "customer_group": "Non-residential", "total_services_gbp": amm2_nonres},
    ]
    df_summary = pd.DataFrame(csv_rows)
    out_fp = os.path.join(OUT_DIR, "payment_summary_AMM_services_incl_reserves.csv")
    df_summary.to_csv(out_fp, index=False)

    # 16) Reserves summary already written in step 1b; just print the per-HH amount
    print("\n[OK] Wrote AMM payment summary CSV to:")
    print(" ", out_fp)
    print("\n[OK] Wrote reserves allocation summary CSV to:")
    print(" ", os.path.join(OUT_DIR, "reserves_allocation_summary.csv"))
    print(
        "[INFO] Per-household reserves charge "
        f"(uniform across all P1–P4 households): "
        f"£{perHH_reserves_gbp_per_month:.2f} / HH / month"
    )
    print(f"[OK] Wrote outputs to: {OUT_DIR}")

if __name__ == "__main__":
    main()
