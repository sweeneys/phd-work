#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NODAL sliding UC/DC-OPF with TRUE BATTERIES (pay-as-bid) + EXPLICIT UP-RESERVES
— Lexicographic solve: (A) minimize shed → (B) minimize reserve slack → (C) minimize cost
— Optional reserve availability payment in objective (config toggle)
— Optional reserve duration (battery deliverability) via reserve_duration_hours
— Soft “local-first” via tiny wheeling charge on absolute line flows (reporting unaffected)
— Node-balance diagnostics & network flow-conservation checks
— Reporting split: local vs. import (reporting only; physics remain DCOPF)
— Ramping DISABLED

Inputs:  data/{network_uk.json, gens_static.csv, gen_profiles.csv, demand/D*_demand.csv, config.json}
Outputs: outputs/run_YYYYmmdd_HHMMSS/YYYY-MM-DD/...
"""

import os, json, glob, traceback
import numpy as np
import pandas as pd
import datetime

from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals, Reals, Binary, Objective,
    Constraint, minimize, value, Expression
)
from pyomo.opt import SolverFactory, TerminationCondition as TC, SolverStatus as SS

# ------------------------- Paths & setup -------------------------
BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data")
OUT_BASE = os.path.join(BASE, "outputs")

def _ensure_ts(df):
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    return df

def _even_step_minutes(ts_index: pd.DatetimeIndex) -> float:
    if len(ts_index) < 2: raise ValueError("Need >=2 timestamps.")
    d = np.diff(ts_index.view("i8"))
    if not np.all(d == d[0]): raise ValueError("Timestamps not evenly spaced.")
    return float(d[0]/1e9/60.0)

def load_cfg():
    p = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(p):
        with open(p,"r") as f: return json.load(f)
    return {}

CFG = load_cfg()

# Performance toggles (defaults are backward compatible)
USE_BIN = bool(CFG.get("use_binary_commitment", True))
DISABLE_MINUPDOWN = bool(CFG.get("disable_min_updown", False))
SINGLE_PASS = False  # overridden by lexicographic solve; keep False

# Fresh run folder (time-stamped)
RUN_TAG = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUT_ROOT = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_ROOT, exist_ok=True)
print(f"[info] Writing under fresh folder: {OUT_ROOT}")

# ------------------------- Input presence checks -------------------------
REQ = [
    ("network_uk.json", False),
    ("gens_static.csv", False),
    ("gen_profiles.csv", False),
    (os.path.join("demand","D*_demand.csv"), True),
]
for rel, is_glob in REQ:
    p = os.path.join(DATA_DIR, rel)
    if is_glob:
        if not glob.glob(p): raise FileNotFoundError(f"Required demand files not found: {p}")
    else:
        if not os.path.exists(p): raise FileNotFoundError(f"Required file not found: {p}")

# ------------------------- Config & constants -------------------------
SBASE = float(CFG.get("sbase_MVA", 1000.0))
SOLVER = CFG.get("solver", "highs")

RESERVE_PCT = float(CFG.get("reserve_requirement_percent", 10.0))
RESERVE_ON_SERVED = bool(CFG.get("reserve_on_served_demand", True))

# Availability payment toggle + duration (hours) for reserve deliverability
INCLUDE_RESERVE_PAY = bool(CFG.get("include_reserve_payment_in_objective", False))
# If omitted, default reserve duration == step length
RESERVE_DURATION_H = float(CFG.get("reserve_duration_hours", 0.0))  # 0.0 → will fallback to STEP_H later

# Reserve availability price (pay-as-bid from CONFIG)
RESERVE_PRICE_VALUE = float(CFG.get("reserve_availability_price_per_MW_h", 0.0))
RESERVE_PRICE_UNITS = str(CFG.get("reserve_availability_price_units", "currency_per_MW_h")).lower()
def _reserve_pay_amount(mw_expr, step_h):
    if RESERVE_PRICE_UNITS in ("currency_per_mw_h", "currency_per_mwh"):
        return RESERVE_PRICE_VALUE * mw_expr * step_h
    elif RESERVE_PRICE_UNITS == "currency_per_mw":
        return RESERVE_PRICE_VALUE * mw_expr
    else:
        return RESERVE_PRICE_VALUE * mw_expr * step_h

RESERVE_ALLOW_DROP_CHARGE = bool(CFG.get("reserve_allow_battery_drop_charge", False))

# Soft “local-first”: tiny penalty on absolute flow (per MWh moved)
FLOW_PENALTY = float(CFG.get("transmission_flow_penalty_per_MWh", 0.0))

BAT_LABELS = [s.lower() for s in CFG.get("battery_tech_labels", ["battery","Battery","BATTERY"])]
ETA_CH_DEF = float(CFG.get("battery_eta_charge", 0.95))
ETA_DIS_DEF = float(CFG.get("battery_eta_discharge", 0.95))
BAT_EXCL = bool(CFG.get("battery_exclusive_mode", False))

SOC_TARGET_FRAC = CFG.get("battery_terminal_soc_frac", 0.5)
SOC_TARGET_PENALTY = float(CFG.get("battery_terminal_soc_penalty_per_MWh", 5.0))
SOC_WINDOW_END_MIN_FRAC = CFG.get("battery_window_end_soc_min_frac", None)

SPILL_PENALTY = float(CFG.get("spill_penalty_per_MWh", 1e-9))
SHED_FIX_TOL = float(CFG.get("shed_fix_tolerance_MWh", 1e-6))
VOLL = float(CFG.get("voll_MWh", 9999.0))

WINDOW_HOURS = float(CFG.get("uc_window_hours", 72.0))
COMMIT_HOURS = float(CFG.get("uc_commit_hours", 24.0))
DISALLOW_LATE_STARTS = bool(CFG.get("disallow_late_starts", True))

MUST_RUN_IDS_CFG  = set(str(x) for x in CFG.get("must_run_gen_ids", []))
MUST_RUN_TECH_LBL = [s.lower() for s in CFG.get("must_run_tech_labels", ["nuclear"])]
MUST_RUN_MODE = str(CFG.get("must_run_mode", "soft")).lower()
MUST_RUN_OFF_PENALTY_PER_H = float(CFG.get("must_run_off_penalty_per_hour", 1e6))

# ------------------------- Incremental output helpers -------------------------
def _append_df(path, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

def _rewrite_filtered(path, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def _rebuild_daily_settlement_and_summary(day_dir, step_h):
    disp_fp = os.path.join(day_dir, "dispatch.csv")
    serv_fp = os.path.join(day_dir, "served_unserved.csv")
    if not (os.path.exists(disp_fp) and os.path.exists(serv_fp)):
        return

    ddisp = pd.read_csv(disp_fp, parse_dates=["timestamp"])
    req_cols = {"timestamp","gen_id","node","p_MW","cost_MWh"}
    missing = req_cols - set(ddisp.columns)
    if missing:
        raise ValueError(f"{disp_fp} missing columns: {sorted(missing)}")

    # --- ENERGY PAY (pay-as-bid on energy) ---
    if "p_dis_MW" in ddisp.columns:
        ddisp["MWh"]     = ddisp["p_MW"] * step_h
        ddisp["dis_MWh"] = ddisp["p_dis_MW"].fillna(0.0) * step_h
        ddisp["pay_energy"] = ddisp.apply(
            lambda r: (r["MWh"] * r["cost_MWh"]) if pd.isna(r.get("p_dis_MW"))
                      else (r["dis_MWh"] * r["cost_MWh"]),
            axis=1
        )
    else:
        ddisp["MWh"] = ddisp["p_MW"] * ddisp["cost_MWh"] * 0.0  # ensure column exists
        ddisp["MWh"] = ddisp["p_MW"] * step_h
        ddisp["pay_energy"] = ddisp["MWh"] * ddisp["cost_MWh"]

    # --- RESERVE PAY-AS-BID (CONFIG PRICE) ---
    ddisp_res = ddisp.copy()
    ddisp_res["reserve_MW"]  = pd.to_numeric(ddisp_res.get("reserve_up_MW", 0.0), errors="coerce").fillna(0.0)

    units = RESERVE_PRICE_UNITS.lower()
    if units in ("currency_per_mw_h", "currency_per_mwh"):
        ddisp_res["reserve_price"] = RESERVE_PRICE_VALUE
        ddisp_res["reserve_MWh"]   = ddisp_res["reserve_MW"] * step_h
        ddisp_res["pay_reserve"]   = ddisp_res["reserve_MWh"] * ddisp_res["reserve_price"]
    elif units == "currency_per_mw":
        ddisp_res["reserve_price"] = RESERVE_PRICE_VALUE
        ddisp_res["reserve_MWh"]   = ddisp_res["reserve_MW"] * step_h
        ddisp_res["pay_reserve"]   = ddisp_res["reserve_MW"] * ddisp_res["reserve_price"]
    else:
        ddisp_res["reserve_price"] = RESERVE_PRICE_VALUE
        ddisp_res["reserve_MWh"]   = ddisp_res["reserve_MW"] * step_h
        ddisp_res["pay_reserve"]   = ddisp_res["reserve_MWh"] * ddisp_res["reserve_price"]

    price_col_name = ("avg_reserve_price_MW_h"
                      if units in ("currency_per_mw_h","currency_per_mwh")
                      else "avg_reserve_price_MW")

    settle = (ddisp_res.groupby("gen_id", as_index=False)
              .agg(node=("node","last"),
                   MWh=("MWh","sum"),
                   reserve_MWh=("reserve_MWh","sum"),
                   avg_energy_price_MWh=("cost_MWh","mean"),
                   **{price_col_name: ("reserve_price","mean")},
                   pay_energy=("pay_energy","sum"),
                   pay_reserve=("pay_reserve","sum"))
              .assign(pay_cost=lambda df: df["pay_energy"] + df["pay_reserve"])
              .sort_values("gen_id")
    )

    settle.to_csv(os.path.join(day_dir, "settlement_pay_as_bid.csv"), index=False)

    dserv = pd.read_csv(serv_fp, parse_dates=["timestamp"])
    served_MWh   = float(dserv["served_MW"].sum()   * step_h)
    unserved_MWh = float(dserv["unserved_MW"].sum() * step_h)
    total_cost   = float(settle["pay_cost"].sum())
    with open(os.path.join(day_dir, "run_summary.txt"), "w") as f:
        f.write(f"Step (minutes): {step_h*60:.0f}\n")
        f.write(f"Total paid MWh (so far): {settle['MWh'].sum():,.2f}\n")
        f.write(f"Total reserve MWh (so far): {settle['reserve_MWh'].sum():,.2f}\n")
        f.write(f"Total pay-as-bid energy (£): {settle['pay_energy'].sum():,.2f}\n")
        f.write(f"Total pay-as-bid reserves (£): {settle['pay_reserve'].sum():,.2f}\n")
        f.write(f"Total pay-as-bid cost (£): {total_cost:,.2f}\n")
        f.write(f"Served MWh (so far): {served_MWh:,.2f}\n")
        f.write(f"Unserved MWh (so far): {unserved_MWh:,.2f}\n")

# ------------------------- Load network (NODAL) -------------------------
with open(os.path.join(DATA_DIR, "network_uk.json"), "r") as f:
    NET = json.load(f)

NODES = [str(n) for n in NET["nodes"]]
EDGES = [(str(a), str(b)) for a,b in NET["edges"]]

CAPMAP = {str(k): float(v) for k,v in NET["edge_capacity"].items()}
VOLT   = NET.get("edge_voltage_kV", None)
LKM    = NET.get("edge_length_km", None)
XPU    = NET.get("edge_reactance_pu", None)

def _get_edge_attr(mapdict, a, b, default=None):
    if mapdict is None: return default
    k1, k2 = f"{a},{b}", f"{b},{a}"
    if k1 in mapdict: return mapdict[k1]
    if k2 in mapdict: return mapdict[k2]
    return default

def _x_from_vl(voltage_kV, length_km, sbase_MVA):
    tbl = {400:0.30, 275:0.40, 132:0.60}  # Ω/km typical
    if voltage_kV is None or length_km is None: return None
    okm = tbl.get(int(voltage_kV))
    if okm is None: return None
    x_ohm = okm*float(length_km)
    return x_ohm * float(sbase_MVA) / (float(voltage_kV)**2)

LINE_ROWS = []
for (a,b) in EDGES:
    cap = _get_edge_attr(CAPMAP,a,b,None)
    if cap is None: raise ValueError(f"Missing edge_capacity for {a}-{b}")
    x = _get_edge_attr(XPU,a,b,None) if XPU is not None else None
    if x is None:
        vk=_get_edge_attr(VOLT,a,b,None) if VOLT else None
        lk=_get_edge_attr(LKM,a,b,None)  if LKM  else None
        x = _x_from_vl(vk, lk, SBASE)
    if x is None or x<=0: x = 1.0
    LINE_ROWS.append({"line_id": f"L{len(LINE_ROWS)+1}", "from_node": a, "to_node": b,
                      "capacity_MW": float(cap), "x": float(x)})
LINES = pd.DataFrame(LINE_ROWS)

# ------------------------- Load data -------------------------
def load_demands_nodal():
    files = sorted(glob.glob(os.path.join(DATA_DIR,"demand","D*_demand.csv")))
    loads = NET.get("loads", {})
    did2node = {did:str(v["node"]) for did,v in loads.items()}
    rows=[]
    for fp in files:
        fname=os.path.basename(fp); did=fname.split("_")[0]
        if did not in did2node:
            raise ValueError(f"{fname}: demand id '{did}' not in network_uk.json loads.")
        node = did2node[did]
        df = pd.read_csv(fp); df=_ensure_ts(df)
        if "total_demand_MW" not in df.columns:
            pcols=[c for c in df.columns if str(c).endswith("_MW")]
            if not pcols: raise ValueError(f"{fname} missing total_demand_MW and no *_MW cols.")
            df["total_demand_MW"] = df[pcols].sum(axis=1)
        df["node"]=node
        rows.append(df[["timestamp","node","total_demand_MW"]].rename(columns={"total_demand_MW":"demand_MW"}))
    d = (pd.concat(rows, ignore_index=True)
           .groupby(["timestamp","node"], as_index=False)["demand_MW"].sum())
    return d

DEMAND = load_demands_nodal()

GSTAT = pd.read_csv(os.path.join(DATA_DIR,"gens_static.csv"))
GSTAT["gen_id"] = GSTAT["gen_id"].astype(str)
GSTAT["node"]   = GSTAT["node"].astype(str)
if "tech" not in GSTAT.columns: GSTAT["tech"] = None
for c in ["Pmax","Pmin","min_up_hours","min_down_hours",
          "E_cap_MWh","P_ch_max_MW","P_dis_max_MW","eta_ch","eta_dis",
          "soc_init_MWh","soc_min_frac","soc_max_frac",
          "energy_MWh","p_charge_max_MW","p_discharge_max_MW","eta_charge","eta_discharge",
          "can_reserve"]:
    if c in GSTAT.columns: GSTAT[c] = pd.to_numeric(GSTAT[c], errors="coerce")

# Eligibility for reserves
if "can_reserve" not in GSTAT.columns:
    GSTAT["can_reserve"] = 1
GSTAT["can_reserve"] = GSTAT["can_reserve"].fillna(1).astype(int)

# Identify batteries
is_batt = pd.Series(False, index=GSTAT.index)
is_batt = is_batt | GSTAT["tech"].astype(str).str.lower().isin(BAT_LABELS)
if "E_cap_MWh" in GSTAT.columns:
    is_batt = is_batt | (GSTAT["E_cap_MWh"].fillna(0) > 0)
if "energy_MWh" in GSTAT.columns:
    is_batt = is_batt | (GSTAT["energy_MWh"].fillna(0) > 0)
GSTAT["is_batt"] = is_batt.astype(bool)

GB = sorted(GSTAT.loc[GSTAT["is_batt"], "gen_id"].unique())   # batteries
GG = sorted(GSTAT.loc[~GSTAT["is_batt"], "gen_id"].unique())  # non-batts

# Must-run by tech/fuel/id among non-batts
mr_mask = pd.Series(False, index=GSTAT.index)
if "tech" in GSTAT.columns:
    mr_mask = mr_mask | GSTAT["tech"].astype(str).str.lower().isin(MUST_RUN_TECH_LBL)
if "fuel" in GSTAT.columns:
    mr_mask = mr_mask | GSTAT["fuel"].astype(str).str.lower().isin(MUST_RUN_TECH_LBL)
mr_mask = mr_mask | GSTAT["gen_id"].astype(str).isin(MUST_RUN_IDS_CFG)
MUST_RUN_IDS_ALL = sorted(GSTAT.loc[mr_mask & (~GSTAT["is_batt"]), "gen_id"].astype(str).unique())

# Profiles (attach node from GSTAT if missing)
GPROF = pd.read_csv(os.path.join(DATA_DIR, "gen_profiles.csv"))
GPROF = _ensure_ts(GPROF)
GPROF["gen_id"] = GPROF["gen_id"].astype(str)
if "node" not in GPROF.columns:
    g_map = GSTAT[["gen_id", "node"]].drop_duplicates()
    GPROF = GPROF.merge(g_map, on="gen_id", how="left")
    if GPROF["node"].isna().any():
        missing = sorted(GPROF.loc[GPROF["node"].isna(), "gen_id"].unique())
        raise ValueError(f"gen_profiles.csv missing node and couldn't map for gen_id(s): {missing[:10]}...")
GPROF["node"] = GPROF["node"].astype(str)
GPROF["avail_MW"] = pd.to_numeric(GPROF["avail_MW"], errors="coerce").fillna(0.0)
GPROF["cost_MWh"] = pd.to_numeric(GPROF.get("cost_MWh", 0.0), errors="coerce").fillna(0.0)

# ------------------------- Time grid -------------------------
ALL_TS = pd.DatetimeIndex(sorted(set(GPROF["timestamp"]) & set(DEMAND["timestamp"]))).tz_localize(None)
STEP_MIN = _even_step_minutes(ALL_TS)
STEP_H = STEP_MIN/60.0
STEPS_PER_WINDOW = int(round(WINDOW_HOURS / STEP_H))
STEPS_PER_COMMIT = int(round(COMMIT_HOURS / STEP_H))

# ------------------------- UC model builder -------------------------
def build_uc_window(ts_list, prev_p, on_state, soc_now, must_run_ids):
    GP = GPROF[GPROF["timestamp"].isin(ts_list)].copy()
    DEM = DEMAND[DEMAND["timestamp"].isin(ts_list)].copy()

    Gg = GG[:]
    Gb = GB[:]
    B  = list(NODES)
    L  = list(LINES["line_id"])
    T  = list(range(len(ts_list)))

    gnode = GSTAT.set_index("gen_id")["node"].to_dict()
    Pmax = GSTAT.set_index("gen_id")["Pmax"].fillna(np.inf).to_dict()
    Pmin = GSTAT.set_index("gen_id")["Pmin"].fillna(0.0).to_dict()
    can_res = GSTAT.set_index("gen_id")["can_reserve"].to_dict()

    # Time-varying availability & energy bids
    avail, bid = {}, {}
    for t_idx, ts in enumerate(ts_list):
        sub = GP[GP["timestamp"]==ts]
        amap = dict(zip(sub["gen_id"], sub["avail_MW"]))
        cmap = dict(zip(sub["gen_id"], sub["cost_MWh"]))
        for g in (Gg + Gb):
            a = float(amap.get(g, 0.0))
            avail[(g,t_idx)] = float(min(a, Pmax.get(g, np.inf)))
            bid[(g,t_idx)]   = float(cmap.get(g, 0.0))

    # Demand per bus,t
    Db = {(b,t):0.0 for b in B for t in T}
    for t_idx, ts in enumerate(ts_list):
        sub = DEM[DEM["timestamp"]==ts]
        s = sub.groupby("node")["demand_MW"].sum().reindex(B).fillna(0.0)
        for b in B:
            Db[(b,t_idx)] = float(s.loc[b])

    CAP_AT = {(g,t): float(min(avail[(g,t)], Pmax.get(g, np.inf))) for g in Gg for t in T}

    # Battery params
    BAT = {}
    for gid in Gb:
        r = GSTAT.loc[GSTAT["gen_id"]==gid].iloc[0]
        E = float(r.get("E_cap_MWh") if pd.notna(r.get("E_cap_MWh")) else r.get("energy_MWh", np.nan))
        if not np.isfinite(E) or E <= 0:
            raise ValueError(f"Battery {gid} requires positive E_cap_MWh/energy_MWh.")
        Pch_max = float(r.get("P_ch_max_MW") if pd.notna(r.get("P_ch_max_MW")) else r.get("p_charge_max_MW", np.nan))
        Pdis_max = float(r.get("P_dis_max_MW") if pd.notna(r.get("P_dis_max_MW")) else r.get("p_discharge_max_MW", np.nan))
        Pmax_row = float(r.get("Pmax")) if pd.notna(r.get("Pmax")) else np.nan
        if not np.isfinite(Pch_max):
            if np.isfinite(Pmax_row): Pch_max = Pmax_row
            else: raise ValueError(f"Battery {gid} missing P_ch_max_MW/Pmax.")
        if not np.isfinite(Pdis_max):
            Pdis_max = Pmax_row if np.isfinite(Pmax_row) else Pch_max
        eta_ch_v = float(r.get("eta_ch") if pd.notna(r.get("eta_ch")) else r.get("eta_charge", ETA_CH_DEF))
        eta_dis_v = float(r.get("eta_dis") if pd.notna(r.get("eta_dis")) else r.get("eta_discharge", ETA_DIS_DEF))
        if not (0 < eta_ch_v <= 1) or not (0 < eta_dis_v <= 1):
            raise ValueError(f"Battery {gid}: invalid eta_ch/eta_dis.")
        soc_min_frac = float(r.get("soc_min_frac", 0.0)) if pd.notna(r.get("soc_min_frac")) else 0.0
        soc_max_frac = float(r.get("soc_max_frac", 1.0)) if pd.notna(r.get("soc_max_frac")) else 1.0
        BAT[gid] = dict(E_cap=E, Pch_max=Pch_max, Pdis_max=Pdis_max,
                        eta_ch=eta_ch_v, eta_dis=eta_dis_v,
                        soc_min_frac=soc_min_frac, soc_max_frac=soc_max_frac)

    PCHmax = {g: BAT[g]["Pch_max"] for g in Gb}
    PDISmax = {g: BAT[g]["Pdis_max"] for g in Gb}
    Ecap    = {g: BAT[g]["E_cap"] for g in Gb}
    eta_ch  = {g: BAT[g]["eta_ch"] for g in Gb}
    eta_dis = {g: BAT[g]["eta_dis"] for g in Gb}
    soc_min = {g: BAT[g]["soc_min_frac"]*Ecap[g] for g in Gb}
    soc_max = {g: BAT[g]["soc_max_frac"]*Ecap[g] for g in Gb}

    # Lines & DC power flow
    line_from = dict(zip(LINES["line_id"], LINES["from_node"]))
    line_to   = dict(zip(LINES["line_id"], LINES["to_node"]))
    cap_line  = dict(zip(LINES["line_id"], LINES["capacity_MW"]))
    x_map     = dict(zip(LINES["line_id"], LINES["x"]))
    REF = B[0]

    # ---- Pyomo model ----
    m = ConcreteModel("nodal_uc_battery_window")
    m.Gg = Set(initialize=Gg, ordered=True)
    m.Gb = Set(initialize=Gb, ordered=True)
    m.B  = Set(initialize=B,  ordered=True)
    m.L  = Set(initialize=L,  ordered=True)
    m.T  = Set(initialize=T,  ordered=True)

    # Vars (commitment mode)
    if USE_BIN:
        m.U = Var(m.Gg, m.T, within=Binary)
        m.Y = Var(m.Gg, m.T, within=Binary)
        m.W = Var(m.Gg, m.T, within=Binary)
    else:
        m.U = Var(m.Gg, m.T, bounds=(0.0,1.0), within=NonNegativeReals)

    m.P = Var(m.Gg, m.T, within=NonNegativeReals)

    m.Pch  = Var(m.Gb, m.T, within=NonNegativeReals)
    m.Pdis = Var(m.Gb, m.T, within=NonNegativeReals)
    m.SOC  = Var(m.Gb, m.T, within=Reals)
    if BAT_EXCL:
        m.Yb = Var(m.Gb, m.T, within=Binary if USE_BIN else NonNegativeReals, bounds=(0,1))

    # Bus angles & flows
    m.Theta = Var(m.B, m.T, within=Reals)
    m.F     = Var(m.L, m.T, within=Reals)               # signed flow
    m.Fpos  = Var(m.L, m.T, within=NonNegativeReals)    # for |F|
    m.Fneg  = Var(m.L, m.T, within=NonNegativeReals)

    # Served/Shed/Spill + reporting split (local vs import)
    m.Served     = Var(m.B, m.T, within=NonNegativeReals)
    m.Shed       = Var(m.B, m.T, within=NonNegativeReals)
    m.Spill      = Var(m.B, m.T, within=NonNegativeReals)
    m.LocalServe = Var(m.B, m.T, within=NonNegativeReals)  # reporting
    m.Import     = Var(m.B, m.T, within=NonNegativeReals)  # reporting

    # Reserves
    m.Rsrv_g = Var(m.Gg, m.T, within=NonNegativeReals)
    m.Rsrv_b = Var(m.Gb, m.T, within=NonNegativeReals)
    m.ResSlack = Var(m.T, within=NonNegativeReals)

    # Must-run
    MUST_RUN_SET = set(must_run_ids)
    if MUST_RUN_MODE == "hard":
        def must_run_rule(m,g,t):
            if g in MUST_RUN_SET: return m.U[g,t] == 1
            return Constraint.Skip
        m.must_run = Constraint(m.Gg, m.T, rule=must_run_rule)
        mr_pen_expr = 0.0
    else:
        m.MR_off = Var(m.Gg, m.T, within=NonNegativeReals)
        def must_run_soft_rule(m,g,t):
            if g in MUST_RUN_SET: return m.U[g,t] + m.MR_off[g,t] >= 1
            return Constraint.Skip
        m.must_run_soft = Constraint(m.Gg, m.T, rule=must_run_soft_rule)
        mr_pen_expr = MUST_RUN_OFF_PENALTY_PER_H * sum(m.MR_off[g,t] * STEP_H for g in m.Gg for t in m.T)

    # Optional late-start handling (only meaningful with binaries)
    MU_h = GSTAT.set_index("gen_id")["min_up_hours"].fillna(0.0).to_dict()
    MD_h = GSTAT.set_index("gen_id")["min_down_hours"].fillna(0.0).to_dict()
    if USE_BIN and DISALLOW_LATE_STARTS:
        def no_late_starts(m,g,t):
            if g in MUST_RUN_SET: return Constraint.Skip
            Lh = int(round(MU_h[g] / STEP_H))
            if Lh <= 0: return Constraint.Skip
            latest_start = min(len(T)-1, int(round(COMMIT_HOURS/STEP_H)) - 1)
            if t > latest_start:
                return m.Y[g,t] == 0
            return Constraint.Skip
        m.no_late_starts = Constraint(m.Gg, m.T, rule=no_late_starts)

    # Reference angle & DC flows
    m.ref_angle = Constraint(m.T, rule=lambda m,t: m.Theta[REF, t] == 0.0)
    def flow_eq(m,l,t):
        fb=line_from[l]; tb=line_to[l]; x=x_map[l]
        return m.F[l,t] == (m.Theta[tb,t] - m.Theta[fb,t])/x
    m.flow_eq = Constraint(m.L, m.T, rule=flow_eq)
    m.flow_up = Constraint(m.L, m.T, rule=lambda m,l,t:  m.F[l,t] <=  cap_line[l])
    m.flow_dn = Constraint(m.L, m.T, rule=lambda m,l,t: -m.F[l,t] <=  cap_line[l])

    # Split F into positive/negative to form |F|
    m.flow_split = Constraint(m.L, m.T, rule=lambda m,l,t: m.F[l,t] == m.Fpos[l,t] - m.Fneg[l,t])

    # Non-battery bounds (no ramping)
    def _p_cap(m,g,t): return m.P[g,t] <= CAP_AT[(g,t)] * m.U[g,t]
    m.p_cap = Constraint(m.Gg, m.T, rule=_p_cap)
    m.p_min = Constraint(m.Gg, m.T, rule=lambda m,g,t: m.P[g,t] >= Pmin[g] * m.U[g,t])

    # Start/stop and min up/down (if binaries and not disabled)
    if USE_BIN:
        def start_stop(m,g,t):
            if t==0: return m.U[g,0] - on_state[g] == m.Y[g,0] - m.W[g,0]
            return m.U[g,t] - m.U[g,t-1] == m.Y[g,t] - m.W[g,t]
        m.start_stop = Constraint(m.Gg, m.T, rule=start_stop)

        if not DISABLE_MINUPDOWN:
            def min_up(m,g,t):
                Lh = int(round(MU_h[g]/STEP_H))
                if Lh<=0 or t+Lh-1 >= len(T): return Constraint.Skip
                return sum(m.U[g,k] for k in range(t, t+Lh)) >= Lh * m.Y[g,t]
            def min_down(m,g,t):
                Lh = int(round(MD_h[g]/STEP_H))
                if Lh<=0 or t+Lh-1 >= len(T): return Constraint.Skip
                return sum(1 - m.U[g,k] for k in range(t, t+Lh)) >= Lh * m.W[g,t]
            m.min_up = Constraint(m.Gg, m.T, rule=min_up)
            m.min_down = Constraint(m.Gg, m.T, rule=min_down)

    # Batteries (no ramping; only power caps + SOC)
    m.b_dis_cap = Constraint(m.Gb, m.T, rule=lambda m,g,t: m.Pdis[g,t] <= min(PDISmax[g], avail[(g,t)]))
    m.b_ch_cap  = Constraint(m.Gb, m.T, rule=lambda m,g,t: m.Pch[g,t]  <= PCHmax[g])
    if BAT_EXCL:
        m.b_excl_1 = Constraint(m.Gb, m.T, rule=lambda m,g,t: m.Pdis[g,t] <= PDISmax[g] * (1 - (m.Yb[g,t] if USE_BIN else m.Yb[g,t])))
        m.b_excl_2 = Constraint(m.Gb, m.T, rule=lambda m,g,t: m.Pch[g,t]  <= PCHmax[g]  * (m.Yb[g,t]))

    def soc_link(m,g,t):
        if t==0:
            return m.SOC[g,0] == soc_now[g] + eta_ch[g]*m.Pch[g,0]*STEP_H - (1.0/eta_dis[g])*m.Pdis[g,0]*STEP_H
        return m.SOC[g,t] == m.SOC[g,t-1] + eta_ch[g]*m.Pch[g,t]*STEP_H - (1.0/eta_dis[g])*m.Pdis[g,t]*STEP_H
    m.soc_link = Constraint(m.Gb, m.T, rule=soc_link)
    m.soc_lo = Constraint(m.Gb, m.T, rule=lambda m,g,t: m.SOC[g,t] >= soc_min[g])
    m.soc_hi = Constraint(m.Gb, m.T, rule=lambda m,g,t: m.SOC[g,t] <= soc_max[g])

    # --------- EXPLICIT RESERVES ----------
    def _rg_headroom(m,g,t):
        if can_res.get(g,1) != 1:
            return m.Rsrv_g[g,t] <= 0.0
        return m.Rsrv_g[g,t] <= (CAP_AT[(g,t)] - m.P[g,t])
    m.rsrv_headroom_g = Constraint(m.Gg, m.T, rule=_rg_headroom)

    m.rsrv_commit_g = Constraint(m.Gg, m.T, rule=lambda m,g,t:
        m.Rsrv_g[g,t] <= CAP_AT[(g,t)] * m.U[g,t]
    )

    def _rb_headroom(m,g,t):
        if can_res.get(g,1) != 1:
            return m.Rsrv_b[g,t] <= 0.0
        if RESERVE_ALLOW_DROP_CHARGE:
            return m.Rsrv_b[g,t] <= (PDISmax[g] - m.Pdis[g,t] + m.Pch[g,t])
        else:
            return m.Rsrv_b[g,t] <= (PDISmax[g] - m.Pdis[g,t])
    m.rsrv_headroom_b = Constraint(m.Gb, m.T, rule=_rb_headroom)

    # Reserve duration: if not given, default to step length
    _RES_H = RESERVE_DURATION_H if RESERVE_DURATION_H > 0 else STEP_H
    m.rsrv_soc_b = Constraint(m.Gb, m.T, rule=lambda m,g,t:
        m.Rsrv_b[g,t] <= eta_dis[g] * m.SOC[g,t] / max(_RES_H, 1e-6)
    )

    # Serve-local-first accounting & KCL
    m.served_le_demand = Constraint(m.B, m.T, rule=lambda m,b,t: m.Served[b,t] <= Db[(b,t)])
    m.shed_le_demand   = Constraint(m.B, m.T, rule=lambda m,b,t: m.Shed[b,t]   <= Db[(b,t)])

    # Local vs Import split (reporting only)
    m.local_cap = Constraint(m.B, m.T, rule=lambda m,b,t:
        m.LocalServe[b,t] <=
        sum(m.P[g,t] for g in m.Gg if gnode[g]==b) +
        sum(m.Pdis[g,t] - m.Pch[g,t] for g in m.Gb if gnode[g]==b)
    )
    m.serve_split = Constraint(m.B, m.T, rule=lambda m,b,t:
        m.Served[b,t] == m.LocalServe[b,t] + m.Import[b,t]
    )

    def node_supply_equals_use(m,b,t):
        gen_nb = sum(m.P[g,t] for g in m.Gg if gnode[g]==b)
        gen_b  = sum(m.Pdis[g,t] - m.Pch[g,t] for g in m.Gb if gnode[g]==b)
        inflow = sum(+m.F[l,t] for l in m.L if line_to[l]==b) \
               + sum(-m.F[l,t] for l in m.L if line_from[l]==b)
        return gen_nb + gen_b + inflow == m.Served[b,t] + m.Spill[b,t]
    m.node_supply_equals_use = Constraint(m.B, m.T, rule=node_supply_equals_use)
    m.shed_def = Constraint(m.B, m.T, rule=lambda m,b,t: m.Shed[b,t] == Db[(b,t)] - m.Served[b,t])

    # System reserve requirement
    def reserve_req(m,t):
        Dtot = sum(Db[(b,t)] for b in B)
        if RESERVE_ON_SERVED:
            Shed_t = sum(m.Shed[b,t] for b in B)
            req = (RESERVE_PCT/100.0) * max(0.0, Dtot) - (RESERVE_PCT/100.0) * Shed_t
        else:
            req = (RESERVE_PCT/100.0) * max(0.0, Dtot)
        elig_g = sum(m.Rsrv_g[g,t] for g in Gg if can_res.get(g,1)==1)
        elig_b = sum(m.Rsrv_b[g,t] for g in Gb if can_res.get(g,1)==1)
        return (elig_g + elig_b + m.ResSlack[t]) >= req
    m.reserve_req = Constraint(m.T, rule=reserve_req)

    # Terminal SOC
    last_t = T[-1]
    soc_pen  = 0.0
    if SOC_TARGET_FRAC is not None:
        m.SOCshort = Var(m.Gb, within=NonNegativeReals)
        def soc_target_def(m,g):
            target = float(SOC_TARGET_FRAC) * Ecap[g]
            return m.SOCshort[g] >= target - m.SOC[g, last_t]
        m.soc_target_def = Constraint(m.Gb, rule=soc_target_def)
        soc_pen = SOC_TARGET_PENALTY * sum(m.SOCshort[g] for g in m.Gb)
    if SOC_WINDOW_END_MIN_FRAC is not None:
        def soc_window_min(m,g):
            return m.SOC[g, last_t] >= float(SOC_WINDOW_END_MIN_FRAC) * Ecap[g]
        m.soc_window_min = Constraint(m.Gb, rule=soc_window_min)

    # Cost pieces (NO reserve slack penalty here; handled lexicographically)
    energy_nonbat = sum(bid[(g,t)] * m.P[g,t]     * STEP_H for g in Gg for t in T)
    energy_bat    = sum(bid[(g,t)] * m.Pdis[g,t]  * STEP_H for g in Gb for t in T)
    spill_pen     = SPILL_PENALTY * sum(m.Spill[b,t] * STEP_H for b in B for t in T)

    # Optional availability payment for reserves
    reserve_pay = (
        sum(_reserve_pay_amount(m.Rsrv_g[g,t], STEP_H) for g in Gg for t in T if can_res.get(g,1)==1) +
        sum(_reserve_pay_amount(m.Rsrv_b[g,t], STEP_H) for g in Gb for t in T if can_res.get(g,1)==1)
    )
    if not INCLUDE_RESERVE_PAY:
        reserve_pay = 0.0

    # |Flow| penalty (soft local-first)
    wheel_cost = FLOW_PENALTY * sum((m.Fpos[l,t] + m.Fneg[l,t]) * STEP_H for l in m.L for t in m.T)

    # Summaries
    m.total_cost = Expression(expr=energy_nonbat + energy_bat + soc_pen + spill_pen
                              + mr_pen_expr + reserve_pay + wheel_cost)
    m.total_shed = Expression(expr=sum(m.Shed[b,t] * STEP_H for b in B for t in T))
    # Time-weighted reserve slack for clear magnitude
    m.total_reserve_slack = Expression(expr=sum(m.ResSlack[t] * STEP_H for t in m.T))

    # We will add Objectives dynamically in the solve loop (lexicographic)
    meta = dict(Gg=Gg, Gb=Gb, B=B, L=L, T=T, ts_list=ts_list,
                line_from=line_from, line_to=line_to, gnode=gnode)
    return m, meta

# ------------------------- Sliding driver (lexicographic) -------------------------
def run_uc_sliding(global_start="2024-01-01 00:00", global_end=None, robust=True):
    if global_end is None:
        global_end = CFG.get("target_end_ts", "2024-12-31 23:30")

    ts_all = ALL_TS[(ALL_TS>=pd.Timestamp(global_start)) & (ALL_TS<=pd.Timestamp(global_end))]
    if len(ts_all)==0: raise ValueError("No overlapping timestamps in requested window.")

    # Rolling state (no ramping)
    prev_p   = {g: 0.0 for g in GG}
    on_state = {g: 0   for g in GG}

    # Battery SOC init
    soc_now = {}
    for g in GB:
        row = GSTAT.loc[GSTAT["gen_id"] == g].iloc[0]
        E = row.get("E_cap_MWh")
        if pd.isna(E) and "energy_MWh" in GSTAT.columns:
            E = row.get("energy_MWh")
        if pd.isna(E) or float(E) <= 0:
            raise ValueError(f"Battery {g} requires 'E_cap_MWh' (or legacy 'energy_MWh') > 0.")
        E = float(E)
        s0 = row.get("soc_init_MWh")
        if pd.isna(s0): s0 = 0.5 * E
        smf = float(row.get("soc_min_frac")) if pd.notna(row.get("soc_min_frac")) else 0.0
        smx = float(row.get("soc_max_frac")) if pd.notna(row.get("soc_max_frac")) else 1.0
        soc_now[g] = min(max(smf*E, float(s0)), smx*E)

    # Must-run init
    Pmin_map = GSTAT.set_index("gen_id")["Pmin"].fillna(0.0).to_dict()
    must_run_ids = [g for g in MUST_RUN_IDS_ALL if g in GG]
    for g in must_run_ids:
        on_state[g] = 1
        prev_p[g]   = float(Pmin_map.get(g, 0.0))

    days_touched = set()
    log_path = os.path.join(OUT_ROOT, "run_errors.log")

    def solve_or_raise(opt, m, tag):
        res = opt.solve(m, tee=False)
        term = res.solver.termination_condition
        status = res.solver.status
        ok = (term in (TC.optimal, TC.feasible, TC.locallyOptimal)) and (status == SS.ok)
        if not ok:
            raise RuntimeError(f"Solve {tag} failed: status={status}, term={term}")
        return res

    t_list = list(ts_all); n = len(t_list); k = 0
    while True:
        start_idx = k * STEPS_PER_COMMIT
        if start_idx >= n: break
        end_idx = min(n-1, start_idx + STEPS_PER_WINDOW - 1)
        window_ts = t_list[start_idx:end_idx+1]
        if not window_ts: break

        try:
            print(f"[window] {window_ts[0]} -> {window_ts[-1]}  ({len(window_ts)} steps)")
            m, meta = build_uc_window(window_ts, prev_p, on_state, soc_now, must_run_ids)

            opt = SolverFactory(SOLVER)
            timelim = int(CFG.get("solver_time_limit_s", 600))
            s = SOLVER.lower()
            try:
                if s == "glpk":               opt.options["tmlim"] = timelim
                elif s == "cbc":              opt.options["seconds"] = timelim
                elif s in ("highs","highs_persistent"):
                    opt.options["time_limit"] = float(timelim)
                    if USE_BIN:
                        opt.options["mip_rel_gap"] = float(CFG.get("solver_mip_gap", 0.02))
                elif s == "gurobi":
                    opt.options["TimeLimit"] = timelim
                    if USE_BIN:
                        opt.options["MIPGap"] = float(CFG.get("solver_mip_gap", 0.02))
                elif s == "cplex":
                    opt.options["timelimit"] = timelim
                    if USE_BIN:
                        opt.options["mip tolerances mipgap"] = float(CFG.get("solver_mip_gap", 0.02))
            except Exception:
                pass

            # --- LEXICOGRAPHIC SOLVING: Stage A → Stage B → Stage C ---
            # Stage A: minimize energy shed
            if hasattr(m, "obj"): m.del_component(m.obj)
            m.obj_shed = Objective(expr=m.total_shed, sense=minimize)
            solve_or_raise(opt, m, "stage-A (shed)")
            min_shed = float(value(m.total_shed))

            # Stage B: with shed fixed, minimize reserve slack
            m.del_component(m.obj_shed)
            if hasattr(m, "shed_cap"): m.del_component(m.shed_cap)
            m.shed_cap = Constraint(expr=m.total_shed <= min_shed + SHED_FIX_TOL)
            m.obj_rsrv_slack = Objective(expr=m.total_reserve_slack, sense=minimize)
            solve_or_raise(opt, m, "stage-B (reserve slack)")
            min_slack = float(value(m.total_reserve_slack))

            # Stage C: with shed & slack fixed, minimize cost
            m.del_component(m.obj_rsrv_slack)
            if hasattr(m, "rsrv_cap"): m.del_component(m.rsrv_cap)
            m.rsrv_cap = Constraint(expr=m.total_reserve_slack <= min_slack + 1e-9)
            m.obj_cost = Objective(expr=m.total_cost, sense=minimize)
            solve_or_raise(opt, m, "stage-C (cost)")

            ts_list = meta["ts_list"]; Gg=meta["Gg"]; Gb=meta["Gb"]; Lm=meta["L"]; Bm=meta["B"]

            # Commit only first COMMIT window; 
            commit_len = min(STEPS_PER_COMMIT, len(ts_list))
            commit_T_idx = list(range(commit_len))

            # Collect outputs
            block_disp, block_flow, block_serv = [], [], []
            block_check, block_flowcon = [], []
            for ti in commit_T_idx:
                tstamp = pd.Timestamp(ts_list[ti])

                # Dispatch (gens)
                for g in Gg:
                    p = float(value(m.P[g,ti]))
                    r_up = float(value(m.Rsrv_g[g,ti]))
                    try:
                        cost = float(GPROF.loc[(GPROF["timestamp"]==tstamp)&(GPROF["gen_id"]==g),"cost_MWh"].values[0])
                    except Exception:
                        cost = 0.0
                    b = GSTAT.loc[GSTAT["gen_id"]==g,"node"].values[0]
                    block_disp.append({
                        "timestamp": tstamp, "gen_id": g, "node": b,
                        "p_MW": p, "cost_MWh": cost,
                        "reserve_up_MW": r_up
                    })

                # Dispatch (batteries)
                for g in Gb:
                    pdis = float(value(m.Pdis[g,ti])); pch = float(value(m.Pch[g,ti]))
                    r_up = float(value(m.Rsrv_b[g,ti]))
                    try:
                        cost = float(GPROF.loc[(GPROF["timestamp"]==tstamp)&(GPROF["gen_id"]==g),"cost_MWh"].values[0])
                    except Exception:
                        cost = 0.0
                    b = GSTAT.loc[GSTAT["gen_id"]==g,"node"].values[0]
                    block_disp.append({
                        "timestamp": tstamp, "gen_id": g, "node": b,
                        "p_MW": (pdis - pch), "cost_MWh": cost,
                        "p_dis_MW": pdis, "p_ch_MW": pch,
                        "reserve_up_MW": r_up
                    })

                # Flows
                for l in Lm:
                    block_flow.append({"timestamp": tstamp, "line_id": l,
                                       "from_node": meta["line_from"][l], "to_node": meta["line_to"][l],
                                       "flow_MW": float(value(m.F[l,ti]))})

                # served/unserved and diagnostics (+ local/import reporting)
                Db_now = DEMAND[DEMAND["timestamp"]==tstamp].groupby("node")["demand_MW"].sum().reindex(Bm).fillna(0.0)
                net_inflow_sum_t = 0.0

                for b in Bm:
                    demand = float(Db_now.loc[b])
                    servedv = float(value(m.Served[b,ti])); shedv = float(value(m.Shed[b,ti])); spillv = float(value(m.Spill[b,ti]))
                    servedv = min(max(servedv, 0.0), demand)
                    shedv   = min(max(shedv,   0.0), demand)

                    localv  = float(value(m.LocalServe[b,ti]))
                    importv = float(value(m.Import[b,ti]))

                    block_serv.append({"timestamp": tstamp, "node": b, "demand_MW": demand,
                                       "served_MW": servedv, "unserved_MW": shedv,
                                       "local_serve_MW": localv, "import_MW": importv})

                    gen_nb = sum(float(value(m.P[g,ti])) for g in Gg if meta["gnode"][g] == b)
                    gen_b  = sum(float(value(m.Pdis[g,ti]) - value(m.Pch[g,ti])) for g in Gb if meta["gnode"][g] == b)
                    inflow = sum(+float(value(m.F[l,ti])) for l in Lm if meta["line_to"][l] == b) \
                           + sum(-float(value(m.F[l,ti])) for l in Lm if meta["line_from"][l] == b)

                    lhs = gen_nb + gen_b + inflow
                    rhs = servedv + spillv
                    residual = lhs - rhs

                    flags = []
                    if servedv > demand + 1e-6: flags.append("served>demand")
                    if shedv   > demand + 1e-6: flags.append("unserved>demand")
                    if spillv  < -1e-6:         flags.append("spill<0")
                    if abs(residual) > 1e-3:    flags.append("node_balance_mismatch")

                    block_check.append({
                        "timestamp": tstamp,
                        "node": b,
                        "gen_nonbat_MW": gen_nb,
                        "batt_net_MW": gen_b,
                        "net_inflow_MW": inflow,
                        "supply_MW": lhs,
                        "demand_MW": demand,
                        "served_MW": servedv,
                        "unserved_MW": shedv,
                        "spill_MW": spillv,
                        "lhs_minus_rhs_MW": residual,
                        "flags": ";".join(flags)
                    })

                    net_inflow_sum_t += inflow

                block_flowcon.append({
                    "timestamp": tstamp,
                    "sum_net_inflow_MW": net_inflow_sum_t,
                    "flow_conservation_ok": abs(net_inflow_sum_t) <= 1e-3
                })

            # Write per-day files
            if block_disp or block_flow or block_serv or block_check or block_flowcon:
                bd = pd.DataFrame(block_disp) if block_disp else pd.DataFrame(columns=["timestamp"])
                bf = pd.DataFrame(block_flow) if block_flow else pd.DataFrame(columns=["timestamp"])
                bs = pd.DataFrame(block_serv) if block_serv else pd.DataFrame(columns=["timestamp"])
                bc = pd.DataFrame(block_check) if block_check else pd.DataFrame(columns=["timestamp"])
                bfc= pd.DataFrame(block_flowcon) if block_flowcon else pd.DataFrame(columns=["timestamp"])

                for df in (bd, bf, bs, bc, bfc):
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                        df["day"] = df["timestamp"].dt.date

                days_to_write = set()
                for df in (bd, bf, bs, bc, bfc):
                    if not df.empty: days_to_write |= set(df["day"])

                for day in sorted(days_to_write):
                    out_dir = os.path.join(OUT_ROOT, str(day)); os.makedirs(out_dir, exist_ok=True)
                    if not bd.empty:
                        _append_df(os.path.join(out_dir, "dispatch.csv"), bd[bd["day"]==day].drop(columns=["day"]))
                    if not bf.empty:
                        _append_df(os.path.join(out_dir, "flows.csv"), bf[bf["day"]==day].drop(columns=["day"]))
                    if not bs.empty:
                        _append_df(os.path.join(out_dir, "served_unserved.csv"), bs[bs["day"]==day].drop(columns=["day"]))
                    if not bc.empty:
                        _append_df(os.path.join(out_dir, "node_balance_check.csv"), bc[bc["day"]==day].drop(columns=["day"]))
                    if not bfc.empty:
                        _append_df(os.path.join(out_dir, "flow_conservation_check.csv"), bfc[bfc["day"]==day].drop(columns=["day"]))
                    _rebuild_daily_settlement_and_summary(out_dir, STEP_H)
                    days_touched.add(str(day))

            # Update rolling state at last committed step
            if commit_T_idx:
                last_ti = commit_T_idx[-1]
                for g in Gg:
                    prev_p[g] = float(value(m.P[g,last_ti]))
                    on_state[g] = 1 if prev_p[g] > 0.0 else 0
                for g in Gb:
                    soc_now[g] = float(value(m.SOC[g,last_ti]))

        except Exception as e:
            if robust:
                with open(log_path,"a") as L:
                    L.write(f"\n=== ERROR window {window_ts[0]}->{window_ts[-1]} ===\n{traceback.format_exc()}\n")
                print(f"[ERR] window {window_ts[0]}->{window_ts[-1]} failed: {e}")
            else:
                raise

        k += 1  # slide

    # Finalize touched days
    for day in sorted(days_touched):
        out_dir = os.path.join(OUT_ROOT, day)
        _rebuild_daily_settlement_and_summary(out_dir, STEP_H)
        print(f"[ok] Finalized day -> {OUT_ROOT}/{day}")

def run(start="2024-01-01 00:00", end=None, robust=True):
    run_uc_sliding(start, end, robust)

# ------------------------- Entry -------------------------
if __name__ == "__main__":
    run("2024-01-01 00:00", CFG.get("target_end_ts", "2024-12-31 23:30"), robust=True)
