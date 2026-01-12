#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DA UC (MILP or LP) + RT SCED (LP, native 30-min) with TRUE batteries, EXPLICIT reserves & LMP pricing
— Ramping DISABLED in both passes
— Reserves: EXPLICIT allocation R[g,t] for ALL eligible units (gas + batteries)
— Serve-local-first (optional) for congestion nudging
— Writes *both* RT-style files (sced_*.csv, settlement_RT.csv) and generic files
    (dispatch.csv, flows.csv, served_unserved.csv, node_balance_check.csv)
— NEW:
    * Pay ONLY for allocated reserves at system price (config)
    * Optional: allow charging batteries to contribute up-reserve by dropping charge
      (config: "reserve_allow_battery_drop_charge": true|false; default false)

RUNTIME / COMPUTATION TRACKING (NEW):
— Writes out_root/runtime.csv (one row per day window)
— Writes out_root/runtime_by_step.csv (one row per RT interval; optional via --runtime-by-step)
— Tracks:
    * DA build/solve time, RT build/solve time, pricing time
    * model sizes (#vars, #constraints) for DA and RT
    * objective values (DA & RT)
    * delivered energy stats (served/unserved/spill in MWh)
    * reserve shortfall (max MW and MWh-equivalent)
    * pricing method (dual vs finite_difference)
    * failure stage & termination condition
"""

import os, json, glob, argparse
import time
import platform
import hashlib
import sys

import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals, Reals, Binary, Constraint,
    Objective, minimize, Suffix, value
)
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.opt import ProblemFormat as PF  # for LP/MPS dumps on infeasible runs

# ---------------- small helpers ----------------

def _check_even_step(timestamps) -> float:
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) < 2:
        raise ValueError("Need >=2 timestamps to infer step size.")
    deltas = np.diff(ts.view('i8'))
    if not np.all(deltas == deltas[0]):
        raise ValueError("Timestamps not evenly spaced.")
    return float(deltas[0] / 1e9 / 60.0)  # minutes

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df
    first = df.columns[0]
    return df.rename(columns={first: "timestamp"})

def _utc_naive(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True).dt.tz_convert(None)
    return df

def _series_or_const(df: pd.DataFrame, col: str, const: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(const)
    return pd.Series(const, index=df.index, dtype="float64")

def _get_edge_attr(mapdict, a, b, default=None):
    k1, k2 = f"{a},{b}", f"{b},{a}"
    if mapdict is None:
        return default
    if k1 in mapdict:
        return mapdict[k1]
    if k2 in mapdict:
        return mapdict[k2]
    return default

def _x_from_voltage_length(voltage_kV, length_km, sbase_MVA):
    tbl = {400: 0.30, 275: 0.40, 132: 0.60}  # Ω/km proxy
    if voltage_kV is None or length_km is None:
        return None
    ohm_per_km = tbl.get(int(voltage_kV))
    if ohm_per_km is None:
        return None
    x_ohm = ohm_per_km * float(length_km)
    return x_ohm * float(sbase_MVA) / (float(voltage_kV) ** 2)

def dump_model(model, stem):
    try:
        model.write(stem + ".lp", format=PF.cpxlp,
                    io_options={'symbolic_solver_labels': True})
    except Exception:
        pass
    try:
        model.write(stem + ".mps", format=PF.mps,
                    io_options={'symbolic_solver_labels': True})
    except Exception:
        pass

# ---------------- runtime/computation helpers (NEW) ----------------

def _model_size(m):
    """Return (n_vars, n_cons) for active components."""
    try:
        nvars = sum(1 for _ in m.component_data_objects(Var, active=True))
        ncons = sum(1 for _ in m.component_data_objects(Constraint, active=True))
        return int(nvars), int(ncons)
    except Exception:
        return np.nan, np.nan

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _ts_hash(ts_index: pd.DatetimeIndex):
    """Short hash to identify windows without dumping all timestamps."""
    try:
        s = ",".join(pd.DatetimeIndex(ts_index).strftime("%Y-%m-%dT%H:%M"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    except Exception:
        return ""

def _energy_MWh_from_MW_series(values_MW, step_h):
    return float(np.nansum(values_MW) * float(step_h))

def _now_iso_utc():
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------- I/O ----------------

def load_config(data_dir: str):
    p = os.path.join(data_dir, "config.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}

def load_network_as_nodes(path_json: str, sbase_MVA=1000.0):
    import json as _json
    with open(path_json, "r") as f:
        net = _json.load(f)
    nodes = [str(n) for n in net["nodes"]]
    edges = [(str(a), str(b)) for a, b in net["edges"]]
    capmap = {str(k): float(v) for k, v in net["edge_capacity"].items()}
    voltmap = net.get("edge_voltage_kV", None)
    lenmap = net.get("edge_length_km", None)
    xmap_pu = net.get("edge_reactance_pu", None)

    rows = []
    lid = 1
    for a, b in edges:
        cap = _get_edge_attr(capmap, a, b, None)
        if cap is None:
            raise ValueError(f"Missing edge_capacity for {a}-{b}")
        x = _get_edge_attr(xmap_pu, a, b, None) if xmap_pu else None
        if x is None:
            vk = _get_edge_attr(voltmap, a, b, None) if voltmap else None
            lk = _get_edge_attr(lenmap, a, b, None) if lenmap else None
            x = _x_from_voltage_length(vk, lk, sbase_MVA)
        if x is None or x <= 0:
            x = 1.0
        rows.append({
            "line_id": f"L{lid}",
            "from_bus": a,
            "to_bus": b,
            "capacity_MW": float(cap),
            "x": float(x)
        })
        lid += 1
    lines_df = pd.DataFrame(rows)
    return sorted(nodes), lines_df

def load_static_generators(data_dir: str):
    g = pd.read_csv(os.path.join(data_dir, "gens_static.csv"))
    g["gen_id"] = g["gen_id"].astype(str)
    g["node"] = g["node"].astype(str)
    if "tech" not in g.columns:
        g["tech"] = None
    for c in ["Pmax", "Pmin", "ramp_up_MW_per_h", "ramp_down_MW_per_h",
              "E_cap_MWh", "P_ch_max_MW", "P_dis_max_MW", "eta_ch", "eta_dis",
              "energy_MWh", "p_charge_max_MW", "p_discharge_max_MW", "eta_charge", "eta_discharge",
              "soc_init_MWh", "soc_min_frac", "soc_max_frac", "can_reserve"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    if "can_reserve" not in g.columns:
        g["can_reserve"] = 1
    g["can_reserve"] = g["can_reserve"].fillna(1).astype(int)
    g["Pmax_numeric"] = pd.to_numeric(g.get("Pmax", np.inf), errors="coerce")
    g["Pmax_cap_for_model"] = g["Pmax_numeric"].fillna(np.inf)
    g["bus_id"] = g["node"].astype(str)
    return g

def load_gen_profiles(data_dir: str):
    gp = pd.read_csv(os.path.join(data_dir, "gen_profiles.csv"))
    gp = _ensure_timestamp(gp)
    gp = _utc_naive(gp)
    gp["gen_id"] = gp["gen_id"].astype(str)
    if "node" not in gp.columns:
        stat = pd.read_csv(os.path.join(data_dir, "gens_static.csv"))
        stat["gen_id"] = stat["gen_id"].astype(str)
        gp = gp.merge(stat[["gen_id", "node"]], on="gen_id", how="left")
    gp["node"] = gp["node"].astype(str)
    gp["avail_MW"] = pd.to_numeric(gp.get("avail_MW", 0.0), errors="coerce").fillna(0.0)
    gp["cost_MWh"] = pd.to_numeric(gp.get("cost_MWh", 0.0), errors="coerce").fillna(0.0)
    if "tech" not in gp.columns:
        gp["tech"] = None
    gp["bus_id"] = gp["node"].astype(str)
    return gp

def load_loads_from_files(data_dir: str, network_json_path: str, subfolder="demand"):
    import json as _json
    demand_dir = os.path.join(data_dir, subfolder)
    with open(network_json_path, "r") as f:
        net = _json.load(f)
    if "loads" not in net:
        raise ValueError("network_uk.json missing 'loads'")
    did2node = {did: str(v["node"]) for did, v in net["loads"].items()}
    files = sorted(glob.glob(os.path.join(demand_dir, "D*_demand.csv")))
    if not files:
        raise FileNotFoundError(f"No D*_demand.csv in {demand_dir}")
    rows = []
    for fp in files:
        fname = os.path.basename(fp)
        did = fname.split("_")[0]
        if did not in did2node:
            raise ValueError(f"{fname}: demand id '{did}' not in network_uk.json loads")
        node = did2node[did]
        df = pd.read_csv(fp)
        df = _ensure_timestamp(df)
        df = _utc_naive(df)
        if "total_demand_MW" not in df.columns:
            pcols = [c for c in df.columns if str(c).endswith("_MW")]
            if not pcols:
                raise ValueError(f"{fname} missing total_demand_MW and no *_MW cols")
            df["total_demand_MW"] = df[pcols].sum(axis=1)
        df["node"] = node
        df["bus_id"] = node
        rows.append(df[["timestamp", "bus_id", "total_demand_MW"]].rename(columns={"total_demand_MW": "demand_MW"}))
    load_prof = (pd.concat(rows, ignore_index=True)
                 .sort_values(["timestamp", "bus_id"])
                 .reset_index(drop=True))
    return load_prof

# --------------- partition helpers ---------------

def split_batteries(gens_static: pd.DataFrame):
    gs = gens_static.set_index("gen_id")
    tech = gs["tech"].astype(str).str.lower()
    by_label = tech.str.contains("batt", na=False)
    e1 = _series_or_const(gs, "E_cap_MWh", 0.0)
    e2 = _series_or_const(gs, "energy_MWh", 0.0)
    is_batt = (by_label | (e1 > 0) | (e2 > 0))
    GB = list(gs.index[is_batt].astype(str))
    GG = list(gs.index[~is_batt].astype(str))
    return GG, GB

# ---------------- reserves detail export (ALLOCATED) ----------------

def compute_and_write_reserves_detail(
    out_dir: str,
    gens_static: pd.DataFrame,
    gen_h_pub: pd.DataFrame,
    rt_ts: pd.DatetimeIndex,
    commit_status_at_ts: dict,
    meta_rt: dict,
    m_rt,  # solved RT model
    step_h: float
):
    """
    Writes out_dir/reserves_detail_RT.csv with per-generator reserve timeseries.

    Reports the ALLOCATED reserve (paid): R[g,t] from the RT solve.
    Also includes deliverability diagnostics (headroom / energy limit).
    """
    gs = gens_static.copy()
    gs["gen_id"] = gs["gen_id"].astype(str)
    gs = gs.set_index("gen_id")

    def _get(g, col, default=np.nan):
        if col in gs.columns:
            v = gs.at[g, col]
            return float(v) if pd.notna(v) else default
        return default

    def Pmax_cap(g):
        v = _get(g, "Pmax_cap_for_model", np.nan)
        if not np.isfinite(v):
            v = _get(g, "Pmax", np.inf)
        return v

    def Pdis_max(g):
        v = _get(g, "P_dis_max_MW", np.nan)
        if not np.isfinite(v):
            v = Pmax_cap(g)
        return v

    def eta_dis(g):
        v = _get(g, "eta_dis", np.nan)
        if not np.isfinite(v):
            v = _get(g, "eta_discharge", 0.95)
        return v if np.isfinite(v) and v > 0 else 0.95

    def is_battery(g):
        return g in set(meta_rt["GB"])

    gen_h_pub2 = gen_h_pub.copy()
    gen_h_pub2["gen_id"] = gen_h_pub2["gen_id"].astype(str)
    gen_h_pub2 = gen_h_pub2.set_index(["gen_id", "timestamp"]).sort_index()

    rows = []
    G_all = list(gs.index.astype(str))
    gen_node = {g: meta_rt["gen_bus"].get(g, str(gs.at[g, "node"])) for g in G_all}
    tech_map = {g: str(gs.at[g, "tech"]) if "tech" in gs.columns else "" for g in G_all}

    def val(x):
        try:
            return float(x())
        except Exception:
            return float(x)

    def cap_avail_at(g, ts):
        try:
            av = float(gen_h_pub2.loc[(g, ts), "avail_MW"])
        except KeyError:
            av = 0.0
        return float(min(Pmax_cap(g), max(0.0, av)))

    for ti, ts in enumerate(meta_rt["rt_ts"]):
        # GAS (non-battery)
        for g in meta_rt["GG"]:
            p = val(m_rt.P[g, ti])
            r = val(m_rt.R[g, ti])
            cap = cap_avail_at(g, ts)
            rows.append({
                "timestamp": ts,
                "gen_id": g,
                "node": gen_node[g],
                "tech": tech_map[g],
                "is_battery": 0,
                "dispatch_net_MW": p,
                "allocated_r_MW": r,
                "deliverable_headroom_MW": max(0.0, cap - p)
            })
        # BATTERIES
        for g in meta_rt["GB"]:
            pdis = val(m_rt.Pdis[g, ti])
            pch = val(m_rt.Pch[g, ti])
            soc = val(m_rt.SOC[g, ti])
            r = val(m_rt.R[g, ti])
            rows.append({
                "timestamp": ts,
                "gen_id": g,
                "node": gen_node[g],
                "tech": tech_map[g],
                "is_battery": 1,
                "dispatch_net_MW": pdis - pch,
                "p_dis_MW": pdis,
                "p_ch_MW": pch,
                "SOC_MWh": soc,
                "allocated_r_MW": r,
                "max_power_headroom_MW": max(0.0, Pdis_max(g) - pdis),
                "max_energy_limit_MW": max(0.0, eta_dis(g) * soc / step_h)
            })

    out = pd.DataFrame(rows).sort_values(["timestamp", "node", "gen_id"])
    out.to_csv(os.path.join(out_dir, "reserves_detail_RT.csv"), index=False)
    print(f"[ok] Wrote reserves_detail_RT.csv with {len(out)} rows → {out_dir}")

# --------------- DA UC (explicit reserves) ---------------

def build_uc_da_model(B, L, lines_df, gens_static, gen_h, load_h, timestamps, cfg,
                      soc0_override: dict | None = None):
    step_m = _check_even_step(pd.DatetimeIndex(timestamps))
    step_h = step_m / 60.0
    voll = float(cfg.get("voll_MWh", 5000.0))
    spill_penalty = float(cfg.get("spill_penalty_per_MWh", cfg.get("pricing_spill_penalty_per_MWh", 5.0)))

    reserve_percent = float(cfg.get("reserve_requirement_percent", 10.0)) / 100.0
    reserve_on_served = bool(cfg.get("reserve_on_served_demand", False))
    reserve_slack_cost = float(cfg.get("reserve_shortfall_cost_per_MW", 6000.0))
    reserve_price = float(cfg.get("reserve_availability_price_per_MW_h", 0.0))
    allow_drop_charge = bool(cfg.get("reserve_allow_battery_drop_charge", False))

    batt_cap_from_profile = bool(cfg.get("battery_profile_caps_discharge", False))
    soc_target_frac = cfg.get("battery_terminal_soc_frac", None)
    soc_target_pen = float(cfg.get("battery_terminal_soc_penalty_per_MWh", 5.0))
    cycle_cost = float(cfg.get("battery_cycle_cost_MWh", 0.0))

    da_energy_neutral = bool(cfg.get("battery_da_energy_neutral", False))
    da_energy_neutral_hard = bool(cfg.get("battery_da_energy_neutral_hard", False))

    gs = gens_static.set_index("gen_id")
    G_all = list(gs.index.astype(str))
    GG, GB = split_batteries(gens_static)

    gen_bus = {g: str(gs.loc[g, "bus_id"]) for g in G_all}
    Pmax = {g: float(gs.loc[g, "Pmax_cap_for_model"]) for g in G_all}
    Pmin = {g: float(gs.loc[g, "Pmin"]) if pd.notna(gs.loc[g, "Pmin"]) else 0.0 for g in G_all}
    can_res = {g: int(gs.loc[g, "can_reserve"]) if pd.notna(gs.loc[g, "can_reserve"]) else 1 for g in G_all}

    # time-varying avail/cost
    cap_avail, cost_t = {}, {}
    for ti, ts in enumerate(timestamps):
        sub = gen_h[gen_h["timestamp"] == ts]
        amap = dict(zip(sub["gen_id"].astype(str), sub["avail_MW"]))
        cmap = dict(zip(sub["gen_id"].astype(str), sub["cost_MWh"]))
        for g in G_all:
            cap_avail[(g, ti)] = min(float(amap.get(g, 0.0)), Pmax[g])
            cost_t[(g, ti)] = float(cmap.get(g, 0.0))

    # demand per bus
    D = {(b, ti): 0.0 for b in B for ti in range(len(timestamps))}
    for ti, ts in enumerate(timestamps):
        subs = load_h[load_h["timestamp"] == ts]
        for _, r in subs.iterrows():
            D[(str(r["bus_id"]), ti)] += float(r["demand_MW"])

    # lines
    LIDS = list(lines_df["line_id"])
    line_from = {r["line_id"]: r["from_bus"] for _, r in lines_df.iterrows()}
    line_to = {r["line_id"]: r["to_bus"] for _, r in lines_df.iterrows()}
    cap_line = {r["line_id"]: float(r["capacity_MW"]) for _, r in lines_df.iterrows()}
    x_map = {r["line_id"]: float(r["x"]) for _, r in lines_df.iterrows()}
    ref_bus = B[0]

    # battery params/util
    gs_idx = gens_static.set_index("gen_id")

    def Pdis_max(g):
        v = gs_idx.get("P_dis_max_MW", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(v):
            v = gs_idx.get("Pmax_cap_for_model", pd.Series(index=[])).get(g, 0.0)
        return float(v)

    def Pch_max(g):
        v = gs_idx.get("P_ch_max_MW", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(v):
            v = gs_idx.get("Pmax_cap_for_model", pd.Series(index=[])).get(g, 0.0)
        return float(v)

    def eta_ch(g):
        e = gs_idx.get("eta_ch", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(e):
            e = gs_idx.get("eta_charge", pd.Series(index=[])).get(g, 0.95)
        return float(e)

    def eta_dis(g):
        e = gs_idx.get("eta_dis", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(e):
            e = gs_idx.get("eta_discharge", pd.Series(index=[])).get(g, 0.95)
        return float(e)

    def Ecap(g):
        e = gs_idx.get("E_cap_MWh", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(e):
            e = gs_idx.get("energy_MWh", pd.Series(index=[])).get(g, np.nan)
        return float(e)

    def soc0_default(g):
        s = gs_idx.get("soc_init_MWh", pd.Series(index=[])).get(g, np.nan)
        E = Ecap(g)
        if not np.isfinite(s):
            s = 0.5 * E
        return float(s)

    # ---- model ----
    m = ConcreteModel("DA_UC")
    m.B = Set(initialize=B, ordered=True)
    m.G_all = Set(initialize=G_all, ordered=True)
    m.GG = Set(initialize=GG, ordered=True)
    m.GB = Set(initialize=GB, ordered=True)
    m.L = Set(initialize=LIDS, ordered=True)
    m.T = Set(initialize=list(range(len(timestamps))), ordered=True)

    m.u = Var(m.GG, m.T, within=Binary)
    m.P = Var(m.GG, m.T, within=NonNegativeReals)
    m.R = Var(m.G_all, m.T, within=NonNegativeReals)  # explicit reserve for ALL eligible units

    m.Pch = Var(m.GB, m.T, within=NonNegativeReals)
    m.Pdis = Var(m.GB, m.T, within=NonNegativeReals)
    m.SOC = Var(m.GB, m.T, within=Reals)

    if bool(cfg.get("battery_exclusive_mode", False)):
        m.Yb = Var(m.GB, m.T, bounds=(0.0, 1.0), within=NonNegativeReals)

    m.Theta = Var(m.B, m.T, within=Reals)
    m.F = Var(m.L, m.T, within=Reals)

    m.Served = Var(m.B, m.T, within=NonNegativeReals)
    m.Shed = Var(m.B, m.T, within=NonNegativeReals)
    m.Spill = Var(m.B, m.T, within=NonNegativeReals)
    m.ResShort = Var(m.T, within=NonNegativeReals)

    # DC flows
    m.ref = Constraint(m.T, rule=lambda m, ti: m.Theta[ref_bus, ti] == 0.0)
    m.flow = Constraint(m.L, m.T,
                        rule=lambda m, lid, ti: m.F[lid, ti] ==
                        (m.Theta[line_to[lid], ti] - m.Theta[line_from[lid], ti]) / x_map[lid])
    m.flow_up = Constraint(m.L, m.T, rule=lambda m, l, ti: m.F[l, ti] <= cap_line[l])
    m.flow_dn = Constraint(m.L, m.T, rule=lambda m, l, ti: -m.F[l, ti] <= cap_line[l])

    # GAS / NON-BATTERY: dispatch + reserve within available committed headroom
    m.head = Constraint(m.GG, m.T,
                        rule=lambda m, g, ti: m.P[g, ti] + m.R[g, ti] <= cap_avail[(g, ti)] * m.u[g, ti])
    m.pmin = Constraint(m.GG, m.T, rule=lambda m, g, ti: m.P[g, ti] >= Pmin[g] * m.u[g, ti])

    # BATTERIES: power caps, SOC, optional exclusivity
    m.b_dis = Constraint(m.GB, m.T, rule=lambda m, g, ti:
                         m.Pdis[g, ti] <= (min(Pdis_max(g), cap_avail[(g, ti)]) if batt_cap_from_profile else Pdis_max(g)))
    m.b_ch = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.Pch[g, ti] <= Pch_max(g))
    if bool(cfg.get("battery_exclusive_mode", False)):
        m.b_ex1 = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.Pdis[g, ti] <= Pdis_max(g) * (1 - m.Yb[g, ti]))
        m.b_ex2 = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.Pch[g, ti] <= Pch_max(g) * m.Yb[g, ti])

    # initial SOC (carry-over)
    def soc0(g):
        if soc0_override and g in soc0_override:
            return float(soc0_override[g])
        return soc0_default(g)

    m.soc = Constraint(m.GB, m.T, rule=lambda m, g, ti:
                       (m.SOC[g, ti] ==
                        (soc0(g) if ti == 0 else m.SOC[g, ti - 1]) +
                        eta_ch(g) * m.Pch[g, ti] * step_h - (1.0 / eta_dis(g)) * m.Pdis[g, ti] * step_h))
    m.soc_lo = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.SOC[g, ti] >= 0.0)
    m.soc_hi = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.SOC[g, ti] <= Ecap(g))

    # BATTERY reserve deliverability (DA)
    if allow_drop_charge:
        m.b_res_headroom_DA = Constraint(m.GB, m.T,
                                         rule=lambda m, g, ti: m.R[g, ti] <= (Pdis_max(g) - m.Pdis[g, ti] + m.Pch[g, ti]))
    else:
        m.b_res_headroom_DA = Constraint(m.GB, m.T,
                                         rule=lambda m, g, ti: m.R[g, ti] <= Pdis_max(g) - m.Pdis[g, ti])
    m.b_res_soc_DA = Constraint(m.GB, m.T,
                                rule=lambda m, g, ti: m.R[g, ti] <= eta_dis(g) * m.SOC[g, ti] / step_h)

    # KCL
    m.served_le_d = Constraint(m.B, m.T, rule=lambda m, b, ti: m.Served[b, ti] <= D[(b, ti)])
    m.shed_def = Constraint(m.B, m.T, rule=lambda m, b, ti: m.Shed[b, ti] == D[(b, ti)] - m.Served[b, ti])
    m.balance = Constraint(m.B, m.T, rule=lambda m, b, ti:
                           sum(m.P[g, ti] for g in m.GG if gen_bus[g] == b) +
                           sum(m.Pdis[g, ti] - m.Pch[g, ti] for g in m.GB if gen_bus[g] == b) +
                           sum(+m.F[l, ti] for l in m.L if line_to[l] == b) -
                           sum(+m.F[l, ti] for l in m.L if line_from[l] == b) +
                           m.Shed[b, ti] == D[(b, ti)] + m.Spill[b, ti])

    # SYSTEM RESERVE REQUIREMENT (EXPLICIT R ONLY)
    def reserve_req(m, ti):
        Dtot = sum(D[(b, ti)] for b in B)
        served_adj = (max(0.0, Dtot) - sum(m.Shed[b, ti] for b in B)) if reserve_on_served else max(0.0, Dtot)
        req = reserve_percent * served_adj
        return (sum(m.R[g, ti] for g in m.G_all if can_res[g] == 1) + m.ResShort[ti]) >= req

    m.reserve_req = Constraint(m.T, rule=reserve_req)

    # DA terminal SOC / neutrality
    soc_pen_expr = 0.0
    last_t = list(m.T)[-1]
    if da_energy_neutral:
        if da_energy_neutral_hard:
            def soc_neutral_da(m, g):
                return m.SOC[g, last_t] == soc0(g)
            m.soc_neutral_da = Constraint(m.GB, rule=soc_neutral_da)
        else:
            m.SOCdev_da = Var(m.GB, within=NonNegativeReals)

            def soc_dev_da(m, g):
                return m.SOCdev_da[g] >= soc0(g) - m.SOC[g, last_t]
            m.soc_dev_da = Constraint(m.GB, rule=soc_dev_da)
            soc_pen_expr += soc_target_pen * sum(m.SOCdev_da[g] for g in m.GB)
    else:
        if soc_target_frac is not None:
            m.SOCshort = Var(m.GB, within=NonNegativeReals)

            def soc_target(m, g):
                return m.SOCshort[g] >= float(soc_target_frac) * Ecap(g) - m.SOC[g, last_t]
            m.soc_target = Constraint(m.GB, rule=soc_target)
            soc_pen_expr = soc_target_pen * sum(m.SOCshort[g] for g in m.GB)

    # linear cycle cost
    cycle_pen = cycle_cost * sum((m.Pch[g, ti] + m.Pdis[g, ti]) * step_h for g in m.GB for ti in m.T)

    # reserve availability payment (DA)
    res_pay = reserve_price * sum(m.R[g, ti] * step_h for g in m.G_all for ti in m.T)

    def obj(m):
        energy = (sum(cost_t[(g, ti)] * m.P[g, ti] * step_h for g in m.GG for ti in m.T) +
                  sum(cost_t[(g, ti)] * m.Pdis[g, ti] * step_h for g in m.GB for ti in m.T))
        shedc = voll * sum(m.Shed[b, ti] * step_h for b in m.B for ti in m.T)
        spillc = spill_penalty * sum(m.Spill[b, ti] * step_h for b in m.B for ti in m.T)
        rshort = reserve_slack_cost * sum(m.ResShort[ti] for ti in m.T)
        return energy + shedc + spillc + rshort + soc_pen_expr + cycle_pen + res_pay

    m.obj = Objective(rule=obj, sense=minimize)

    meta = dict(B=B, L=LIDS, timestamps=timestamps, step_h=step_h, gen_bus=gen_bus,
                line_from=line_from, line_to=line_to, D=D, GG=GG, GB=GB)
    return m, meta

# --------------- RT SCED (explicit reserves) ---------------

def build_rt_sced_lp(
    B, L, lines_df, gens_static, gen_h_pub, load_pub, rt_ts,
    commit_status_at_ts, soc0_map, cfg, demand_bumps=None
):
    step_m = _check_even_step(pd.DatetimeIndex(rt_ts))
    step_h = step_m / 60.0
    voll = float(cfg.get("voll_MWh", 5000.0))
    spill_penalty = float(cfg.get("spill_penalty_per_MWh", cfg.get("pricing_spill_penalty_per_MWh", 20.0)))
    reserve_percent = float(cfg.get("reserve_requirement_percent", 10.0)) / 100.0
    reserve_on_served = bool(cfg.get("reserve_on_served_demand", False))
    reserve_slack_cost = float(cfg.get("reserve_shortfall_cost_per_MW", 6000.0))
    reserve_price = float(cfg.get("reserve_availability_price_per_MW_h", 0.0))
    allow_drop_charge = bool(cfg.get("reserve_allow_battery_drop_charge", False))

    wire_cost_exp = float(cfg.get("local_first_export_cost_MWh", cfg.get("local_first_wire_cost_MWh", 1.0)))
    wire_cost_imp = float(cfg.get("local_first_import_cost_MWh", cfg.get("local_first_wire_cost_import_MWh", 0.0)))

    batt_cap_from_profile = bool(cfg.get("battery_profile_caps_discharge", False))
    cycle_cost = float(cfg.get("battery_cycle_cost_MWh", 0.0))

    rt_target_frac = cfg.get("battery_rt_terminal_soc_frac",
                             cfg.get("battery_terminal_soc_frac", None))
    rt_target_pen = float(cfg.get("battery_rt_terminal_soc_penalty_per_MWh",
                                  cfg.get("battery_terminal_soc_penalty_per_MWh", 5.0)))
    rt_energy_neutral = bool(cfg.get("battery_rt_energy_neutral", False))
    rt_energy_neutral_hard = bool(cfg.get("battery_rt_energy_neutral_hard", False))

    gs = gens_static.set_index("gen_id")
    G_all = list(gs.index.astype(str))
    GG, GB = split_batteries(gens_static)
    gen_bus = {g: str(gs.loc[g, "bus_id"]) for g in G_all}
    Pmax = {g: float(gs.loc[g, "Pmax_cap_for_model"]) for g in G_all}
    can_res = {g: int(gs.loc[g, "can_reserve"]) if pd.notna(gs.loc[g, "can_reserve"]) else 1 for g in G_all}

    # time-varying avail/cost
    cap_avail, cost_t = {}, {}
    for g in G_all:
        sub = gen_h_pub[gen_h_pub["gen_id"] == g].set_index("timestamp").sort_index()
        av = sub["avail_MW"].reindex(rt_ts, fill_value=0.0)
        co = sub["cost_MWh"].reindex(rt_ts, fill_value=0.0)
        for ti, ts in enumerate(rt_ts):
            cap_avail[(g, ti)] = min(float(av.loc[ts]), Pmax[g])
            cost_t[(g, ti)] = float(co.loc[ts])

    # demand (optionally bumped for FD)
    D = {(b, ti): 0.0 for b in B for ti in range(len(rt_ts))}
    for b in B:
        sub = load_pub[load_pub["bus_id"] == b].set_index("timestamp").sort_index()
        d = sub["demand_MW"].reindex(rt_ts, fill_value=0.0)
        for ti, ts in enumerate(rt_ts):
            D[(b, ti)] = float(d.loc[ts])
    if demand_bumps:
        for (b, ti), add in demand_bumps.items():
            D[(b, ti)] = float(D[(b, ti)] + add)

    # lines
    LIDS = list(lines_df["line_id"])
    line_from = {r["line_id"]: r["from_bus"] for _, r in lines_df.iterrows()}
    line_to = {r["line_id"]: r["to_bus"] for _, r in lines_df.iterrows()}
    cap_line = {r["line_id"]: float(r["capacity_MW"]) for _, r in lines_df.iterrows()}
    x_map = {r["line_id"]: float(r["x"]) for _, r in lines_df.iterrows()}
    ref_bus = B[0]

    # model
    m = ConcreteModel("RT_SCED_LP")
    m.B = Set(initialize=B, ordered=True)
    m.G_all = Set(initialize=G_all, ordered=True)
    m.GG = Set(initialize=GG, ordered=True)
    m.GB = Set(initialize=GB, ordered=True)
    m.L = Set(initialize=LIDS, ordered=True)
    m.T = Set(initialize=list(range(len(rt_ts))), ordered=True)

    m.P = Var(m.GG, m.T, within=NonNegativeReals)
    m.R = Var(m.G_all, m.T, within=NonNegativeReals)
    m.Pch = Var(m.GB, m.T, within=NonNegativeReals)
    m.Pdis = Var(m.GB, m.T, within=NonNegativeReals)
    m.SOC = Var(m.GB, m.T, within=Reals)

    m.Theta = Var(m.B, m.T, within=Reals)
    m.F = Var(m.L, m.T, within=Reals)
    m.Served = Var(m.B, m.T, within=NonNegativeReals)
    m.Shed = Var(m.B, m.T, within=NonNegativeReals)
    m.Spill = Var(m.B, m.T, within=NonNegativeReals)
    m.ResShort = Var(m.T, within=NonNegativeReals)

    # import/export proxies for "serve-local-first"
    m.ImportPlus = Var(m.B, m.T, within=NonNegativeReals)
    m.ExportPlus = Var(m.B, m.T, within=NonNegativeReals)

    # DC flow
    m.ref = Constraint(m.T, rule=lambda m, ti: m.Theta[ref_bus, ti] == 0.0)
    m.flow = Constraint(m.L, m.T,
                        rule=lambda m, lid, ti: m.F[lid, ti] ==
                        (m.Theta[line_to[lid], ti] - m.Theta[line_from[lid], ti]) / x_map[lid])
    m.flow_up = Constraint(m.L, m.T, rule=lambda m, l, ti: m.F[l, ti] <= cap_line[l])
    m.flow_dn = Constraint(m.L, m.T, rule=lambda m, l, ti: -m.F[l, ti] <= cap_line[l])

    # commitment from DA for non-batteries
    def u_at(g, ti):
        return float(commit_status_at_ts.get((g, rt_ts[ti]), 0.0))

    m.head = Constraint(m.GG, m.T,
                        rule=lambda m, g, ti: m.P[g, ti] + m.R[g, ti] <= cap_avail[(g, ti)] * u_at(g, ti))
    m.pmin = Constraint(m.GG, m.T, rule=lambda m, g, ti: m.P[g, ti] >= 0.0)

    # storage params
    gs_idx = gens_static.set_index("gen_id")

    def Pdis_max(g):
        v = gs_idx.get("P_dis_max_MW", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(v):
            v = gs_idx.get("Pmax_cap_for_model", pd.Series(index=[])).get(g, 0.0)
        return float(v)

    def Pch_max(g):
        v = gs_idx.get("P_ch_max_MW", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(v):
            v = gs_idx.get("Pmax_cap_for_model", pd.Series(index=[])).get(g, 0.0)
        return float(v)

    def eta_ch(g):
        e = gs_idx.get("eta_ch", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(e):
            e = gs_idx.get("eta_charge", pd.Series(index=[])).get(g, 0.95)
        return float(e)

    def eta_dis(g):
        e = gs_idx.get("eta_dis", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(e):
            e = gs_idx.get("eta_discharge", pd.Series(index=[])).get(g, 0.95)
        return float(e)

    def Ecap(g):
        e = gs_idx.get("E_cap_MWh", pd.Series(index=[])).get(g, np.nan)
        if not np.isfinite(e):
            e = gs_idx.get("energy_MWh", pd.Series(index=[])).get(g, np.nan)
        return float(e)

    # battery power caps (profile cap only if toggled)
    m.b_dis = Constraint(m.GB, m.T, rule=lambda m, g, ti:
                         m.Pdis[g, ti] <= (min(Pdis_max(g), cap_avail[(g, ti)]) if batt_cap_from_profile else Pdis_max(g)))
    m.b_ch = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.Pch[g, ti] <= Pch_max(g))

    # SOC with carry-in for t=0
    m.soc = Constraint(m.GB, m.T, rule=lambda m, g, ti:
                       (m.SOC[g, ti] ==
                        (soc0_map.get(g, 0.5 * Ecap(g)) if ti == 0 else m.SOC[g, ti - 1]) +
                        eta_ch(g) * m.Pch[g, ti] * step_h - (1.0 / eta_dis(g)) * m.Pdis[g, ti] * step_h))
    m.soc_lo = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.SOC[g, ti] >= 0.0)
    m.soc_hi = Constraint(m.GB, m.T, rule=lambda m, g, ti: m.SOC[g, ti] <= Ecap(g))

    # battery reserve deliverability (RT)
    if allow_drop_charge:
        m.b_res_headroom = Constraint(m.GB, m.T,
                                      rule=lambda m, g, ti: m.R[g, ti] <= (Pdis_max(g) - m.Pdis[g, ti] + m.Pch[g, ti]))
    else:
        m.b_res_headroom = Constraint(m.GB, m.T,
                                      rule=lambda m, g, ti: m.R[g, ti] <= Pdis_max(g) - m.Pdis[g, ti])
    m.b_res_soc = Constraint(m.GB, m.T,
                             rule=lambda m, g, ti: m.R[g, ti] <= eta_dis(g) * m.SOC[g, ti] / step_h)

    # KCL
    m.served_le_d = Constraint(m.B, m.T, rule=lambda m, b, ti: m.Served[b, ti] <= D[(b, ti)])
    m.shed_def = Constraint(m.B, m.T, rule=lambda m, b, ti: m.Shed[b, ti] == D[(b, ti)] - m.Served[b, ti])
    m.balance = Constraint(m.B, m.T, rule=lambda m, b, ti:
                           sum(m.P[g, ti] for g in m.GG if gen_bus[g] == b) +
                           sum(m.Pdis[g, ti] - m.Pch[g, ti] for g in m.GB if gen_bus[g] == b) +
                           sum(+m.F[l, ti] for l in m.L if line_to[l] == b) -
                           sum(+m.F[l, ti] for l in m.L if line_from[l] == b) +
                           m.Shed[b, ti] == D[(b, ti)] + m.Spill[b, ti])

    # import/export proxies
    m.import_lower = Constraint(m.B, m.T, rule=lambda m, b, ti:
                                m.ImportPlus[b, ti] >=
                                (sum(+m.F[l, ti] for l in m.L if line_to[l] == b) -
                                 sum(+m.F[l, ti] for l in m.L if line_from[l] == b)))
    m.export_lower = Constraint(m.B, m.T, rule=lambda m, b, ti:
                                m.ExportPlus[b, ti] >=
                                (sum(+m.F[l, ti] for l in m.L if line_from[l] == b) -
                                 sum(+m.F[l, ti] for l in m.L if line_to[l] == b)))

    # SYSTEM RESERVE REQUIREMENT (EXPLICIT R ONLY)
    def reserve_req(m, ti):
        Dtot = sum(D[(b, ti)] for b in B)
        served_adj = (max(0.0, Dtot) - sum(m.Shed[b, ti] for b in B)) if reserve_on_served else max(0.0, Dtot)
        req = reserve_percent * served_adj
        return (sum(m.R[g, ti] for g in m.G_all if can_res[g] == 1) + m.ResShort[ti]) >= req

    m.reserve_req = Constraint(m.T, rule=reserve_req)

    # cycle cost
    cycle_pen = cycle_cost * sum((m.Pch[g, ti] + m.Pdis[g, ti]) * step_h for g in m.GB for ti in m.T)

    # RT terminal SOC / neutrality
    soc_pen_rt = 0.0
    if rt_target_frac is not None:
        m.SOCshort_rt = Var(m.GB, m.T, within=NonNegativeReals)
        last_t = list(m.T)[-1]

        def soc_target_rt(m, g):
            return m.SOCshort_rt[g, last_t] >= float(rt_target_frac) * Ecap(g) - m.SOC[g, last_t]
        m.soc_target_rt = Constraint(m.GB, rule=soc_target_rt)
        soc_pen_rt = rt_target_pen * sum(m.SOCshort_rt[g, last_t] for g in m.GB)

    if rt_energy_neutral:
        last_t = list(m.T)[-1]
        if rt_energy_neutral_hard:
            def soc_neutral_hard(m, g):
                return m.SOC[g, last_t] == soc0_map.get(g, 0.5 * Ecap(g))
            m.soc_neutral_hard = Constraint(m.GB, rule=soc_neutral_hard)
        else:
            m.SOCdev = Var(m.GB, within=NonNegativeReals)

            def soc_dev(m, g):
                return m.SOCdev[g] >= soc0_map.get(g, 0.5 * Ecap(g)) - m.SOC[g, last_t]
            m.soc_dev = Constraint(m.GB, rule=soc_dev)
            soc_pen_rt += rt_target_pen * sum(m.SOCdev[g] for g in m.GB)

    # reserve availability payment (RT)
    res_pay = reserve_price * sum(m.R[g, ti] * step_h for g in m.G_all for ti in m.T)

    def obj(m):
        energy = (sum(cost_t[(g, ti)] * m.P[g, ti] * step_h for g in m.GG for ti in m.T) +
                  sum(cost_t[(g, ti)] * m.Pdis[g, ti] * step_h for g in m.GB for ti in m.T))
        shedc = voll * sum(m.Shed[b, ti] * step_h for b in m.B for ti in m.T)
        spillc = spill_penalty * sum(m.Spill[b, ti] * step_h for b in m.B for ti in m.T)
        rshort = reserve_slack_cost * sum(m.ResShort[ti] for ti in m.T)
        wire = (wire_cost_exp * sum(m.ExportPlus[b, ti] * step_h for b in m.B for ti in m.T) +
                wire_cost_imp * sum(m.ImportPlus[b, ti] * step_h for b in m.B for ti in m.T))
        return energy + shedc + spillc + rshort + wire + cycle_pen + soc_pen_rt + res_pay

    m.obj = Objective(rule=obj, sense=minimize)

    # enable duals for LMP if available
    m.dual = Suffix(direction=Suffix.IMPORT)

    meta = dict(B=B, L=LIDS, rt_ts=rt_ts, step_h=step_h, gen_bus=gen_bus,
                line_from=line_from, line_to=line_to, D=D, GG=GG, GB=GB)
    return m, meta

# --------------- solve helpers, pricing ---------------

def solve_lp(model, cfg, solver_name=None):
    solver = solver_name or cfg.get("solver", "highs")
    opt = SolverFactory(solver)
    if opt is None or not opt.available(False):
        raise RuntimeError(f"Solver '{solver}' not available.")
    try:
        tlim = int(cfg.get("solver_time_limit_s", 600))
        if hasattr(opt, "options"):
            for k in ("time_limit", "timelimit", "TimeLimit", "tmlim", "seconds"):
                try:
                    opt.options[k] = tlim
                    break
                except Exception:
                    pass
    except Exception:
        pass
    res = opt.solve(model, tee=False)
    return res, solver

def duals_imported(model, constraint_block):
    if not hasattr(model, "dual"):
        return False
    try:
        for idx in constraint_block:
            if model.dual.get(constraint_block[idx], None) is not None:
                return True
    except Exception:
        pass
    return False

def price_via_finite_difference(build_model_fn, base_args, cfg, rt_ts, B, step_h, epsilon_MW=0.1):
    prices = {(ts, b): np.nan for ts in rt_ts for b in B}
    m_base, _ = build_model_fn(**base_args, demand_bumps=None)
    res_base, solver_used = solve_lp(m_base, cfg)
    tc = res_base.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
        raise RuntimeError(f"Base RT SCED not optimal for FD pricing (termination={tc}).")
    obj0 = float(m_base.obj())

    for ti, ts in enumerate(rt_ts):
        for b in B:
            bumps = {(b, ti): epsilon_MW}
            m_eps, _ = build_model_fn(**base_args, demand_bumps=bumps)
            res_eps, _ = solve_lp(m_eps, cfg, solver_name=solver_used)
            tc2 = res_eps.solver.termination_condition
            if tc2 not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
                continue
            obj1 = float(m_eps.obj())
            prices[(ts, b)] = (obj1 - obj0) / (epsilon_MW * step_h)
    return prices

# --------------- main ---------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DA+RT SCED with EXPLICIT reserves (gas+battery) and LMPs (+ runtime logs)")
    ap.add_argument("--start", default="2024-01-01 00:00")
    ap.add_argument("--end", default="2024-12-31 23:30")
    ap.add_argument("--out-root", default="outputs/uc_da_rt_sced_bat_reserves_runtime")
    ap.add_argument("--data-dir", default="marketExecution_Actual/data")
    ap.add_argument("--runtime-by-step", action="store_true",
                    help="Write runtime_by_step.csv (one row per RT interval).")
    args = ap.parse_args()

    base = os.getcwd()
    data_dir = os.path.join(base, args.data_dir)
    out_root = os.path.join(base, args.out_root)
    os.makedirs(out_root, exist_ok=True)

    cfg = load_config(data_dir)
    sbase = float(cfg.get("sbase_MVA", 1000.0))
    net_path = os.path.join(data_dir, "network_uk.json")

    # carry toggle
    carry_soc = bool(cfg.get("battery_carry_soc_across_days", True))
    fd_eps = float(cfg.get("rt_fd_epsilon_MW", cfg.get("pricing_eps", 0.1)))

    nodes, lines_df = load_network_as_nodes(net_path, sbase_MVA=sbase)
    gens_static = load_static_generators(data_dir)
    gen_prof = load_gen_profiles(data_dir)
    load_prof = load_loads_from_files(data_dir, net_path, subfolder="demand")

    gen_prof, load_prof = _utc_naive(gen_prof.copy()), _utc_naive(load_prof.copy())
    ts_all = pd.DatetimeIndex(sorted(set(gen_prof["timestamp"]) & set(load_prof["timestamp"])))
    ts_mask = (ts_all >= pd.Timestamp(args.start)) & (ts_all <= pd.Timestamp(args.end))
    ts_idx = ts_all[ts_mask]
    if len(ts_idx) == 0:
        raise ValueError("No overlapping timestamps in requested window.")
    step_minutes = _check_even_step(ts_idx)
    step_h = step_minutes / 60.0
    print(f"[info] {len(ts_idx)} timestamps, step={step_minutes} min.")

    # simple per-day windows
    by_day = {}
    for t in ts_idx:
        by_day.setdefault(t.date(), []).append(t)
    windows = [(d, pd.DatetimeIndex(sorted(v))) for d, v in sorted(by_day.items())]
    solver_name = cfg.get("solver", "highs")

    # SOC carry map (end-of-day from DA → next-day DA & RT)
    soc_carry = {}  # {gen_id: SOC_MWh}
    GB_all = set(split_batteries(gens_static)[1])

    # runtime collectors (NEW)
    runtime_rows = []
    runtime_by_step_rows = []  # optional, via --runtime-by-step
    host = platform.node()
    pyver = sys.version.split()[0]
    run_started_iso = _now_iso_utc()

    for (d, pub_ts) in windows:
        out_dir = os.path.join(out_root, str(d))
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[info] Day {d}: {len(pub_ts)} steps")

        day_wall_t0 = time.perf_counter()
        day_hash = _ts_hash(pub_ts)

        gen_h = gen_prof[gen_prof["timestamp"].isin(pub_ts)].copy()
        load_h = load_prof[load_prof["timestamp"].isin(pub_ts)].copy()

        # ---------------- DA build + solve (timed) ----------------
        t0 = time.perf_counter()
        m_uc, meta_uc = build_uc_da_model(
            nodes, list(lines_df["line_id"]), lines_df,
            gens_static, gen_h, load_h, list(pub_ts), cfg,
            soc0_override=(soc_carry if carry_soc else None)
        )
        t1 = time.perf_counter()
        res_uc, _ = solve_lp(m_uc, cfg, solver_name)
        t2 = time.perf_counter()

        build_seconds_DA = t1 - t0
        solve_seconds_DA = t2 - t1
        obj_DA = _safe_float(m_uc.obj(), np.nan)
        nvars_DA, ncons_DA = _model_size(m_uc)

        tc = res_uc.solver.termination_condition
        print(f"[DA] term={tc}")
        if tc not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            dump_model(m_uc, os.path.join(out_dir, "DA_infeasible"))
            day_wall_t1 = time.perf_counter()
            runtime_rows.append({
                "run_started_utc": run_started_iso,
                "row_written_utc": _now_iso_utc(),
                "date": str(d),
                "window_hash": day_hash,
                "mechanism": "LMP",
                "solver": str(solver_name),
                "host": host,
                "python": pyver,
                "n_steps": int(len(pub_ts)),
                "step_minutes": float(step_minutes),

                "build_seconds_DA": build_seconds_DA,
                "solve_seconds_DA": solve_seconds_DA,
                "build_seconds_RT": np.nan,
                "solve_seconds_RT": np.nan,
                "pricing_seconds": np.nan,
                "pricing_method": "",

                "window_seconds_total": day_wall_t1 - day_wall_t0,
                "seconds_per_interval_equiv": (day_wall_t1 - day_wall_t0) / max(1, len(pub_ts)),

                "n_vars_DA": nvars_DA,
                "n_cons_DA": ncons_DA,
                "n_vars_RT": np.nan,
                "n_cons_RT": np.nan,

                "obj_DA": obj_DA,
                "obj_RT": np.nan,

                "served_MWh": np.nan,
                "unserved_MWh": np.nan,
                "spill_MWh": np.nan,

                "reserve_shortfall_MWh_equiv": np.nan,
                "reserve_shortfall_max_MW": np.nan,

                "failed": 1,
                "fail_stage": "DA",
                "termination": str(tc),
            })
            continue

        # DA commitment for RT
        GG = meta_uc["GG"]
        GB = meta_uc["GB"]
        rows_commit = []
        for ti, ts in enumerate(meta_uc["timestamps"]):
            if ts not in set(pub_ts):
                continue
            for g in GG:
                rows_commit.append({
                    "timestamp": ts,
                    "gen_id": g,
                    "u": int(round(value(m_uc.u[g, ti]) or 0.0))
                })
        da_commitment = pd.DataFrame(rows_commit).sort_values(["timestamp", "gen_id"])
        da_commitment.to_csv(os.path.join(out_dir, "commitment_DA.csv"), index=False)

        # end-of-day SOC for carry and RT start
        soc0_for_rt = {}
        last_t = list(m_uc.T)[-1]
        for g in GB:
            try:
                soc_end = float(value(m_uc.SOC[g, last_t]) or 0.0)
            except Exception:
                Ecap = gens_static.set_index("gen_id").get("E_cap_MWh", pd.Series(index=[])).get(g, np.nan)
                if not np.isfinite(Ecap):
                    Ecap = gens_static.set_index("gen_id").get("energy_MWh", pd.Series(index=[])).get(g, 0.0)
                soc_end = 0.5 * float(Ecap)
            soc0_for_rt[g] = soc_end
            if carry_soc and g in GB_all:
                soc_carry[g] = soc_end

        gen_h_pub = gen_h.copy()
        load_pub = load_h.copy()
        rt_ts = pd.DatetimeIndex(sorted(pub_ts))

        commit_status_at_ts = {(row["gen_id"], row["timestamp"]): int(row["u"])
                               for _, row in da_commitment.iterrows()}

        base_args = dict(
            B=nodes, L=list(lines_df["line_id"]), lines_df=lines_df,
            gens_static=gens_static, gen_h_pub=gen_h_pub, load_pub=load_pub,
            rt_ts=list(rt_ts), commit_status_at_ts=commit_status_at_ts,
            soc0_map=soc0_for_rt, cfg=cfg
        )

        # ---------------- RT build + solve (timed) ----------------
        t3 = time.perf_counter()
        m_rt, meta_rt = build_rt_sced_lp(**base_args, demand_bumps=None)
        t4 = time.perf_counter()
        res_rt, solver_used = solve_lp(m_rt, cfg)
        t5 = time.perf_counter()

        build_seconds_RT = t4 - t3
        solve_seconds_RT = t5 - t4
        obj_RT = _safe_float(m_rt.obj(), np.nan)
        nvars_RT, ncons_RT = _model_size(m_rt)

        tc_rt = res_rt.solver.termination_condition
        print(f"[RT] solver={solver_used}, term={tc_rt}")
        if tc_rt not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            dump_model(m_rt, os.path.join(out_dir, "RT_infeasible"))
            day_wall_t1 = time.perf_counter()
            runtime_rows.append({
                "run_started_utc": run_started_iso,
                "row_written_utc": _now_iso_utc(),
                "date": str(d),
                "window_hash": day_hash,
                "mechanism": "LMP",
                "solver": str(solver_used),
                "host": host,
                "python": pyver,
                "n_steps": int(len(pub_ts)),
                "step_minutes": float(step_minutes),

                "build_seconds_DA": build_seconds_DA,
                "solve_seconds_DA": solve_seconds_DA,
                "build_seconds_RT": build_seconds_RT,
                "solve_seconds_RT": solve_seconds_RT,
                "pricing_seconds": np.nan,
                "pricing_method": "",

                "window_seconds_total": day_wall_t1 - day_wall_t0,
                "seconds_per_interval_equiv": (day_wall_t1 - day_wall_t0) / max(1, len(pub_ts)),

                "n_vars_DA": nvars_DA,
                "n_cons_DA": ncons_DA,
                "n_vars_RT": nvars_RT,
                "n_cons_RT": ncons_RT,

                "obj_DA": obj_DA,
                "obj_RT": obj_RT,

                "served_MWh": np.nan,
                "unserved_MWh": np.nan,
                "spill_MWh": np.nan,

                "reserve_shortfall_MWh_equiv": np.nan,
                "reserve_shortfall_max_MW": np.nan,

                "failed": 1,
                "fail_stage": "RT",
                "termination": str(tc_rt),
            })
            continue

        # --- Write RT dispatch/flows/node balance (both sced_* and comparable files) ---
        rows_rt_g, rows_rt_f, rows_rt_nb = [], [], []
        rows_disp_generic, rows_flow_generic = [], []
        rows_serv_generic, rows_nbcheck_generic = [], []
        rows_flowcon = []

        # optional per-step runtime: here we only have total solve times; still useful to emit per step with same values
        # (if you later move to rolling SCED per interval, this table becomes truly per-step)

        for ti, ts in enumerate(meta_rt["rt_ts"]):
            # gens
            for g in meta_rt["GG"]:
                rows_rt_g.append({
                    "timestamp": ts,
                    "gen_id": g,
                    "node": meta_rt["gen_bus"][g],
                    "p_MW": float(value(m_rt.P[g, ti]) or 0.0),
                    "r_MW": float(value(m_rt.R[g, ti]) or 0.0),
                    "is_battery": 0
                })
            for g in meta_rt["GB"]:
                pdis = float(value(m_rt.Pdis[g, ti]) or 0.0)
                pch = float(value(m_rt.Pch[g, ti]) or 0.0)
                rows_rt_g.append({
                    "timestamp": ts,
                    "gen_id": g,
                    "node": meta_rt["gen_bus"][g],
                    "p_MW": pdis - pch,
                    "p_dis_MW": pdis,
                    "p_ch_MW": pch,
                    "r_MW": float(value(m_rt.R[g, ti]) or 0.0),
                    "is_battery": 1
                })

            # flows
            for lid in meta_rt["L"]:
                flow = float(value(m_rt.F[lid, ti]) or 0.0)
                rows_rt_f.append({
                    "timestamp": ts,
                    "line_id": lid,
                    "from_node": meta_rt["line_from"][lid],
                    "to_node": meta_rt["line_to"][lid],
                    "flow_MW": flow
                })

            # node balance
            for b in meta_rt["B"]:
                gen_nb = sum(float(value(m_rt.P[g, ti]) or 0.0)
                             for g in meta_rt["GG"] if meta_rt["gen_bus"][g] == b)
                gen_b = sum(float(value(m_rt.Pdis[g, ti]) or 0.0) - float(value(m_rt.Pch[g, ti]) or 0.0)
                            for g in meta_rt["GB"] if meta_rt["gen_bus"][g] == b)
                inflow = (sum(+float(value(m_rt.F[l, ti]) or 0.0) for l in meta_rt["L"] if meta_rt["line_to"][l] == b)
                          - sum(+float(value(m_rt.F[l, ti]) or 0.0) for l in meta_rt["L"] if meta_rt["line_from"][l] == b))
                Dbt = meta_rt["D"][(b, ti)]
                sv = float(value(m_rt.Served[b, ti]) or 0.0)
                sh = float(value(m_rt.Shed[b, ti]) or 0.0)
                sp = float(value(m_rt.Spill[b, ti]) or 0.0)

                rows_rt_nb.append({
                    "timestamp": ts,
                    "node": b,
                    "gen_nonbat_MW": gen_nb,
                    "gen_batt_net_MW": gen_b,
                    "net_inflow_MW": inflow,
                    "demand_nominal_MW": Dbt,
                    "served_MW": sv,
                    "unserved_MW": sh,
                    "spill_MW": sp,
                    "lhs_MW": gen_nb + gen_b + inflow + sh,
                    "rhs_MW": Dbt + sp,
                    "kcl_diff_MW": gen_nb + gen_b + inflow + sh - (Dbt + sp)
                })

                # comparable files
                rows_serv_generic.append({"timestamp": ts, "node": b, "served_MW": sv, "unserved_MW": sh, "demand_MW": Dbt})
                lhs = gen_nb + gen_b + inflow
                rhs = sv + sp
                rows_nbcheck_generic.append({
                    "timestamp": ts,
                    "node": b,
                    "gen_nonbat_MW": gen_nb,
                    "batt_net_MW": gen_b,
                    "net_inflow_MW": inflow,
                    "supply_MW": lhs,
                    "demand_MW": Dbt,
                    "served_MW": sv,
                    "unserved_MW": sh,
                    "spill_MW": sp,
                    "lhs_minus_rhs_MW": lhs - rhs
                })

            # comparable dispatch/flow rows
            for r in rows_rt_g[-(len(meta_rt["GG"]) + len(meta_rt["GB"])):]:
                rows_disp_generic.append({
                    "timestamp": r["timestamp"],
                    "gen_id": r["gen_id"],
                    "node": r["node"],
                    "p_MW": r["p_MW"],
                    "p_dis_MW": r.get("p_dis_MW", np.nan),
                    "p_ch_MW": r.get("p_ch_MW", np.nan)
                })
            for r in rows_rt_f[-len(meta_rt["L"]):]:
                rows_flow_generic.append({
                    "timestamp": r["timestamp"],
                    "line_id": r["line_id"],
                    "from_node": r["from_node"],
                    "to_node": r["to_node"],
                    "flow_MW": r["flow_MW"]
                })

            rows_flowcon.append({"timestamp": ts, "sum_net_inflow_MW": 0.0, "flow_conservation_ok": True})

        # save RT-style
        rt_dispatch = pd.DataFrame(rows_rt_g).sort_values(["timestamp", "node", "gen_id"])
        rt_flows = pd.DataFrame(rows_rt_f).sort_values(["timestamp", "line_id"])
        rt_nodebal = pd.DataFrame(rows_rt_nb).sort_values(["timestamp", "node"])
        rt_dispatch.to_csv(os.path.join(out_dir, "sced_dispatch_RT.csv"), index=False)
        rt_flows.to_csv(os.path.join(out_dir, "sced_flows_RT.csv"), index=False)
        rt_nodebal.to_csv(os.path.join(out_dir, "sced_node_balance_RT.csv"), index=False)

        # save comparable set
        disp_generic = pd.DataFrame(rows_disp_generic).sort_values(["timestamp", "node", "gen_id"])
        flow_generic = pd.DataFrame(rows_flow_generic).sort_values(["timestamp", "line_id"])
        serv_generic = pd.DataFrame(rows_serv_generic).sort_values(["timestamp", "node"])
        nbcheck_generic = pd.DataFrame(rows_nbcheck_generic).sort_values(["timestamp", "node"])
        flowcon = pd.DataFrame(rows_flowcon).sort_values(["timestamp"])

        disp_generic.to_csv(os.path.join(out_dir, "dispatch.csv"), index=False)
        flow_generic.to_csv(os.path.join(out_dir, "flows.csv"), index=False)
        serv_generic.to_csv(os.path.join(out_dir, "served_unserved.csv"), index=False)
        nbcheck_generic.to_csv(os.path.join(out_dir, "node_balance_check.csv"), index=False)
        flowcon.to_csv(os.path.join(out_dir, "flow_conservation_check.csv"), index=False)

        # >>> ALLOCATED reserve detail export <<<
        compute_and_write_reserves_detail(
            out_dir=out_dir,
            gens_static=gens_static,
            gen_h_pub=gen_h_pub,
            rt_ts=rt_ts,
            commit_status_at_ts=commit_status_at_ts,
            meta_rt=meta_rt,
            m_rt=m_rt,
            step_h=meta_rt["step_h"]
        )
        # <<< END NEW >>>

        # --- aggregate outcome metrics (NEW) ---
        served_MWh = _energy_MWh_from_MW_series([r["served_MW"] for r in rows_serv_generic], meta_rt["step_h"])
        unserved_MWh = _energy_MWh_from_MW_series([r["unserved_MW"] for r in rows_serv_generic], meta_rt["step_h"])
        spill_MWh = _energy_MWh_from_MW_series([r.get("spill_MW", 0.0) for r in rows_rt_nb], meta_rt["step_h"])

        try:
            res_short_MW = [float(value(m_rt.ResShort[ti]) or 0.0) for ti in meta_rt["T"]]
        except Exception:
            res_short_MW = [np.nan] * len(meta_rt["rt_ts"])
        reserve_shortfall_MWh_equiv = _energy_MWh_from_MW_series(res_short_MW, meta_rt["step_h"])
        reserve_shortfall_max_MW = float(np.nanmax(res_short_MW)) if len(res_short_MW) else np.nan

        # ---- LMPs (duals or FD) ---- (timed, NEW)
        tp0 = time.perf_counter()
        pricing_method = ""
        have_duals = duals_imported(m_rt, m_rt.balance)

        if have_duals:
            pricing_method = "dual"
            lmp_rows = []
            for ti, ts in enumerate(meta_rt["rt_ts"]):
                for b in meta_rt["B"]:
                    mu = m_rt.dual.get(m_rt.balance[b, ti], None)
                    price = (mu / meta_rt["step_h"]) if (mu is not None and meta_rt["step_h"] > 0) else np.nan
                    lmp_rows.append({"timestamp": ts, "node": b, "lmp": price})
            lmps_rt = pd.DataFrame(lmp_rows).sort_values(["timestamp", "node"])
        else:
            pricing_method = "finite_difference"
            print("[warn] Duals not available; computing FD LMPs.")
            fd_prices = price_via_finite_difference(
                build_model_fn=build_rt_sced_lp,
                base_args=dict(
                    B=nodes, L=list(lines_df["line_id"]), lines_df=lines_df,
                    gens_static=gens_static, gen_h_pub=gen_h_pub, load_pub=load_pub,
                    rt_ts=list(rt_ts), commit_status_at_ts=commit_status_at_ts,
                    soc0_map=soc0_for_rt, cfg=cfg
                ),
                cfg=cfg, rt_ts=list(rt_ts), B=nodes, step_h=meta_rt["step_h"],
                epsilon_MW=fd_eps
            )
            lmps_rt = pd.DataFrame([
                {"timestamp": ts, "node": b, "lmp": fd_prices.get((ts, b), np.nan)}
                for ts in meta_rt["rt_ts"] for b in nodes
            ])

        tp1 = time.perf_counter()
        pricing_seconds = tp1 - tp0

        cap_price = float(cfg.get("settlement_price_cap_MWh", 6000.0))
        lmps_rt["settlement_lmp"] = lmps_rt["lmp"].clip(lower=0.0, upper=cap_price)
        lmps_rt.to_csv(os.path.join(out_dir, "lmps_RT.csv"), index=False)
        lmps_rt.to_csv(os.path.join(out_dir, "settlement_RT.csv"), index=False)

        # optional runtime_by_step
        if args.runtime_by_step:
            # note: this SCED is solved as one horizon LP; we still write per-interval rows for downstream analysis
            # (if you later switch to rolling/interval-by-interval solves, this becomes true per-step runtime)
            for ts in meta_rt["rt_ts"]:
                runtime_by_step_rows.append({
                    "run_started_utc": run_started_iso,
                    "row_written_utc": _now_iso_utc(),
                    "date": str(d),
                    "window_hash": day_hash,
                    "timestamp": ts,
                    "mechanism": "LMP",
                    "solver": str(solver_used),
                    "host": host,
                    "python": pyver,
                    "step_minutes": float(step_minutes),
                    "build_seconds_DA": build_seconds_DA,
                    "solve_seconds_DA": solve_seconds_DA,
                    "build_seconds_RT": build_seconds_RT,
                    "solve_seconds_RT": solve_seconds_RT,
                    "pricing_seconds": pricing_seconds,
                    "pricing_method": pricing_method,
                    "n_vars_DA": nvars_DA,
                    "n_cons_DA": ncons_DA,
                    "n_vars_RT": nvars_RT,
                    "n_cons_RT": ncons_RT,
                    "obj_DA": obj_DA,
                    "obj_RT": obj_RT,
                })

        # ----- finalize runtime row for this window (NEW) -----
        day_wall_t1 = time.perf_counter()
        window_seconds_total = day_wall_t1 - day_wall_t0

        runtime_rows.append({
            "run_started_utc": run_started_iso,
            "row_written_utc": _now_iso_utc(),
            "date": str(d),
            "window_hash": day_hash,
            "mechanism": "LMP",
            "solver": str(solver_used),
            "host": host,
            "python": pyver,
            "n_steps": int(len(pub_ts)),
            "step_minutes": float(step_minutes),

            "build_seconds_DA": build_seconds_DA,
            "solve_seconds_DA": solve_seconds_DA,
            "build_seconds_RT": build_seconds_RT,
            "solve_seconds_RT": solve_seconds_RT,
            "pricing_seconds": pricing_seconds,
            "pricing_method": pricing_method,

            "window_seconds_total": window_seconds_total,
            "seconds_per_interval_equiv": window_seconds_total / max(1, len(pub_ts)),

            "n_vars_DA": nvars_DA,
            "n_cons_DA": ncons_DA,
            "n_vars_RT": nvars_RT,
            "n_cons_RT": ncons_RT,

            "obj_DA": obj_DA,
            "obj_RT": obj_RT,

            "served_MWh": served_MWh,
            "unserved_MWh": unserved_MWh,
            "spill_MWh": spill_MWh,

            "reserve_shortfall_MWh_equiv": reserve_shortfall_MWh_equiv,
            "reserve_shortfall_max_MW": reserve_shortfall_max_MW,

            "failed": 0,
            "fail_stage": "",
            "termination": str(tc_rt),
        })

        print(f"[ok] Wrote DA & RT outputs for {d} → {out_dir}")

    # ---- write runtime logs (NEW) ----
    if runtime_rows:
        rt_df = pd.DataFrame(runtime_rows)
        rt_df.to_csv(os.path.join(out_root, "runtime.csv"), index=False)
        print(f"[ok] Wrote runtime.csv → {os.path.join(out_root, 'runtime.csv')}")
    else:
        print("[warn] No runtime rows collected; nothing written for runtime.csv")

    if args.runtime_by_step and runtime_by_step_rows:
        rts_df = pd.DataFrame(runtime_by_step_rows)
        rts_df.to_csv(os.path.join(out_root, "runtime_by_step.csv"), index=False)
        print(f"[ok] Wrote runtime_by_step.csv → {os.path.join(out_root, 'runtime_by_step.csv')}")
    elif args.runtime_by_step:
        print("[warn] --runtime-by-step set but no rows collected; nothing written.")
