#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-varying generator clustering + cluster Shapley allocation (serve-max DC flow)

Inputs
------
- Network JSON: availabilityPayments/data/network_uk.json
- Generator profiles (single file or dir):
    availabilityPayments/data/gen_profiles_expost.csv
    (long form with columns including: timestamp, gen_id, avail_MW; wide also accepted)
- Demand files (CSV): availabilityPayments/data/demand/D0_demand.csv ... D7_demand.csv

Outputs
-------
Per-day folder: availabilityPayments/outputs/YYYY-MM-DD/
  - final_clusters_<YYYY-MM-DD_HHMM>.json
  - cf_cluster_<YYYY-MM-DD_HHMM>.csv
  - shapley_cluster_<YYYY-MM-DD_HHMM>.csv
  - allocated_from_clusters_<YYYY-MM-DD_HHMM>.csv

Append-only long files:
  - availabilityPayments/outputs/long/cf_cluster.csv
      columns: timestamp, coalition, served_load, served_prop, unserved_load, unserved_prop
  - availabilityPayments/outputs/long/shapley_allocations_generators.csv
      columns: timestamp, gen_id, shap_from_cluster, shap_norm, v_all_cluster
"""

import os
import re
import glob
import json
import math
import time
import heapq
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
import pyomo.environ as pyo
from tqdm import tqdm

# ========= Paths & Config =========
DATA_ROOT    = os.path.join("availabilityPayments", "data")
NETWORK_PATH = os.path.join(DATA_ROOT, "network_uk.json")
GEN_SOURCE   = os.path.join(DATA_ROOT, "gen_profiles_expost.csv")  # single CSV or a directory
DEMAND_DIR   = os.path.join(DATA_ROOT, "demand")

OUTPUT_ROOT  = os.path.join("availabilityPayments", "outputs")
OUTPUT_DIR   = OUTPUT_ROOT
LONG_DIR     = os.path.join(OUTPUT_ROOT, "long")

TRUNK_NODES     = {'N1', 'N16', 'N17'}
HOP_THRESHOLD   = 2
USE_WIDEST_PATH = False          # True => widest-path capacity; False => shortest-path bottleneck
SOLVER          = "glpk"         # must be on PATH; change to "highs" or another if preferred

# ========= Small helpers =========
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def infer_timestamp_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError(f"Could not infer a timestamp column. Columns={list(df.columns)}")

def parse_load_id_from_filename(path: str) -> str:
    # ".../D0_demand.csv" -> "D0"
    base = os.path.basename(path)
    m = re.match(r"(D\d+)", base)
    if not m:
        raise ValueError(f"Cannot infer load ID from filename: {base}")
    return m.group(1)

def to_utc_naive(series: pd.Series) -> pd.Series:
    """
    Parse datetimes, coerce to UTC tz-aware, then drop tz -> tz-naive.
    - tz-naive inputs are assumed UTC.
    - tz-aware inputs are converted to UTC.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)  # tz-aware UTC
    return s.dt.tz_convert("UTC").dt.tz_localize(None)     # drop tz -> naive

def _coalesce_numeric_col(df: pd.DataFrame, candidates: list[str], default_col: str | None = None) -> pd.Series:
    """
    First non-null numeric among candidates; if none and default_col provided, use default_col.
    Returns float Series.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    if default_col and default_col in df.columns:
        out = out.fillna(pd.to_numeric(df[default_col], errors="coerce"))
    return out

# ========= IO: network, gens, loads =========
def build_graph(net):
    nodes = net['nodes']
    edges = [tuple(e) for e in net['edges']]
    # parse edge capacities from "u,v"
    edge_capacity = {}
    for k, cap in net['edge_capacity'].items():
        u, v = k.split(',')
        edge_capacity[(u, v)] = float(cap)
        edge_capacity[(v, u)] = float(cap)  # undirected

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.set_edge_attributes(G, edge_capacity, 'capacity')
    hop_len = dict(nx.all_pairs_shortest_path_length(G))
    return G, nodes, edge_capacity, hop_len

def read_gen_profiles(source: str) -> pd.DataFrame:
    """
    Reads generator availability and returns long-format:
      columns: timestamp, gen_id, avail_MW
    Accepts either:
      - a single CSV file path (e.g., gen_profiles_expost.csv), or
      - a directory containing one/more CSVs (tries to prefer gen_profiles_expost.csv)
    """
    paths = []
    if os.path.isdir(source):
        # Prefer expost file if present; otherwise any CSV in the dir (use with care)
        preferred = glob.glob(os.path.join(source, "gen_profiles_expost.csv"))
        if preferred:
            paths = preferred
        else:
            paths = sorted(glob.glob(os.path.join(source, "*.csv")))
    else:
        paths = [source]

    if not paths:
        raise FileNotFoundError(f"No generator CSVs found at {source}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        ts_col = infer_timestamp_col(df)
        df[ts_col] = to_utc_naive(df[ts_col])

        non_ts = [c for c in df.columns if c != ts_col]
        # LONG form: must have gen_id + some availability-like column
        if "gen_id" in non_ts:
            cand = [c for c in ["avail_MW", "availability_MW", "availability", "available_MW", "Pmax_MW", "power_MW"] if c in non_ts]
            if not cand:
                raise ValueError(f"{os.path.basename(path)}: no availability column found among {non_ts}")
            sub = df[[ts_col, "gen_id"] + cand].copy()
            sub["avail_MW"] = _coalesce_numeric_col(sub, cand)
            sub = sub[[ts_col, "gen_id", "avail_MW"]]
            sub.rename(columns={ts_col: "timestamp"}, inplace=True)
        else:
            # WIDE form: timestamp + gen columns
            wide = df.set_index(ts_col)
            wide.columns = [str(c) for c in wide.columns]
            sub = wide.stack().reset_index()
            sub.columns = ["timestamp", "gen_id", "avail_MW"]
            sub["timestamp"] = to_utc_naive(sub["timestamp"])

        sub["avail_MW"] = pd.to_numeric(sub["avail_MW"], errors="coerce").fillna(0.0).clip(lower=0.0)
        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)
    out = (
        long_df.groupby(["timestamp", "gen_id"], as_index=False)["avail_MW"]
        .sum()
    )
    return out

def read_all_loads_from_dir(demand_dir: str) -> pd.DataFrame:
    """
    Reads ALL CSVs like D*_demand.csv in demand_dir and returns long DF:
      columns: timestamp, load_id, power_MW
    Prefers 'total_demand_MW'; fallbacks: 'power_MW', 'demand_MW', first non-timestamp column.
    """
    files = sorted(glob.glob(os.path.join(demand_dir, "D*_demand.csv")))
    if not files:
        raise FileNotFoundError(f"No demand CSVs found in {demand_dir}")
    frames = []
    for path in files:
        ld = pd.read_csv(path)
        ts_col = infer_timestamp_col(ld)
        ld[ts_col] = to_utc_naive(ld[ts_col])  # normalize to UTC-naive

        non_ts = [c for c in ld.columns if c != ts_col]
        if "total_demand_MW" in non_ts:
            pcol = "total_demand_MW"
        elif "power_MW" in non_ts:
            pcol = "power_MW"
        elif "demand_MW" in non_ts:
            pcol = "demand_MW"
        elif non_ts:
            pcol = non_ts[0]
        else:
            raise ValueError(f"No demand column found in {path}")

        sub = ld[[ts_col, pcol]].copy()
        sub.columns = ["timestamp", "power_MW"]
        sub["timestamp"] = to_utc_naive(sub["timestamp"])  # ensure normalized
        sub["load_id"] = parse_load_id_from_filename(path)
        sub["power_MW"] = pd.to_numeric(sub["power_MW"], errors="coerce").fillna(0.0).clip(lower=0.0)
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    return out

# ========= Clustering internals =========
def widest_path_capacity(G, source, target):
    """Maximize bottleneck capacity along a path using a max-heap."""
    best = {source: float('inf')}
    pq = [(-best[source], source)]
    while pq:
        b_u, u = heapq.heappop(pq); b_u = -b_u
        if u == target:
            return b_u
        for v, attrs in G[u].items():
            cap = attrs.get('capacity', 0.0)
            b_v = min(b_u, cap)
            if b_v > best.get(v, 0.0):
                best[v] = b_v
                heapq.heappush(pq, (-b_v, v))
    return 0.0

def shortest_path_bottleneck(G, u, v, edge_capacity):
    path = nx.shortest_path(G, u, v)
    return min(edge_capacity[(path[k], path[k+1])] for k in range(len(path)-1))

def nearest_trunk_map(gen_nodes: dict, hop_len) -> dict:
    parent = {}
    for g, bus in gen_nodes.items():
        best, dist = None, float('inf')
        for t in TRUNK_NODES:
            d = hop_len.get(bus, {}).get(t, float('inf'))
            if d < dist:
                best, dist = t, d
        parent[g] = best
    return parent

def cluster_generators_for_timestamp(G, edge_capacity, hop_len, gen_nodes: dict, gen_avail: dict):
    """
    Greedy clustering per-timestamp with:
      - same trunk branch
      - hop distance ≤ HOP_THRESHOLD
      - path bottleneck ≥ max(avail_i, avail_j) for at least one pair
    """
    # co-location clusters (same bus)
    bus_to_gens = defaultdict(list)
    for g, bus in gen_nodes.items():
        bus_to_gens[bus].append(g)
    clusters = [set(gs) for gs in bus_to_gens.values()]
    parent_map = nearest_trunk_map(gen_nodes, hop_len)

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                # trunk branch filter
                if parent_map[next(iter(c1))] != parent_map[next(iter(c2))]:
                    continue
                # hop filter
                dist = min(hop_len[gen_nodes[g1]][gen_nodes[g2]] for g1 in c1 for g2 in c2)
                if dist > HOP_THRESHOLD:
                    continue
                # capacity feasibility
                can_merge = False
                for g1 in c1:
                    for g2 in c2:
                        u, v = gen_nodes[g1], gen_nodes[g2]
                        cap = (widest_path_capacity(G, u, v)
                               if USE_WIDEST_PATH else
                               shortest_path_bottleneck(G, u, v, edge_capacity))
                        need = max(gen_avail.get(g1, 0.0), gen_avail.get(g2, 0.0))
                        if cap >= need:
                            can_merge = True
                            break
                    if can_merge:
                        break
                if can_merge:
                    clusters[i] = c1 | c2
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    final_clusters = [sorted(list(c)) for c in clusters]
    gen_to_cluster = {g: idx for idx, c in enumerate(final_clusters) for g in c}
    return final_clusters, gen_to_cluster

# ========= DC serve-max model =========
def solve_optimal_flow(nodes, edge_capacity, gen_nodes, load_nodes, avail_by_gen, demand_by_load):
    """Maximize total served load subject to DC balance and line capacities."""
    model = pyo.ConcreteModel()
    model.edges = pyo.Set(initialize=list(edge_capacity.keys()), dimen=2)
    model.nodes = pyo.Set(initialize=nodes)
    model.gen_idx = pyo.Set(initialize=list(avail_by_gen.keys()))
    model.load_idx = pyo.Set(initialize=list(demand_by_load.keys()))

    model.flow = pyo.Var(model.edges, domain=pyo.Reals)
    model.gen_power = pyo.Var(model.gen_idx, domain=pyo.NonNegativeReals)
    model.load_served = pyo.Var(model.load_idx, domain=pyo.NonNegativeReals)

    def cap_rule(model, i, j):
        return pyo.inequality(-edge_capacity[(i, j)], model.flow[i, j], edge_capacity[(i, j)])
    model.capacity = pyo.Constraint(model.edges, rule=cap_rule)

    model.gen_limit = pyo.Constraint(model.gen_idx, rule=lambda model, g: model.gen_power[g] <= avail_by_gen[g])
    model.load_limit = pyo.Constraint(model.load_idx, rule=lambda model, d: model.load_served[d] <= demand_by_load[d])

    def balance_rule(model, n):
        inflow = sum(model.flow[i, j] for (i, j) in model.edges if j == n)
        outflow = sum(model.flow[i, j] for (i, j) in model.edges if i == n)
        gen = sum(model.gen_power[g] for g, bus in gen_nodes.items() if bus == n)
        demand = sum(model.load_served[d] for d, bus in load_nodes.items() if bus == n)
        return inflow + gen == outflow + demand
    model.balance = pyo.Constraint(model.nodes, rule=balance_rule)

    model.obj = pyo.Objective(
        expr=sum(model.load_served[d] for d in model.load_idx),
        sense=pyo.maximize
    )
    pyo.SolverFactory(SOLVER).solve(model, tee=False)

    served_total = sum(pyo.value(model.load_served[d]) for d in model.load_idx)
    return served_total

def precompute_cf(nodes, edge_capacity, gen_nodes, load_nodes, avail_by_gen, demand_by_load,
                  player_sets, grouping):
    """Characteristic function v(S) over cluster coalitions."""
    cf = {}
    for S in tqdm(player_sets, desc="Precomputing CF", unit="coalition"):
        gens = [g for cl in S for g in grouping[cl]]
        # zero out non-members
        avail = {g: 0.0 for g in avail_by_gen.keys()}
        for g in gens:
            avail[g] = avail_by_gen.get(g, 0.0)
        vS = solve_optimal_flow(nodes, edge_capacity, gen_nodes, load_nodes, avail, demand_by_load)
        cf[tuple(sorted(S))] = vS
    return cf

def all_subsets(players):
    for k in range(len(players) + 1):
        for combo in itertools.combinations(players, k):
            yield combo

# ========= Shapley (clusters) =========
def shapley_from_cf(players, cf_dict):
    """
    Compute Shapley value for each player given cf_dict over all subsets.
    players: list of player labels (hashable)
    cf_dict: dict with keys as tuples of players (coalitions) -> v(S)
    """
    n = len(players)
    fact = math.factorial
    denom = fact(n)
    index = {p: i for i, p in enumerate(players)}
    phi = {p: 0.0 for p in players}

    for p in players:
        others = [q for q in players if q != p]
        for S in all_subsets(others):
            S_key = tuple(sorted(S))
            S_with_p_key = tuple(sorted(S + (p,)))
            vS = cf_dict.get(S_key, 0.0)
            vSp = cf_dict.get(S_with_p_key, 0.0)
            k = len(S)
            weight = fact(k) * fact(n - k - 1) / denom
            phi[p] += weight * (vSp - vS)
    return phi

# ========= Main pipeline =========
def main():
    ensure_dirs(OUTPUT_DIR, LONG_DIR)

    # --- Load network ---
    log(f"Loading network: {NETWORK_PATH}")
    with open(NETWORK_PATH, 'r') as f:
        net = json.load(f)

    G, nodes, edge_capacity, hop_len = build_graph(net)
    gen_nodes_static = {g: info['node'] for g, info in net['generators'].items()}
    load_nodes_static = {d: info['node'] for d, info in net['loads'].items()}

    # --- Load generator profiles (single file or directory) ---
    log(f"Loading generator profiles from: {GEN_SOURCE}")
    df_gen = read_gen_profiles(GEN_SOURCE)  # timestamp, gen_id, avail_MW
    # Keep only gens known to the network
    df_gen = df_gen[df_gen['gen_id'].isin(gen_nodes_static.keys())]

    # --- Load loads ---
    log(f"Loading demand files from: {DEMAND_DIR}")
    df_loads = read_all_loads_from_dir(DEMAND_DIR)  # timestamp, load_id, power_MW
    # Keep only loads known to the network
    df_loads = df_loads[df_loads['load_id'].isin(load_nodes_static.keys())]

    # Log coverage (after normalization)
    log(f"gen time range:  {df_gen['timestamp'].min()} -> {df_gen['timestamp'].max()} (unique={df_gen['timestamp'].nunique()})")
    log(f"load time range: {df_loads['timestamp'].min()} -> {df_loads['timestamp'].max()} (unique={df_loads['timestamp'].nunique()})")

    # Align timestamps: intersection only — keep pandas.Timestamp type to match groupby keys
    ts_idx_gen  = pd.Index(df_gen['timestamp'])
    ts_idx_load = pd.Index(df_loads['timestamp'])
    timestamps = ts_idx_gen.intersection(ts_idx_load).unique().sort_values()

    log(f"overlap timestamps: {len(timestamps)}")
    if len(timestamps) == 0:
        g_samp = list(ts_idx_gen.unique().sort_values()[:5])
        l_samp = list(ts_idx_load.unique().sort_values()[:5])
        log(f"sample gen ts:  {g_samp}")
        log(f"sample load ts: {l_samp}")
        raise ValueError("No overlapping timestamps between gen profiles and load files (after UTC normalization).")

    # Pre-index for speed — keys are pandas.Timestamp (matching 'timestamps')
    gen_by_ts = {
        ts: sub[['gen_id', 'avail_MW']].set_index('gen_id')['avail_MW'].to_dict()
        for ts, sub in df_gen.groupby('timestamp', sort=False)
    }
    load_by_ts = {
        ts: sub[['load_id', 'power_MW']].set_index('load_id')['power_MW'].to_dict()
        for ts, sub in df_loads.groupby('timestamp', sort=False)
    }

    # Prepare append-only long CSVs
    long_cf_cluster = os.path.join(LONG_DIR, "cf_cluster.csv")
    long_shap_gen = os.path.join(LONG_DIR, "shapley_allocations_generators.csv")
    if not os.path.exists(long_cf_cluster):
        pd.DataFrame(columns=['timestamp','coalition','served_load','served_prop','unserved_load','unserved_prop']).to_csv(long_cf_cluster, index=False)
    if not os.path.exists(long_shap_gen):
        pd.DataFrame(columns=['timestamp','gen_id','shap_from_cluster','shap_norm','v_all_cluster']).to_csv(long_shap_gen, index=False)

    # Iterate timestamps (already pandas.Timestamp)
    for ts in tqdm(timestamps, desc="Processing timestamps"):
        ts_dt = ts
        ts_str = ts_dt.strftime("%Y-%m-%d_%H%M")
        day_folder = os.path.join(OUTPUT_DIR, ts_dt.strftime("%Y-%m-%d"))
        ensure_dirs(day_folder)

        # Availability and demand dicts for this ts
        avail_by_gen = {g: float(v) for g, v in gen_by_ts.get(ts, {}).items()}
        demand_by_load = {d: float(v) for d, v in load_by_ts.get(ts, {}).items()}

        # Ensure all known entities present (fill missing with 0)
        avail_by_gen = {g: max(0.0, float(avail_by_gen.get(g, 0.0))) for g in gen_nodes_static.keys()}
        demand_by_load = {d: max(0.0, float(demand_by_load.get(d, 0.0))) for d in load_nodes_static.keys()}

        # Quick diagnostics
        tot_avail = float(sum(avail_by_gen.values()))
        tot_demand = float(sum(demand_by_load.values()))
        log(f"{ts_str}: total_gen_avail={tot_avail:.3f} MW, total_demand={tot_demand:.3f} MW, gens={len(avail_by_gen)}, loads={len(demand_by_load)}")
        if tot_avail == 0.0:
            missing_g = [g for g in gen_nodes_static.keys() if g not in gen_by_ts.get(ts, {})]
            if missing_g:
                log(f"WARNING {ts_str}: all-zero availability; {len(missing_g)} gens missing profiles at this ts. Example: {missing_g[:5]}")
        if tot_demand == 0.0:
            missing_d = [d for d in load_nodes_static.keys() if d not in load_by_ts.get(ts, {})]
            if missing_d:
                log(f"WARNING {ts_str}: all-zero demand; {len(missing_d)} loads missing demand at this ts. Example: {missing_d[:5]}")

        # Build time-specific cluster partition
        final_clusters, gen_to_cluster = cluster_generators_for_timestamp(
            G, edge_capacity, hop_len,
            gen_nodes=gen_nodes_static,
            gen_avail=avail_by_gen
        )

        # Save clusters for this timestamp
        with open(os.path.join(day_folder, f"final_clusters_{ts_str}.json"), 'w') as f:
            json.dump(final_clusters, f, indent=2)

        # Build grouping dict keyed by cluster label strings
        cluster_labels = [",".join(c) for c in final_clusters]
        grouping = {label: final_clusters[i] for i, label in enumerate(cluster_labels)}
        players = cluster_labels

        # If no clusters or trivial, skip gracefully
        if len(players) == 0:
            log(f"{ts_str}: No clusters — skipping.")
            continue

        # Characteristic function v(S) for all coalitions
        player_sets = list(all_subsets(players))
        cf_dict = precompute_cf(
            nodes, edge_capacity, gen_nodes_static, load_nodes_static,
            avail_by_gen, demand_by_load,
            player_sets, grouping
        )

        # Save CF table for this timestamp
        total_demand_ts = sum(demand_by_load.values())
        cf_rows = []
        for S_key, vS in cf_dict.items():
            served = float(vS)
            unserved = max(0.0, total_demand_ts - served)
            served_prop = (served / total_demand_ts) if total_demand_ts > 0 else np.nan
            unserved_prop = (unserved / total_demand_ts) if total_demand_ts > 0 else np.nan
            cf_rows.append({
                "timestamp": ts_dt,
                "coalition": ",".join(S_key),
                "served_load": served,
                "served_prop": served_prop,
                "unserved_load": unserved,
                "unserved_prop": unserved_prop
            })
        df_cf = pd.DataFrame(cf_rows).sort_values("coalition")
        df_cf.to_csv(os.path.join(day_folder, f"cf_cluster_{ts_str}.csv"), index=False)
        # Append to long
        df_cf.to_csv(long_cf_cluster, mode="a", header=False, index=False)

        # Shapley at cluster level
        phi_cluster = shapley_from_cf(players, cf_dict)

        # Save cluster Shapley
        df_phi = pd.DataFrame({
            "timestamp": [ts_dt] * len(players),
            "cluster": players,
            "phi_cluster": [float(phi_cluster[p]) for p in players]
        })
        df_phi.to_csv(os.path.join(day_folder, f"shapley_cluster_{ts_str}.csv"), index=False)

        # Grand-coalition value (for reference)
        v_all = cf_dict.get(tuple(sorted(players)), 0.0)

        # Allocate cluster Shapley down to generators (proportional to availability in cluster)
        gen_rows = []
        for label in players:
            members = grouping[label]
            # availability weights
            avs = np.array([max(0.0, avail_by_gen.get(g, 0.0)) for g in members], dtype=float)
            if np.all(avs == 0.0):
                # equal split if no availability
                weights = np.ones(len(members)) / len(members)
            else:
                weights = avs / avs.sum()

            phi_c = float(phi_cluster[label])
            allocs = weights * phi_c

            for g, a in zip(members, allocs):
                gen_rows.append({"timestamp": ts_dt, "gen_id": g, "shap_from_cluster": float(a)})

        df_gen_alloc = pd.DataFrame(gen_rows)

        # Normalized non-negative share across ALL generators at this timestamp
        # (use max(0, a) then divide by sum to get shap_norm)
        df_gen_alloc["shap_pos"] = df_gen_alloc["shap_from_cluster"].clip(lower=0.0)
        ssum = df_gen_alloc["shap_pos"].sum()
        if ssum > 0:
            df_gen_alloc["shap_norm"] = df_gen_alloc["shap_pos"] / ssum
        else:
            # fallback: uniform across present gens
            n = len(df_gen_alloc)
            df_gen_alloc["shap_norm"] = (1.0 / n) if n > 0 else np.nan

        df_gen_alloc["v_all_cluster"] = float(v_all)

        # Save per-timestamp generator allocations (named to match your downstream consumer)
        df_gen_alloc.rename(columns={"shap_from_cluster": "shap_from_cluster"}, inplace=True)
        df_gen_alloc.to_csv(os.path.join(day_folder, f"allocated_from_clusters_{ts_str}.csv"), index=False)

        # Append to long (columns: timestamp, gen_id, shap_from_cluster, shap_norm, v_all_cluster)
        df_gen_alloc[["timestamp","gen_id","shap_from_cluster","shap_norm","v_all_cluster"]].to_csv(
            long_shap_gen, mode="a", header=False, index=False
        )

    log("All timestamps processed successfully.")

# --------- Entry point ---------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        raise
