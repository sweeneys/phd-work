#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LMP (nodal vs socialised) vs AMM — Per-HH & Totals, Geography, and Burden→Cost Links
(ABSOLUTE controllable energy version: uses kWh/MWh, not shares)

UPDATED: Loads FOUR AMM subscription series and shows 5-series bar charts:
- LMP socialised
- Base Individual (AMM1)
- Base Aggregate (AMM1)
- Delta Individual (AMM2)
- Delta Aggregate (AMM2)

Important constraint satisfied:
- Output filenames are unchanged.
- Existing single-AMM columns remain and are populated from the PRIMARY AMM series
  (Base Individual (AMM1)) so downstream logic remains compatible.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite

# ===================== EDIT THESE =====================
LMP_DIR = Path("marketExecution_Actual/analysis/lmp_household_costs/run_20250919_230229")
AMM_DIR = Path("marketExecution_AMM/availabilityPayments/analysis/run_20251206_120636")

AMM_VARIANT = "NonFuelOpexCapExIndividualTS_Base"  # kept for backward compatibility (primary series)
AMM_VARIANT_FILES = {
    "NonFuelOpexCapExIndividualTS_Base":  "per_product_subscription_flat_NonFuelOpexCapExIndividualTS_Base.csv",
    "NonFuelOpexCapExIndividualTS_Delta": "per_product_subscription_flat_NonFuelOpexCapExIndividualTS_Delta.csv",
    "NonFuelOpexCapExAggregate_Base":     "per_product_subscription_flat_NonFuelOpexCapExAggregate_Base.csv",
    "NonFuelOpexCapExAggregate_Delta":    "per_product_subscription_flat_NonFuelOpexCapExAggregate_Delta.csv",
}

# Optional (not required for the absolute-energy analyses)
GEN_EXPOST = Path("/mnt/data/gen_profiles_expost.csv")

# Products & household counts (for totals & per-HH normalisation)
PRODS = ["P1", "P2", "P3", "P4"]
HOUSEHOLDS = {"P1": 19_000_000, "P2": 6_000_000, "P3": 2_500_000, "P4": 1_500_000}

OUT_DIR = Path("./comparison_out")
# =====================================================

# ---- NEW: AMM multi-series configuration ----
AMM_SERIES_ORDER = [
    ("Base Individual (AMM1)",  "NonFuelOpexCapExIndividualTS_Base"),
    ("Base Aggregate (AMM1)",   "NonFuelOpexCapExAggregate_Base"),
    ("Delta Individual (AMM2)", "NonFuelOpexCapExIndividualTS_Delta"),
    ("Delta Aggregate (AMM2)",  "NonFuelOpexCapExAggregate_Delta"),
]
PRIMARY_AMM_LABEL = "Base Individual (AMM1)"
PRIMARY_AMM_KEY   = "NonFuelOpexCapExIndividualTS_Base"


# ---------- helpers ----------
def choose_col(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None

def safe_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def pearson_r(x, y):
    x = np.asarray([safe_num(v) for v in x], dtype=float)
    y = np.asarray([safe_num(v) for v in y], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x**2).sum() * (y**2).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


# ---------- readers (robust per-HH) ----------
def read_lmp(lmp_dir: Path):
    nodes_fp = lmp_dir / "household_costs_annual.csv"
    soc_fp   = lmp_dir / "product_total_costs_socialised.csv"
    if not nodes_fp.exists():
        raise FileNotFoundError(nodes_fp)
    if not soc_fp.exists():
        raise FileNotFoundError(soc_fp)

    raw = pd.read_csv(nodes_fp)
    soc = pd.read_csv(soc_fp)

    raw["product"] = raw["product"].astype(str)
    raw["load_id"] = raw["load_id"].astype(str)
    soc["product"] = soc["product"].astype(str)

    # ---- FIX: robust detection of per-HH column in nodal file ----
    # Preferred explicit per-HH names
    perhh_col = choose_col(raw, [
        "gbp_per_HH_year", "per_HH_gbp_year", "perHH_gbp_year", "gbp_per_HH_yr",
        "total_cost_per_HH", "total_cost_per_hh", "cost_per_hh", "gbp_per_hh_year",
        # pandas duplicate-column suffix
        "total_cost.1"
    ])

    # If not found explicitly, try to infer from duplicated total_cost columns
    if perhh_col is None and ("total_cost" in raw.columns) and ("total_cost.1" in raw.columns):
        m0 = pd.to_numeric(raw["total_cost"], errors="coerce").median()
        m1 = pd.to_numeric(raw["total_cost.1"], errors="coerce").median()
        # Heuristic: per-HH should be "small" relative to total
        if np.isfinite(m0) and np.isfinite(m1):
            if (m0 > 1e6) and (m1 < 1e6):
                perhh_col = "total_cost.1"
            elif (m1 > 1e6) and (m0 < 1e6):
                perhh_col = "total_cost"

    if perhh_col is not None:
        raw["LMP_nodal_perHH_yr"] = pd.to_numeric(raw[perhh_col], errors="coerce").fillna(0.0)
    else:
        # Fallback: derive from total / households if available
        total_col = choose_col(raw, ["total_cost", "total_gbp", "gbp_total"])
        hh_col    = choose_col(raw, ["households", "num_households", "hh_count", "n_households"])
        if total_col and hh_col:
            tot = pd.to_numeric(raw[total_col], errors="coerce").fillna(0.0)
            hh  = pd.to_numeric(raw[hh_col],    errors="coerce").replace({0: np.nan})
            raw["LMP_nodal_perHH_yr"] = (tot / hh).fillna(0.0)
        else:
            raise ValueError(
                "Could not find/derive a nodal per-HH column. "
                "Expected an explicit per-HH column, or duplicated total_cost.1, or totals+households."
            )

    # Socialised LMP per-HH (unchanged)
    soc_col = choose_col(soc, ["gbp_per_HH_year", "gbp_per_HH_yr"])
    if not soc_col:
        raise ValueError("LMP socialised file needs gbp_per_HH_year (or gbp_per_HH_yr).")

    lmp_soc = soc[["product", soc_col]].rename(columns={soc_col: "LMP_socialised_perHH_yr"})
    lmp_soc["LMP_socialised_perHH_yr"] = pd.to_numeric(lmp_soc["LMP_socialised_perHH_yr"], errors="coerce").fillna(0.0)

    lmp_nodes = raw[["load_id", "product", "LMP_nodal_perHH_yr"]].copy()
    return lmp_nodes, lmp_soc


def read_amm(amm_dir: Path, variant_key: str):
    """
    Reads ONE AMM subscription file (variant_key) + UC energy/power timeseries.
    This is kept for backwards compatibility; the multi-series reader below
    loads the other 3 subscription variants too.
    """
    flat_fp = amm_dir / AMM_VARIANT_FILES[variant_key]
    uc_energy_fp = amm_dir / "per_product_UC_energy_timeseries_RES_served.csv"
    uc_power_fp  = amm_dir / "per_product_UC_power_timeseries_RES_served.csv"  # optional

    if not flat_fp.exists():
        raise FileNotFoundError(flat_fp)
    if not uc_energy_fp.exists():
        raise FileNotFoundError(uc_energy_fp)

    sub = pd.read_csv(flat_fp)
    sub["product"] = sub["product"].astype(str)
    sub_col = choose_col(sub, ["flat_subscription_gbp_per_HH_per_month"])
    if not sub_col:
        raise ValueError("Missing AMM 'flat_subscription_gbp_per_HH_per_month'.")
    sub["AMM_perHH_yr"] = pd.to_numeric(sub[sub_col], errors="coerce").fillna(0.0) * 12.0
    amm_sub = sub[["product", "AMM_perHH_yr"]].copy()

    uc_energy = pd.read_csv(uc_energy_fp)
    uc_energy["product"] = uc_energy["product"].astype(str)
    uc_energy["u_served_MWh"] = pd.to_numeric(uc_energy["u_served_MWh"], errors="coerce").fillna(0.0)
    uc_energy["c_served_MWh"] = pd.to_numeric(uc_energy["c_served_MWh"], errors="coerce").fillna(0.0)

    uc_power = None
    if uc_power_fp.exists():
        uc_power = pd.read_csv(uc_power_fp)
        uc_power["product"] = uc_power["product"].astype(str)
        uc_power["u_MW"] = pd.to_numeric(uc_power["u_MW"], errors="coerce").fillna(0.0)
        uc_power["c_MW"] = pd.to_numeric(uc_power["c_MW"], errors="coerce").fillna(0.0)

    return amm_sub, uc_energy, uc_power


def read_amm_all_subscriptions(amm_dir: Path):
    """
    Load all four AMM subscription series (per-product flat subscriptions).
    Returns: dict[label] -> DataFrame(product, AMM_perHH_yr)
    """
    out = {}
    for label, key in AMM_SERIES_ORDER:
        fp = amm_dir / AMM_VARIANT_FILES[key]
        if not fp.exists():
            raise FileNotFoundError(fp)
        df = pd.read_csv(fp)
        df["product"] = df["product"].astype(str)
        sub_col = choose_col(df, ["flat_subscription_gbp_per_HH_per_month"])
        if not sub_col:
            raise ValueError(f"Missing 'flat_subscription_gbp_per_HH_per_month' in {fp.name}")
        df["AMM_perHH_yr"] = pd.to_numeric(df[sub_col], errors="coerce").fillna(0.0) * 12.0
        out[label] = df[["product", "AMM_perHH_yr"]].copy()
    return out


def read_availability(gen_expost_path: Path):
    if not gen_expost_path.exists():
        return None
    gf = pd.read_csv(gen_expost_path)
    if not {"tech", "avail_MW"}.issubset(gf.columns):
        return None
    gf["tech"] = gf["tech"].astype(str).str.lower()
    gf["avail_MW"] = pd.to_numeric(gf["avail_MW"], errors="coerce").fillna(0.0)
    isU = gf["tech"].str.contains("wind") | gf["tech"].str.contains("wnd")
    U = float(gf.loc[isU, "avail_MW"].sum())
    C = float(gf.loc[~isU, "avail_MW"].sum())
    tot = U + C
    if tot <= 0:
        return None
    return {"U_share": U / tot, "C_share": C / tot, "U_MW_sum": U, "C_MW_sum": C, "Total_MW_sum": tot}


def availability_from_amm_energy(uc_energy: pd.DataFrame):
    agg = uc_energy.groupby("product", as_index=False).agg(
        U_MWh=("u_served_MWh", "sum"),
        C_MWh=("c_served_MWh", "sum"),
    )
    U = float(agg["U_MWh"].sum())
    C = float(agg["C_MWh"].sum())
    tot = U + C
    if tot <= 0:
        return {"U_share": np.nan, "C_share": np.nan, "U_MW_sum": U, "C_MW_sum": C, "Total_MW_sum": tot}
    return {"U_share": U / tot, "C_share": C / tot, "U_MW_sum": U, "C_MW_sum": C, "Total_MW_sum": tot}


# ---------- builders ----------
def behaviour_absolute_energy(uc_energy: pd.DataFrame) -> pd.DataFrame:
    """Absolute controllable energy per product (MWh / kWh) and per household."""
    g = uc_energy.groupby("product", as_index=False).agg(
        U_MWh=("u_served_MWh", "sum"),
        C_MWh=("c_served_MWh", "sum"),
    )
    g["C_kWh_total"] = g["C_MWh"] * 1000.0
    g["C_kWh_per_HH"] = g.apply(
        lambda r: (r["C_kWh_total"] / HOUSEHOLDS.get(str(r["product"]), np.nan)),
        axis=1,
    )
    return g[["product", "C_MWh", "C_kWh_total", "C_kWh_per_HH", "U_MWh"]]


def build_cost_levels_multi(lmp_nodes: pd.DataFrame, lmp_soc: pd.DataFrame, amm_subs: dict) -> pd.DataFrame:
    """
    Build per-product cost table with:
      - LMP nodal median
      - LMP socialised
      - AMM series columns for all 4 variants
      - Backward compatible AMM_perHH_yr = PRIMARY AMM series (Base Individual)
    """
    rows = []
    for p in PRODS:
        nodal_vals = lmp_nodes.loc[lmp_nodes["product"] == p, "LMP_nodal_perHH_yr"].values
        lmp_nodal_median = float(np.nanmedian(nodal_vals)) if nodal_vals.size else np.nan

        lmp_socialised = (
            float(lmp_soc.loc[lmp_soc["product"] == p, "LMP_socialised_perHH_yr"].values[0])
            if (lmp_soc["product"] == p).any()
            else np.nan
        )

        row = {
            "product": p,
            "LMP_nodal_median_perHH_yr": lmp_nodal_median,
            "LMP_socialised_perHH_yr": lmp_socialised,
        }

        # Add each AMM series as its own column
        for label, _key in AMM_SERIES_ORDER:
            s = amm_subs[label]
            v = float(s.loc[s["product"] == p, "AMM_perHH_yr"].values[0]) if (s["product"] == p).any() else np.nan
            row[f"AMM_{label}_perHH_yr"] = v

        # Backward compatible single-AMM column
        row["AMM_perHH_yr"] = row[f"AMM_{PRIMARY_AMM_LABEL}_perHH_yr"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_per_node_comparison(lmp_nodes: pd.DataFrame, lmp_soc: pd.DataFrame, amm_primary: pd.DataFrame) -> pd.DataFrame:
    """
    Per-node comparisons remain based on the PRIMARY AMM series (Base Individual),
    so geo deltas and existing outputs are unchanged in meaning.
    """
    soc = lmp_soc[["product", "LMP_socialised_perHH_yr"]]
    amm = amm_primary[["product", "AMM_perHH_yr"]]
    base = lmp_nodes[["load_id", "product", "LMP_nodal_perHH_yr"]].copy()
    out = base.merge(soc, on="product", how="left").merge(amm, on="product", how="left")
    out["Δ_nodal_minus_AMM"] = out["LMP_nodal_perHH_yr"] - out["AMM_perHH_yr"]
    out["Δ_socialised_minus_AMM"] = out["LMP_socialised_perHH_yr"] - out["AMM_perHH_yr"]
    out["ratio_nodal_over_AMM"] = np.where(out["AMM_perHH_yr"] > 0, out["LMP_nodal_perHH_yr"] / out["AMM_perHH_yr"], np.nan)
    out["ratio_socialised_over_AMM"] = np.where(out["AMM_perHH_yr"] > 0, out["LMP_socialised_perHH_yr"] / out["AMM_perHH_yr"], np.nan)

    # AMM nodal "volatility" placeholder (0 because AMM is flat per product across nodes here)
    out["AMM_nodal_volatility_£perHHyr"] = 0.0
    return out


def summarise_geo(per_node: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PRODS:
        d = per_node.loc[per_node["product"] == p, "Δ_nodal_minus_AMM"].dropna().values
        if d.size == 0:
            continue
        pct = lambda q: float(np.nanpercentile(d, q))
        rows.append(
            {
                "product": p,
                "nodes_count": int((per_node["product"] == p).sum()),
                "mean_delta": float(np.nanmean(d)),
                "median_delta": float(np.nanmedian(d)),
                "std_delta": float(np.nanstd(d)),
                "p05": pct(5),
                "p25": pct(25),
                "p75": pct(75),
                "p95": pct(95),
                "share_nodes_LMPnodal_gt_AMM": float(np.mean(d > 0.0)),
                "share_nodes_within_±100": float(np.mean(np.abs(d) <= 100.0)),
                "share_at_zero": float(np.mean(d <= 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_power_burden(uc_power: pd.DataFrame | None) -> pd.DataFrame:
    """Summarise OPTIONAL AMM U/C power timeseries (MW)."""
    if uc_power is None or uc_power.empty:
        return pd.DataFrame(columns=["product", "peak_c_MW", "mean_c_MW", "peak_u_MW", "mean_u_MW"])
    return uc_power.groupby("product", as_index=False).agg(
        peak_c_MW=("c_MW", "max"),
        mean_c_MW=("c_MW", "mean"),
        peak_u_MW=("u_MW", "max"),
        mean_u_MW=("u_MW", "mean"),
    )


def power_burden_perHH(power_df: pd.DataFrame) -> pd.DataFrame:
    """Convert MW burden to kW per household (OPTIONAL; needs power_df + HOUSEHOLDS)."""
    if power_df is None or power_df.empty:
        return pd.DataFrame(
            columns=["product", "mean_c_kW_per_HH", "peak_c_kW_per_HH", "mean_u_kW_per_HH", "peak_u_kW_per_HH"]
        )
    rows = []
    for p in PRODS:
        row = power_df[power_df["product"] == p]
        if row.empty:
            continue
        n = float(HOUSEHOLDS.get(p, np.nan))
        if not np.isfinite(n) or n <= 0:
            continue
        rows.append(
            {
                "product": p,
                "mean_c_kW_per_HH": float(row["mean_c_MW"].iloc[0]) * 1000.0 / n,
                "peak_c_kW_per_HH": float(row["peak_c_MW"].iloc[0]) * 1000.0 / n,
                "mean_u_kW_per_HH": float(row["mean_u_MW"].iloc[0]) * 1000.0 / n,
                "peak_u_kW_per_HH": float(row["peak_u_MW"].iloc[0]) * 1000.0 / n,
            }
        )
    return pd.DataFrame(rows)


# ---------- plotting ----------
def plot_costs_LMPsoc_vs_AMM(cost_levels: pd.DataFrame, out_dir: Path):
    """
    UPDATED: 5-series bar chart:
      LMP + 4 AMM series
    Keeps filename the same.
    """
    x = np.arange(len(PRODS))
    series_labels = ["LMP socialised"] + [lab for lab, _ in AMM_SERIES_ORDER]
    series_cols = ["LMP_socialised_perHH_yr"] + [f"AMM_{lab}_perHH_yr" for lab, _ in AMM_SERIES_ORDER]

    k = len(series_cols)
    group_w = 0.85
    bar_w = group_w / k
    offsets = (np.arange(k) - (k - 1) / 2.0) * bar_w

    plt.figure(figsize=(14, 6))
    for j, (lab, col) in enumerate(zip(series_labels, series_cols)):
        vals = [float(cost_levels.loc[cost_levels["product"] == p, col].values[0]) for p in PRODS]
        plt.bar(x + offsets[j], vals, width=bar_w, label=lab)

    plt.xticks(x, PRODS)
    plt.ylabel("£ / HH / year")
    plt.title("Per-household annual cost — LMP socialised vs AMM variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_costs_LMPsocialised_vs_AMM.pdf", dpi=150)
    plt.close()


def plot_totals_LMPsoc_vs_AMM(cost_levels: pd.DataFrame, out_dir: Path):
    """
    UPDATED: 5-series totals bar chart:
      LMP + 4 AMM series
    Keeps filename the same.
    """
    # Build totals table internally
    rows = []
    for p in PRODS:
        n = float(HOUSEHOLDS.get(p, np.nan))
        row = {"product": p}
        row["LMP_socialised_total"] = float(cost_levels.loc[cost_levels["product"] == p, "LMP_socialised_perHH_yr"].values[0]) * n
        for lab, _ in AMM_SERIES_ORDER:
            perhh = float(cost_levels.loc[cost_levels["product"] == p, f"AMM_{lab}_perHH_yr"].values[0])
            row[f"AMM_{lab}_total"] = perhh * n
        rows.append(row)
    df = pd.DataFrame(rows)

    x = np.arange(len(PRODS))
    series_labels = ["LMP socialised"] + [lab for lab, _ in AMM_SERIES_ORDER]
    series_cols = ["LMP_socialised_total"] + [f"AMM_{lab}_total" for lab, _ in AMM_SERIES_ORDER]

    k = len(series_cols)
    group_w = 0.85
    bar_w = group_w / k
    offsets = (np.arange(k) - (k - 1) / 2.0) * bar_w

    plt.figure(figsize=(14, 6))
    for j, (lab, col) in enumerate(zip(series_labels, series_cols)):
        plt.bar(x + offsets[j], df[col].values, width=bar_w, label=lab)

    plt.xticks(x, PRODS)
    plt.ylabel("£ / year (TOTAL by product)")
    plt.title("TOTAL annual cost — LMP socialised vs AMM variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_totals_LMPsocialised_vs_AMM.pdf", dpi=150)
    plt.close()


def plot_geo_boxplots(per_node: pd.DataFrame, out_dir: Path):
    data = []
    amm = []
    for p in PRODS:
        sub = per_node[per_node["product"] == p]
        data.append(sub["LMP_nodal_perHH_yr"].values)
        amm.append(float(sub["AMM_perHH_yr"].iloc[0]) if not sub.empty else np.nan)

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=PRODS, showfliers=True)
    for i, y in enumerate(amm, 1):
        if np.isfinite(y):
            plt.plot([i - 0.3, i + 0.3], [y, y], linewidth=2.5, label="_nolegend_")
    plt.plot([], [], color="C1", linewidth=2.5, label="AMM per-product flat (primary)")
    plt.ylabel("£ / HH / year")
    plt.title("Geographic spread: LMP nodal per product (AMM primary overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_geo_boxplot_LMPnodal_vs_AMM.pdf", dpi=150)
    plt.close()


def plot_geo_cdf_deltas(per_node: pd.DataFrame, out_dir: Path, geo_summary: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    for p in PRODS:
        d = per_node[per_node["product"] == p]["Δ_nodal_minus_AMM"].dropna().values
        if d.size == 0:
            continue
        x = np.sort(d)
        y = np.linspace(0, 1, len(x), endpoint=True)
        plt.plot(x, y, label=p)
        row = geo_summary[geo_summary["product"] == p]
        if not row.empty:
            med = float(row["median_delta"].iloc[0])
            sh0 = float(row["share_at_zero"].iloc[0])
            plt.scatter([med], [0.5], s=35)
            plt.scatter([0], [sh0], s=35)

    plt.axvline(0.0, color="k", linewidth=1)
    plt.xlabel("Δ (LMP nodal − AMM primary)  [£ / HH / year]")
    plt.ylabel("Cumulative fraction of nodes")
    plt.title("CDF of node-level deltas: LMP nodal vs AMM primary (markers: median, share ≤ 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_geo_cdf_delta_LMPnodal_minus_AMM.pdf", dpi=150)
    plt.close()


def plot_scatter(x, y, labels, xlabel, ylabel, title, out_path: Path):
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y)
    for xi, yi, lab in zip(x, y, labels):
        if isfinite(xi) and isfinite(yi):
            plt.annotate(str(lab), (xi, yi), textcoords="offset points", xytext=(6, 4))
    if len(x) >= 2 and all(map(isfinite, x)) and all(map(isfinite, y)):
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 100)
        ys = coef[0] * xs + coef[1]
        plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_power_burden(power_df: pd.DataFrame, out_dir: Path):
    """Bar chart of mean/peak controllable MW by product (OPTIONAL)."""
    if power_df is None or power_df.empty:
        return
    df = power_df.set_index("product").reindex(PRODS)
    x = np.arange(len(PRODS))
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - w / 2, df["mean_c_MW"], width=w, label="Mean controllable MW")
    plt.bar(x + w / 2, df["peak_c_MW"], width=w, label="Peak controllable MW")
    plt.xticks(x, PRODS)
    plt.ylabel("MW")
    plt.title("AMM controllable power burden by product")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_amm_power_burden.pdf", dpi=150)
    plt.close()


def plot_power_burden_perHH(perHH_df: pd.DataFrame, out_dir: Path):
    """Bar chart of mean/peak controllable kW per HH by product (OPTIONAL)."""
    if perHH_df is None or perHH_df.empty:
        return
    df = perHH_df.set_index("product").reindex(PRODS)
    x = np.arange(len(PRODS))
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - w / 2, df["mean_c_kW_per_HH"], width=w, label="Mean controllable kW / HH")
    plt.bar(x + w / 2, df["peak_c_kW_per_HH"], width=w, label="Peak controllable kW / HH")
    plt.xticks(x, PRODS)
    plt.ylabel("kW / household")
    plt.title("AMM controllable power burden — per household")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_amm_power_burden_perHH.pdf", dpi=150)
    plt.close()


# ----- AMM nodal volatility & effective price plots -----
def plot_amm_nodal_volatility(per_node: pd.DataFrame, out_dir: Path):
    if "AMM_nodal_volatility_£perHHyr" not in per_node.columns:
        return

    plt.figure(figsize=(12, 6))
    base_x = np.arange(len(PRODS))
    for i, p in enumerate(PRODS):
        sub = per_node[per_node["product"] == p]
        if sub.empty:
            continue
        vol = sub["AMM_nodal_volatility_£perHHyr"].values
        jitter = (np.random.rand(len(vol)) - 0.5) * 0.15
        xs = base_x[i] + jitter
        plt.scatter(xs, vol, alpha=0.6, label=p if i == 0 else "_nolegend_")

    plt.xticks(base_x, PRODS)
    plt.ylabel("AMM nodal volatility (std dev, £ / HH / year)")
    plt.title("AMM node-by-node price volatility (flat per product in this experiment)")
    plt.axhline(0.0, color="k", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_amm_nodal_volatility.pdf", dpi=150)
    plt.close()


def plot_amm_effective_price_per_product(cost_levels: pd.DataFrame, out_dir: Path):
    """
    Keep filename the same. Uses PRIMARY AMM to match previous meaning.
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(PRODS))
    amm_vals = [
        float(cost_levels.loc[cost_levels["product"] == p, "AMM_perHH_yr"].values[0])
        if (cost_levels["product"] == p).any()
        else np.nan
        for p in PRODS
    ]
    plt.bar(x, amm_vals, width=0.5)
    plt.xticks(x, PRODS)
    plt.ylabel("£ / HH / year")
    plt.title("Effective AMM nodal price by product\n(primary AMM; flat across nodes in this experiment)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_amm_nodal_effective_price_per_product.pdf", dpi=150)
    plt.close()


# ---------- runner ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load LMP
    lmp_nodes, lmp_soc = read_lmp(LMP_DIR)

    # 2) Load AMM UC energy/power once + PRIMARY subscription via existing reader (back-compat)
    #    NOTE: AMM_VARIANT is ignored for multi-series labels; primary is fixed by PRIMARY_AMM_KEY.
    amm_primary, uc_energy, uc_power = read_amm(AMM_DIR, PRIMARY_AMM_KEY)

    # 3) Load all four AMM subscriptions (for 5-series bar charts + extra columns)
    amm_subs = read_amm_all_subscriptions(AMM_DIR)

    # 4) Availability (not used directly in absolute-energy correlations)
    availability = read_availability(GEN_EXPOST) or availability_from_amm_energy(uc_energy)

    # 5) Absolute controllable energy per product
    energy_abs = behaviour_absolute_energy(uc_energy)  # C_MWh, C_kWh_total, C_kWh_per_HH

    # 6) Core comparisons (cost_levels now includes all four AMM columns)
    cost_levels = build_cost_levels_multi(lmp_nodes, lmp_soc, amm_subs)

    # Per-node comparisons remain tied to PRIMARY AMM (same output meaning)
    per_node    = build_per_node_comparison(lmp_nodes, lmp_soc, amm_primary)
    geo_summary = summarise_geo(per_node)

    # 7) Power burden (OPTIONAL if power timeseries exists)
    power_df    = build_power_burden(uc_power)
    perHH_power = power_burden_perHH(power_df)

    # 8) Per-product summary (include absolute controllable energy)
    #    Keep existing column names, using PRIMARY AMM for AMM_perHH_yr and deltas.
    rows = []
    for p in PRODS:
        cl = cost_levels[cost_levels["product"] == p]
        ea = energy_abs[energy_abs["product"] == p]
        n  = float(HOUSEHOLDS.get(p, np.nan))

        LMP_soc = float(cl["LMP_socialised_perHH_yr"].values[0]) if not cl.empty else np.nan
        AMM     = float(cl["AMM_perHH_yr"].values[0]) if not cl.empty else np.nan  # PRIMARY

        C_MWh = float(ea["C_MWh"].values[0]) if not ea.empty else np.nan
        C_kWh_total = float(ea["C_kWh_total"].values[0]) if not ea.empty else np.nan
        C_kWh_per_HH = float(ea["C_kWh_per_HH"].values[0]) if not ea.empty else np.nan

        row = {
            "product": p,
            "LMP_socialised_perHH_yr": LMP_soc,
            "AMM_perHH_yr": AMM,  # PRIMARY
            "Δ_AMM_minus_LMPsocialised_perHH": AMM - LMP_soc,
            "LMP_socialised_TOTAL_gbp": LMP_soc * n if np.isfinite(n) else np.nan,
            "AMM_TOTAL_gbp": AMM * n if np.isfinite(n) else np.nan,
            "Δ_AMM_minus_LMPsocialised_TOTAL_gbp": (AMM - LMP_soc) * n if np.isfinite(n) else np.nan,
            "C_MWh_total": C_MWh,
            "C_kWh_total": C_kWh_total,
            "C_kWh_per_HH": C_kWh_per_HH,
        }

        # Also add the four AMM series perHH (so comparisons_per_product.csv carries them too)
        for lab, _ in AMM_SERIES_ORDER:
            row[f"AMM_{lab}_perHH_yr"] = float(cl[f"AMM_{lab}_perHH_yr"].values[0]) if not cl.empty else np.nan
            row[f"AMM_{lab}_TOTAL_gbp"] = row[f"AMM_{lab}_perHH_yr"] * n if np.isfinite(n) else np.nan

        rows.append(row)

    comparisons = pd.DataFrame(rows)

    # 9) Correlations (approach-specific and Δ) — keep as before (PRIMARY AMM)
    r_AMM_perHH = pearson_r(comparisons["C_kWh_per_HH"], comparisons["AMM_perHH_yr"])
    r_LMP_perHH = pearson_r(comparisons["C_kWh_per_HH"], comparisons["LMP_socialised_perHH_yr"])
    r_AMM_total = pearson_r(comparisons["C_MWh_total"], comparisons["AMM_TOTAL_gbp"])
    r_LMP_total = pearson_r(comparisons["C_MWh_total"], comparisons["LMP_socialised_TOTAL_gbp"])
    r_Δ_perHH   = pearson_r(comparisons["C_kWh_per_HH"], comparisons["Δ_AMM_minus_LMPsocialised_perHH"])
    r_Δ_total   = pearson_r(comparisons["C_MWh_total"], comparisons["Δ_AMM_minus_LMPsocialised_TOTAL_gbp"])

    def slope_and_intercept(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if len(x) < 2:
            return np.nan, np.nan
        a, b = np.polyfit(x, y, 1)
        return float(a), float(b)

    s_AMM_perHH, b_AMM_perHH = slope_and_intercept(comparisons["C_kWh_per_HH"], comparisons["AMM_perHH_yr"])
    s_LMP_perHH, b_LMP_perHH = slope_and_intercept(comparisons["C_kWh_per_HH"], comparisons["LMP_socialised_perHH_yr"])
    s_AMM_total, b_AMM_total  = slope_and_intercept(comparisons["C_MWh_total"], comparisons["AMM_TOTAL_gbp"])
    s_LMP_total, b_LMP_total  = slope_and_intercept(comparisons["C_MWh_total"], comparisons["LMP_socialised_TOTAL_gbp"])

    # 10) Save tables (filenames unchanged)
    cost_levels.to_csv(OUT_DIR / "per_product_cost_levels.csv", index=False)
    per_node.to_csv(OUT_DIR / "per_node_comparison.csv", index=False)
    geo_summary.to_csv(OUT_DIR / "geo_summary_by_product.csv", index=False)
    energy_abs.to_csv(OUT_DIR / "energy_burden_absolute.csv", index=False)
    if power_df is not None and not power_df.empty:
        power_df.to_csv(OUT_DIR / "amm_power_burden.csv", index=False)
    if perHH_power is not None and not perHH_power.empty:
        perHH_power.to_csv(OUT_DIR / "amm_power_burden_perHH.csv", index=False)
    comparisons.to_csv(OUT_DIR / "comparisons_per_product.csv", index=False)

    # Approach-specific summary CSVs (unchanged meaning: PRIMARY AMM vs LMP socialised)
    pd.DataFrame(
        {
            "approach": ["AMM_primary", "LMP_socialised"],
            "corr_r_with_C_kWh_per_HH": [r_AMM_perHH, r_LMP_perHH],
            "slope_£perHH_per_(kWh_perHH)": [s_AMM_perHH, s_LMP_perHH],
            "intercept_£perHH": [b_AMM_perHH, b_LMP_perHH],
        }
    ).to_csv(OUT_DIR / "approach_vs_controllable_kWh.csv", index=False)

    pd.DataFrame(
        {
            "approach": ["AMM_primary", "LMP_socialised"],
            "corr_r_with_C_MWh_total": [r_AMM_total, r_LMP_total],
            "slope_£per_MWh_total": [s_AMM_total, s_LMP_total],
            "intercept_£total": [b_AMM_total, b_LMP_total],
        }
    ).to_csv(OUT_DIR / "approach_totals_vs_controllable_MWh.csv", index=False)

    pd.DataFrame(
        {
            "metric": ["Pearson(C_kWh_per_HH, Δ_perHH)", "Pearson(C_MWh_total, Δ_total)"],
            "r": [r_Δ_perHH, r_Δ_total],
        }
    ).to_csv(OUT_DIR / "burden_vs_cost_delta.csv", index=False)

    pd.DataFrame([availability]).to_csv(OUT_DIR / "availability_summary.csv", index=False)

    # 11) Plots (filenames unchanged)
    plot_costs_LMPsoc_vs_AMM(cost_levels, OUT_DIR)   # now 5-series
    plot_totals_LMPsoc_vs_AMM(cost_levels, OUT_DIR)  # now 5-series
    plot_geo_boxplots(per_node, OUT_DIR)
    plot_geo_cdf_deltas(per_node, OUT_DIR, geo_summary)

    # Scatter plots (unchanged filenames; uses PRIMARY AMM)
    plot_scatter(
        x=comparisons["C_kWh_per_HH"].tolist(),
        y=comparisons["AMM_perHH_yr"].tolist(),
        labels=comparisons["product"].tolist(),
        xlabel="Controllable energy per HH (kWh / year)",
        ylabel="AMM (PRIMARY) cost ( £ / HH / year )",
        title=f"AMM PRIMARY: cost vs controllable kWh/HH  r={r_AMM_perHH:.3f}",
        out_path=OUT_DIR / "fig_scatter_AMM_cost_vs_CkWhperHH.pdf",
    )
    plot_scatter(
        x=comparisons["C_kWh_per_HH"].tolist(),
        y=comparisons["LMP_socialised_perHH_yr"].tolist(),
        labels=comparisons["product"].tolist(),
        xlabel="Controllable energy per HH (kWh / year)",
        ylabel="LMP socialised cost ( £ / HH / year )",
        title=f"LMP socialised: cost vs controllable kWh/HH  r={r_LMP_perHH:.3f}",
        out_path=OUT_DIR / "fig_scatter_LMPsoc_cost_vs_CkWhperHH.pdf",
    )
    plot_scatter(
        x=comparisons["C_MWh_total"].tolist(),
        y=comparisons["AMM_TOTAL_gbp"].tolist(),
        labels=comparisons["product"].tolist(),
        xlabel="Controllable energy (MWh / year) — TOTAL per product",
        ylabel="AMM PRIMARY TOTAL ( £ / year )",
        title=f"AMM PRIMARY TOTAL vs controllable MWh  r={r_AMM_total:.3f}",
        out_path=OUT_DIR / "fig_scatter_AMM_total_vs_CMWh.pdf",
    )
    plot_scatter(
        x=comparisons["C_MWh_total"].tolist(),
        y=comparisons["LMP_socialised_TOTAL_gbp"].tolist(),
        labels=comparisons["product"].tolist(),
        xlabel="Controllable energy (MWh / year) — TOTAL per product",
        ylabel="LMP socialised TOTAL ( £ / year )",
        title=f"LMP socialised TOTAL vs controllable MWh  r={r_LMP_total:.3f}",
        out_path=OUT_DIR / "fig_scatter_LMPsoc_total_vs_CMWh.pdf",
    )

    # Optional Δ plots (unchanged; uses PRIMARY AMM)
    plot_scatter(
        x=comparisons["C_kWh_per_HH"].tolist(),
        y=comparisons["Δ_AMM_minus_LMPsocialised_perHH"].tolist(),
        labels=comparisons["product"].tolist(),
        xlabel="Controllable energy per HH (kWh / year)",
        ylabel="AMM PRIMARY − LMP socialised ( £ / HH / year )",
        title=f"Do higher controllable kWh/HH pay more under AMM PRIMARY than LMP socialised?  r={r_Δ_perHH:.3f}",
        out_path=OUT_DIR / "fig_scatter_CkWhperHH_vs_costDelta.pdf",
    )
    plot_scatter(
        x=comparisons["C_MWh_total"].tolist(),
        y=comparisons["Δ_AMM_minus_LMPsocialised_TOTAL_gbp"].tolist(),
        labels=comparisons["product"].tolist(),
        xlabel="Controllable energy (MWh / year) — TOTAL per product",
        ylabel="AMM PRIMARY − LMP socialised ( £ / year ) — TOTAL",
        title=f"Total: do controllable-heavy products pay more under AMM PRIMARY?  r={r_Δ_total:.3f}",
        out_path=OUT_DIR / "fig_scatter_CMWhTotal_vs_costDeltaTotal.pdf",
    )

    # OPTIONAL power burden figures
    plot_power_burden(power_df, OUT_DIR)
    plot_power_burden_perHH(perHH_power, OUT_DIR)

    # AMM nodal stability figures (unchanged filenames; primary meaning)
    plot_amm_nodal_volatility(per_node, OUT_DIR)
    plot_amm_effective_price_per_product(cost_levels, OUT_DIR)

    # 12) Interpretation notes (unchanged filename; primary meaning)
    md = f"""
# Interpretation — approach-specific vs absolute controllable energy (PRIMARY AMM)

We examine whether each approach **on its own** charges more when a product uses more **controllable energy**.

PRIMARY AMM series used throughout correlations/deltas/geo: **{PRIMARY_AMM_LABEL}**.

**Per household**
- AMM PRIMARY: r(C_kWh/HH, AMM_perHH) = {r_AMM_perHH:.3f}, slope = {s_AMM_perHH:.4f} £/HH per (kWh/HH)
- LMP socialised: r(C_kWh/HH, LMP_perHH) = {r_LMP_perHH:.3f}, slope = {s_LMP_perHH:.4f} £/HH per (kWh/HH)

**Totals (system scale)**
- AMM PRIMARY TOTAL vs C_MWh_total: r = {r_AMM_total:.3f}, slope = {s_AMM_total:.3f} £ per extra MWh controllable
- LMP socialised TOTAL vs C_MWh_total: r = {r_LMP_total:.3f}, slope = {s_LMP_total:.3f} £ per extra MWh controllable

Δ results (AMM PRIMARY − LMP socialised) are also saved for context.
(N≈4 products ⇒ treat correlations as indicative.)

NOTE: The bar charts compare **LMP socialised** against all **four AMM variants**
(Base/Delta × Individual/Aggregate), while the scatter/correlation sections remain tied
to PRIMARY AMM to keep filenames and semantics stable.
""".strip()
    (OUT_DIR / "INTERPRETATION.md").write_text(md, encoding="utf-8")

    print(f"[OK] Wrote outputs to: {OUT_DIR.resolve()}")
    print(f"Per-HH r (PRIMARY): AMM={r_AMM_perHH:.3f}, LMP={r_LMP_perHH:.3f} | TOTAL r (PRIMARY): AMM={r_AMM_total:.3f}, LMP={r_LMP_total:.3f}")
    print(f"Δ r (PRIMARY context): per-HH={r_Δ_perHH:.3f}, total={r_Δ_total:.3f}")


if __name__ == "__main__":
    main()
