#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-axis burden→cost check for AMM vs LMP socialised.

Loads:
  comparison_out/comparisons_per_product.csv

Adds contract parameters (Pmax, monthly entitlement),
builds annual entitlement (kWh/HH/year),
fits multivariate linear models:

  cost_perHH = b0 + bE * E_ent_year + bP * Pmax

Outputs:
  comparison_out/two_axis_fit_summary.csv
  comparison_out/two_axis_contrasts.csv
  comparison_out/fig_3d_planes_cost_vs_contractspace.pdf
  comparison_out/fig_cost_surface_predictions.pdf

Notes:
- N=4 products → treat all statistics as descriptive / sanity checks.
- LOOCV SSE is included because it's meaningful even for tiny N.
- A tiny exact permutation test (24 permutations) is included for "does P add
  explanatory power beyond E?" (still descriptive, but avoids overclaiming).
- Household counts are OPTIONAL. If you want population weighting, set HOUSEHOLDS.
"""

from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------- config -----------------
IN_DIR = Path("./comparison_out")
IN_FP  = IN_DIR / "comparisons_per_product.csv"

OUT_SUMMARY   = IN_DIR / "two_axis_fit_summary.csv"
OUT_CONTRASTS = IN_DIR / "two_axis_contrasts.csv"
OUT_FIG_3D    = IN_DIR / "fig_3d_planes_cost_vs_contractspace.pdf"
OUT_FIG_SURF  = IN_DIR / "fig_cost_surface_predictions.pdf"

# Contract parameters from LaTeX table (per household)
CONTRACT = pd.DataFrame({
    "product": ["P1", "P2", "P3", "P4"],
    "Pmax_kW": [2.0, 10.0, 2.0, 10.0],
    "Eent_kWh_per_month": [250.0, 700.0, 500.0, 800.0],
})
CONTRACT["Eent_kWh_per_year"] = CONTRACT["Eent_kWh_per_month"] * 12.0

# Optional: household counts for weighting (leave None to disable)
HOUSEHOLDS = None
# Example if you want weighting:
# HOUSEHOLDS = {"P1": 19_000_000, "P2": 6_000_000, "P3": 2_500_000, "P4": 1_500_000}


# ----------------- helpers -----------------
def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)

def fit_ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """
    Weighted least squares if w is provided (w are weights, not std devs).
    Returns beta (flattened), yhat (flattened), R2 (weighted if w provided).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    if w is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return beta.flatten(), yhat.flatten(), r2

    w = np.asarray(w, dtype=float).reshape(-1)
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.inv(XtW @ X) @ (XtW @ y)
    yhat = X @ beta

    ybar = float((w * y.flatten()).sum() / w.sum())
    ss_res = float((w * (y.flatten() - yhat.flatten()) ** 2).sum())
    ss_tot = float((w * (y.flatten() - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta.flatten(), yhat.flatten(), r2

def loocv_sse(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> float:
    """
    Leave-one-out SSE for linear regression.
    For w != None, we do weighted LS on the training set but evaluate plain SSE
    on held-out point (still useful descriptively).
    """
    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    n = len(y)
    sse = 0.0

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xi = X[mask]
        yi = y[mask]

        wi = None
        if w is not None:
            wi = np.asarray(w, float).reshape(-1)[mask]

        beta, *_ = fit_ols(Xi, yi, w=wi)
        yhat_i = float(X[i] @ beta)
        sse += (y[i] - yhat_i) ** 2

    return float(sse)

def corr(x, y):
    x = _as_1d(x)
    y = _as_1d(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x**2).sum() * (y**2).sum())
    return float((x*y).sum() / den) if den > 0 else np.nan

def partial_corr(y, x1, x2):
    """
    Partial corr(y, x1 | x2) via residualisation.
    """
    y = _as_1d(y)
    x1 = _as_1d(x1)
    x2 = _as_1d(x2)

    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    y, x1, x2 = y[m], x1[m], x2[m]
    if len(y) < 3:
        return np.nan

    X2 = np.column_stack([np.ones(len(x2)), x2])

    by, *_ = np.linalg.lstsq(X2, y, rcond=None)
    ry = y - (X2 @ by)

    bx1, *_ = np.linalg.lstsq(X2, x1, rcond=None)
    rx1 = x1 - (X2 @ bx1)

    return corr(ry, rx1)

def incremental_r2(y, X_base, X_plus, w=None):
    """
    Descriptive incremental R2 from adding predictors.
    """
    _, _, r20 = fit_ols(X_base, y, w=w)
    _, _, r21 = fit_ols(X_plus, y, w=w)
    return r20, r21, (r21 - r20)

def perm_test_inc_r2(y, X_E, X_EP, w=None):
    """
    Exact permutation test (24 perms at N=4) for:
      H0: adding P doesn't improve fit beyond chance alignment.
    Statistic: ΔR2 = R2(EP) - R2(E-only)

    Returns:
      obs_delta, p_ge (>=), p_gt (>), n_perms
    """
    y = _as_1d(y)
    obs_r2E, obs_r2EP, obs_delta = incremental_r2(y, X_E, X_EP, w=w)

    deltas = []
    for perm in itertools.permutations(y.tolist(), len(y)):
        yp = np.array(perm, dtype=float)
        _, _, d = incremental_r2(yp, X_E, X_EP, w=w)
        deltas.append(d)

    deltas = np.array(deltas, dtype=float)
    p_ge = float(np.mean(deltas >= obs_delta))
    p_gt = float(np.mean(deltas >  obs_delta))
    return float(obs_delta), p_ge, p_gt, int(len(deltas))


# ----------------- main -----------------
def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_FP.exists():
        raise FileNotFoundError(f"Missing input: {IN_FP}")

    df = pd.read_csv(IN_FP)

    if "product" not in df.columns:
        raise ValueError("comparisons_per_product.csv must include a 'product' column.")

    # Allow minor column-name variants
    col_amm = "AMM_perHH_yr"
    col_lmp = "LMP_socialised_perHH_yr"
    if col_amm not in df.columns:
        raise ValueError(f"Missing column '{col_amm}' in {IN_FP}")
    if col_lmp not in df.columns:
        raise ValueError(f"Missing column '{col_lmp}' in {IN_FP}")

    # Merge contract parameters
    d = df.merge(CONTRACT, on="product", how="left")

    if d[["Pmax_kW", "Eent_kWh_per_year"]].isna().any().any():
        missing = d.loc[d["Pmax_kW"].isna() | d["Eent_kWh_per_year"].isna(), "product"].tolist()
        raise ValueError(f"Missing contract params for products: {missing}")

    # Enforce product order (P1..P4) for contrasts/plots
    order = ["P1", "P2", "P3", "P4"]
    d["__order"] = d["product"].apply(lambda x: order.index(x) if x in order else 999)
    d = d.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # Dependent variables (per-HH/year)
    y_amm = d[col_amm].astype(float).values
    y_lmp = d[col_lmp].astype(float).values

    # Explanatory variables
    E = d["Eent_kWh_per_year"].astype(float).values
    P = d["Pmax_kW"].astype(float).values

    # Optional weights
    w = None
    if HOUSEHOLDS is not None:
        w = np.array([float(HOUSEHOLDS.get(p, np.nan)) for p in d["product"]], dtype=float)
        if np.any(~np.isfinite(w)):
            raise ValueError("HOUSEHOLDS dict is missing some products.")
        w = w / np.mean(w)

    # Design matrices
    X_EP = np.column_stack([np.ones(len(d)), E, P])
    X_E  = np.column_stack([np.ones(len(d)), E])
    X_P  = np.column_stack([np.ones(len(d)), P])

    # -------- Fits (in-sample) --------
    beta_amm, yhat_amm, r2_amm = fit_ols(X_EP, y_amm, w=w)
    beta_lmp, yhat_lmp, r2_lmp = fit_ols(X_EP, y_lmp, w=w)

    # Incremental R2 diagnostics (descriptive)
    r2E_amm, _, incP_amm = incremental_r2(y_amm, X_E, X_EP, w=w)
    r2P_amm, _, incE_amm = incremental_r2(y_amm, X_P, X_EP, w=w)

    r2E_lmp, _, incP_lmp = incremental_r2(y_lmp, X_E, X_EP, w=w)
    r2P_lmp, _, incE_lmp = incremental_r2(y_lmp, X_P, X_EP, w=w)

    # Partial correlations (descriptive)
    pcorr_amm_P_given_E = partial_corr(y_amm, P, E)
    pcorr_amm_E_given_P = partial_corr(y_amm, E, P)
    pcorr_lmp_P_given_E = partial_corr(y_lmp, P, E)
    pcorr_lmp_E_given_P = partial_corr(y_lmp, E, P)

    # -------- LOOCV (more meaningful at N=4) --------
    loocv_amm_E  = loocv_sse(y_amm, X_E,  w=w)
    loocv_amm_P  = loocv_sse(y_amm, X_P,  w=w)
    loocv_amm_EP = loocv_sse(y_amm, X_EP, w=w)

    loocv_lmp_E  = loocv_sse(y_lmp, X_E,  w=w)
    loocv_lmp_P  = loocv_sse(y_lmp, X_P,  w=w)
    loocv_lmp_EP = loocv_sse(y_lmp, X_EP, w=w)

    # -------- Tiny exact permutation tests (24 perms) --------
    # For "does P add explanatory value beyond E?"
    obs_delta_amm, p_ge_amm, p_gt_amm, nperm = perm_test_inc_r2(y_amm, X_E, X_EP, w=w)
    obs_delta_lmp, p_ge_lmp, p_gt_lmp, _     = perm_test_inc_r2(y_lmp, X_E, X_EP, w=w)

    # Save summary table (one row per approach)
    summary = pd.DataFrame([
        {
            "approach": "AMM",
            "b0": beta_amm[0],
            "bE_per_kWhEntYear": beta_amm[1],
            "bP_per_kWcap": beta_amm[2],
            "R2_EP": r2_amm,
            "R2_E_only": r2E_amm,
            "R2_P_only": r2P_amm,
            "Inc_R2_add_P_given_E": incP_amm,
            "Inc_R2_add_E_given_P": incE_amm,
            "partial_corr_P_given_E": pcorr_amm_P_given_E,
            "partial_corr_E_given_P": pcorr_amm_E_given_P,
            "LOOCV_SSE_E_only": loocv_amm_E,
            "LOOCV_SSE_P_only": loocv_amm_P,
            "LOOCV_SSE_EP": loocv_amm_EP,
            "LOOCV_Delta_SSE_EP_minus_E": (loocv_amm_EP - loocv_amm_E),
            "perm_test_obs_deltaR2_EP_minus_E": obs_delta_amm,
            "perm_test_p_ge": p_ge_amm,
            "perm_test_p_gt": p_gt_amm,
            "perm_test_n_perms": nperm,
        },
        {
            "approach": "LMP_socialised",
            "b0": beta_lmp[0],
            "bE_per_kWhEntYear": beta_lmp[1],
            "bP_per_kWcap": beta_lmp[2],
            "R2_EP": r2_lmp,
            "R2_E_only": r2E_lmp,
            "R2_P_only": r2P_lmp,
            "Inc_R2_add_P_given_E": incP_lmp,
            "Inc_R2_add_E_given_P": incE_lmp,
            "partial_corr_P_given_E": pcorr_lmp_P_given_E,
            "partial_corr_E_given_P": pcorr_lmp_E_given_P,
            "LOOCV_SSE_E_only": loocv_lmp_E,
            "LOOCV_SSE_P_only": loocv_lmp_P,
            "LOOCV_SSE_EP": loocv_lmp_EP,
            "LOOCV_Delta_SSE_EP_minus_E": (loocv_lmp_EP - loocv_lmp_E),
            "perm_test_obs_deltaR2_EP_minus_E": obs_delta_lmp,
            "perm_test_p_ge": p_ge_lmp,
            "perm_test_p_gt": p_gt_lmp,
            "perm_test_n_perms": nperm,
        },
    ])
    summary.to_csv(OUT_SUMMARY, index=False)

    # -------- Contrasts table (simple deltas + DiD) --------
    def _get_y_by_prod(yvals):
        return {p: float(yvals[i]) for i, p in enumerate(d["product"].tolist())}

    yA = _get_y_by_prod(y_amm)
    yL = _get_y_by_prod(y_lmp)

    # Define deltas used for DID diagnostics
    # (P2 - P1) : switch to high Pmax for "lower tier" pairing
    # (P4 - P3) : switch to high Pmax for "higher tier" pairing
    # DID power effect: (P4-P3) - (P2-P1)
    # DID entitlement effect: (P4-P2) - (P3-P1)  (algebraically identical here)
    def contrasts(y):
        d21 = y["P2"] - y["P1"]
        d43 = y["P4"] - y["P3"]
        did_power = d43 - d21
        did_ent   = (y["P4"] - y["P2"]) - (y["P3"] - y["P1"])
        return d21, d43, did_power, did_ent

    amm_d21, amm_d43, amm_didp, amm_dide = contrasts(yA)
    lmp_d21, lmp_d43, lmp_didp, lmp_dide = contrasts(yL)

    contrasts_df = pd.DataFrame([
        {"approach": "AMM",
         "delta_P2_minus_P1": amm_d21,
         "delta_P4_minus_P3": amm_d43,
         "did_power_effect_highE_minus_lowE": amm_didp,
         "did_entitlement_effect_highP_minus_lowP": amm_dide},
        {"approach": "LMP_socialised",
         "delta_P2_minus_P1": lmp_d21,
         "delta_P4_minus_P3": lmp_d43,
         "did_power_effect_highE_minus_lowE": lmp_didp,
         "did_entitlement_effect_highP_minus_lowP": lmp_dide},
    ])
    contrasts_df.to_csv(OUT_CONTRASTS, index=False)

    # ---------- Figure 1: 3D scatter + fitted planes ----------
    with PdfPages(OUT_FIG_3D) as pdf:
        # AMM
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(E, P, y_amm)
        for i, p in enumerate(d["product"].tolist()):
            ax.text(E[i], P[i], y_amm[i], p)

        Egrid = np.linspace(E.min(), E.max(), 10)
        Pgrid = np.linspace(P.min(), P.max(), 10)
        Eg, Pg = np.meshgrid(Egrid, Pgrid)
        Zg = beta_amm[0] + beta_amm[1]*Eg + beta_amm[2]*Pg
        ax.plot_surface(Eg, Pg, Zg, alpha=0.25)

        ax.set_xlabel("Entitlement magnitude E_ent (kWh/HH/year)")
        ax.set_ylabel("Impact cap Pmax (kW)")
        ax.set_zlabel("Cost (£/HH/year)")
        ax.set_title("AMM: cost vs contract space (E_ent, Pmax)")
        pdf.savefig(fig)
        plt.close(fig)

        # LMP
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(E, P, y_lmp)
        for i, p in enumerate(d["product"].tolist()):
            ax.text(E[i], P[i], y_lmp[i], p)

        Zg = beta_lmp[0] + beta_lmp[1]*Eg + beta_lmp[2]*Pg
        ax.plot_surface(Eg, Pg, Zg, alpha=0.25)

        ax.set_xlabel("Entitlement magnitude E_ent (kWh/HH/year)")
        ax.set_ylabel("Impact cap Pmax (kW)")
        ax.set_zlabel("Cost (£/HH/year)")
        ax.set_title("LMP socialised: cost vs contract space (E_ent, Pmax)")
        pdf.savefig(fig)
        plt.close(fig)

    # ---------- Figure 2: predicted cost surfaces (2D heatmaps) ----------
    with PdfPages(OUT_FIG_SURF) as pdf:
        Egrid = np.linspace(E.min(), E.max(), 60)
        Pgrid = np.linspace(P.min(), P.max(), 60)
        Eg, Pg = np.meshgrid(Egrid, Pgrid)

        for title, beta in [("AMM", beta_amm), ("LMP socialised", beta_lmp)]:
            Z = beta[0] + beta[1]*Eg + beta[2]*Pg

            fig = plt.figure(figsize=(8, 6))
            plt.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[Egrid.min(), Egrid.max(), Pgrid.min(), Pgrid.max()],
            )
            plt.colorbar(label="Predicted cost (£/HH/year)")
            plt.scatter(E, P)
            for i, p in enumerate(d["product"].tolist()):
                plt.annotate(p, (E[i], P[i]), textcoords="offset points", xytext=(6, 4))

            plt.xlabel("Entitlement magnitude E_ent (kWh/HH/year)")
            plt.ylabel("Impact cap Pmax (kW)")
            plt.title(f"{title}: predicted cost over contract space (linear fit)")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("[OK] Wrote:")
    print(f" - {OUT_SUMMARY}")
    print(f" - {OUT_CONTRASTS}")
    print(f" - {OUT_FIG_3D}")
    print(f" - {OUT_FIG_SURF}")

if __name__ == "__main__":
    main()
