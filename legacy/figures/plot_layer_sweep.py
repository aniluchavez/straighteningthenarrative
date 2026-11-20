"""
plot_layer_sweep.py
===================
Shows how cross-space decoding quality and curvature/displacement effects
vary across all LLM layers.

Reads per-layer pointwise CSVs produced by run_layer_sweep.sh (using
--force-layer N --match-pcs), one per (lag, layer).

2 rows (lag 0, lag +1) × 2 cols (LLM→neural, neural→LLM).
Each panel shows three layer-profile lines:
  - Gray:       mean decoding r across patients
  - Orange:     partial r(curvature | displacement)
  - Periwinkle: partial r(displacement | curvature)
Vertical dashed line at LOO-optimal layer (5 for hippocampus).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_1samp, pearsonr
from sklearn.linear_model import LinearRegression

BASE   = os.path.dirname(os.path.abspath(__file__))
RDIR   = os.path.join(BASE, "results", "geometry_paper_curvature_alignment")
FDIR   = os.path.join(BASE, "figures", "appendix")
REGION = "hippocampus"
MODEL  = "llama-3.1-8b"
N_LAYERS = 33          # layers 0–32
LOO_LAYER = 5          # LOO-optimal layer for hippocampus

C_CURV  = "#D95319"
C_DISP  = "#5E7CE2"
C_DEC   = "#444444"
C_ZERO  = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

PANEL_LABEL_KW = dict(
    fontsize=28, fontweight="bold", fontfamily="sans-serif",
    va="top", ha="left",
)

DIRECTIONS = ["LLM_to_neural", "neural_to_LLM"]
DIR_LABELS = {"LLM_to_neural": "LLM → neural", "neural_to_LLM": "Neural → LLM"}
LAGS = [0, 1]


def stem(lag, layer):
    return (
        f"curvature_alignment_{MODEL}_hippocampus-ACC_all-patients"
        f"_pca_llmpc64_shufflecv_ridge_both_lagplus{lag}_matchedpcs_layer{layer}"
    )


def resid_ols(y, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    return y - LinearRegression().fit(x, y).predict(x)


def per_patient_metrics(pt, direction):
    """For one lag×direction slice, return per-patient:
      mean_r, partial_r_curv, partial_r_disp
    """
    sub = pt[(pt["region"] == REGION) & pt["valid"] & (pt["direction"] == direction)]
    mean_rs, rc_list, rd_list = [], [], []
    for _, g in sub.groupby("pid"):
        c = g["joint_curvature"].values.astype(float)
        d = g["joint_displacement"].values.astype(float)
        r = g["target_corr"].values.astype(float)
        m = np.isfinite(c) & np.isfinite(d) & np.isfinite(r)
        if m.sum() < 10:
            continue
        c, d, r = c[m], d[m], r[m]
        mean_rs.append(r.mean())
        rc_list.append(pearsonr(resid_ols(c, d), resid_ols(r, d))[0])
        rd_list.append(pearsonr(resid_ols(d, c), resid_ols(r, c))[0])
    return (
        np.array(mean_rs),
        np.array(rc_list),
        np.array(rd_list),
    )


def load_layer(lag, layer):
    path = os.path.join(RDIR, f"{stem(lag, layer)}_pointwise.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_layer_profiles():
    """Returns dict: (lag, direction) → arrays of shape (N_LAYERS,) for each metric."""
    profiles = {}
    for lag in LAGS:
        for direction in DIRECTIONS:
            key = (lag, direction)
            dec_mu  = np.full(N_LAYERS, np.nan)
            dec_se  = np.full(N_LAYERS, np.nan)
            rc_mu   = np.full(N_LAYERS, np.nan)
            rc_se   = np.full(N_LAYERS, np.nan)
            rd_mu   = np.full(N_LAYERS, np.nan)
            rd_se   = np.full(N_LAYERS, np.nan)
            for layer in range(N_LAYERS):
                pt = load_layer(lag, layer)
                if pt is None:
                    continue
                mean_rs, rcs, rds = per_patient_metrics(pt, direction)
                if len(mean_rs) == 0:
                    continue
                dec_mu[layer] = mean_rs.mean()
                dec_se[layer] = scipy_sem(mean_rs)
                rc_mu[layer]  = rcs.mean()
                rc_se[layer]  = scipy_sem(rcs)
                rd_mu[layer]  = rds.mean()
                rd_se[layer]  = scipy_sem(rds)
                print(f"  lag={lag} {direction} layer={layer}: "
                      f"r={dec_mu[layer]:.3f}  "
                      f"rc={rc_mu[layer]:+.3f}  rd={rd_mu[layer]:+.3f}", flush=True)
            profiles[key] = dict(
                dec_mu=dec_mu, dec_se=dec_se,
                rc_mu=rc_mu,   rc_se=rc_se,
                rd_mu=rd_mu,   rd_se=rd_se,
            )
    return profiles


def draw_panel(ax, prof, lag, direction):
    xs = np.arange(N_LAYERS)
    p  = prof[(lag, direction)]

    # ── right axis: decoding r ──
    ax2 = ax.twinx()
    valid = np.isfinite(p["dec_mu"])
    if valid.any():
        ax2.plot(xs[valid], p["dec_mu"][valid],
                 color=C_DEC, lw=2.2, marker=".", ms=4,
                 label="Mean decoding $r$", zorder=3)
        ax2.fill_between(xs[valid],
                         p["dec_mu"][valid] - p["dec_se"][valid],
                         p["dec_mu"][valid] + p["dec_se"][valid],
                         color=C_DEC, alpha=0.10, zorder=2)
    ax2.set_ylabel("Mean decoding $r$", color=C_DEC, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_DEC, labelsize=8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_DEC)
    ax2.spines["top"].set_visible(False)

    # ── left axis: partial r's ──
    valid_rc = np.isfinite(p["rc_mu"])
    if valid_rc.any():
        ax.plot(xs[valid_rc], p["rc_mu"][valid_rc],
                color=C_CURV, lw=2.0, marker="o", ms=4,
                label=r"Partial $r_\kappa$ (ctrl $d$)", zorder=4)
        ax.fill_between(xs[valid_rc],
                        p["rc_mu"][valid_rc] - p["rc_se"][valid_rc],
                        p["rc_mu"][valid_rc] + p["rc_se"][valid_rc],
                        color=C_CURV, alpha=0.15, zorder=3)

    valid_rd = np.isfinite(p["rd_mu"])
    if valid_rd.any():
        ax.plot(xs[valid_rd], p["rd_mu"][valid_rd],
                color=C_DISP, lw=2.0, marker="s", ms=4,
                label=r"Partial $r_d$ (ctrl $\kappa$)", zorder=4)
        ax.fill_between(xs[valid_rd],
                        p["rd_mu"][valid_rd] - p["rd_se"][valid_rd],
                        p["rd_mu"][valid_rd] + p["rd_se"][valid_rd],
                        color=C_DISP, alpha=0.15, zorder=3)

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.axvline(LOO_LAYER, color="#999999", lw=1.2, ls=":", alpha=0.8, zorder=5,
               label=f"LOO-opt. layer ({LOO_LAYER})")

    ax.set_xlabel("LLM layer")
    ax.set_ylabel("Partial correlation ($r$)", fontsize=9)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(np.arange(0, N_LAYERS, 4))

    return ax2


def main():
    os.makedirs(FDIR, exist_ok=True)

    print("Computing layer profiles...", flush=True)
    profiles = compute_layer_profiles()

    n_layers_found = sum(
        np.isfinite(profiles[(LAGS[0], DIRECTIONS[0])]["dec_mu"])
    )
    if n_layers_found == 0:
        print("\nNo layer-sweep data found.")
        print(f"Run:  bash run_layer_sweep.sh")
        print(f"Then re-run this script.")
        return

    print(f"\nFound data for {n_layers_found} layers. Building figure...", flush=True)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.09, right=0.98,
                           bottom=0.10, top=0.88)

    panel_labels = iter("ABCD")

    for row, lag in enumerate(LAGS):
        for col, direction in enumerate(DIRECTIONS):
            ax = fig.add_subplot(gs[row, col])
            ax2 = draw_panel(ax, profiles, lag, direction)

            label = next(panel_labels)
            lag_str = "current state (lag 0)" if lag == 0 else "next state (lag +1)"
            title = f"{DIR_LABELS[direction]}  ·  {lag_str}"
            ax.set_title(title, loc="left", pad=5)
            ax.text(-0.14, 1.08, label, transform=ax.transAxes, **PANEL_LABEL_KW)

            if row == 0 and col == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          loc="upper left", framealpha=0.9,
                          edgecolor="#CCCCCC", fontsize=8)

    stem_fig = "fig_layer_sweep"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FDIR, f"{stem_fig}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
