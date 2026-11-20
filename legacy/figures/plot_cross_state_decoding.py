"""
plot_cross_state_decoding.py
============================
3-panel figure for cross-space decoding, both lags overlaid.

Panel A: Mean decoding r (real vs null=0) for LLM→neural and neural→LLM,
         grouped bars for lag 0 and lag+1.

Panel B: Curvature quintile bins vs mean decoding r,
         4 lines (2 directions × 2 lags), solid=lag0 dashed=lag+1.

Panel C: Partial r(curvature|displacement) for both directions × lags,
         with SEM bars and significance markers.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_1samp, pearsonr, sem as scipy_sem
from sklearn.linear_model import LinearRegression

BASE  = os.path.dirname(os.path.abspath(__file__))
RDIR  = os.path.join(BASE, "results", "geometry_paper_curvature_alignment")
FDIR  = os.path.join(BASE, "figures")
REGION = "hippocampus"

C_LLM    = "#1B8A3C"   # forest green  — LLM→neural
C_NEURAL = "#D4297B"   # deep pink     — neural→LLM
C_NULL   = "#AAAAAA"
C_ZERO   = "#888888"

DIRECTIONS  = ["LLM_to_neural", "neural_to_LLM"]
DIR_LABELS  = {"LLM_to_neural": "LLM → Neural", "neural_to_LLM": "Neural → LLM"}
DIR_COLORS  = {"LLM_to_neural": C_LLM, "neural_to_LLM": C_NEURAL}
LAGS        = [0, 1]
LAG_LABELS  = {0: "lag 0 (current)", 1: "lag +1 (upcoming)"}
LAG_LS      = {0: "-", 1: "--"}
LAG_ALPHA   = {0: 1.0, 1: 0.65}
N_BINS      = 5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "svg.fonttype": "none",
})

PANEL_LABEL_KW = dict(
    fontsize=32, fontweight="bold",
    fontfamily="sans-serif",   # Helvetica-style
    va="top", ha="left",
    transform_rotates_text=False,
)


def resid(y, x):
    x2 = np.asarray(x, float).reshape(-1, 1)
    y2 = np.asarray(y, float)
    return y2 - LinearRegression().fit(x2, y2).predict(x2)


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def load_pointwise(lag):
    stem = (f"curvature_alignment_llama-3.1-8b_hippocampus-ACC_all-patients"
            f"_pca_llmpc64_shufflecv_ridge_both_lagplus{lag}_matchedpcs_pointwise.csv")
    return pd.read_csv(os.path.join(RDIR, stem))


def per_patient_mean_r(df, direction):
    sub = df[(df["region"] == REGION) & df["valid"] & (df["direction"] == direction)]
    return sub.groupby("pid")["target_corr"].mean()


def per_patient_partial_r(df, direction):
    sub = df[(df["region"] == REGION) & df["valid"] & (df["direction"] == direction)]
    rc_list, rd_list = [], []
    for _, g in sub.groupby("pid"):
        c = g["joint_curvature"].values.astype(float)
        d = g["joint_displacement"].values.astype(float)
        r = g["target_corr"].values.astype(float)
        m = np.isfinite(c) & np.isfinite(d) & np.isfinite(r)
        if m.sum() < 10:
            continue
        c, d, r = c[m], d[m], r[m]
        rc_list.append(pearsonr(resid(c, d), resid(r, d))[0])
        rd_list.append(pearsonr(resid(d, c), resid(r, c))[0])
    return np.array(rc_list), np.array(rd_list)


def per_patient_curv_bins(df, direction, n_bins=N_BINS):
    """Returns (bin_centers, mean_r_per_bin) averaged across patients."""
    sub = df[(df["region"] == REGION) & df["valid"] & (df["direction"] == direction)]
    all_bin_r = []
    for _, g in sub.groupby("pid"):
        c = g["joint_curvature"].values.astype(float)
        r = g["target_corr"].values.astype(float)
        m = np.isfinite(c) & np.isfinite(r)
        if m.sum() < n_bins * 3:
            continue
        c, r = c[m], r[m]
        bins = pd.qcut(c, n_bins, labels=False, duplicates="drop")
        bin_means = [r[bins == b].mean() for b in range(n_bins)]
        if len(bin_means) == n_bins:
            all_bin_r.append(bin_means)
    arr = np.array(all_bin_r)   # (n_patients, n_bins)
    return np.arange(1, n_bins + 1), arr.mean(0), scipy_sem(arr, axis=0)


# ── Panel A: mean decoding r ──────────────────────────────────────────────────

def draw_panel_a(ax, data):
    """data: dict (lag, direction) → (mean_r, sem_r, t, p)
    Groups by direction; within each group, lag 0 vs lag+1.
    """
    width    = 0.30
    gap      = 0.10
    group_gap = 0.85

    xtick_xs, xtick_labels = [], []

    for gi, direction in enumerate(DIRECTIONS):
        color = DIR_COLORS[direction]
        x0    = gi * group_gap
        xtick_xs.append(x0 + width + gap / 2)
        xtick_labels.append(DIR_LABELS[direction])

        for li, lag in enumerate(LAGS):
            mu, se, t, p = data[(lag, direction)]
            x = x0 + li * (width + gap)
            edgecolor = "white" if lag == 0 else color
            linewidth = 0.8 if lag == 0 else 1.3
            ax.bar(x, mu, width, color=color, alpha=LAG_ALPHA[lag],
                   edgecolor=edgecolor, linewidth=linewidth, zorder=3)
            ax.errorbar(x, mu, yerr=se, fmt="none",
                        color="#333333", capsize=3, lw=1.2, zorder=4)
            top = mu + se + 0.001
            ax.text(x + width / 2, top, stars(p), ha="center", va="bottom",
                    fontsize=10, color="#333333")

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.set_ylabel("Mean decoding $r$")
    ax.set_xticks(xtick_xs)
    ax.set_xticklabels(xtick_labels, fontsize=9)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=DIR_COLORS["LLM_to_neural"], label="LLM → Neural"),
        Patch(facecolor=DIR_COLORS["neural_to_LLM"],  label="Neural → LLM"),
        Patch(facecolor="#999999", edgecolor="white", label="lag 0 (current)"),
        Patch(facecolor="#999999", alpha=0.65, edgecolor="#666666",
              linewidth=1.3, label="lag +1 (upcoming)"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9,
              edgecolor="#CCCCCC", ncol=1, fontsize=8,
              bbox_to_anchor=(0.0, 0.58))
    ax.text(-0.22, 1.08, "A", transform=ax.transAxes, **PANEL_LABEL_KW)


# ── Panel B: curvature bins ───────────────────────────────────────────────────

def draw_panel_b(ax, bin_data):
    """bin_data: dict (lag, direction) → (xs, mus, sems)"""
    for lag in LAGS:
        for direction in DIRECTIONS:
            xs, mus, sems = bin_data[(lag, direction)]
            color = DIR_COLORS[direction]
            ls    = LAG_LS[lag]
            alpha = LAG_ALPHA[lag]
            ax.plot(xs, mus, color=color, ls=ls, lw=2.0,
                    marker="o", ms=4, alpha=alpha,
                    label=f"{DIR_LABELS[direction]}  {LAG_LABELS[lag]}")
            ax.fill_between(xs, mus - sems, mus + sems,
                            color=color, alpha=0.10 * alpha)

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.set_xlabel("Curvature quintile\n(1 = flattest, 5 = most curved)")
    ax.set_ylabel("Mean decoding $r$")
    ax.set_xticks(np.arange(1, N_BINS + 1))
    ax.legend(loc="lower left", framealpha=0.9,
              edgecolor="#CCCCCC", ncol=1, fontsize=8)
    ax.text(-0.22, 1.08, "B", transform=ax.transAxes, **PANEL_LABEL_KW)


# ── Panel C: paired patient plot — curvature vs displacement, lag+1 ───────────

def draw_panel_c(ax, partial_data):
    """Per-patient lines connecting curvature partial r to displacement partial r,
    for each direction side by side. Lag+1 only.
    Lines slanting left = curvature wins; lines slanting right = displacement wins.
    """
    lag       = 1
    x_curv    = 0.0
    x_disp    = 1.0
    group_gap = 2.6   # separation between LLM→neu and neural→LLM groups

    for gi, direction in enumerate(DIRECTIONS):
        color = DIR_COLORS[direction]
        xc = gi * group_gap + x_curv
        xd = gi * group_gap + x_disp
        rc, rd = partial_data[(lag, direction)]

        # negate so higher = stronger unique contribution
        rc_plot = -rc
        rd_plot = -rd

        # per-patient lines
        for c_val, d_val in zip(rc_plot, rd_plot):
            ax.plot([xc, xd], [c_val, d_val],
                    color=color, lw=1.0, alpha=0.35, zorder=2)
            ax.scatter([xc, xd], [c_val, d_val],
                       color=color, s=18, alpha=0.5, zorder=3)

        # group means ± SEM as thick markers
        for vals, x in [(rc_plot, xc), (rd_plot, xd)]:
            mu, se = vals.mean(), scipy_sem(vals)
            orig   = -vals   # original signed values for significance test
            _, p   = ttest_1samp(orig, 0)
            ax.plot([x - 0.18, x + 0.18], [mu, mu],
                    color=color, lw=3.0, solid_capstyle="round", zorder=4)
            ax.errorbar(x, mu, yerr=se, fmt="none",
                        color=color, capsize=5, lw=2.0, zorder=5)
            top = mu + se + 0.003
            ax.text(x, top, stars(p), ha="center", va="bottom",
                    fontsize=10, color="#333333")


    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=1)
    ax.set_ylabel("Unique decoding contribution\n(−partial $r$, higher = stronger)")
    ax.set_xticks([x_curv, x_disp,
                   group_gap + x_curv, group_gap + x_disp])
    ax.set_xticklabels(["Curv.", "Disp.", "Curv.", "Disp."], fontsize=10)
    ax.set_xlim(-0.5, group_gap + x_disp + 0.5)
    ax.set_ylim(-0.06, 0.20)

    # vertical separator between directions
    ax.axvline((x_disp + group_gap + x_curv) / 2,
               color="#CCCCCC", lw=1.0, ls=":", zorder=1)

    # direction labels below the x-axis ticks
    for gi, (direction, color) in enumerate(zip(DIRECTIONS, [C_LLM, C_NEURAL])):
        mid = gi * group_gap + (x_curv + x_disp) / 2
        ax.text(mid, -0.048, DIR_LABELS[direction], ha="center", va="top",
                fontsize=10, color=color, fontweight="bold")

    ax.text(-0.22, 1.08, "C", transform=ax.transAxes, **PANEL_LABEL_KW)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FDIR, exist_ok=True)

    print("Loading data...", flush=True)
    dfs = {lag: load_pointwise(lag) for lag in LAGS}

    # Panel A data
    panel_a = {}
    for lag, df in dfs.items():
        for direction in DIRECTIONS:
            pp = per_patient_mean_r(df, direction)
            t, p = ttest_1samp(pp, 0)
            panel_a[(lag, direction)] = (pp.mean(), scipy_sem(pp), t, p)
            print(f"  lag={lag} {direction}: r={pp.mean():.4f} p={p:.4f}", flush=True)

    # Panel B data
    print("Computing curvature bins...", flush=True)
    bin_data = {}
    for lag, df in dfs.items():
        for direction in DIRECTIONS:
            bin_data[(lag, direction)] = per_patient_curv_bins(df, direction)

    # Panel C data
    print("Computing partial r...", flush=True)
    partial_data = {}
    for lag, df in dfs.items():
        for direction in DIRECTIONS:
            rc, rd = per_patient_partial_r(df, direction)
            partial_data[(lag, direction)] = (rc, rd)
            t, p = ttest_1samp(rc, 0)
            print(f"  lag={lag} {direction}: partial r(curv|disp)={rc.mean():+.3f} p={p:.4f}", flush=True)

    # Build figure
    fig = plt.figure(figsize=(14, 5.2))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            wspace=0.52, left=0.10, right=0.98,
                            bottom=0.16, top=0.86)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    draw_panel_a(ax_a, panel_a)
    draw_panel_b(ax_b, bin_data)
    draw_panel_c(ax_c, partial_data)

    stem = "fig_cross_state_decoding"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FDIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
