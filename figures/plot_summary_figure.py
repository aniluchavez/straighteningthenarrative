"""
plot_summary_figure.py
======================
Three-panel summary figure:

  A  Within-space: flat trajectory → better within-modality similarity
     (LLM internal cosine sim, neural internal decoding r)

  B  Cross-space: flat trajectory → better cross-space decoding
     (LLM → neural  &  neural → LLM, lag +1)

  C  Curvature vs displacement: curvature is the unique predictor
     (residualized bins, averaged across directions, lag +1)

All panels: x = curvature quintile (flat → curved), SEM shading,
chance/baseline dashed line, significance stars.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem, ttest_1samp, ttest_rel
from sklearn.linear_model import LinearRegression

BASE  = os.path.dirname(os.path.abspath(__file__))
RDIR  = os.path.join(BASE, "results", "geometry_paper_curvature_alignment")
FDIR  = os.path.join(BASE, "figures", "appendix")
REGION = "hippocampus"
MODEL  = "llama-3.1-8b"
N_BINS = 5

C_LLM   = "#1B8A3C"   # forest green  — LLM / curvature
C_NEURAL= "#D4297B"   # deep pink     — neural / displacement
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
    "axes.linewidth": 0.9,
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

def add_panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

XTICK_LABELS = ["1\n(flat)", "2", "3", "4", "5\n(curved)"]


def pstar(p):
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def resid_ols(y, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    return y - LinearRegression().fit(x, y).predict(x)


def add_sig_bracket(ax, xs, ys, p, color, offset=0.012):
    """Small star annotation above the flat bin."""
    star = pstar(p)
    if star and star != "n.s.":
        ax.text(xs[0], ys[0] + offset * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                star, ha="center", va="bottom", color=color, fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# Panels A & B: cross-space decoding, lag 0 and lag+1
# ─────────────────────────────────────────────────────────────────────────────
def per_patient_curvature_bins(pt, direction):
    sub = pt[(pt["region"] == REGION) & pt["valid"] & (pt["direction"] == direction)]
    rows = []
    for _, g in sub.groupby("pid"):
        g = g.dropna(subset=["joint_curvature", "target_corr"]).copy()
        if len(g) < N_BINS * 4:
            continue
        try:
            g["_bin"] = pd.qcut(g["joint_curvature"], q=N_BINS,
                                 labels=False, duplicates="drop")
            b = g.groupby("_bin")["target_corr"].mean().reindex(range(N_BINS))
            if b.isna().sum() <= 1:
                rows.append(b.fillna(b.mean()).values)
        except Exception:
            pass
    return np.array(rows) if rows else None


def draw_cross(ax, lag, panel_label):
    pt = pd.read_csv(os.path.join(RDIR,
        f"curvature_alignment_llama-3.1-8b_hippocampus-ACC_all-patients"
        f"_pca_llmpc64_shufflecv_ridge_both_lagplus{lag}_matchedpcs_pointwise.csv"))
    xs = np.arange(1, N_BINS + 1)

    for direction, color, label, marker, ypos in [
        ("LLM_to_neural", C_LLM,    "LLM → neural", "o", 0.97),
        ("neural_to_LLM", C_NEURAL, "Neural → LLM", "s", 0.82),
    ]:
        pat = per_patient_curvature_bins(pt, direction)
        if pat is None:
            continue
        mu = pat.mean(0)
        se = scipy_sem(pat, axis=0)
        deltas = pat[:, 0] - pat[:, -1]
        t, p   = ttest_1samp(deltas, 0)

        ax.plot(xs, mu, color=color, lw=2.2, marker=marker, ms=5,
                label=label, zorder=4)
        ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.15, zorder=3)
        ax.text(0.97, ypos, f"$\\Delta$={deltas.mean():+.3f} {pstar(p)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=color)

    lag_str = "current state (lag 0)" if lag == 0 else "next state (lag +1)"
    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2,
               label="Chance (r = 0)")
    ax.set_xticks(xs)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel("Curvature quintile")
    ax.set_ylabel("Cross-space decoding $r$")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)
    ax.set_title(f"Cross-space  ·  {lag_str}", loc="left", pad=6)
    add_panel_label(ax, panel_label)


# ─────────────────────────────────────────────────────────────────────────────
# Residualized bin helper
# ─────────────────────────────────────────────────────────────────────────────
def get_residualized_bins(pt, direction, bin_col, control_col):
    sub = pt[(pt["region"] == REGION) & pt["valid"] & (pt["direction"] == direction)]
    pat_bins, pat_deltas = [], []
    for _, g in sub.groupby("pid"):
        g = g.dropna(subset=[bin_col, control_col, "target_corr"]).copy()
        if len(g) < N_BINS * 4:
            continue
        g["resid"] = resid_ols(g["target_corr"].values, g[control_col].values)
        try:
            g["_bin"] = pd.qcut(g[bin_col], q=N_BINS, labels=False, duplicates="drop")
        except Exception:
            continue
        b = g.groupby("_bin")["resid"].mean().reindex(range(N_BINS))
        if b.isna().sum() > 1:
            continue
        b = b.fillna(b.mean()).values
        pat_bins.append(b)
        pat_deltas.append(b[0] - b[-1])
    if not pat_bins:
        return None, None
    return np.array(pat_bins), np.array(pat_deltas)


# ─────────────────────────────────────────────────────────────────────────────
# (layer sweep kept for standalone use)
# ─────────────────────────────────────────────────────────────────────────────
N_LAYERS  = 33
LOO_LAYER = 5


def draw_layer_sweep(ax):
    from scipy.stats import pearsonr

    xs = np.arange(N_LAYERS)
    rc_mu = np.full(N_LAYERS, np.nan)
    rc_se = np.full(N_LAYERS, np.nan)
    rd_mu = np.full(N_LAYERS, np.nan)
    rd_se = np.full(N_LAYERS, np.nan)

    for layer in range(N_LAYERS):
        path = os.path.join(RDIR,
            f"curvature_alignment_{MODEL}_hippocampus-ACC_all-patients"
            f"_pca_llmpc64_shufflecv_ridge_both_lagplus1_matchedpcs_layer{layer}_pointwise.csv")
        if not os.path.exists(path):
            continue
        pt = pd.read_csv(path)
        rcs, rds = [], []
        for direction in ["LLM_to_neural"]:
            sub = pt[(pt["region"] == REGION) & pt["valid"] & (pt["direction"] == direction)]
            for _, g in sub.groupby("pid"):
                c = g["joint_curvature"].values.astype(float)
                d = g["joint_displacement"].values.astype(float)
                r = g["target_corr"].values.astype(float)
                m = np.isfinite(c) & np.isfinite(d) & np.isfinite(r)
                if m.sum() < 10:
                    continue
                c, d, r = c[m], d[m], r[m]
                rcs.append(pearsonr(resid_ols(c, d), resid_ols(r, d))[0])
                rds.append(pearsonr(resid_ols(d, c), resid_ols(r, c))[0])
        if rcs:
            rc_mu[layer] = np.mean(np.abs(rcs));  rc_se[layer] = scipy_sem(np.abs(rcs))
            rd_mu[layer] = np.mean(np.abs(rds));  rd_se[layer] = scipy_sem(np.abs(rds))

    valid_c = np.isfinite(rc_mu)
    valid_d = np.isfinite(rd_mu)

    if valid_c.any():
        ax.plot(xs[valid_c], rc_mu[valid_c], color=C_LLM, lw=2.2, marker="o", ms=4,
                label=r"Partial $r$ — curvature (ctrl $d$)", zorder=4)
        ax.fill_between(xs[valid_c],
                        rc_mu[valid_c] - rc_se[valid_c],
                        rc_mu[valid_c] + rc_se[valid_c],
                        color=C_LLM, alpha=0.15, zorder=3)

    if valid_d.any():
        ax.plot(xs[valid_d], rd_mu[valid_d], color=C_NEURAL, lw=2.2, marker="s", ms=4,
                label=r"Partial $r$ — displacement (ctrl $\kappa$)", zorder=4)
        ax.fill_between(xs[valid_d],
                        rd_mu[valid_d] - rd_se[valid_d],
                        rd_mu[valid_d] + rd_se[valid_d],
                        color=C_NEURAL, alpha=0.15, zorder=3)

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.axvline(LOO_LAYER, color="#999999", lw=1.2, ls=":", alpha=0.8, zorder=5,
               label=f"LOO-opt. layer ({LOO_LAYER})")

    ax.set_xlabel("LLM layer")
    ax.set_ylabel("Partial correlation $|r|$ (unique contribution)")
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(np.arange(0, N_LAYERS, 4))
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="#CCCCCC", fontsize=8.5)
    ax.set_title("Curvature vs. displacement across layers  (LLM→neural, lag +1)",
                 loc="left", pad=6)
    add_panel_label(ax, "C")


# ─────────────────────────────────────────────────────────────────────────────
def draw_cross_single(ax, lag, direction, panel_label):
    """Cross-space decoding bins for one direction."""
    pt = pd.read_csv(os.path.join(RDIR,
        f"curvature_alignment_llama-3.1-8b_hippocampus-ACC_all-patients"
        f"_pca_llmpc64_shufflecv_ridge_both_lagplus{lag}_matchedpcs_pointwise.csv"))
    xs = np.arange(1, N_BINS + 1)

    pat = per_patient_curvature_bins(pt, direction)
    if pat is not None:
        mu = pat.mean(0)
        se = scipy_sem(pat, axis=0)
        deltas = pat[:, 0] - pat[:, -1]
        t, p = ttest_1samp(deltas, 0)
        color = C_LLM if direction == "LLM_to_neural" else C_NEURAL
        ax.plot(xs, mu, color=color, lw=2.2, marker="o", ms=5, zorder=4)
        ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.15, zorder=3)
        ax.text(0.97, 0.97, f"$\\Delta$={deltas.mean():+.3f} {pstar(p)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=color)

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel("Curvature quintile")
    ax.set_ylabel("Cross-space decoding $r$")
    dir_str = "LLM → neural" if direction == "LLM_to_neural" else "Neural → LLM"
    lag_str = "lag 0" if lag == 0 else "lag +1"
    ax.set_title(f"{dir_str}  ·  {lag_str}", loc="left", pad=6)
    add_panel_label(ax, panel_label)


def draw_residualized_single(ax, direction, panel_label):
    """Residualized bins for one direction, lag+1."""
    pt = pd.read_csv(os.path.join(RDIR,
        "curvature_alignment_llama-3.1-8b_hippocampus-ACC_all-patients"
        "_pca_llmpc64_shufflecv_ridge_both_lagplus1_matchedpcs_pointwise.csv"))
    xs = np.arange(1, N_BINS + 1)

    for bin_col, ctrl_col, color, label, marker in [
        ("joint_curvature",    "joint_displacement", C_LLM,    r"Curvature ($\kappa$, ctrl $d$)",    "o"),
        ("joint_displacement", "joint_curvature",    C_NEURAL, r"Displacement ($d$, ctrl $\kappa$)", "s"),
    ]:
        bins, deltas = get_residualized_bins(pt, direction, bin_col, ctrl_col)
        if bins is None:
            continue
        mu = bins.mean(0)
        se = scipy_sem(bins, axis=0)
        t, p = ttest_1samp(deltas, 0)
        ax.plot(xs, mu, color=color, lw=2.2, marker=marker, ms=5,
                label=f"{label}  $\\Delta$={deltas.mean():+.3f}{pstar(p)}", zorder=4)
        ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.15, zorder=3)

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel("Curvature quintile")
    ax.set_ylabel("Residualized decoding $r$")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC", fontsize=8)
    dir_str = "LLM → neural" if direction == "LLM_to_neural" else "Neural → LLM"
    ax.set_title(f"Curvature vs. displacement  ·  {dir_str}", loc="left", pad=6)
    add_panel_label(ax, panel_label)


def main():
    os.makedirs(FDIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw=dict(hspace=0.58, wspace=0.42,
                                             left=0.10, right=0.98,
                                             bottom=0.09, top=0.88))

    draw_cross_single(axes[0, 0], lag=1, direction="LLM_to_neural",  panel_label="A")
    draw_residualized_single(axes[0, 1], direction="LLM_to_neural",  panel_label="B")
    draw_cross_single(axes[1, 0], lag=1, direction="neural_to_LLM",  panel_label="C")
    draw_residualized_single(axes[1, 1], direction="neural_to_LLM",  panel_label="D")


    stem = "fig_summary"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FDIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
