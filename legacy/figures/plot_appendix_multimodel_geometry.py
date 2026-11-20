"""
plot_appendix_multimodel_geometry.py
=====================================
Two-panel appendix figure showing multi-model geometric evidence.

Panel A: Heatmap — mean r(LLM metric, neural metric) for 9 models × 3 geometric
         quantities (Displacement, Curvature, Position), hippocampus only,
         15-word chunks averaged across all layers and patients.

Panel B: Scatter — models with flatter LLM trajectories (lower curvature)
         show stronger hippocampal cross-space readout (r across 9 models).
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_1samp, pearsonr, linregress

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
# P6 results: symlinked into results/ alongside the curvature_alignment dir
P6_RES    = os.path.join(BASE, "results")
ALN_RES   = os.path.join(BASE, "results", "geometry_paper_curvature_alignment")
FDIR      = os.path.join(BASE, "figures", "appendix")

MODEL_AVERAGED_CSV = os.path.join(
    ALN_RES,
    "multimodel_linear_curvature_alignment_all-models_hippocampus-ACC_all-patients"
    "_pca_llmpc64_shufflecv_ridge_both_lagplus1_direction_averaged_model_summary.csv",
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":         12,
    "axes.titlesize":    12,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

PANEL_LABEL_KW = dict(
    fontsize=28, fontweight="bold", fontfamily="sans-serif",
    va="top", ha="left",
)

def add_panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

# ── Model metadata ─────────────────────────────────────────────────────────────
MODEL_ORDER = [
    "bert-base", "roberta-base",
    "gpt2", "gpt2-medium", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]
MODEL_LABELS = {
    "bert-base":    "BERT-base",
    "roberta-base": "RoBERTa-base",
    "gpt2":         "GPT-2",
    "gpt2-medium":  "GPT-2 medium",
    "opt-350m":     "OPT-350m",
    "llama-2-7b":   "LLaMA-2 7B",
    "llama-3.1-8b": "LLaMA-3.1 8B",
    "gemma-2-9b":   "Gemma-2 9B",
    "mistral-7b":   "Mistral-7B",
}
MODEL_COLORS = {
    "gpt2":         "#4C78A8",
    "gpt2-medium":  "#72B7B2",
    "opt-350m":     "#54A24B",
    "bert-base":    "#8A8A8A",
    "roberta-base": "#B279A2",
    "llama-2-7b":   "#F58518",
    "llama-3.1-8b": "#E45756",
    "gemma-2-9b":   "#9D755D",
    "mistral-7b":   "#7F6AD7",
}
BIDIRECTIONAL = {"bert-base", "roberta-base"}

METRICS = [
    dict(label="Displacement", r_key_ch="r_disp_layers_ch"),
    dict(label="Curvature κ",  r_key_ch="r_curv_layers_ch"),
    dict(label="Position",     r_key_ch="r_pos_layers_ch"),
]


def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ── Panel A: heatmap ──────────────────────────────────────────────────────────

def build_heatmap_matrix():
    n_models  = len(MODEL_ORDER)
    n_metrics = len(METRICS)
    mean_mat = np.full((n_models, n_metrics), np.nan)
    sig_mat  = np.full((n_models, n_metrics), "", dtype=object)

    for mi, model in enumerate(MODEL_ORDER):
        path = os.path.join(P6_RES, f"cross_space_layers_{model}_hippocampus.json")
        if not os.path.exists(path):
            continue
        recs = json.load(open(path))
        for qi, metric in enumerate(METRICS):
            key = metric["r_key_ch"]
            pat_means = []
            for rec in recs:
                if key not in rec:
                    continue
                arr = np.array(rec[key], dtype=float)
                if np.isfinite(arr).any():
                    pat_means.append(np.nanmean(arr))
            if not pat_means:
                continue
            pat_means = np.array(pat_means)
            mean_mat[mi, qi] = np.mean(pat_means)
            if len(pat_means) > 1:
                _, p = ttest_1samp(pat_means, 0)
                sig_mat[mi, qi] = _stars(p)

    return mean_mat, sig_mat


def draw_heatmap(ax, mean_mat, sig_mat):
    vals = mean_mat.ravel()
    vals = vals[np.isfinite(vals)]
    vmax = np.percentile(np.abs(vals), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(mean_mat, cmap="RdYlBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m["label"] for m in METRICS], fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
    ax.axhline(1.5, color="white", lw=2.0)
    ax.spines[:].set_visible(False)

    for mi in range(len(MODEL_ORDER)):
        for qi in range(len(METRICS)):
            val = mean_mat[mi, qi]
            if np.isnan(val):
                continue
            star = sig_mat[mi, qi]
            txt  = f"{val:+.3f}{star}"
            bg   = norm(val)
            fg   = "white" if abs(bg - 0.5) > 0.30 else "#222222"
            ax.text(qi, mi, txt, ha="center", va="center",
                    fontsize=8, color=fg, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.07, pad=0.03)
    cbar.set_label("Mean $r$ (LLM ↔ Neural)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Cross-space geometric coupling  ·  Hippocampus\n"
                 "15-word chunks, mean $r$ across layers and patients",
                 loc="left", fontsize=10)
    add_panel_label(ax, "A")


# ── Panel B: scatter ──────────────────────────────────────────────────────────

def draw_scatter(ax, model_df):
    hpc = model_df[model_df["region"] == "hippocampus"].copy()

    x = hpc["mean_llm_curvature_mean"].to_numpy(dtype=float)
    y = hpc["mean_decoding_corr_mean"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    r, p = pearsonr(x, y)
    fit  = linregress(x, y)
    xl   = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, fit.intercept + fit.slope * xl,
            color="#227C70", lw=1.6, alpha=0.85, zorder=2)

    for _, row in hpc.iterrows():
        model = row["model"]
        is_ctrl = model in BIDIRECTIONAL
        ax.scatter(
            row["mean_llm_curvature_mean"],
            row["mean_decoding_corr_mean"],
            s=65 if not is_ctrl else 70,
            marker="s" if is_ctrl else "o",
            facecolor="white" if is_ctrl else MODEL_COLORS[model],
            edgecolor=MODEL_COLORS[model],
            linewidth=1.4, alpha=0.95, zorder=3,
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (row["mean_llm_curvature_mean"], row["mean_decoding_corr_mean"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=8, color=MODEL_COLORS[model],
        )

    p_str = "p < .001" if p < 0.001 else f"p = {p:.2f}"
    ax.text(0.05, 0.95, f"$r = {r:.2f}$,  {p_str}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color="#222222")

    ax.set_xlabel("LLM curvature (lower = flatter)")
    ax.set_ylabel("Mean cross-space readout ($r$)")
    ax.set_title("Flatter models show stronger hippocampal readout",
                 loc="left", fontsize=10)
    add_panel_label(ax, "B")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FDIR, exist_ok=True)

    print("Building heatmap matrix...", flush=True)
    mean_mat, sig_mat = build_heatmap_matrix()

    print("Loading model-averaged data...", flush=True)
    model_df = pd.read_csv(MODEL_AVERAGED_CSV)

    fig = plt.figure(figsize=(12.5, 5.0))
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            wspace=0.48, left=0.10, right=0.97,
                            bottom=0.13, top=0.87)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    draw_heatmap(ax_a, mean_mat, sig_mat)
    draw_scatter(ax_b, model_df)

    stem = "fig_appendix_multimodel_geometry"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FDIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
