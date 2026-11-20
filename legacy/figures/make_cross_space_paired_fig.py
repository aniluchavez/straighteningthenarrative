"""
make_cross_space_paired_fig.py
================================
Two-panel figure:

  Panel A — Paired dot plot: per-patient r(LLM, neural) for curvature,
             displacement, and position at 15-word chunks, averaged across
             all layers. LLaMA-3.1-8B hippocampus.
             Significance brackets from paired t-tests.

  Panel B — Chunk-size sweep for curvature and displacement only (position
             is flat and omitted for clarity), averaged across all models
             and patients. Shows that displacement peaks at short windows
             while curvature rises and plateaus at longer ones.

Run:
    cd /scratch/aniluchavez/neural_network_similarity/Experiments/p6/geometry_paper
    conda run -n gpt2_embed python3 make_cross_space_paired_fig.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_rel, ttest_1samp, sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION   = "hippocampus"
MODEL    = "llama-3.1-8b"   # kept for reference; Panel A now averages all models

ALL_MODELS = [
    "gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         13,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_POS  = "#3DAA6C"


# ── helpers ───────────────────────────────────────────────────────────────────

def _sig_bracket(ax, x0, x1, y, h, p, fontsize=8.5):
    """Draw a significance bracket between x0 and x1 at height y."""
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=1.0, color="k", clip_on=False)
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    pstr = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text((x0 + x1) / 2, y + h * 1.15,
            f"{stars} ({pstr})", ha="center", va="bottom",
            fontsize=fontsize, color="k")


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  → {p}")
    plt.close(fig)


CHUNK_SIZE_PANEL_A = 15   # elbow from piecewise linear fit

# ── Panel A data ──────────────────────────────────────────────────────────────

def load_panel_a():
    """
    Per-patient r values at CHUNK_SIZE_PANEL_A words, averaged across all
    layers and models.  Reads directly from the chunk-sweep JSON so no layer
    recompute is needed when changing the chunk size.
    """
    sweep = json.load(open(os.path.join(RESULTS, "cross_space_chunk_sweep.json")))
    cs_data = sweep[str(CHUNK_SIZE_PANEL_A)]  # {model: {metric: [r per patient]}}

    # pool per-patient values across models; align by position (same patient order)
    # First, determine n_patients from any model
    n_pat = max(len(v.get("curv", [])) for v in cs_data.values())

    curv_by_pat = [[] for _ in range(n_pat)]
    disp_by_pat = [[] for _ in range(n_pat)]
    pos_by_pat  = [[] for _ in range(n_pat)]

    for model in ALL_MODELS:
        md = cs_data.get(model, {})
        for i, (c, d, p) in enumerate(zip(
                md.get("curv", []), md.get("disp", []), md.get("pos", []))):
            if c is not None and not np.isnan(c): curv_by_pat[i].append(c)
            if d is not None and not np.isnan(d): disp_by_pat[i].append(d)
            if p is not None and not np.isnan(p): pos_by_pat[i].append(p)

    curv_pat = np.array([np.mean(v) if v else np.nan for v in curv_by_pat])
    disp_pat = np.array([np.mean(v) if v else np.nan for v in disp_by_pat])
    pos_pat  = np.array([np.mean(v) if v else np.nan for v in pos_by_pat])
    return curv_pat, disp_pat, pos_pat


# ── Panel B data ──────────────────────────────────────────────────────────────

def load_panel_b():
    """Mean r vs chunk size for curvature and displacement, across all models."""
    path = os.path.join(RESULTS, "cross_space_chunk_sweep.json")
    data = json.load(open(path))
    chunk_sizes = sorted(int(k) for k in data.keys())

    curv_by_cs, disp_by_cs = [], []
    for cs in chunk_sizes:
        c_vals, d_vals = [], []
        for model_data in data[str(cs)].values():
            c_vals.extend([v for v in model_data.get("curv", [])
                           if v is not None and not np.isnan(v)])
            d_vals.extend([v for v in model_data.get("disp", [])
                           if v is not None and not np.isnan(v)])
        curv_by_cs.append((np.mean(c_vals), scipy_sem(c_vals)) if c_vals else (np.nan, np.nan))
        disp_by_cs.append((np.mean(d_vals), scipy_sem(d_vals)) if d_vals else (np.nan, np.nan))

    return chunk_sizes, curv_by_cs, disp_by_cs


# ── Figure ────────────────────────────────────────────────────────────────────

def main():
    curv, disp, pos = load_panel_a()
    chunk_sizes, curv_sweep, disp_sweep = load_panel_b()

    fig = plt.figure(figsize=(9.5, 4.2))
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.09, right=0.97, bottom=0.14, top=0.88,
                            wspace=0.42, width_ratios=[1.1, 1.6])
    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])

    # ── Panel A: paired dot plot ───────────────────────────────────────────────
    groups  = [curv, disp, pos]
    colors  = [C_CURV, C_DISP, C_POS]
    labels  = ["Curvature (κ)", "Displacement (d)", "Position"]
    x_pos   = [0, 1, 2]
    n_pat   = len(curv)

    # Bars (mean)
    for x, vals, col in zip(x_pos, groups, colors):
        mu = np.nanmean(vals)
        ax_A.bar(x, mu, width=0.55, color=col, alpha=0.35, zorder=2,
                 edgecolor="none")
        ax_A.bar(x, mu, width=0.55, color="none", zorder=2,
                 edgecolor=col, linewidth=1.2)

    # Lines connecting paired patients (draw first so dots sit on top)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, n_pat)
    for i in range(n_pat):
        ys = [curv[i], disp[i], pos[i]]
        xs = [x + jitter[i] for x in x_pos]
        ax_A.plot(xs, ys, color="#AAAAAA", lw=0.7, zorder=3, alpha=0.6)

    # Dots
    for x, vals, col in zip(x_pos, groups, colors):
        xs = [x + j for j in jitter]
        ax_A.scatter(xs, vals, color=col, s=28, zorder=4,
                     edgecolors="white", linewidths=0.5)

    # Mean ± SE error bars
    for x, vals, col in zip(x_pos, groups, colors):
        mu = np.nanmean(vals)
        se = scipy_sem(vals[np.isfinite(vals)])
        ax_A.errorbar(x, mu, yerr=se, fmt="none", color=col,
                      capsize=3, lw=1.6, zorder=5)

    ax_A.axhline(0, color="k", lw=0.5, ls=":")

    # Significance brackets
    y_top = max(np.nanmax(curv), np.nanmax(disp), np.nanmax(pos))
    gap   = y_top * 0.12

    _, p_cd = ttest_rel(curv[np.isfinite(curv) & np.isfinite(disp)],
                        disp[np.isfinite(curv) & np.isfinite(disp)])
    _, p_cp = ttest_rel(curv[np.isfinite(curv) & np.isfinite(pos)],
                        pos[np.isfinite(curv) & np.isfinite(pos)])

    b1 = y_top + gap * 0.8
    b2 = b1 + gap * 1.8
    _sig_bracket(ax_A, 0, 1, b1, gap * 0.3, p_cd)
    _sig_bracket(ax_A, 0, 2, b2, gap * 0.3, p_cp)

    ax_A.set_xticks(x_pos)
    ax_A.set_xticklabels(labels, fontsize=10)
    ax_A.set_ylabel("Semantic ↔ neural Pearson r", fontsize=10)
    ax_A.set_title("A  Cross-space geometry alignment\n"
                   "all models · hippocampus · 15-word chunks",
                   loc="left", fontsize=9.5)

    # ── Panel B: chunk sweep ───────────────────────────────────────────────────
    xs = np.array(chunk_sizes)

    for sweep, color, label in [
        (curv_sweep, C_CURV, "Curvature (κ)"),
        (disp_sweep, C_DISP, "Displacement (d)"),
    ]:
        mus = np.array([v[0] for v in sweep])
        ses = np.array([v[1] for v in sweep])
        ax_B.plot(xs, mus, color=color, lw=2.0, marker="o", ms=4.5, label=label)
        ax_B.fill_between(xs, mus - ses, mus + ses, color=color, alpha=0.15)

    ax_B.axvline(15, color="#555555", lw=1.1, ls="--", label="15-word elbow (Panel A)")
    ax_B.axhline(0,  color="k", lw=0.4, ls=":")

    ax_B.set_xlabel("Chunk size (words)")
    ax_B.set_ylabel("Mean r  (LLM ↔ Neural)")
    ax_B.set_title("B  Coupling vs chunk size\n"
                   "all models · hippocampus · avg across layers",
                   loc="left", fontsize=9.5)
    ax_B.set_xticks(xs)
    ax_B.legend(frameon=False, fontsize=9)

    _save(fig, "fig_cross_space_paired")
    print("Done.")


if __name__ == "__main__":
    main()
