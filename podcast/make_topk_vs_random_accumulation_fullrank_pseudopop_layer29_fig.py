"""
make_topk_vs_random_accumulation_fullrank_pseudopop_layer29_fig.py
=====================================================================
Full-rank version of make_topk_vs_random_accumulation_pseudopop_layer29_fig.py
(semantic up to 1600 PCs, neural up to 475; right of the dotted line the
neural side is fixed at all 475 PCs).
Row 1: top-K vs. random PCs added in both spaces (median, 50% of orderings).
Row 2: random orderings split at each K by whether semantic PC1 is included yet.
Grey = shuffle null 95% (top-K).
Data: results/curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_fullrank_pseudopop.json

Saves: figures/fig_topk_vs_random_accumulation_fullrank_pseudopop_layer29.{pdf,png,svg}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = ["curv", "disp", "pos"]
MNAME = dict(curv="Curvature", disp="Displacement", pos="Position (first PC)")
TOP_COL, RND_COL, IN_COL, OUT_COL = "#c0392b", "#5d6d7e", "#8e44ad", "#16a085"

d = json.load(open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_fullrank_pseudopop.json")))
KS = np.array(d["config"]["KS"])
NN = d["config"]["n_neu"]
rb = d["random"]["both"]
arr = np.array(rb["r"])
p1 = np.array(rb["pc1_pos_sem"])
inc = p1[:, None] <= KS[None, :]          # orderings x K: semantic PC1 included


def band(ax, v, col, label):
    ok = ~np.isnan(v).all(axis=0)
    ax.fill_between(KS[ok], *np.nanpercentile(v[:, ok], [25, 75], axis=0), color=col, alpha=0.3, lw=0)
    ax.plot(KS[ok], np.nanmedian(v[:, ok], axis=0), color=col, lw=2, label=label)


fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True, sharex=True)
for c, m in enumerate(METRICS):
    top = np.array([d["top"][str(K)][m]["r"] for K in KS])
    for r in range(2):
        ax = axes[r, c]
        ax.fill_between(KS, [d["top"][str(K)][m]["null_lo"] for K in KS],
                        [d["top"][str(K)][m]["null_hi"] for K in KS], color="0.88", lw=0, label="shuffle null 95%")
        ax.plot(KS, top, color=TOP_COL, lw=2.5, marker="o", ms=4, label="top-K PCs")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(NN, color="k", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256, NN, 1600])
        ax.set_xticklabels([1, 4, 16, 64, 256, NN, 1600])
        ax.set_title(f"{'ABCDEF'[3 * r + c]}  {MNAME[m]}", loc="left")
    v = arr[:, :, c]
    band(axes[0, c], v, RND_COL, "random PCs (median, 50%)")
    band(axes[1, c], np.where(inc, v, np.nan), IN_COL, "random, semantic PC1 included")
    band(axes[1, c], np.where(~inc, v, np.nan), OUT_COL, "random, semantic PC1 not included")
    axes[1, c].set_xlabel("PCs per space (K)\n(neural fixed at 475 right of dotted line)")
for r in range(2):
    axes[r, 0].set_ylabel("Semantic–neural r")
    axes[r, 0].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle(f"Top-K vs. random PCs, full rank (semantic {d['config']['n_sem']}, neural {NN}) — pseudopopulation, "
             f"gpt2-xl layer {d['config']['LAYER']} ({d['config']['N_ORDER']} random orderings)",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_topk_vs_random_accumulation_fullrank_pseudopop_layer29"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
