"""
make_topk_vs_random_accumulation_pseudopop_layer29_fig.py
============================================================
Top-K PCs (ordered by variance, both spaces) vs. random PCs added one at a
time (independent random orderings on both spaces), pseudopopulation,
gpt2-xl layer 29. One panel per metric; grey = shuffle null 95% (top-K).
Data: results/curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.json

Saves: figures/fig_topk_vs_random_accumulation_pseudopop_layer29.{pdf,png,svg}
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
TOP_COL, RND_COL = "#c0392b", "#5d6d7e"

d = json.load(open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.json")))
KS = np.array(d["config"]["KS"])
arr = np.array(d["random"]["both"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for c, m in enumerate(METRICS):
    ax = axes[c]
    top = np.array([d["top"][str(K)][m]["r"] for K in KS])
    ax.fill_between(KS, [d["top"][str(K)][m]["null_lo"] for K in KS], [d["top"][str(K)][m]["null_hi"] for K in KS],
                    color="0.88", lw=0, label="shuffle null 95%")
    v = arr[:, :, c]
    ax.fill_between(KS, *np.percentile(v, [25, 75], axis=0), color=RND_COL, alpha=0.3, lw=0,
                    label="random PCs, 50% of orderings")
    ax.plot(KS, np.median(v, axis=0), color=RND_COL, lw=2, label="random PCs, median")
    ax.plot(KS, top, color=TOP_COL, lw=2.5, marker="o", ms=4, label="top-K PCs")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 200])
    ax.set_xticklabels([1, 4, 16, 64, 200])
    ax.set_xlabel("PCs per space (K)")
    ax.set_title(f"{'ABC'[c]}  {MNAME[m]}", loc="left")
axes[0].set_ylabel("Semantic–neural r")
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle(f"Top-K vs. random PCs added in both spaces — pseudopopulation, gpt2-xl layer {d['config']['LAYER']} "
             f"({d['config']['N_ORDER']} random orderings)", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_topk_vs_random_accumulation_pseudopop_layer29"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
