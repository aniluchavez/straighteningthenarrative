"""
make_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop_fig.py
============================================================================
Figure for compute_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.py.
Rows: random scheme (both sides / semantic only / neural only random).
Columns: curvature, displacement, position. Median and 50%/95% bands across
random orderings vs. K; black = top-K reference; grey = shuffle null 95% of top-K.

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.{pdf,png,svg}
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
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = ["curv", "disp", "pos"]
MNAME = dict(curv="Curvature", disp="Displacement", pos="Position (first PC)")
MCOL = dict(curv="#c0392b", disp="#2471a3", pos="#7d8c2b")
SCHEMES = [("both", "Random PCs, both spaces"),
           ("sem", "Random semantic PCs, neural top-K"),
           ("neu", "Random neural PCs, semantic top-K")]

d = json.load(open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.json")))
KS = np.array(d["config"]["KS"])
top = {m: np.array([d["top"][str(K)][m]["r"] for K in KS]) for m in METRICS}
nlo = {m: np.array([d["top"][str(K)][m]["null_lo"] for K in KS]) for m in METRICS}
nhi = {m: np.array([d["top"][str(K)][m]["null_hi"] for K in KS]) for m in METRICS}

fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
for r, (sch, title) in enumerate(SCHEMES):
    arr = np.array(d["random"][sch])  # orderings x K x metric
    for c, m in enumerate(METRICS):
        ax = axes[r, c]
        v = arr[:, :, c]
        ax.fill_between(KS, nlo[m], nhi[m], color="0.85", lw=0, label="shuffle null 95%")
        ax.fill_between(KS, *np.percentile(v, [2.5, 97.5], axis=0), color=MCOL[m], alpha=0.15, lw=0,
                        label="random 95%")
        ax.fill_between(KS, *np.percentile(v, [25, 75], axis=0), color=MCOL[m], alpha=0.35, lw=0,
                        label="random 50%")
        ax.plot(KS, np.median(v, axis=0), color=MCOL[m], lw=2, label="random median")
        ax.plot(KS, top[m], color="k", lw=1.5, ls="--", label="top-K")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 200])
        ax.set_xticklabels([1, 4, 16, 64, 200])
        ax.set_title(f"{'ABCDEFGHI'[3 * r + c]}  {MNAME[m]}", loc="left")
        if c == 0:
            ax.set_ylabel(f"{title}\nsemantic–neural r")
        if r == 2:
            ax.set_xlabel("PCs added (K)")
        if r == 0:
            ax.legend(frameon=False, loc="upper left")

fig.suptitle("Random PC accumulation — curvature / displacement / position alignment, pseudopopulation, "
             f"gpt2-xl layer {d['config']['LAYER']} ({d['config']['N_ORDER']} random orderings)",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
