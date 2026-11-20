"""
make_curv_disp_pos_gpt2xl_layer29_factorial_AB_fig.py
========================================================
Subset of make_curv_disp_pos_gpt2xl_layer29_factorial_fig.py: semantic-neural
r only (row 1), duration-window counts vs. duration-window rates (columns A, B).

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29_factorial_AB.{pdf,png,svg}
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
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position", "#2E9E6C")]
PANELS = [("dur_count", "A  Spike counts\n[onset+80, offset+40] window"),
          ("dur_rate", "B  Firing rates\n(count / window length)")]

d = json.load(open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_factorial.json")))
KS = d["config"]["KS"]
P = list(d["patients"].values())
x = np.arange(len(KS))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, (w, title) in zip(axes, PANELS):
    for m, lab, col in METRICS:
        v = np.array([[p[f"{w}|{K}"][m]["r"] for K in KS] for p in P])
        ax.errorbar(x, v.mean(axis=0), yerr=v.std(axis=0, ddof=1) / np.sqrt(len(P)),
                    color=col, marker="o", ms=4, lw=1.8, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {K}" if K != "full" else "full" for K in KS])
    ax.set_xlabel("Dimensions (matched top-K PCs, or full space)")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("Semantic–neural r")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle(f"gpt2-xl layer {d['config']['LAYER']}, per patient (n = {len(P)}), "
             f"chunk = {d['config']['CHUNK']}, mean ± SEM", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_pos_gpt2xl_layer29_factorial_AB"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
