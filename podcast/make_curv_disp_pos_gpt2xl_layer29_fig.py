"""
make_curv_disp_pos_gpt2xl_layer29_fig.py
===========================================
Paired bar+strip figure (curvature/displacement/position), PC8 matched,
chunk=3, dur80/40, gpt2-xl LAYER 29 only (the RSA-peak layer), all 13
podcast patients. Same style as make_tentative_paired_pc8_chunk3.py's
all-layers/all-models version, but single model + single layer.

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29.{pdf,png,svg}
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

data = json.load(open(os.path.join(BASE_DIR, "results", "curv_disp_pos_gpt2xl_layer29.json")))

per_patient = {"curv": [], "disp": [], "pos": []}
for pid, rec in data.items():
    for metric in ["curv", "disp", "pos"]:
        v = rec[metric]
        if v is not None and np.isfinite(v):
            per_patient[metric].append(v)

METRICS = [("pos", "Position", "#2E9E6C"),
           ("disp", "Displacement (d)", "#5B8FD4"),
           ("curv", "Curvature ($\\kappa$)", "#E67E22")]

fig, ax = plt.subplots(figsize=(7, 6.2))
xpos = np.arange(3)
all_lo, all_hi = [], []
for i, (key, label, color) in enumerate(METRICS):
    vals = np.array(per_patient[key])
    mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals))
    t, p = ttest_1samp(vals, 0.0)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.bar(xpos[i], mu, yerr=se, color=color, alpha=0.55, width=0.6, capsize=4)
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(vals)) - 0.5) * 0.18
    ax.scatter(xpos[i] + jitter, vals, color=color, edgecolor="white", s=34, zorder=3)
    ytop = max(vals.max(), mu + se) + 0.02
    ax.text(xpos[i], ytop, f"{sig}\n(p={p:.1e})", ha="center", fontsize=11)
    all_lo.append(vals.min()); all_hi.append(ytop + 0.03)
    print(f"{key}: mean={mu:+.4f} p={p:.4e} n={len(vals)}")

ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xticks(xpos); ax.set_xticklabels([m[1] for m in METRICS])
ax.set_ylabel("Semantic $\\leftrightarrow$ neural Pearson $r$")
ax.set_ylim(min(all_lo) - 0.03, max(all_hi))
ax.set_title("Cross-space geometry alignment (PC8, matched)\n"
              "dur80/40 window, 13 patients, 3-word chunks, gpt2-xl LAYER 29 only",
              loc="left", pad=14, fontsize=13.5)
fig.tight_layout()

os.makedirs(FIGURES, exist_ok=True)
for ext in ("pdf", "png", "svg"):
    path = os.path.join(FIGURES, f"fig_curv_disp_pos_gpt2xl_layer29.{ext}")
    fig.savefig(path)
    print(f"  -> {path}")
