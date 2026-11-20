"""
make_pc_distribution_ridgeline_pseudopop_layer29_fig.py
==========================================================
How the trajectory geometry of each space changes as PCs are added.
Rows: semantic (gpt2-xl layer 29) and neural pseudopopulation
(results/pseudopop_full_sequence_dur8040_rate.npz, from
compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.py).
Columns:
  A/D  cumulative variance explained by the top-K PCs
  B/E  per-word curvature (deg) on valid words, top-K PCs, one ridge per K
  C/F  per-word displacement / median displacement, one ridge per K
Ridges are Gaussian KDEs; white line / black tick = median.

Saves: figures/fig_pc_distribution_ridgeline_pseudopop_layer29.{pdf,png,svg}
"""
import os
import sys

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from podcast_loader import load_podcast_patient
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import (
    CACHE_PATH, REF_PID, REGION, MODEL, LAYER, pca_pool, features,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

KS = [2, 4, 8, 16, 32, 64, 128, 200]
OVERLAP = 1.6

c = np.load(CACHE_PATH)
neu_z, common, valid = c["neu_z"], c["common"], c["valid"]
data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
emb = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
spaces = [("Semantic (gpt2-xl L29)", plt.cm.Blues, *pca_pool(emb)),
          ("Neural pseudopop", plt.cm.Oranges, *pca_pool(neu_z))]


def ridgeline(ax, data, grid, cmap, xlabel):
    cols = cmap(np.linspace(0.4, 0.95, len(KS)))
    for i, (K, v) in enumerate(zip(KS, data)):
        y0 = (len(KS) - 1 - i)
        d = gaussian_kde(v, bw_method=0.12)(grid)
        d = OVERLAP * d / d.max()
        ax.fill_between(grid, y0, y0 + d, color=cols[i], alpha=1.0, lw=0, zorder=2 * i)
        ax.plot(grid, y0 + d, color="k", lw=0.6, zorder=2 * i + 1)
        m = np.median(v)
        ax.plot([m, m], [y0, y0 + np.interp(m, grid, d)], color="w", lw=1.5, zorder=2 * i + 1)
        ax.plot([m], [y0], marker="|", ms=8, mew=1.5, color="k", zorder=2 * i + 1)
    ax.set_yticks(np.arange(len(KS))[::-1])
    ax.set_yticklabels([f"K={K}" for K in KS])
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.1, len(KS) - 1 + OVERLAP + 0.1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)


fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
for r, (name, cmap, Y, evr) in enumerate(spaces):
    ks = [K for K in KS if K <= Y.shape[1]]
    feats = [features(Y[:, :K], valid) for K in ks]

    ax = axes[r, 0]
    cum = np.cumsum(evr) * 100
    ax.plot(np.arange(1, len(cum) + 1), cum, color=cmap(0.8), lw=2)
    ax.scatter(ks, cum[np.array(ks) - 1], color=cmap(np.linspace(0.4, 0.95, len(KS))), ec="k", lw=0.5, s=40, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 200])
    ax.set_xticklabels([1, 4, 16, 64, 200])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of PCs (K)")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title(f"{'AD'[r]}  {name}\nPC1 = {evr[0] * 100:.1f}%", loc="left")

    ridgeline(axes[r, 1], [f["curv"] for f in feats], np.linspace(0, 180, 361), cmap, "Curvature (deg)")
    axes[r, 1].axvline(90, color="k", lw=0.6, ls=":", zorder=100)
    axes[r, 1].set_title(f"{'BE'[r]}  Curvature", loc="left")
    ridgeline(axes[r, 2], [f["disp"] / np.median(f["disp"]) for f in feats], np.linspace(0, 3.5, 351), cmap,
              "Displacement / median")
    axes[r, 2].axvline(1, color="k", lw=0.6, ls=":", zorder=100)
    axes[r, 2].set_title(f"{'CF'[r]}  Displacement", loc="left")
    for K, f in zip(ks, feats):
        print(f"  {name:24s} K={K:>3}  cumvar={cum[K - 1]:5.1f}%  curv med={np.median(f['curv']):6.1f}  "
              f"disp CV={f['disp'].std() / f['disp'].mean():.2f}")

fig.suptitle(f"Trajectory geometry as PCs are added — pseudopopulation ({neu_z.shape[1]} ch, "
             f"{int(valid.sum())} valid words), gpt2-xl layer {LAYER}", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_pc_distribution_ridgeline_pseudopop_layer29"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
