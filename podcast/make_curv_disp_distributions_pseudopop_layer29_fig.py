"""
make_curv_disp_distributions_pseudopop_layer29_fig.py
========================================================
Distributions of per-word curvature (deg) and displacement on valid words,
pseudopopulation neural (results/pseudopop_full_sequence_dur8040_rate.npz,
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.py) and gpt2-xl
layer 29 semantic, for top-K PCs at each K. Semantic side also shown with
PC1 dropped (PCs 2..K+1).

Saves: figures/fig_curv_disp_distributions_pseudopop_layer29.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from podcast_loader import load_podcast_patient
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import (
    CACHE_PATH, REF_PID, REGION, MODEL, LAYER, KS, pca_pool, features,
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

c = np.load(CACHE_PATH)
neu_z, common, valid = c["neu_z"], c["common"], c["valid"]
data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
emb = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
Ys, _ = pca_pool(emb)
Yn, _ = pca_pool(neu_z)

kcol = {K: plt.cm.Oranges(v) for K, v in zip(KS, np.linspace(0.35, 1.0, len(KS)))}
feats = {}
for K in KS:
    feats[("sem", K)] = features(Ys[:, :K], valid)
    feats[("sem_noPC1", K)] = features(Ys[:, 1:K + 1], valid)
    feats[("neu", K)] = features(Yn[:, :K], valid)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
cols = [("sem", "Semantic (gpt2-xl L29), top-K PCs"),
        ("sem_noPC1", "Semantic, PC1 dropped (PCs 2..K+1)"),
        ("neu", "Neural pseudopop, top-K PCs")]
summary = {}
for j, (side, title) in enumerate(cols):
    ax = axes[0, j]
    for K in KS:
        v = feats[(side, K)]["curv"]
        ax.hist(v, bins=np.linspace(0, 180, 61), histtype="step", lw=1.8, color=kcol[K], density=True,
                label=f"K={K}  median {np.median(v):.0f}°")
    ax.axvline(90, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Curvature (deg)")
    ax.set_ylabel("Density")
    ax.set_title(f"{'ABC'[j]}  {title}", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, j]
    for K in KS:
        v = feats[(side, K)]["disp"]
        v = v / np.median(v)
        ax.hist(v, bins=np.linspace(0, 4, 61), histtype="step", lw=1.8, color=kcol[K], density=True,
                label=f"K={K}  CV {feats[(side, K)]['disp'].std() / feats[(side, K)]['disp'].mean():.2f}")
    ax.set_xlabel("Displacement / median displacement")
    ax.set_ylabel("Density")
    ax.set_title(f"{'DEF'[j]}  {title}", loc="left")
    ax.legend(frameon=False, loc="upper right")
    for K in KS:
        f = feats[(side, K)]
        summary[f"{side}_K{K}"] = dict(curv_median=float(np.median(f["curv"])), curv_sd=float(f["curv"].std()),
                                       disp_median=float(np.median(f["disp"])),
                                       disp_cv=float(f["disp"].std() / f["disp"].mean()))

for row in axes:
    hi = max(a.get_ylim()[1] for a in row)
    for a in row:
        a.set_ylim(0, hi)

fig.suptitle(f"Per-word curvature and displacement distributions vs. K — pseudopopulation "
             f"({neu_z.shape[1]} ch, {int(valid.sum())} valid words), gpt2-xl layer {LAYER}",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_distributions_pseudopop_layer29"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
json.dump(summary, open(os.path.join(RESULTS, "curv_disp_distributions_pseudopop_layer29.json"), "w"), indent=1)
for k, v in summary.items():
    print(f"  {k:16s} curv median={v['curv_median']:6.1f} sd={v['curv_sd']:5.1f}   "
          f"disp median={v['disp_median']:7.2f} CV={v['disp_cv']:.2f}")
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
