"""
make_poisson_normalized_geometry_pc_sweep_fig.py
==================================================
Same layout/style as make_curv_disp_pos_gpt2xl_layer29_factorial_AB_fig.py,
for compute_poisson_normalized_geometry.py results: per-patient semantic-neural
r (chunk 3, mean ± SEM, n = 13) vs. top-K PCs / full space, one panel per
neural normalization; rows = window (dur80/40, fixed 250 ms).

argv[1] = semantic variant: raw (default) | sphere_tan

Saves: figures/fig_poisson_norm_geometry_pc_sweep_sem{raw|sphere_tan}.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")
SEM = sys.argv[1] if len(sys.argv) > 1 else "raw"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position", "#2E9E6C")]
KS = ["8", "16", "32", "64", "128", "full"]
VARIANTS = [("z", "z-scored counts\n(previous pipeline)"),
            ("sqrt", "√(k + 3/8)"),
            ("sphere_euc", "√ → L2-normalized\nEuclidean"),
            ("sphere_tan", "√ → L2-normalized\ntangent curvature"),
            ("sphere_tan_cv", "√ → L2 + tangent,\nthinning-debiased disp.")]
WINDOWS = [("dur8040", "[onset+80, offset+40]"), ("fixed250", "[onset+80, onset+330]")]


def load(window, K):
    tag = "" if K == "full" else f"_pc{K}"
    return json.load(open(os.path.join(RESULTS, f"poisson_norm_geometry_perpatient_{window}{tag}.json")))["per_patient"]


data = {(w, K): load(w, K) for w, _ in WINDOWS for K in KS}
pids = sorted(data[("dur8040", "full")])
x = np.arange(len(KS))

fig, axes = plt.subplots(len(WINDOWS), len(VARIANTS), figsize=(3.3 * len(VARIANTS), 3.9 * len(WINDOWS)),
                         sharey=True)
letters = iter("ABCDEFGHIJ")
for ri, (w, wlab) in enumerate(WINDOWS):
    for ci, (v, vlab) in enumerate(VARIANTS):
        ax = axes[ri, ci]
        pair = f"{v}|{SEM}"
        for m, lab, col in METRICS:
            M = np.array([[data[(w, K)][p]["r"][pair][m] for K in KS] for p in pids], float)
            mu = np.nanmean(M, axis=0)
            se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.isfinite(M).sum(axis=0))
            ax.errorbar(x, mu, yerr=se, color=col, marker="o", ms=4, lw=1.8, capsize=3, label=lab)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"top {K}" if K != "full" else "full" for K in KS])
        ax.set_title(f"{next(letters)}  {vlab}", loc="left", fontsize=10)
        if ri == len(WINDOWS) - 1:
            ax.set_xlabel("Dimensions (top-K PCs, or full space)")
    axes[ri, 0].set_ylabel(f"Semantic–neural r\n{wlab} window")
axes[0, 0].legend(frameon=False, loc="upper left")
fig.suptitle(f"Poisson-normalized neural geometry vs. gpt2-xl layer 29 (semantic: {SEM}), per patient "
             f"(n = {len(pids)}), chunk = 3, spike counts, mean ± SEM", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = f"fig_poisson_norm_geometry_pc_sweep_sem{SEM}"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
