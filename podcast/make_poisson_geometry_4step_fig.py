"""
make_poisson_geometry_4step_fig.py
====================================
4-step normalization ablation (compute_poisson_normalized_geometry.py ... four):
raw counts -> sqrt only -> L2 only -> sqrt + L2, window [onset+80, offset+40],
semantic = gpt2-xl layer 29 raw embedding, chunk = 3, vs. top-K PCs / full.
Style follows make_curv_disp_pos_gpt2xl_layer29_factorial_AB_fig.py.

Figure 1 (perpatient): mean ± SEM across patients.
Figure 2 (pseudopop): single r per K; filled marker = circular-shift p < 0.05,
                      open marker = n.s.

Saves: figures/fig_poisson_geometry_4step_{perpatient,pseudopop}.{pdf,png,svg}
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
VARIANTS = [("raw", "A  Raw spike counts"),
            ("sqrt", "B  Square root  √k"),
            ("l2", "C  L2-normalized  k / ‖k‖"),
            ("sqrt_l2", "D  Square root + L2  √k / ‖√k‖")]
WINDOW = sys.argv[1] if len(sys.argv) > 1 else "dur8040"
RATE = len(sys.argv) > 2 and sys.argv[2] == "rate"
RT = "_rate" if RATE else ""
WIN = {"dur8040": "[onset+80, offset+40]", "fixed250": "[onset+80, onset+330]",
       "fixed300": "[onset+80, onset+380]"}[WINDOW] + (" window, firing rates (count / window length)" if RATE else " window, spike counts")
if RATE:
    VARIANTS = [("raw", "A  Raw firing rates  λ = k / T"), ("sqrt", "B  Square root  √λ"),
                ("l2", "C  L2-normalized  λ / ‖λ‖"), ("sqrt_l2", "D  Square root + L2  √λ / ‖√λ‖")]
x = np.arange(len(KS))
XT = [f"top {K}" if K != "full" else "full" for K in KS]


def load(mode, K):
    tag = "full" if K == "full" else f"pc{K}"
    return json.load(open(os.path.join(RESULTS, f"poisson_norm_geometry_4step{RT}_{mode}_{WINDOW}_{tag}.json")))


def finish(fig, axes, mode, stem, title):
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(XT)
        ax.set_xlabel("Dimensions (top-K PCs, or full space)")
    axes[0].set_ylabel("Semantic–neural r")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


# ---------------- per patient
D = {K: load("perpatient", K)["per_patient"] for K in KS}
pids = sorted(D["full"])
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
for ax, (v, title) in zip(axes, VARIANTS):
    for m, lab, col in METRICS:
        M = np.array([[D[K][p]["r"][f"{v}|raw"][m] for K in KS] for p in pids], float)
        se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.isfinite(M).sum(axis=0))
        ax.errorbar(x, np.nanmean(M, axis=0), yerr=se, color=col, marker="o", ms=4, lw=1.8, capsize=3, label=lab)
    ax.set_title(title, loc="left")
finish(fig, axes, "perpatient", "fig_poisson_geometry_4step_perpatient" + ("" if WINDOW == "dur8040" else f"_{WINDOW}") + RT,
       f"Per patient (n = {len(pids)}), gpt2-xl layer 29, {WIN}, chunk = 3, mean ± SEM")

# ---------------- pseudopop
P = {K: load("pseudopop", K) for K in KS}
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
for ax, (v, title) in zip(axes, VARIANTS):
    for m, lab, col in METRICS:
        r = np.array([P[K]["coupling"][f"{v}|raw"][m]["r"] for K in KS])
        p = np.array([P[K]["coupling"][f"{v}|raw"][m]["p"] for K in KS])
        ax.plot(x, r, color=col, lw=1.8, label=lab)
        ax.scatter(x[p < 0.05], r[p < 0.05], color=col, s=30, zorder=3)
        ax.scatter(x[p >= 0.05], r[p >= 0.05], facecolor="white", edgecolor=col, s=30, lw=1.4, zorder=3)
    ax.set_title(title, loc="left")
n_units = P["full"]["n_units"]
finish(fig, axes, "pseudopop", "fig_poisson_geometry_4step_pseudopop" + ("" if WINDOW == "dur8040" else f"_{WINDOW}") + RT,
       f"Pseudopopulation ({n_units} units, {len(P['full']['patients'])} patients), gpt2-xl layer 29, {WIN}, "
       f"chunk = 3; filled = circular-shift p < 0.05, open = n.s.")
