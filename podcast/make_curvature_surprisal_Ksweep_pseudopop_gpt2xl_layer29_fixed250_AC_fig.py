"""
make_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29_fixed250_AC_fig.py
==============================================================================
Panels A and C of fig_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29_fixed250,
replotted from results/curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29_fixed250.json
with clearer labels. Neural = pseudopopulation, fixed [onset+80, onset+330]
window spike counts; semantic = gpt2-xl layer 29.

Saves: figures/fig_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29_fixed250_AC.{pdf,png,svg}
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

COL = dict(sem="#5B8FD4", neu="#E67E22")
LAB = dict(sem="Semantic (gpt2-xl layer 29)", neu="Neural pseudopopulation (475 units)")

res = json.load(open(os.path.join(RESULTS, "curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29_fixed250.json")))
n_valid = np.load(os.path.join(RESULTS, "pseudopop_full_sequence_fixed250_count.npz"))["valid"].sum()

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax = axes[0]
for side in ("sem", "neu"):
    R = res[side]
    ks = [int(k) for k in R if k != "full"]
    x = np.array(ks)
    ax.fill_between(x, [R[str(k)]["lo"] for k in ks], [R[str(k)]["hi"] for k in ks], color=COL[side], alpha=0.2, lw=0)
    ax.plot(x, [R[str(k)]["rho"] for k in ks], "-o", color=COL[side], ms=4, label=LAB[side])
    ax.axhline(R["full"]["rho"], color=COL[side], ls="--", lw=1)
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks([2, 8, 32, 128, 475, 1600])
ax.set_xticklabels([2, 8, 32, 128, 475, 1600])
ax.set_xlabel("Number of top PCs used to build the trajectory\n(dashed line = full space, no PCA)")
ax.set_ylabel("Spearman ρ between a word's surprisal\nand the trajectory's turning angle at that word")
ax.set_title("A  Does the trajectory turn more sharply at surprising words?", loc="left")
ax.text(0.02, 0.26,
        "ρ > 0: more surprising words → sharper turns\nρ < 0: more surprising words → straighter path\n"
        "Band = 95% bootstrap CI",
        transform=ax.transAxes, fontsize=8.5, va="bottom", color="0.3")
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
x = np.arange(1, 11)
for side in ("sem", "neu"):
    ax.plot(x, res[side]["full"]["deciles"], "-o", color=COL[side], ms=4.5, label=f"{LAB[side]}, full space")
ax.set_xticks(x)
ax.set_xlabel("Surprisal decile\n(1 = 10% least surprising words, 10 = 10% most surprising)")
ax.set_ylabel("Mean turning angle at the word (°)")
ax.set_title("C  Turning angle by surprisal, full space", loc="left")
ax.legend(frameon=False, loc="upper left")

fig.suptitle(f"Curvature vs. surprisal — neural: fixed 250 ms window [onset+80, onset+330] spike counts; "
             f"{int(n_valid)} valid words common to all 13 patients", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29_fixed250_AC"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
