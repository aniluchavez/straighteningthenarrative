"""
make_simulate_poisson_llm_raw_fig.py
=====================================
Raw-counts panel only from simulate_poisson_llm_geometry.py results: one panel
per tuning depth beta; solid = Poisson counts (mean ± SD over seeds), dashed =
noise-free expected counts (lambda). Position plotted as |r| (PC1 sign is
arbitrary, so the signed r flips between seeds).

Saves: figures/fig_simulate_poisson_llm_raw.{pdf,png,svg}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 10, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position |r|", "#2E9E6C")]

d = json.load(open(os.path.join(BASE_DIR, "results", "simulate_poisson_llm_geometry.json")))
KS = d["Ks"]
x = np.arange(len(KS))
fig, axes = plt.subplots(1, len(d["betas"]), figsize=(4.6 * len(d["betas"]), 4.4), sharey=True)
for ax, beta, letter in zip(axes, d["betas"], "ABC"):
    R = d["results"][str(beta)]
    for m, lab, col in METRICS:
        M = np.array(R["noisy"]["raw"][m])
        nf = np.array(R["noisefree"]["raw"][m])
        if m == "pos":
            M, nf = np.abs(M), np.abs(nf)
        ax.errorbar(x, M.mean(0), yerr=M.std(0, ddof=1), color=col, marker="o", ms=4, lw=1.8, capsize=3,
                    label=f"{lab}, Poisson counts")
        ax.plot(x, nf, "--", color=col, lw=1.2, label=f"{lab}, noise-free λ")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {K}" if K != "full" else "full" for K in KS])
    ax.set_xlabel("Dimensions (top-K PCs, or full space)")
    ax.set_title(f"{letter}  Tuning depth β = {beta}", loc="left")
axes[0].set_ylabel("Simulated neurons – gpt2-xl r")
axes[0].legend(frameon=False, loc="lower left", fontsize=7.5)
fig.suptitle(f"Raw counts: simulated Poisson neurons driven by gpt2-xl layer 29 ({d['n_units']} units, real mean "
             f"counts; {d['n_words']} words, chunk = 3; mean ± SD over {d['n_seed']} seeds)", x=0.01, ha="left",
             fontsize=12)
fig.tight_layout()
stem = os.path.join(BASE_DIR, "figures", "fig_simulate_poisson_llm_raw")
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{stem}.{ext}")
print(f"  -> {stem}.{{pdf,png,svg}}")
for beta in d["betas"]:
    R = d["results"][str(beta)]
    print(f"beta={beta}: " + "  ".join(
        f"{K}: c {np.mean(R['noisy']['raw']['curv'], 0)[i]:+.2f}[{R['noisefree']['raw']['curv'][i]:+.2f}] "
        f"d {np.mean(R['noisy']['raw']['disp'], 0)[i]:+.2f}[{R['noisefree']['raw']['disp'][i]:+.2f}] "
        f"|p| {np.mean(np.abs(R['noisy']['raw']['pos']), 0)[i]:.2f}" for i, K in enumerate(KS)))
