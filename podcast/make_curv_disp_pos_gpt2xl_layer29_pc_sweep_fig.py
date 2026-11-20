"""
make_curv_disp_pos_gpt2xl_layer29_pc_sweep_fig.py
====================================================
Plots results/curv_disp_pos_gpt2xl_layer29_pc_sweep.json
(compute_curv_disp_pos_gpt2xl_layer29_pc_sweep.py):

  A  random 8-of-50 PC draws: distribution of group-mean r per metric,
     top-8 reference as dashed line
  B  per-PC inclusion effect, neural side (PC1..20)
  C  per-PC inclusion effect, semantic side (PC1..20)
  D-F  PC dropout (PCs j+1..j+8): neural-only, semantic-only, both;
       mean +/- SEM across patients

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29_pc_sweep.{pdf,png,svg}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = [("pos", "Position", "#2E9E6C"),
           ("disp", "Displacement", "#5B8FD4"),
           ("curv", "Curvature", "#E67E22")]
N_INCL = 20

d = json.load(open(os.path.join(BASE_DIR, "results", "curv_disp_pos_gpt2xl_layer29_pc_sweep.json")))
cfg = d["config"]
draws = d["random"]["draws"]
incl = d["random"]["inclusion_effect"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

# A: random-draw distributions
ax = axes[0, 0]
allv = np.concatenate([[dr["group"][k]["mean"] for dr in draws] for k, _, _ in METRICS])
bins = np.linspace(allv.min(), max(allv.max(), d["top8"]["group"]["curv"]["mean"]) + 0.005, 45)
for k, lab, col in METRICS:
    v = [dr["group"][k]["mean"] for dr in draws]
    ax.hist(v, bins=bins, color=col, alpha=0.55, label=lab)
    ax.axvline(d["top8"]["group"][k]["mean"], color=col, ls="--", lw=1.5)
ax.axvline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel("Group-mean r (13 patients)")
ax.set_ylabel("Draws")
ax.set_title(f"A  Random {cfg['K']} of {cfg['K_POOL']} PCs, {len(draws)} draws\n(dashed = top-{cfg['K']} PCs)",
             loc="left")
ax.legend(frameon=False)

# B, C: inclusion effects
x = np.arange(1, N_INCL + 1)
w = 0.27
for ax, side, title in ((axes[0, 1], "neu", "B  Inclusion effect, neural PCs"),
                        (axes[0, 2], "sem", "C  Inclusion effect, semantic PCs")):
    for i, (k, lab, col) in enumerate(METRICS):
        ax.bar(x + (i - 1) * w, incl[side][k][:N_INCL], width=w, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(j) if j in (1, 5, 10, 15, 20) else "" for j in x])
    ax.set_xlabel("PC index")
    ax.set_ylabel("r(with PC) − r(without PC)")
    ax.set_title(title, loc="left")
axes[0, 1].legend(frameon=False)
lo = min(axes[0, 1].get_ylim()[0], axes[0, 2].get_ylim()[0])
hi = max(axes[0, 1].get_ylim()[1], axes[0, 2].get_ylim()[1])
for ax in axes[0, 1:]:
    ax.set_ylim(lo, hi)

# D-F: dropout
for ax, scheme, title in ((axes[1, 0], "neu", "D  Drop top neural PCs\n(semantic fixed at top 8)"),
                          (axes[1, 1], "sem", "E  Drop top semantic PCs\n(neural fixed at top 8)"),
                          (axes[1, 2], "both", "F  Drop top PCs on both sides")):
    rows = d["dropout"][scheme]
    off = np.array([r["offset"] for r in rows])
    for k, lab, col in METRICS:
        mu = np.array([r["group"][k]["mean"] for r in rows])
        se = np.array([r["group"][k]["sem"] for r in rows])
        ax.plot(off, mu, "-o", color=col, lw=2, ms=3.5, label=lab)
        ax.fill_between(off, mu - se, mu + se, color=col, alpha=0.2, lw=0)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(f"PCs dropped (uses PCs j+1..j+{cfg['K']})")
    ax.set_ylabel("Mean r ± SEM")
    ax.set_title(title, loc="left")
    if scheme != "sem":
        n = np.array([r["group"]["curv"]["n"] for r in rows])
        cut = off[np.argmax(n < 13)] if (n < 13).any() else None
        if cut is not None:
            ax.axvspan(cut - 0.5, off[-1] + 0.5, color="0.5", alpha=0.1, lw=0)
            ax.text(cut, ax.get_ylim()[1], " n<13 patients\n (neural pool exhausted)",
                    va="top", ha="left", fontsize=9, color="0.35")
axes[1, 0].legend(frameon=False, loc="upper right")

fig.suptitle(f"Curvature / displacement / position vs. PC choice — gpt2-xl layer {cfg['LAYER']}, "
             f"chunk={cfg['CHUNK_SIZE']}, dur80/40, 13 patients", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_pos_gpt2xl_layer29_pc_sweep"
for ext in ("pdf", "png", "svg"):
    path = os.path.join(FIGURES, f"{stem}.{ext}")
    fig.savefig(path)
    print(f"  -> {path}")
