"""
make_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop_fig.py
==============================================================
Plots results/curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.json
(compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.py).

Row 1  A  top-K r vs K (error bars = circular-shift null 95%)
       B  random K-of-200 draws: r distribution per K and metric
       C  curvature per-PC inclusion effect (PC1-20), semantic vs neural, per K
Row 2  curvature PC dropout per K: neural-only / semantic-only / both
Row 3  all three metrics PC dropout at K=REF_K, with null 95% bands

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.{pdf,png,svg}
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
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = [("pos", "Position", "#2E9E6C"),
           ("disp", "Displacement", "#5B8FD4"),
           ("curv", "Curvature", "#E67E22")]
REF_K = 32
N_INCL = 20

d = json.load(open(os.path.join(BASE_DIR, "results", "curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.json")))
cfg = d["config"]
Ks = [int(k) for k in d["byK"]]
kcol = {K: plt.cm.Oranges(v) for K, v in zip(Ks, np.linspace(0.35, 1.0, len(Ks)))}
SCHEMES = [("neu", "Drop top neural PCs\n(semantic fixed at top K)"),
           ("sem", "Drop top semantic PCs\n(neural fixed at top K)"),
           ("both", "Drop top PCs on both sides")]

fig, axes = plt.subplots(3, 3, figsize=(16, 13))

# A: top-K vs K
ax = axes[0, 0]
xk = np.arange(len(Ks))
for i, (m, lab, col) in enumerate(METRICS):
    r = np.array([d["byK"][str(K)]["top"][m]["r"] for K in Ks])
    lo = np.array([d["byK"][str(K)]["top"][m]["null_lo"] for K in Ks])
    hi = np.array([d["byK"][str(K)]["top"][m]["null_hi"] for K in Ks])
    off = (i - 1) * 0.18
    ax.vlines(xk + off, lo, hi, color=col, alpha=0.35, lw=6)
    ax.plot(xk + off, r, "o-", color=col, lw=2, ms=6, label=lab)
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xticks(xk); ax.set_xticklabels(Ks)
ax.set_xlabel("K (top PCs per side)")
ax.set_ylabel("r")
ax.set_title("A  Top-K PCs\n(bars = circular-shift null 95%)", loc="left")
ax.legend(frameon=False)

# B: random-draw distributions
ax = axes[0, 1]
for i, (m, lab, col) in enumerate(METRICS):
    vals = [[dr[m] for dr in d["byK"][str(K)]["random"]["draws"]] for K in Ks]
    bp = ax.boxplot(vals, positions=xk + (i - 1) * 0.25, widths=0.2, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=1))
    for b in bp["boxes"]:
        b.set_facecolor(col); b.set_alpha(0.7); b.set_edgecolor(col)
    for part in ("whiskers", "caps"):
        for w in bp[part]:
            w.set_color(col)
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xticks(xk); ax.set_xticklabels(Ks)
ax.set_xlabel("K (random PCs per side, from top 200)")
ax.set_ylabel("r")
ax.set_title(f"B  Random K of {cfg['K_POOL']} PCs, {cfg['N_DRAWS']} draws per K", loc="left")

# C: curvature inclusion effect
ax = axes[0, 2]
x = np.arange(1, N_INCL + 1)
for K in Ks:
    inc = d["byK"][str(K)]["random"]["inclusion_effect"]
    ax.plot(x, inc["sem"]["curv"][:N_INCL], "-o", color=kcol[K], lw=1.6, ms=3.5, label=f"K={K}")
    ax.plot(x, inc["neu"]["curv"][:N_INCL], "--", color=kcol[K], lw=1.2, alpha=0.8)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("PC index")
ax.set_ylabel("curvature r(with PC) − r(without PC)")
ax.set_title("C  Curvature inclusion effect\n(solid = semantic PC, dashed = neural PC)", loc="left")
ax.legend(frameon=False, ncol=2)

# Row 2: curvature dropout by K
for ax, (scheme, title) in zip(axes[1], SCHEMES):
    nlo, nhi = [], []
    for K in Ks:
        rows = d["byK"][str(K)]["dropout"][scheme]
        off = [r["offset"] for r in rows]
        ax.plot(off, [r["curv"]["r"] for r in rows], "-", color=kcol[K], lw=1.6, label=f"K={K}")
        nlo.append(np.mean([r["curv"]["null_lo"] for r in rows]))
        nhi.append(np.mean([r["curv"]["null_hi"] for r in rows]))
    ax.axhspan(np.mean(nlo), np.mean(nhi), color="0.5", alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("PCs dropped (uses PCs j+1..j+K)")
    ax.set_ylabel("Curvature r")
    ax.set_title(f"{'DEF'[SCHEMES.index((scheme, title))]}  Curvature: {title}", loc="left")
axes[1, 0].legend(frameon=False, loc="lower left")
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(lo, hi)

# Row 3: all metrics at REF_K
for ax, (scheme, title) in zip(axes[2], SCHEMES):
    rows = d["byK"][str(REF_K)]["dropout"][scheme]
    off = np.array([r["offset"] for r in rows])
    for m, lab, col in METRICS:
        ax.fill_between(off, [r[m]["null_lo"] for r in rows], [r[m]["null_hi"] for r in rows],
                        color=col, alpha=0.12, lw=0)
        ax.plot(off, [r[m]["r"] for r in rows], "-", color=col, lw=1.6, label=lab)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(f"PCs dropped (uses PCs j+1..j+{REF_K})")
    ax.set_ylabel("r")
    ax.set_title(f"{'GHI'[SCHEMES.index((scheme, title))]}  K={REF_K}: {title}", loc="left")
axes[2, 0].legend(frameon=False, loc="lower left")
lo = min(a.get_ylim()[0] for a in axes[2]); hi = max(a.get_ylim()[1] for a in axes[2])
for a in axes[2]:
    a.set_ylim(lo, hi)

fig.suptitle(f"Curvature / displacement / position vs. PC choice — pseudopopulation "
             f"({cfg['n_channels']} hippocampal channels, 13 patients, {cfg['n_valid']} words), "
             f"gpt2-xl layer {cfg['LAYER']}, chunk={cfg['CHUNK_SIZE']}, dur80/40 rate",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop"
for ext in ("pdf", "png", "svg"):
    path = os.path.join(FIGURES, f"{stem}.{ext}")
    fig.savefig(path)
    print(f"  -> {path}")
