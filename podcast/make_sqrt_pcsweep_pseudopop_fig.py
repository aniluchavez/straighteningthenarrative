"""
make_sqrt_pcsweep_pseudopop_fig.py
==================================
Figure for the pseudopop top-K PCA sweeps (raw GPT-2 XL L29 vs 2*sqrt(count)):
r(neural, semantic) vs K for curvature, displacement, position (= magnitude |y_t|),
two PCA versions:
  centered PCA   (origin = centroid -> |y_t| = distance from centroid)
  uncentered PCA (TruncatedSVD, origin = 0 -> |y_t| = magnitude)
Full space (no PCA) as a dashed line. Filled markers: circular-shift p < 0.05.

Reads: results/curv_disp_pos_sqrt_pcsweep_gpt2xl_layer29_pseudopop.json
       results/curv_disp_pos_sqrt_uncentered_pcsweep_gpt2xl_layer29_pseudopop.json
Saves: figures/sqrt_pcsweep_pseudopop_gpt2xl_layer29.png/.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE_DIR, "results")
FIG = os.path.join(BASE_DIR, "figures")
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {"centered PCA": ("curv_disp_pos_sqrt_pcsweep_gpt2xl_layer29_pseudopop.json", "#2a78d6", "o"),
          "uncentered PCA": ("curv_disp_pos_sqrt_uncentered_pcsweep_gpt2xl_layer29_pseudopop.json", "#eb6834", "s")}
PANELS = [("curv", "Curvature"), ("disp", "Displacement ‖x$_{t+1}$ − x$_t$‖"), ("mag", "Position = magnitude ‖y$_t$‖")]


def main():
    data = {name: json.load(open(os.path.join(R, f))) for name, (f, _, _) in SERIES.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=BG, sharey=True)
    for ax, (m, title) in zip(axes, PANELS):
        ax.set_facecolor(BG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, lw=0.8)
        full = None
        for name, (_, col, mk) in SERIES.items():
            byk = data[name]["by_k"]
            ks = sorted(int(k) for k in byk if k != "full")
            r = [byk[str(k)][m]["r"] for k in ks]
            p = [byk[str(k)][m]["p"] for k in ks]
            ax.plot(ks, r, "-", color=col, lw=2, label=name, zorder=2)
            for k, ri, pi in zip(ks, r, p):
                ax.scatter(k, ri, s=46, marker=mk, zorder=3, linewidths=1.4, edgecolors=col,
                           facecolors=col if pi < 0.05 else BG)
            full = byk["full"][m]["r"]
        ax.axhline(full, ls="--", color=INK, lw=1.2)
        ax.text(ks[0], full, f" full space r = {full:+.3f}", va="top", fontsize=8, color=INK)
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("number of PCs K (same in both spaces)", fontsize=9, color=INK2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("r(neural, semantic), 3-word chunks", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[2].text(0.02, 0.02, "centered: ‖y$_t$‖ = distance from centroid\nuncentered: ‖y$_t$‖ = distance from 0",
                 transform=axes[2].transAxes, fontsize=7.5, color=INK2, va="bottom")
    fig.suptitle("Pseudopop (12 patients, 436 units): GPT-2 XL L29 raw vs hippocampus 2√count — top-K PCs vs full space"
                 "   (filled = circular-shift p < 0.05)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    base = os.path.join(FIG, "sqrt_pcsweep_pseudopop_gpt2xl_layer29")
    fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".pdf")
    print("saved", base + ".png")


if __name__ == "__main__":
    main()
