"""
make_curvature_dominance_fig.py
==================================
Two figures showing curvature-vs-displacement dominance across the
spike-integration window sweep (WIN_SIZES 100-2000ms), chunk=15, all 9
models averaged across all layers, raw r (matches the official
fig_cross_space_paired.pdf methodology, just varying window instead of
chunk size):

  fig_curvature_dominance_convo.pdf  -- orig10 vs new4, two panels
  fig_curvature_dominance_podcast.pdf -- single panel, all 9 podcast patients

Data hardcoded from compute_{orig10,new4,podcast}_window_scan_fast.py logs
(no permutation null in this scan -- raw r comparison only, see session
notes for the significance caveat on any single point).

Run:
    conda run -n gpt2_embed python3 make_curvature_dominance_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         13,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

C_CURV = "#D95319"
C_DISP = "#5E7CE2"

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]

# ---- hard-coded data from the window-scan logs ----
ORIG10 = dict(
    curv=[0.0491, 0.0794, 0.0918, 0.1357, 0.1029, 0.1035, 0.0542, 0.0243],
    disp=[0.0129, 0.0496, 0.0663, 0.0890, 0.0975, 0.0894, 0.0949, 0.0905],
)
NEW4 = dict(
    curv=[0.0487, 0.0370, 0.0587, 0.0275, 0.0147, 0.0118, -0.0046, -0.0166],
    disp=[0.0185, 0.0492, 0.0816, 0.0982, 0.1003, 0.1127, 0.1116, 0.1074],
)
PODCAST = dict(
    curv=[0.0279, 0.0418, 0.0289, 0.0226, 0.0241, 0.0339, 0.0096, 0.0104],
    disp=[0.0139, 0.0651, 0.0955, 0.1292, 0.1402, 0.1490, 0.1478, 0.1439],
)


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  -> {p}")
    plt.close(fig)


def _plot_panel(ax, data, title, show_legend=False):
    xs = np.array(WIN_SIZES, dtype=float)
    curv = np.array([v if v is not None else np.nan for v in data["curv"]])
    disp = np.array([v if v is not None else np.nan for v in data["disp"]])

    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.plot(xs, curv, color=C_CURV, marker="o", ms=4.5, label="Curvature (κ)")
    ax.plot(xs, disp, color=C_DISP, marker="o", ms=4.5, label="Displacement (d)")

    # shade the region where curvature wins
    both = np.isfinite(curv) & np.isfinite(disp)
    ax.fill_between(xs, curv, disp, where=(curv > disp) & both,
                     color=C_CURV, alpha=0.12, interpolate=True)
    ax.fill_between(xs, curv, disp, where=(curv <= disp) & both,
                     color=C_DISP, alpha=0.12, interpolate=True)

    ax.set_xscale("log")
    ax.set_xticks(xs[np.isfinite(curv) | np.isfinite(disp)])
    ax.set_xticklabels([f"{int(w)}" for w in xs[np.isfinite(curv) | np.isfinite(disp)]])
    ax.set_xlabel("Spike-integration window (ms)")
    ax.set_ylabel("Mean r (LLM ↔ Neural)")
    ax.set_title(title, loc="left", fontsize=10.5)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="upper right")


def main():
    os.makedirs(FIGURES, exist_ok=True)

    # ---- Convo: orig10 vs new4, two panels ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    _plot_panel(axes[0], ORIG10, "A  Original 10 patients\nall models · hippocampus · 15-word chunks", show_legend=True)
    _plot_panel(axes[1], NEW4, "B  New 4 patients\nall models · hippocampus · 15-word chunks")
    fig.suptitle("Curvature vs. displacement dominance by spike-integration window (conversation)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_curvature_dominance_convo")

    # ---- Podcast: single panel ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(5.2, 4.2))
    _plot_panel(ax2, PODCAST, "Podcast · 9 patients\nall models · hippocampus · 15-word chunks", show_legend=True)
    fig2.suptitle("Curvature vs. displacement dominance by spike-integration window (podcast)",
                  fontsize=11, y=1.02)
    fig2.tight_layout()
    _save(fig2, "fig_curvature_dominance_podcast")

    print("\nDone.")


if __name__ == "__main__":
    main()
