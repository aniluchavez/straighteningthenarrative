"""
make_flat_curvy_decoding_figure_pseudopop.py
====================================
Plots compute_flat_curvy_decoding_test_pseudopop.py's results: cross-space
next-state decoding R^2 (held-out, mean +/- SE over 20 random train/test
splits), flat vs curvy source trajectory, across K (PCs), for both
directions. Neural side = 13-patient pseudopopulation (475 channels,
dur80/40 rate-normalized); semantic side = YEY's gpt2-xl layer 20.

Saves: figures/fig_flat_curvy_decoding_pseudopop.{pdf,png,svg}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(PARENT_DIR, "figures")

C_FLAT = "#2980B9"   # blue  -- flat (low curvature)
C_CURVY = "#C0392B"  # red   -- curvy (high curvature)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def _panel(ax, data, title, n_total):
    ks = sorted(int(k) for k in data.keys())
    flat = np.array([data[str(k)]["r2_flat_mean"] for k in ks])
    flat_se = np.array([data[str(k)]["r2_flat_se"] for k in ks])
    curvy = np.array([data[str(k)]["r2_curvy_mean"] for k in ks])
    curvy_se = np.array([data[str(k)]["r2_curvy_se"] for k in ks])
    diff = [data[str(k)]["diff"] for k in ks]

    ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.6)
    ax.plot(ks, flat, color=C_FLAT, lw=1.8, marker="o", ms=5, label="Flat (low curvature)")
    ax.fill_between(ks, flat - flat_se, flat + flat_se, color=C_FLAT, alpha=0.15)
    ax.plot(ks, curvy, color=C_CURVY, lw=1.8, marker="o", ms=5, label="Curvy (high curvature)")
    ax.fill_between(ks, curvy - curvy_se, curvy + curvy_se, color=C_CURVY, alpha=0.15)
    ax.set_xlabel("K (PCs)", fontsize=10)
    ax.set_ylabel("Held-out R² (next-state decoding)", fontsize=9.5)
    ax.set_title(f"{title}\n$n$ = {n_total} words, 20 splits", loc="left", fontsize=10.5, pad=4)
    ax.tick_params(labelsize=9)
    ax.legend(loc="best", fontsize=8, framealpha=0.8, edgecolor="#DDDDDD")

    ax2 = ax.twinx()
    ax2.plot(ks, diff, color="#7F7F7F", lw=1.2, ls="--", marker="s", ms=3.5, alpha=0.8)
    ax2.set_ylabel("diff (flat − curvy)", fontsize=8.5, color="#7F7F7F")
    ax2.tick_params(axis="y", labelsize=8, colors="#7F7F7F")
    ax2.axhline(0, color="#7F7F7F", lw=0.5, ls=":", alpha=0.4)


def main():
    d = json.load(open(os.path.join(RESULTS, "flat_curvy_decoding_test_pseudopop.json")))
    n_total = d["neural_to_semantic"]["3"]["n_total"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.14, top=0.82, wspace=0.5)

    _panel(axes[0], d["neural_to_semantic"], "A  neural[t] → semantic[t+1]", n_total)
    _panel(axes[1], d["semantic_to_neural"], "B  semantic[t] → neural[t+1]", n_total)

    fig.suptitle("Podcast (pseudopopulation, 475 channels) · flat vs. curvy trajectory → cross-space next-state decoding",
                 fontsize=9.5, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_flat_curvy_decoding_pseudopop"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
