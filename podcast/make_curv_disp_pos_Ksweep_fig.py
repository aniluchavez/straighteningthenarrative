"""
make_curv_disp_pos_Ksweep_fig.py
====================================
Figure for the K-sweep result (definitive_curv_disp_pos_Ksweep.py):
OOS unique R2 (pooled, beyond Zipf freq + duration + sentence position,
leave-one-patient-out CV) for curvature/displacement/position, across
K=[5,8,20,50,native], for llama-3.1-8b and gpt2-large.

Saves: figures/fig_curv_disp_pos_Ksweep.{pdf,png,svg}
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(PARENT_DIR, "results")
FIGURES = os.path.join(PARENT_DIR, "figures")

KS = ["5", "8", "20", "50", "native"]
FEATURES = [("curvature", "Curvature", "#E67E22"),
            ("displacement", "Displacement", "#5B8FD4"),
            ("position", "Position", "#2E9E6C")]

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


def main():
    data = json.load(open(os.path.join(RESULTS, "definitive_curv_disp_pos_Ksweep.json")))
    models = list(data.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5.0), sharey=False)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(KS))
    win_tally = {lbl: 0 for lbl, *_ in FEATURES}
    pos_tally = {lbl: 0 for lbl, *_ in FEATURES}
    n_cells = 0

    for ax, model in zip(axes, models):
        for key, label, color in FEATURES:
            vals = [data[model][k][key]["oos_unique_pooled"] for k in KS]
            ax.plot(x, vals, marker="o", lw=1.8, ms=6, color=color, label=label)
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(KS)
        ax.set_xlabel("K (PCs, matched semantic ↔ neural)")
        ax.set_ylabel("OOS unique R² (beyond freq/dur/sentpos)")
        ax.set_title(f"{model}", loc="left", fontsize=12)

        for k in KS:
            n_cells += 1
            cell = {lbl: data[model][k][lbl]["oos_unique_pooled"] for lbl, *_ in FEATURES}
            winner = max(cell.items(), key=lambda kv: kv[1])[0]
            win_tally[winner] += 1
            for lbl, v in cell.items():
                if v > 0:
                    pos_tally[lbl] += 1

    axes[0].legend(fontsize=9, framealpha=0.9, edgecolor="#DDDDDD", loc="upper left")

    summary = "  ·  ".join(
        f"{lbl}: wins {win_tally[lbl]}/{n_cells}, positive {pos_tally[lbl]}/{n_cells}"
        for lbl, *_ in FEATURES
    )
    fig.suptitle("Curvature vs displacement vs position: unique contribution to neural activity, across K\n"
                 f"podcast, dur8040 raw-count window, LOPO CV  ·  {summary}",
                 fontsize=9.5, fontweight="bold", y=1.05)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curv_disp_pos_Ksweep"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
