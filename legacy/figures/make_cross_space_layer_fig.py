"""
make_cross_space_layer_fig.py
==============================
Layer-profile figure: r(LLM metric, neural metric) vs normalised layer
position [0, 1] for three geometric quantities:

  displacement  — step-to-step Euclidean distance
  curvature     — turning angle κ
  position      — first-PC coordinate time series

All 9 models overlaid (same colour / linestyle scheme as fig2_model_comparison).
Each panel = one geometric quantity; mean ± SE across patients.

Saves: figures/fig_cross_space_layer_profiles.{pdf,png,svg}

Run:
    cd /scratch/aniluchavez/neural_network_similarity/Experiments/p6/geometry_paper
    conda run -n gpt2_embed python3 make_cross_space_layer_fig.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures", "appendix")
REGION   = "hippocampus"

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":         12,
    "axes.titlesize":    12,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   9,
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

PANEL_LABEL_KW = dict(
    fontsize=28, fontweight="bold", fontfamily="sans-serif",
    va="top", ha="left",
)


def add_panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

MODEL_STYLE = {
    "bert-base":    dict(color="#E67E22", label="BERT-base",     ls=(0,(3,1,1,1))),
    "roberta-base": dict(color="#F39C12", label="RoBERTa-base",  ls=(0,(5,2))),
    "gpt2":         dict(color="#2980B9", label="GPT-2",         ls="-"),
    "gpt2-medium":  dict(color="#1F618D", label="GPT-2 medium",  ls="--"),
    "opt-350m":     dict(color="#884EA0", label="OPT-350m",      ls=(0,(3,1))),
    "llama-2-7b":   dict(color="#C0392B", label="LLaMA-2 7B",    ls="--"),
    "llama-3.1-8b": dict(color="#E91E63", label="LLaMA-3.1 8B",  ls="-"),
    "gemma-2-9b":   dict(color="#148F77", label="Gemma-2 9B",    ls=(0,(4,1))),
    "mistral-7b":   dict(color="#D35400", label="Mistral-7B",    ls=(0,(2,1))),
}

QUANTITIES = [
    dict(key="disp", label="Displacement",
         ylabel="r(LLM displacement, neural displacement)", panel="A"),
    dict(key="curv", label="Curvature κ",
         ylabel="r(LLM curvature, neural curvature)",      panel="B"),
    dict(key="pos",  label="Position (PC1)",
         ylabel="r(LLM PC1, neural PC1)",                  panel="C"),
]

GRID = np.linspace(0, 1, 200)


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  → {p}")
    plt.close(fig)


def _make_panel_fig(r_key_suffix, title_tag, out_name):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.80,
                        wspace=0.46)

    for ax, q in zip(axes, QUANTITIES):
        r_key = f"r_{q['key']}_layers{r_key_suffix}"
        for model, st in MODEL_STYLE.items():
            path = os.path.join(RESULTS,
                                f"cross_space_layers_{model}_{REGION}.json")
            if not os.path.exists(path):
                continue
            recs = json.load(open(path))

            mats = []
            for rec in recs:
                if r_key not in rec:
                    continue
                arr = np.array(rec[r_key], dtype=float)
                n_l = len(arr)
                xs  = np.linspace(0, 1, n_l)
                mats.append(np.interp(GRID, xs, arr))

            if not mats:
                continue
            mat = np.array(mats)
            mu  = np.nanmean(mat, axis=0)
            se  = np.array([scipy_sem(mat[:, i][np.isfinite(mat[:, i])])
                            for i in range(mat.shape[1])])

            ax.plot(GRID, mu, color=st["color"], ls=st["ls"],
                    lw=1.4, label=st["label"])
            ax.fill_between(GRID, mu - se, mu + se,
                            color=st["color"], alpha=0.10)

        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Normalised layer position")
        ax.set_ylabel(q["ylabel"], fontsize=9)
        ax.set_title(f"{q['label']}\nall models · hippocampus",
                     loc="left", fontsize=10)
        add_panel_label(ax, q["panel"])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.6, columnspacing=0.8, fontsize=8.5)

    _save(fig, out_name)


def main():
    _make_panel_fig(
        r_key_suffix = "",
        title_tag    = "word-by-word",
        out_name     = "fig_cross_space_layer_profiles_word",
    )
    _make_panel_fig(
        r_key_suffix = "_ch",
        title_tag    = f"10-word chunks",
        out_name     = "fig_cross_space_layer_profiles_chunk",
    )
    print("Done.")


if __name__ == "__main__":
    main()
