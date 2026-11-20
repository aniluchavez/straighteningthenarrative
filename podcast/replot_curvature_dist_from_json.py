"""
replot_curvature_dist_from_json.py
====================================
Re-renders fig_curvature_dist_all_models_all_layers{suffix}.png/.pdf from
the already-computed results/curvature_dist_all_models_all_layers{suffix}.json,
without rerunning the expensive computation. Used to pick up rcParams
changes (e.g. pdf.fonttype=42 for proper Illustrator-editable text) after
the fact.

Run:
    conda run -n gpt2_embed python3 -u replot_curvature_dist_from_json.py --suffix _llama2
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(PARENT_DIR, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    in_path = os.path.join(RESULTS_DIR, f"curvature_dist_all_models_all_layers{args.suffix}.json")
    with open(in_path) as f:
        payload = json.load(f)

    centers = np.array(payload["bin_centers"])
    d_neu = np.array(payload["neural_density"])
    d_llm = np.array(payload["semantic_density"])
    n_neural = payload["n_neural"]
    n_semantic = payload["n_semantic"]
    neural_mean = payload["neural_mean"]
    semantic_mean = payload["semantic_mean"]
    models = payload["models"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    neural_color = "#148F77"
    llm_color = "#5DADE2"

    ax.fill_between(centers, d_neu, step="mid", color=neural_color, alpha=0.20)
    ax.plot(centers, d_neu, color=neural_color, lw=1.6,
            label=f"Hippocampal neural ($N={n_neural:,}$)")
    ax.fill_between(centers, d_llm, step="mid", color=llm_color, alpha=0.18)
    ax.plot(centers, d_llm, color=llm_color, lw=1.6,
            label=f"{'+'.join(models)} (all layers, $N={n_semantic:,}$)")

    ax.axvline(neural_mean, color=neural_color, lw=1.0)
    ax.axvline(semantic_mean, color=llm_color, lw=1.0)

    ax.set_xlim(0, 180)
    ax.set_xlabel("Curvature $\\kappa$ (deg)")
    ax.set_ylabel("Density")
    ax.set_title(f"Curvature distribution\naggregated across {'+'.join(models)}, all layers (podcast, n=13)",
                  loc="left", pad=8)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.85, edgecolor="#DDDDDD")
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"fig_curvature_dist_all_models_all_layers{args.suffix}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
