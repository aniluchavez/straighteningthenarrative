"""
replot_curvature_hilo_from_json.py
====================================
Re-renders fig_curvature_hilo_all_models_all_layers{suffix}.png/.pdf from
the already-computed results/curvature_hilo_all_models_all_layers{suffix}.json,
without rerunning the expensive computation. Used to pick up rcParams
changes (e.g. pdf.fonttype=42 for proper Illustrator-editable text).

Run:
    conda run -n gpt2_embed python3 -u replot_curvature_hilo_from_json.py --suffix _llama2
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

    in_path = os.path.join(RESULTS_DIR, f"curvature_hilo_all_models_all_layers{args.suffix}.json")
    with open(in_path) as f:
        payload = json.load(f)

    models = list(payload["per_model_hi"].keys())
    hi_all = np.concatenate([np.array(v) for v in payload["per_model_hi"].values()])
    lo_all = np.concatenate([np.array(v) for v in payload["per_model_lo"].values()])

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
    C_HI, C_LO = "#C0392B", "#2980B9"
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.bar(["High surprisal\n(>=75th pct)", "Low surprisal\n(<=25th pct)"],
           [hi_all.mean(), lo_all.mean()], color=[C_HI, C_LO], alpha=0.85,
           yerr=[hi_all.std(ddof=1) / np.sqrt(len(hi_all)), lo_all.std(ddof=1) / np.sqrt(len(lo_all))],
           capsize=6)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_ylabel("r(neural curvature, semantic curvature)")
    ax.set_title(f"Curvature-surprisal coupling, aggregated\n{'+'.join(models)}, all layers x all patients "
                 f"(n={len(hi_all)} triples)", loc="left", pad=8, fontsize=10.5)
    ax.text(0.02, 0.02, f"pooled hi: r={payload['r_hi_pooled']:+.3f}, p={payload['p_hi_pooled']:.2e}\n"
                        f"pooled lo: r={payload['r_lo_pooled']:+.3f}, p={payload['p_lo_pooled']:.2e}",
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"fig_curvature_hilo_all_models_all_layers{args.suffix}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
