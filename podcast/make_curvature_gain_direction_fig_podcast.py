"""
make_curvature_gain_direction_fig_podcast.py
=============================================
Podcast version of ../make_curvature_gain_direction_fig.py.
Source: results/clean_fig4_podcast_pca20/curvature_gain_direction_control_podcast_summary.csv
(produced by run_curvature_gain_direction_control_podcast.py)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "clean_fig4_podcast_pca20", "curvature_gain_direction_control_podcast_summary.csv")
FIGURES  = os.path.join(BASE_DIR, "figures")

C_LINE  = "#8E44AD"
C_PATIENT = "#B0B0B0"

COLS   = ["beta_surp_base", "beta_surp_gain", "beta_surp_dir", "beta_surp_gain_dir"]
LABELS = ["Base", "+ Gain", "+ Direction", "+ Gain\n+ Direction"]

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
    df = pd.read_csv(CSV_PATH)
    xs = np.arange(len(COLS))

    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    for _, row in df.iterrows():
        ys = [row[c] for c in COLS]
        ax.plot(xs, ys, color=C_PATIENT, lw=0.9, alpha=0.6, marker="o", ms=3, zorder=2)

    mu = [df[c].mean() for c in COLS]
    se = [scipy_sem(df[c].dropna()) for c in COLS]
    ax.errorbar(xs, mu, yerr=se, color=C_LINE, lw=2.4, marker="o", ms=7,
                capsize=3, zorder=3, label=f"Mean $\\pm$ SEM ($n$={len(df)} patients)")

    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(r"$\beta$ (surprisal $\rightarrow$ neural curvature)")
    ax.set_title("PODCAST -- Curvature-surprisal effect under gain/direction control\nhippocampus, llama-3.1-8b",
                  loc="left", fontsize=10.5, pad=8)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75, edgecolor="#DDDDDD")

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_gain_direction_control_podcast"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
