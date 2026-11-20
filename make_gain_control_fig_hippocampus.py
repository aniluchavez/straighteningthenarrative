"""
make_gain_control_fig_hippocampus.py
======================================
Hippocampus-only version of make_gain_control_fig.py. Tests whether the
r_high > r_low "scissors" pattern survives once surprisal-locked gain in the
raw population vector is removed before curvature is computed:

  additive       -- subtract the mean population vector within each
                     surprisal quintile (removes common-mode, surprisal-
                     locked gain).
  multiplicative -- normalize each population vector to unit norm (removes
                     per-word amplitude/gain entirely).

Data: ../results/gain_control_surprisal_split_results.json
(produced by run_gain_control_surprisal_split.py)

Run:
    conda run -n gpt2_embed python3 make_gain_control_fig_hippocampus.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_IN  = os.path.join(BASE_DIR, "..", "results", "gain_control_surprisal_split_results.json")
FIGURES     = os.path.join(BASE_DIR, "figures", "appendix")

REGIONS  = ["hippocampus"]
VARIANTS = [
    ("raw",            "Raw (no control)",       "#333333", "-",  2.2),
    ("additive",       "Additive-gain removed",  "#3C82C4", "--", 1.8),
    ("multiplicative", "Gain-normalized",         "#D85A30", ":",  1.8),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PANEL_LABEL_KW = dict(
    fontsize=16, fontweight="bold", fontfamily="sans-serif",
    va="top", ha="left",
)


def add_panel_label(ax, letter):
    ax.text(-0.16, 1.10, letter, transform=ax.transAxes, **PANEL_LABEL_KW)


def draw_panel(ax, recs, key):
    n_layers = recs[0]["n_layers"]
    layers_x = np.arange(n_layers)
    for variant, label, color, ls, lw in VARIANTS:
        vrecs = [r for r in recs if r["variant"] == variant]
        if not vrecs:
            continue
        mat = np.array([r[key] for r in vrecs], dtype=float)
        mu  = np.nanmean(mat, axis=0)
        se_ = sem(mat, axis=0, nan_policy="omit")
        ax.plot(layers_x, mu, color=color, lw=lw, ls=ls, label=label)
        ax.fill_between(layers_x, mu - se_, mu + se_, color=color, alpha=0.12)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Layer")


def main():
    with open(RESULTS_IN) as f:
        records = json.load(f)

    fig, axes = plt.subplots(len(REGIONS), 2, figsize=(11.0, 4.6 * len(REGIONS)),
                              squeeze=False)
    fig.suptitle(
        "Gain controls do not explain surprisal-locked curvature coupling\n"
        "(LLaMA-3.1-8B, hippocampus)  r_diff = r(neural κ, semantic κ | hi) − r(· | lo)",
        fontsize=12,
    )

    letters = iter("AB")
    for ri, region in enumerate(REGIONS):
        recs = [r for r in records if r["region"] == region]
        if not recs:
            for ci in range(2):
                axes[ri, ci].set_visible(False)
            continue
        n_pat = len({r["patient"] for r in recs})

        ax_word = axes[ri, 0]
        draw_panel(ax_word, recs, "r_diff")
        ax_word.set_title(f"{region}  (word-level, n={n_pat} patients)")
        add_panel_label(ax_word, next(letters))
        ax_word.set_ylabel("r_diff  (r_high − r_low)")
        ax_word.legend(loc="upper left", frameon=False)

        ax_sent = axes[ri, 1]
        sent_recs = [r for r in recs if r["n_sent_hi"] > 0]
        draw_panel(ax_sent, sent_recs, "sent_r_diff")
        ax_sent.set_title(f"{region}  (sentence-level)")
        add_panel_label(ax_sent, next(letters))
        ax_sent.set_ylabel("r_diff  (r_high − r_low)")
        ax_sent.legend(loc="upper left", frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_gain_control_surprisal_split_hippocampus"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        if ext == "svg":
            with open(path, "r") as fh:
                txt = fh.read().replace("TeX Gyre Termes", "Times New Roman")
            with open(path, "w") as fh:
                fh.write(txt)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
