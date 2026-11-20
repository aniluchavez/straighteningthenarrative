"""
make_freq_control_fig.py
=========================
Figure for the word-frequency control on the surprisal-split neural-semantic
curvature coupling (LLaMA-3.1-8B). Tests whether the r_high > r_low pattern
reflects genuine contextual surprise rather than lexical frequency:

  raw               -- split by surprisal percentile (baseline).
  freq_residualized -- split by surprisal residualized on log-frequency
                        (surprise left over after frequency is removed).
  freq_only         -- split directly by log-frequency, ignoring surprisal
                        (rare words = "hi" analog, common words = "lo"
                        analog).

Data: ../results/freq_control_surprisal_split_results.json
(produced by run_freq_control_surprisal_split.py)

Run:
    conda run -n gpt2_embed python3 make_freq_control_fig.py
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
RESULTS_IN  = os.path.join(BASE_DIR, "..", "results", "freq_control_surprisal_split_results.json")
FIGURES     = os.path.join(BASE_DIR, "figures", "appendix")

REGIONS  = ["hippocampus", "ACC"]
VARIANTS = [
    ("raw",               "Raw (surprisal split)",        "#333333", "-",  2.2),
    ("freq_residualized", "Surprisal | freq. removed",    "#3C82C4", "--", 1.8),
    ("freq_only",         "Frequency split (no surprisal)", "#D85A30", ":",  1.8),
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

    fig, axes = plt.subplots(2, len(REGIONS), figsize=(5.2 * len(REGIONS), 7.6),
                              sharex=True)
    fig.suptitle(
        "Word-frequency control on surprisal-locked curvature coupling (LLaMA-3.1-8B)\n"
        "r_diff = r(neural κ, semantic κ | hi) − r(· | lo); see legend for split criterion",
        fontsize=12,
    )

    letters = iter("ABCD")
    for ci, region in enumerate(REGIONS):
        recs = [r for r in records if r["region"] == region]
        if not recs:
            for ri in range(2):
                axes[ri, ci].set_visible(False)
            continue
        n_pat = len({r["patient"] for r in recs})

        ax_word = axes[0, ci]
        draw_panel(ax_word, recs, "r_diff")
        ax_word.set_title(f"{region}  (word-level, n={n_pat} patients)")
        add_panel_label(ax_word, next(letters))
        if ci == 0:
            ax_word.set_ylabel("r_diff  (hi − lo)")
        ax_word.legend(loc="upper left", frameon=False)

        ax_sent = axes[1, ci]
        sent_recs = [r for r in recs if r["n_sent_hi"] > 0]
        draw_panel(ax_sent, sent_recs, "sent_r_diff")
        ax_sent.set_title(f"{region}  (sentence-level)")
        add_panel_label(ax_sent, next(letters))
        if ci == 0:
            ax_sent.set_ylabel("r_diff  (hi − lo)")
        ax_sent.legend(loc="upper left", frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_freq_control_surprisal_split"
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
