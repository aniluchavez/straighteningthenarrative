"""
plot_decoding_panels_fig.py
============================
Standalone two-panel figure: just the cross-space decoding-vs-curvature
panels (A: LLM -> neural, B: Neural -> LLM) from plot_summary_figure.py,
side by side, with the flat-vs-curved delta test and the hippocampus
decoding chance-null annotation.

Run:
    cd /scratch/aniluchavez/straighteningthenarrative/figures
    conda run -n gpt2_embed python3 plot_decoding_panels_fig.py
"""

from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_summary_figure import draw_cross_single, BASE

FDIR = os.path.join(BASE, "figures")


def main():
    os.makedirs(FDIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                              gridspec_kw=dict(hspace=0.45, wspace=0.35,
                                                left=0.07, right=0.98,
                                                bottom=0.08, top=0.92))

    draw_cross_single(axes[0, 0], lag=1, direction="LLM_to_neural", panel_label="A")
    draw_cross_single(axes[0, 1], lag=1, direction="neural_to_LLM", panel_label="B")
    draw_cross_single(axes[1, 0], lag=1, direction="LLM_to_neural", panel_label="C", square=True)
    draw_cross_single(axes[1, 1], lag=1, direction="neural_to_LLM", panel_label="D", square=True)

    stem = "fig_cross_space_decoding_AB"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FDIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
