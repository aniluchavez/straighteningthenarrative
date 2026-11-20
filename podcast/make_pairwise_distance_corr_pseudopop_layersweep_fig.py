"""
make_pairwise_distance_corr_pseudopop_layersweep_fig.py
===========================================================
Plots results/pairwise_distance_corr_gpt2xl_pseudopop_layersweep.json --
the all-word-pairs neural-vs-semantic distance correlation (pseudopopulation,
13 patients) across all 49 gpt2-xl layers.

Saves: figures/fig_pairwise_distance_corr_pseudopop_layersweep.{pdf,png,svg}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 12,
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


def main():
    d = json.load(open(os.path.join(RESULTS, "pairwise_distance_corr_gpt2xl_pseudopop_layersweep.json")))
    all_layers = sorted(int(k) for k in d.keys())
    n_layers_total = max(all_layers)
    # Drop the final layer: its pre-LM-head geometry is a known outlier/artifact
    # (large jump discontinuous with the smooth trend), not a genuine continuation.
    layers = [L for L in all_layers if L != n_layers_total]
    rs = [d[str(L)]["r"] for L in layers]

    peak_L = max(layers, key=lambda L: d[str(L)]["r"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(layers, rs, "-o", color="#2C3E90", lw=1.6, ms=3.5)
    ax.axvline(peak_L, color="#C0392B", lw=1, ls="--", alpha=0.7)
    ax.annotate(f"peak: layer {peak_L}, r={d[str(peak_L)]['r']:.4f}",
                xy=(peak_L, d[str(peak_L)]["r"]), xytext=(peak_L - 22, d[str(peak_L)]["r"] - 0.0015),
                fontsize=9, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8))

    ax.set_xlabel(f"gpt2-xl layer  (0 = embeddings, {n_layers_total - 1} = last transformer block)")
    ax.set_ylabel("All-word-pairs distance correlation $r$\n(semantic vs. neural cosine distance)")
    ax.set_title("Pairwise semantic-neural distance alignment by layer\n"
                  "Pseudopopulation (13 patients, 475 combined hippocampal channels), $n$=16.15M pairs\n"
                  "(final layer excluded -- its pre-LM-head geometry is a known outlier)",
                  loc="left", fontsize=10, pad=12)
    ax.set_xlim(-1, layers[-1] + 1)
    ax.set_ylim(top=max(rs) + 0.0025)

    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_pairwise_distance_corr_pseudopop_layersweep"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
