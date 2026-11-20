"""
Recolor fig_multimodel_layersweep_surprisal_curvature_pca20_podcast.{pdf,png,svg}
using the SAME fixed model->color mapping as
podcast/make_tentative_ABC_pca20_nochunk.py (MODEL_STYLE), instead of
tab10-by-peak-rank, so model colors match tent_contentfig1.pdf's Panel B.
Reads cached results/multimodel_layersweep_surprisal_curvature_pca20_podcast.json
-- no recompute.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GP = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(GP, "results")
FIGURES = BASE_DIR

MODELS = ["bert-base", "roberta-base", "gpt2", "gpt2-medium", "gpt2-xl",
          "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
COLORS = cm.tab10(np.linspace(0, 1, 10))
MODEL_COLOR = {m: COLORS[i] for i, m in enumerate(MODELS)}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

results = json.load(open(os.path.join(RESULTS, "multimodel_layersweep_surprisal_curvature_pca20_podcast.json")))
valid_results = {k: v for k, v in results.items() if v is not None}
ranked = sorted(valid_results.values(), key=lambda r: -r["r_all_best"])

fig, ax = plt.subplots(figsize=(8.5, 6.0))
for r in ranked:
    n_layers = r["n_layers"]
    x_norm = np.linspace(0, 1, n_layers)
    color = MODEL_COLOR.get(r["model"], "gray")
    ax.plot(x_norm, r["r_all_by_layer"], color=color, lw=1.6,
            label=f"{r['model']} (peak={r['r_all_best']:+.3f} @ L{r['best_layer']}/{n_layers-1})")
ax.axhline(0, color="#BBBBBB", lw=0.7, ls=":")
ax.set_xlabel("Normalized layer position")
ax.set_ylabel("r(semantic curvature, neural curvature), all words")
ax.set_title("Neural-semantic curvature correlation by layer, across models (PCA-20)\npodcast, hippocampus", loc="left", fontsize=11)
ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#DDDDDD", loc="upper right")

stem = "fig_multimodel_layersweep_surprisal_curvature_pca20_podcast"
for ext in ("pdf", "png", "svg"):
    path = os.path.join(FIGURES, f"{stem}.{ext}")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
plt.close(fig)
