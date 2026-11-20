"""
make_pairwise_distance_scatter_gpt2xl.py
===========================================
Reproduces the classic "all word-pairs" neural-vs-semantic distance scatter
(cf. Goldstein et al.-style figure): for one podcast patient, computes
cosine distance between EVERY pair of words in both neural (hippocampal
population firing-rate) space and gpt2-xl semantic embedding space, then
scatters a random subsample of pairs with a regression line, matching the
reference figure's r/p annotation style.

Patient choice: YFK (n=7346 words) -- n(n-1)/2 = 26,977,985 pairs, i.e.
essentially the same word-pair count as the reference figure's
26,978,185, so this is the natural patient to match it against.

NOTE ON PHI: per established project convention, transcript/word content
is PHI and must never be displayed. The reference figure labels individual
word pairs in text; this version deliberately omits any such text labels
and shows only the (de-identified) geometric quantities -- cosine
distances and their correlation.

Saves: figures/fig_pairwise_distance_scatter_gpt2xl.{pdf,png,svg}

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_pairwise_distance_scatter_gpt2xl.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")

PID = "YFK"
REGION = "hippocampus"
MODEL = "gpt2-xl"
LAYER = 36
N_SCATTER = 20000
SEED = 0

sys.path.insert(0, PARENT_DIR)
from podcast_loader import load_podcast_patient

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


def cosine_distance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    Xn = X / norms
    sim = Xn @ Xn.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def main():
    data = load_podcast_patient(PID, region=REGION, model_tag=MODEL, context_words=200)
    n = data["n_words"]
    neu_vecs = data["neu_vecs"][:n]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
    del emb_mm

    print(f"{PID}: n_words={n}  neural dim={neu_vecs.shape[1]}  semantic dim={emb_L.shape[1]}")

    print("Computing neural cosine distance matrix...")
    D_neu = cosine_distance_matrix(neu_vecs)
    print("Computing semantic cosine distance matrix...")
    D_sem = cosine_distance_matrix(emb_L)

    iu = np.triu_indices(n, k=1)
    d_neu_all = D_neu[iu]
    d_sem_all = D_sem[iu]
    n_pairs = len(d_neu_all)
    print(f"n_pairs = {n_pairs:,}")

    r, p = pearsonr(d_sem_all, d_neu_all)
    print(f"r = {r:.4f}  p = {p:.4e}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_pairs, size=min(N_SCATTER, n_pairs), replace=False)
    x_plot, y_plot = d_sem_all[idx], d_neu_all[idx]

    # regression line over the FULL data (not just the plotted subsample)
    slope, intercept = np.polyfit(d_sem_all, d_neu_all, 1)
    xs = np.linspace(d_sem_all.min(), d_sem_all.max(), 200)
    ys = slope * xs + intercept

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x_plot, y_plot, s=2, color="black", alpha=0.15, linewidths=0, rasterized=True)
    ax.plot(xs, ys, color="#D6006C", lw=2.2, zorder=5)

    ax.text(0.04, 0.94, f"$r$ = {r:.3f}\n$p$ < 0.0001" if p < 1e-4 else f"$r$ = {r:.3f}\n$p$ = {p:.2e}",
            transform=ax.transAxes, fontsize=12, va="top")

    ax.set_xlabel(f"{MODEL} semantic distance (layer {LAYER}, cosine)")
    ax.set_ylabel("Neural cosine distance (hippocampus)")
    ax.set_title(f"All word-pair distances, patient {PID}\n"
                 f"$n$ = {n:,} words, {n_pairs:,} pairs", loc="left", fontsize=11)

    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_pairwise_distance_scatter_gpt2xl"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
