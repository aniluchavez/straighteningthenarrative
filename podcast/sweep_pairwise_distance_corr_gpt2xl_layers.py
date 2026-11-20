"""
sweep_pairwise_distance_corr_gpt2xl_layers.py
=================================================
The single-layer (L20) all-pairs distance correlation
(make_pairwise_distance_scatter_gpt2xl.py) came out essentially null
(r=-0.005) vs. the reference figure's r=0.064. Before concluding the
effect just isn't there, sweep all 49 gpt2-xl layers to check whether a
different layer shows a stronger RSA-style (global pairwise distance)
alignment -- this is a different quantity than the local trajectory-
curvature layer sweep done elsewhere in this project, so the best layer
need not match.

Patient: YFK (n=7346 words, matching the reference figure's exact
26,978,185-pair count).

Saves: results/pairwise_distance_corr_gpt2xl_layersweep_YFK.json

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u sweep_pairwise_distance_corr_gpt2xl_layers.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")

PID = "YFK"
REGION = "hippocampus"
MODEL = "gpt2-xl"

sys.path.insert(0, PARENT_DIR)
from podcast_loader import load_podcast_patient


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

    print("Computing neural cosine distance matrix (fixed across layers)...")
    D_neu = cosine_distance_matrix(neu_vecs)
    iu = np.triu_indices(n, k=1)
    d_neu_all = D_neu[iu]
    del D_neu

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]
    print(f"n_layers={n_layers}  n_words={n}  n_pairs={len(d_neu_all):,}")

    results = {}
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        D_sem = cosine_distance_matrix(emb_L)
        d_sem_all = D_sem[iu]
        del D_sem
        r, p = pearsonr(d_sem_all, d_neu_all)
        results[L] = dict(r=float(r), p=float(p))
        print(f"  layer {L:2d}/{n_layers-1}: r={r:+.4f}  p={p:.2e}", flush=True)
        json.dump(results, open(os.path.join(RESULTS, f"pairwise_distance_corr_gpt2xl_layersweep_{PID}.json"), "w"), indent=2)

    best_L = max(results, key=lambda k: results[k]["r"])
    print(f"\nBest layer: {best_L}  r={results[best_L]['r']:.4f}  p={results[best_L]['p']:.2e}")


if __name__ == "__main__":
    main()
