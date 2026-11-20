"""
sweep_pairwise_distance_corr_gpt2xl_pseudopop_layers.py
===========================================================
Reuses the cached pseudopopulation neural pairwise-distance setup
(results/pseudopop_neural_cache.npz, built by
make_pairwise_distance_scatter_gpt2xl_pseudopop*.py) to quickly sweep
gpt2-xl layers -- including early layers -- without re-doing the
expensive 13-patient difflib alignment + neural loading each time.

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u sweep_pairwise_distance_corr_gpt2xl_pseudopop_layers.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")

REF_PID = "YEY"
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
    cache = np.load(os.path.join(RESULTS, "pseudopop_neural_cache.npz"))
    neu_z = cache["neu_z"]
    common_valid = cache["common_valid"]
    n = neu_z.shape[0]
    print(f"n_words={n}  neural dim={neu_z.shape[1]}")

    print("Computing neural cosine distance matrix (fixed)...")
    D_neu = cosine_distance_matrix(neu_z)
    iu = np.triu_indices(n, k=1)
    d_neu_all = D_neu[iu]
    del D_neu

    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]

    results = {}
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, common_valid], dtype=np.float64)
        D_sem = cosine_distance_matrix(emb_L)
        d_sem_all = D_sem[iu]
        del D_sem
        r, p = pearsonr(d_sem_all, d_neu_all)
        results[L] = dict(r=float(r), p=float(p))
        print(f"  layer {L:2d}/{n_layers-1}: r={r:+.4f}  p={p:.2e}", flush=True)
        json.dump(results, open(os.path.join(RESULTS, "pairwise_distance_corr_gpt2xl_pseudopop_layersweep.json"), "w"), indent=2)

    best_L = max(results, key=lambda k: results[k]["r"])
    print(f"\nBest layer: {best_L}  r={results[best_L]['r']:.4f}  p={results[best_L]['p']:.2e}")


if __name__ == "__main__":
    main()
