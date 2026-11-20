"""
compute_podcast_chunk_context_sweep.py
========================================
Podcast version of compute_cross_space_chunk_sweep.py, crossed with context
length: does the context-length effect on semantic-neural coupling (higher r
at shorter context, seen at the word level) survive once you chunk-average
words together, or does chunking wash it out?

For gpt2-xl (best-correlating model from the context sweep), for each
context length (200/100/50/10/3) x chunk size (3-30 words), compute
r(LLM curvature, neural curvature) and r(LLM displacement, neural
displacement), averaged across layers, per patient -- same _chunk_r logic
as the convo version, no permutation null (kept fast; significance already
established via the paired t-tests on the raw word-level correlations).

Saves: results/podcast_chunk_context_sweep.json
Structure: { context: { chunk_size: { metric: [r_per_patient] } } }

Run:
    conda run -n gpt2_embed python3 -u compute_podcast_chunk_context_sweep.py
"""
import json, os, sys
import numpy as np
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

RESULTS_DIR  = os.path.join(BASE_DIR, "results")
REGION       = "hippocampus"
MODEL        = "gpt2-xl"
CONTEXTS     = [200, 100, 50, 10, 3]
CHUNK_SIZES  = [3, 5, 8, 10, 15, 20, 25, 30]


def _chunk_r(x, y, chunk_size, min_chunks=3):
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < chunk_size // 2:
            continue
        xa.append(np.mean(x[sl][fin]))
        ya.append(np.mean(y[sl][fin]))
    if len(xa) < min_chunks:
        return float("nan")
    r, _ = pearsonr(xa, ya)
    return float(r)


def main():
    out_path = os.path.join(RESULTS_DIR, "podcast_chunk_context_sweep.json")
    results = {ctx: {cs: {"curv": [], "disp": []} for cs in CHUNK_SIZES} for ctx in CONTEXTS}

    for ctx in CONTEXTS:
        print(f"\ncontext={ctx}")
        for pid in PODCAST_PATIENTS:
            data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=ctx)
            if data is None:
                continue
            n = data["n_words"]
            surp, sp, is_turn = data["surp"], data["sent_pos"], data["is_turn"]
            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < 20:
                continue

            Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
            if Y_neu is None:
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]

            layer_r = {cs: {"curv": [], "disp": []} for cs in CHUNK_SIZES}
            for L in range(n_layers):
                emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                c_sem = trajectory_curvature(emb_L)[valid]
                d_sem = trajectory_displacement(emb_L)[:n - 2][valid]
                for cs in CHUNK_SIZES:
                    layer_r[cs]["curv"].append(_chunk_r(c_sem, c_neu, cs))
                    layer_r[cs]["disp"].append(_chunk_r(d_sem, d_neu, cs))
            del emb_mm

            for cs in CHUNK_SIZES:
                for metric in ("curv", "disp"):
                    vals = [v for v in layer_r[cs][metric] if not np.isnan(v)]
                    results[ctx][cs][metric].append(float(np.mean(vals)) if vals else float("nan"))

            print(f"  {pid}: done ({n_layers} layers)")

    out = {str(ctx): {str(cs): results[ctx][cs] for cs in CHUNK_SIZES} for ctx in CONTEXTS}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(out_path, "w"))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
