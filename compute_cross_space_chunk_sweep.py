"""
compute_cross_space_chunk_sweep.py
====================================
Sweep chunk sizes to show that the cross-space coupling pattern is stable
and not artefactually tied to the choice of 10-word chunks.

For each chunk size in CHUNK_SIZES:
  For each model × patient, average all layers:
    chunk-average LLM and neural metrics → pearsonr

Saves: results/cross_space_chunk_sweep.json

Structure:
  { chunk_size: { model: { metric: [r_per_patient] } } }

Run:
    cd /scratch/aniluchavez/straighteningthenarrative
    conda run -n gpt2_embed python3 compute_cross_space_chunk_sweep.py
"""

import json, os, sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_PERM = 200   # word-order shuffles for null distribution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement,
    build_valid_mask, fit_neural_pca, load_patient,
    PATIENTS, MIN_PAIRS
)

RESULTS     = os.path.join(BASE_DIR, "results")
REGION      = "hippocampus"
CHUNK_SIZES = [3, 5, 8, 10, 15, 20, 25, 30]

MODELS = [
    "gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]


def _pc1_series(X):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return np.full(X.shape[0], np.nan)
    Xs = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=1).fit_transform(Xs)[:, 0]
    s = np.std(pc1)
    return (pc1 - pc1.mean()) / s if s > 1e-12 else np.full(len(pc1), np.nan)


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
    out_path = os.path.join(RESULTS, "cross_space_chunk_sweep.json")

    rng = np.random.default_rng(42)

    # results[chunk_size][model][metric] = [r per patient]
    # null_results stores per-patient null mean and std (averaged across layers)
    results = {cs: {m: {"disp": [], "curv": [], "pos": [],
                         "null_curv_mean": [], "null_curv_std": [],
                         "null_disp_mean": [], "null_disp_std": []}
                    for m in MODELS}
               for cs in CHUNK_SIZES}

    for model in MODELS:
        print(f"\n  {model}")
        for pid, preg in PATIENTS.items():
            if REGION not in preg:
                continue
            data = load_patient(pid, preg, REGION, model)
            if data is None:
                continue

            n       = data["n_words"]
            surp    = data["surp"]
            sp      = data["sent_pos"]
            is_turn = data["is_turn"]

            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < MIN_PAIRS:
                continue

            interior = np.arange(1, n - 1)

            Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
            if Y_neu is None:
                continue

            # neural metrics (fixed across all layers and chunk sizes)
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
            p_neu = _pc1_series(Y_neu)[interior][valid]

            emb_mm   = np.load(data["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]

            # Pass 1: accumulate real r values per layer, store c_sem/d_sem for null
            layer_r  = {cs: {"disp": [], "curv": [], "pos": []} for cs in CHUNK_SIZES}
            c_sems, d_sems = [], []

            for L in range(n_layers):
                emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                c_sem = trajectory_curvature(emb_L)[valid]
                d_sem = trajectory_displacement(emb_L)[:n - 2][valid]
                p_sem = _pc1_series(emb_L)[interior][valid]
                c_sems.append(c_sem)
                d_sems.append(d_sem)

                for cs in CHUNK_SIZES:
                    layer_r[cs]["curv"].append(_chunk_r(c_sem, c_neu, cs))
                    layer_r[cs]["disp"].append(_chunk_r(d_sem, d_neu, cs))
                    layer_r[cs]["pos"].append( _chunk_r(p_sem, p_neu, cs))

            del emb_mm

            # average across layers, store one value per patient
            for cs in CHUNK_SIZES:
                for metric in ("disp", "curv", "pos"):
                    vals = [v for v in layer_r[cs][metric] if not np.isnan(v)]
                    results[cs][model][metric].append(
                        float(np.mean(vals)) if vals else float("nan"))

            # Pass 2: shuffle null — permute neural word order N_PERM times,
            # apply to all layers, average across layers per permutation.
            # Each permutation is a consistent shuffle across all layers.
            for cs in CHUNK_SIZES:
                null_curv_perms, null_disp_perms = [], []
                for _ in range(N_PERM):
                    perm = rng.permutation(len(c_neu))
                    rs_c = [_chunk_r(c_sems[L], c_neu[perm], cs) for L in range(n_layers)]
                    rs_d = [_chunk_r(d_sems[L], d_neu[perm], cs) for L in range(n_layers)]
                    null_curv_perms.append(float(np.nanmean(rs_c)))
                    null_disp_perms.append(float(np.nanmean(rs_d)))
                results[cs][model]["null_curv_mean"].append(float(np.nanmean(null_curv_perms)))
                results[cs][model]["null_curv_std"].append(float(np.nanstd(null_curv_perms)))
                results[cs][model]["null_disp_mean"].append(float(np.nanmean(null_disp_perms)))
                results[cs][model]["null_disp_std"].append(float(np.nanstd(null_disp_perms)))

            print(f"    {pid}: done ({n_layers} layers)")

        # checkpoint after every model, so a killed/interrupted run doesn't
        # lose already-completed models (each model's inner loop is the
        # expensive part -- ~N_PERM permutations x chunk sizes x layers x patients)
        out = {str(cs): results[cs] for cs in CHUNK_SIZES}
        json.dump(out, open(out_path, "w"))
        print(f"  checkpoint saved -> {out_path}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
