"""
compute_cross_space_chunk_sweep_pca20_podcast.py
====================================
PCA20 version of the cross-space chunk sweep for PODCAST (9 patients),
matching compute_cross_space_chunk_sweep_pca20.py's structure/conventions
(default win=500ms via load_podcast_patient, ctx200). Each of 9 models,
each chunk size, per-patient r (curvature, displacement) with the semantic
side PCA-reduced to a FIXED 20 components per layer (averaged across
layers per model). No permutation null.

Saves: results/podcast_cross_space_chunk_sweep_pca20.json
Structure: { chunk_size: { model: { metric: [r_per_patient] } } }
"""
import json, os, sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

RESULTS     = os.path.join(BASE_DIR, "results")
REGION      = "hippocampus"
CHUNK_SIZES = [3, 5, 8, 10, 15, 20, 25, 30]
PCA_N       = 20

MODELS = [
    "gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]

CACHE_DIR = os.path.join(BASE_DIR, "cache_perword_podcast_pca20_bymodel")
os.makedirs(CACHE_DIR, exist_ok=True)


def fit_pca_fixed(X, n_components):
    X = np.asarray(X, dtype=np.float64)
    max_pc = min(X.shape[1], X.shape[0] - 1, n_components)
    if X.std() == 0 or max_pc < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
    return pca.transform(Xs)[:, :max_pc]


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


def get_neural_perword(pid):
    cache_path = os.path.join(CACHE_DIR, f"{pid}_neural.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["c_neu"], z["d_neu"], z["valid"], int(z["n"])

    data = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data is None or data["neu_vecs"] is None:
        return None
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    np.savez(cache_path, c_neu=c_neu, d_neu=d_neu, valid=valid, n=n)
    return c_neu, d_neu, valid, n


def get_semantic_perword_pca20(pid, model, n, valid):
    cache_path = os.path.join(CACHE_DIR, f"{pid}_{model}_pca{PCA_N}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["c_sem"], z["d_sem"]

    data_m = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
    if data_m is None:
        return None
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]
    c_layers, d_layers = [], []
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        emb_L_pca = fit_pca_fixed(emb_L, PCA_N)
        if emb_L_pca is None:
            continue
        c_layers.append(trajectory_curvature(emb_L_pca)[valid])
        d_layers.append(trajectory_displacement(emb_L_pca)[:n - 2][valid])
    del emb_mm
    if not c_layers:
        return None
    c_sem = np.mean(c_layers, axis=0)
    d_sem = np.mean(d_layers, axis=0)
    np.savez(cache_path, c_sem=c_sem, d_sem=d_sem)
    return c_sem, d_sem


def main():
    out_path = os.path.join(RESULTS, "podcast_cross_space_chunk_sweep_pca20.json")
    results = {cs: {m: {"disp": [], "curv": []} for m in MODELS} for cs in CHUNK_SIZES}

    for pid in PODCAST_PATIENTS:
        print(f"\n  {pid}", flush=True)
        neural = get_neural_perword(pid)
        if neural is None:
            print(f"    SKIP (neural)")
            for cs in CHUNK_SIZES:
                for m in MODELS:
                    results[cs][m]["curv"].append(float("nan"))
                    results[cs][m]["disp"].append(float("nan"))
            continue
        c_neu, d_neu, valid, n = neural

        for model in MODELS:
            sem = get_semantic_perword_pca20(pid, model, n, valid)
            if sem is None:
                print(f"    {model}: SKIP")
                for cs in CHUNK_SIZES:
                    results[cs][model]["curv"].append(float("nan"))
                    results[cs][model]["disp"].append(float("nan"))
                continue
            c_sem, d_sem = sem
            for cs in CHUNK_SIZES:
                results[cs][model]["curv"].append(_chunk_r(c_sem, c_neu, cs))
                results[cs][model]["disp"].append(_chunk_r(d_sem, d_neu, cs))
            print(f"    {model}: done ({len(c_sem)} words)", flush=True)

        out = {str(cs): results[cs] for cs in CHUNK_SIZES}
        json.dump(out, open(out_path, "w"))
        print(f"  checkpoint saved -> {out_path}", flush=True)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
