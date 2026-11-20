"""
compute_podcast_position_only_sweep.py
====================================
Computes ONLY the "position" metric (PC1 of the trajectory, z-scored) for
PODCAST (9 patients), all 9 models, all chunk sizes -- to add a Position
column to the PCA20 cross-space figure.

Position is mathematically identical whether computed on the raw embedding
or on a PCA-reduced-to-K version (PC1 of the raw embedding == the first
column of any PCA(K>=1) reduction), so this uses a cheap single-component
PC1 fit directly on raw embeddings -- no need to redo the expensive
20-component reduction.

Saves: results/podcast_position_only_sweep.json
Structure: { chunk_size: { model: { "pos": [r_per_patient] } } }
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

from run_geometry_analysis import build_valid_mask, fit_neural_pca
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

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
    pc1 = PCA(n_components=1, svd_solver="randomized", random_state=0).fit_transform(Xs)[:, 0]
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
    out_path = os.path.join(RESULTS, "podcast_position_only_sweep.json")
    results = {cs: {m: {"pos": []} for m in MODELS} for cs in CHUNK_SIZES}

    for pid in PODCAST_PATIENTS:
        print(f"\n  {pid}", flush=True)
        data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
        if data0 is None or data0["neu_vecs"] is None:
            print("    SKIP (neural)")
            for cs in CHUNK_SIZES:
                for m in MODELS:
                    results[cs][m]["pos"].append(float("nan"))
            continue
        n = data0["n_words"]
        valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
        if valid.sum() < 50:
            print("    SKIP (valid)")
            for cs in CHUNK_SIZES:
                for m in MODELS:
                    results[cs][m]["pos"].append(float("nan"))
            continue
        Y_neu, _ = fit_neural_pca(data0["neu_vecs"], n)
        if Y_neu is None:
            print("    SKIP (neural pca)")
            for cs in CHUNK_SIZES:
                for m in MODELS:
                    results[cs][m]["pos"].append(float("nan"))
            continue
        interior = np.arange(1, n - 1)
        p_neu = _pc1_series(Y_neu)[interior][valid]

        for model in MODELS:
            data_m = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
            if data_m is None:
                print(f"    {model}: SKIP")
                for cs in CHUNK_SIZES:
                    results[cs][model]["pos"].append(float("nan"))
                continue
            emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]
            p_layers = []
            for L in range(n_layers):
                emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                p_layers.append(_pc1_series(emb_L)[interior][valid])
            del emb_mm
            p_sem = np.mean(p_layers, axis=0)

            for cs in CHUNK_SIZES:
                results[cs][model]["pos"].append(_chunk_r(p_sem, p_neu, cs))
            print(f"    {model}: done ({len(p_sem)} words)", flush=True)

        out = {str(cs): results[cs] for cs in CHUNK_SIZES}
        json.dump(out, open(out_path, "w"))
        print(f"  checkpoint saved -> {out_path}", flush=True)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
