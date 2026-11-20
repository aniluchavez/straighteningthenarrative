"""
compute_jabberwocky_cross_space_chunk_sweep.py
==================================================
Jabberwocky version of ../compute_cross_space_chunk_sweep.py: sweep chunk
sizes, for each model x patient average all layers, chunk-average LLM and
neural metrics -> pearsonr. Same methodology as the convo figure this
mirrors (fig_cross_space_paired.pdf) -- WIN_MS=500 (jabberwocky_loader
default), 9 models, all layers averaged (not LOO-selected).

Saves: results/jabberwocky_cross_space_chunk_sweep.json
Structure: { chunk_size: { model: { metric: [r_per_patient] } } }
"""
import json, os, sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_PERM = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient

RESULTS     = os.path.join(BASE_DIR, "results")
REGION      = "hippocampus"
CHUNK_SIZES = [3, 5, 8, 10, 15, 20, 25, 30]
MIN_PAIRS   = 50

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
    out_path = os.path.join(RESULTS, "jabberwocky_cross_space_chunk_sweep.json")
    rng = np.random.default_rng(42)

    results = {cs: {m: {"disp": [], "curv": [], "pos": [],
                         "null_curv_mean": [], "null_curv_std": [],
                         "null_disp_mean": [], "null_disp_std": []}
                    for m in MODELS}
               for cs in CHUNK_SIZES}

    for model in MODELS:
        print(f"\n  {model}", flush=True)
        for pid in JABBERWOCKY_PATIENTS:
            data = load_jabberwocky_patient(pid, region=REGION, model_tag=model, context_words=200)
            if data is None:
                print(f"    {pid}: SKIP (no data)")
                continue

            n       = data["n_words"]
            surp    = data["surp"]
            sp      = data["sent_pos"]
            is_turn = data["is_turn"]

            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < MIN_PAIRS:
                print(f"    {pid}: SKIP (too few valid words)")
                continue

            interior = np.arange(1, n - 1)

            Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
            if Y_neu is None:
                print(f"    {pid}: SKIP (PCA failed)")
                continue

            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
            p_neu = _pc1_series(Y_neu)[interior][valid]

            emb_mm   = np.load(data["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]

            layer_r = {cs: {"disp": [], "curv": [], "pos": []} for cs in CHUNK_SIZES}
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

            for cs in CHUNK_SIZES:
                for metric in ("disp", "curv", "pos"):
                    vals = [v for v in layer_r[cs][metric] if not np.isnan(v)]
                    results[cs][model][metric].append(
                        float(np.mean(vals)) if vals else float("nan"))

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

            print(f"    {pid}: done ({n_layers} layers)", flush=True)

    out = {str(cs): results[cs] for cs in CHUNK_SIZES}
    json.dump(out, open(out_path, "w"))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
