"""
compute_model_peaklayer_randomdim_check.py
====================================
Tests whether each model's peak-layer curvature correlation
(compute_model_comparison_bootstrap_dur8040.py -- almost every model peaks
at layer 0/1, the near-raw embedding layer, suspiciously suggesting a
lexical artifact rather than semantic alignment) depends on picking the
TOP-20 PCA components specifically, or holds for ANY 20-dim subspace.

Two random-dimension schemes at each model's peak layer, pooled across all
13 patients (same chunk-pair + pooled-pearsonr machinery as
compute_podcast_randomdim_bootstrap.py):
  "raw" -- random 20 of the raw embedding dims, no PCA.
  "pca" -- random 20 components out of a K_MAX=300 master PCA fit.
Compared against the deterministic top-20 PCA reference (what the peak-layer
ranking used).

Neural side: dur80/40 (onset+80ms/offset+40ms), rate-normalized.

Saves: results/model_peaklayer_randomdim_check.json
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

from run_geometry_analysis_pca20 import trajectory_curvature, build_valid_mask, fit_neural_pca
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

REGION = "hippocampus"
CHUNK = 15
K_MAX = 300
SEM_K = 20
N_BOOT = 300
SEED = 0
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "model_peaklayer_randomdim_check.json")

PEAK_LAYER = {
    "roberta-base": 1, "llama-2-7b": 1, "llama-3.1-8b": 1, "opt-350m": 6,
    "gemma-2-9b": 0, "mistral-7b": 2, "gpt2-medium": 1, "bert-base": 1,
    "gpt2-xl": 3, "gpt2": 1,
}


def _chunk_pairs(x, y, cs, min_frac=0.5):
    n = len(x)
    out = []
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < cs * min_frac:
            continue
        out.append((np.mean(x[sl][fin]), np.mean(y[sl][fin])))
    return out


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def get_neural_curv(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    preg = _convo_regions(pid)
    n = min(data0["n_words"], len(onset_ms))
    neu_vecs = windowed_neu_vecs(pid, REGION, preg, onset_ms[:n], offset_ms[:n],
                                  ONSET_PAD_MS, OFFSET_PAD_MS, rate=True)
    if neu_vecs is None:
        return None
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, _ = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    return n, valid, c_neu


def prep_layer(pid, model, layer, n):
    data_m = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
    if data_m is None:
        return None
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    if layer >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_L = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    if emb_L.std() == 0:
        return None
    Xs = StandardScaler().fit_transform(emb_L)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, K_MAX)
    if max_pc < SEM_K:
        return None
    pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
    Xpca = pca.transform(Xs)
    return dict(raw=Xs, pca=Xpca)


def pool_and_corr(neural, sem_curv_by_pid):
    rows_s, rows_n = [], []
    for pid, c_sem in sem_curv_by_pid.items():
        n, valid, c_neu = neural[pid]
        pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
        if len(pairs) < 3:
            continue
        s = zscore([p[0] for p in pairs]); nz = zscore([p[1] for p in pairs])
        rows_s += list(s); rows_n += list(nz)
    if len(rows_s) < 10:
        return None
    r, _ = pearsonr(rows_s, rows_n)
    return float(r)


def main():
    rng = np.random.default_rng(SEED)
    neural = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_curv(pid)
        if res is not None:
            neural[pid] = res
    print(f"{len(neural)} patients with neural data", flush=True)

    results = {}
    for model, layer in PEAK_LAYER.items():
        print(f"\n=== {model} (layer {layer}) ===", flush=True)
        layer_cache = {}
        for pid in neural:
            prep = prep_layer(pid, model, layer, neural[pid][0])
            if prep is not None:
                layer_cache[pid] = prep
        pids = list(layer_cache.keys())
        if len(pids) < 3:
            results[model] = None
            continue

        # deterministic top-20 reference
        sem_topk = {pid: trajectory_curvature(layer_cache[pid]["pca"][:, :SEM_K])[neural[pid][1]] for pid in pids}
        topk_r = pool_and_corr(neural, sem_topk)

        boot = {"raw": [], "pca": []}
        for b in range(N_BOOT):
            for scheme in ("raw", "pca"):
                sem_b = {}
                for pid in pids:
                    X = layer_cache[pid][scheme]
                    if X.shape[1] < SEM_K:
                        continue
                    cols = rng.choice(X.shape[1], size=SEM_K, replace=False)
                    sem_b[pid] = trajectory_curvature(X[:, cols])[neural[pid][1]]
                r_b = pool_and_corr(neural, sem_b)
                if r_b is not None:
                    boot[scheme].append(r_b)
            if (b + 1) % 100 == 0:
                print(f"  boot {b+1}/{N_BOOT}", flush=True)

        out = dict(layer=layer, topk_r=topk_r)
        for scheme in ("raw", "pca"):
            arr = np.array(boot[scheme])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[scheme] = dict(boot_mean=float(arr.mean()), ci_lo=float(lo), ci_hi=float(hi), n=len(arr))
            in_ci = lo <= topk_r <= hi
            print(f"  {scheme}: boot_mean={arr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  top-20 r={topk_r:+.4f}  in_CI={in_ci}", flush=True)
        results[model] = out
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
