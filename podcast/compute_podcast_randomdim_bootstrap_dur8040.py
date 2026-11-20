"""
compute_podcast_randomdim_bootstrap_dur8040.py
====================================
compute_podcast_randomdim_bootstrap.py, but the neural side uses the
per-word DURATION-LOCKED window [onset+80ms, offset+40ms] with spike RATE
(count / window duration) instead of the pipeline-default fixed 500ms
window with raw spike count -- per compute_pca_sweep_podcast_dur80_40_rate.py's
finding that this window/rate combo is the unconfounded version (raw count
in a duration-locked window mechanically inflates displacement; rate
normalization removes that artifact while curvature is unaffected either way).

Two random-dimension schemes (both run):
  "raw" -- random subset of the RAW gpt2 embedding dims (768), no PCA at all.
  "pca" -- random subset of components out of a K_MAX=300 master PCA fit.

Three alignment metrics: curvature, displacement, position (see
compute_podcast_randomdim_bootstrap.py for definitions -- unchanged here).

Model: gpt2 only. Region: hippocampus. Dataset: podcast, all 13 patients.

Saves: results/podcast_randomdim_bootstrap_dur8040.json
"""
import json, os, sys
import numpy as np
import pandas as pd
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
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

MODEL = "gpt2"
REGION = "hippocampus"
CHUNK = 15
K_MAX = 300
N_BOOT = 500
SEED = 0
ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40

RESULTS = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)
OUT_PATH = os.path.join(RESULTS, "podcast_randomdim_bootstrap_dur8040.json")


def position(X):
    """||X_t - mean(X)||, length n. Rotation/translation-invariant 0th-deriv
    counterpart to curvature/displacement."""
    X = np.asarray(X, dtype=np.float64)
    c = X.mean(axis=0, keepdims=True)
    return np.linalg.norm(X - c, axis=1)


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


def get_neural_data(pid):
    try:
        data0 = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    except (AssertionError, FileNotFoundError, OSError):
        return None
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
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    p_neu = position(Y_neu)[1:-1][valid]
    return dict(n=n, valid=valid, c_neu=c_neu, d_neu=d_neu, p_neu=p_neu, n_pcs=n_pcs)


def metrics_from_X(X, valid, n):
    c = trajectory_curvature(X)[valid]
    d = trajectory_displacement(X)[:n - 2][valid]
    p = position(X)[1:-1][valid]
    return c, d, p


def prep_patient_layers(pid, n, matched_k):
    """Cache per-layer standardized raw embedding + master PCA transform once
    per patient; random draws only re-slice columns (cheap)."""
    data_m = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    if data_m is None:
        return None
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]
    raw_layers, pca_layers = [], []
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        if emb_L.std() == 0:
            raw_layers.append(None); pca_layers.append(None)
            continue
        Xs = StandardScaler().fit_transform(emb_L)
        raw_layers.append(Xs)
        max_pc = min(Xs.shape[1], Xs.shape[0] - 1, K_MAX)
        if max_pc < matched_k:
            pca_layers.append(None)
            continue
        pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
        pca_layers.append(pca.transform(Xs))
    del emb_mm
    return dict(raw=raw_layers, pca=pca_layers)


def topk_metrics(layers_transformed, matched_k, valid, n):
    """Deterministic top-K reference (top raw dims / top PCA components)."""
    c_all, d_all, p_all = [], [], []
    for X in layers_transformed:
        if X is None or X.shape[1] < matched_k:
            continue
        Xk = X[:, :matched_k]
        c, d, p = metrics_from_X(Xk, valid, n)
        c_all.append(c); d_all.append(d); p_all.append(p)
    if not c_all:
        return None
    return np.mean(c_all, axis=0), np.mean(d_all, axis=0), np.mean(p_all, axis=0)


def randomdim_metrics(layers_transformed, matched_k, valid, n, rng):
    c_all, d_all, p_all = [], [], []
    for X in layers_transformed:
        if X is None or X.shape[1] < matched_k:
            continue
        cols = rng.choice(X.shape[1], size=matched_k, replace=False)
        Xk = X[:, cols]
        c, d, p = metrics_from_X(Xk, valid, n)
        c_all.append(c); d_all.append(d); p_all.append(p)
    if not c_all:
        return None
    return np.mean(c_all, axis=0), np.mean(d_all, axis=0), np.mean(p_all, axis=0)


def pool_and_corr(neural, sem_by_pid):
    """Cross-patient pearsonr for each metric, pooling chunk-averaged pairs."""
    rows = {"curv": [], "disp": [], "pos": []}
    for pid, (c_sem, d_sem, p_sem) in sem_by_pid.items():
        c_neu, d_neu, p_neu = neural[pid]["c_neu"], neural[pid]["d_neu"], neural[pid]["p_neu"]
        for key, sem_arr, neu_arr in (("curv", c_sem, c_neu), ("disp", d_sem, d_neu), ("pos", p_sem, p_neu)):
            pairs = _chunk_pairs(sem_arr, neu_arr, CHUNK)
            if len(pairs) < 3:
                continue
            s = zscore([p[0] for p in pairs]); nz = zscore([p[1] for p in pairs])
            rows[key] += list(zip(s, nz))
    out = {}
    for key, pts in rows.items():
        if len(pts) < 10:
            out[key] = None
            continue
        s = np.array([p[0] for p in pts]); nz = np.array([p[1] for p in pts])
        r, _ = pearsonr(s, nz)
        out[key] = float(r)
    return out


def main():
    rng = np.random.default_rng(SEED)

    neural = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP (no neural data)")
            continue
        neural[pid] = res
        print(f"  {pid}: n_words={res['n']}  neural_n_pcs={res['n_pcs']}", flush=True)

    print("\nCaching per-layer semantic embeddings (raw + master PCA)...", flush=True)
    layer_cache = {}
    for pid, ndata in neural.items():
        prep = prep_patient_layers(pid, ndata["n"], ndata["n_pcs"])
        if prep is None:
            print(f"  {pid}: SKIP (no semantic embedding)")
            continue
        layer_cache[pid] = prep
        print(f"  {pid}: cached", flush=True)

    pids = [p for p in neural if p in layer_cache]
    print(f"\n{len(pids)} patients with both neural + semantic data: {pids}")

    results = {}
    for scheme in ("raw", "pca"):
        print(f"\n=== scheme: {scheme} ===", flush=True)

        # deterministic top-K reference
        sem_topk = {}
        for pid in pids:
            layers = layer_cache[pid][scheme]
            r = topk_metrics(layers, neural[pid]["n_pcs"], neural[pid]["valid"], neural[pid]["n"])
            if r is not None:
                sem_topk[pid] = r
        topk_r = pool_and_corr(neural, sem_topk)
        print(f"  top-K reference: {topk_r}", flush=True)

        # random-dimension bootstrap
        boot_r = {"curv": [], "disp": [], "pos": []}
        for b in range(N_BOOT):
            sem_b = {}
            for pid in pids:
                layers = layer_cache[pid][scheme]
                r = randomdim_metrics(layers, neural[pid]["n_pcs"], neural[pid]["valid"], neural[pid]["n"], rng)
                if r is not None:
                    sem_b[pid] = r
            r_b = pool_and_corr(neural, sem_b)
            for key in boot_r:
                if r_b[key] is not None:
                    boot_r[key].append(r_b[key])
            if (b + 1) % 50 == 0:
                print(f"  boot {b+1}/{N_BOOT}", flush=True)

        scheme_out = {}
        for key in ("curv", "disp", "pos"):
            arr = np.array(boot_r[key])
            if len(arr) < 10:
                scheme_out[key] = None
                continue
            lo, hi = np.percentile(arr, [2.5, 97.5])
            scheme_out[key] = dict(
                topk=topk_r[key],
                boot_mean=float(arr.mean()),
                boot_ci_lo=float(lo),
                boot_ci_hi=float(hi),
                n_boot=int(len(arr)),
            )
        results[scheme] = scheme_out
        json.dump(results, open(OUT_PATH, "w"), indent=2)
        print(f"  checkpoint saved -> {OUT_PATH}", flush=True)

    print(f"\nSaved -> {OUT_PATH}\n\nSummary:")
    for scheme, scheme_out in results.items():
        print(f"\n  scheme={scheme}")
        for key, r in scheme_out.items():
            if r is None:
                print(f"    {key}: insufficient data")
                continue
            in_ci = r["boot_ci_lo"] <= r["topk"] <= r["boot_ci_hi"]
            print(f"    {key}: top-K r={r['topk']:+.4f}  random-dim boot mean={r['boot_mean']:+.4f} "
                  f"[{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]  top-K in-CI={in_ci}")


if __name__ == "__main__":
    main()
