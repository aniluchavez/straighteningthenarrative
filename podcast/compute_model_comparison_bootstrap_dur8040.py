"""
compute_model_comparison_bootstrap_dur8040.py
====================================
Fairer model-comparison test than picking each model's single best-looking
layer (compute_cross_space_layers_podcast_pca20_dur8040.py's peak per model
is a noisy single point, not validated). For each model, at that model's own
peak-curvature layer (already identified), pool chunk-level (curvature)
pairs across all 13 patients and run a patient-cluster bootstrap (resample
patients with replacement, 2000 draws) to get r_curv +/- 95% CI. Ranks
models by bootstrap mean and reports whether CIs overlap the top model's.

Neural side: dur80/40 (onset+80ms/offset+40ms), rate-normalized -- same
validated windowing as the rest of this investigation.
Semantic side: PCA20-reduced per layer (sem_pca_reduce), matching the
pca20 pipeline convention.

Saves: results/model_comparison_bootstrap_dur8040.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import (
    trajectory_curvature, build_valid_mask, fit_neural_pca, sem_pca_reduce,
)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

REGION = "hippocampus"
CHUNK = 15
N_BOOT = 2000
SEED = 0
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "model_comparison_bootstrap_dur8040.json")

# Each model's peak-curvature layer, from cross_space_layers_podcast_pca20_dur8040_*.json
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


def bootstrap_r(df, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    patients = df.patient.unique()
    r_obs, _ = pearsonr(df.sem_z, df.neu_z)
    boot_r = []
    for _ in range(n_boot):
        samp = rng.choice(patients, size=len(patients), replace=True)
        bd = pd.concat([df[df.patient == p] for p in samp])
        if len(bd) < 3:
            continue
        r_b, _ = pearsonr(bd.sem_z, bd.neu_z)
        boot_r.append(r_b)
    boot_r = np.array(boot_r)
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    return float(r_obs), float(boot_r.mean()), float(lo), float(hi)


def main():
    neural = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_curv(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        print(f"  {pid}: n_words={res[0]}", flush=True)

    results = {}
    for model, layer in PEAK_LAYER.items():
        rows = []
        for pid, (n, valid, c_neu) in neural.items():
            data_m = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
            if data_m is None:
                continue
            emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
            if layer >= emb_mm.shape[0]:
                del emb_mm
                continue
            emb_L = np.array(emb_mm[layer, :n], dtype=np.float64)
            del emb_mm
            emb_L = sem_pca_reduce(emb_L)
            c_sem = trajectory_curvature(emb_L)[valid]
            pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
            if len(pairs) < 3:
                continue
            s = zscore([p[0] for p in pairs]); nz = zscore([p[1] for p in pairs])
            rows += [(pid, si, nzi) for si, nzi in zip(s, nz)]

        if len(rows) < 10:
            results[model] = None
            print(f"  {model}: insufficient data")
            continue
        df = pd.DataFrame(rows, columns=["patient", "sem_z", "neu_z"])
        r_obs, r_boot_mean, lo, hi = bootstrap_r(df)
        results[model] = dict(layer=layer, r_obs=r_obs, boot_mean=r_boot_mean, ci_lo=lo, ci_hi=hi, n_chunks=len(rows))
        print(f"  {model:15s} layer={layer:3d}  r={r_obs:+.4f}  boot_mean={r_boot_mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")
    print("\nRanked by bootstrap mean:")
    ranked = sorted([(k, v) for k, v in results.items() if v], key=lambda kv: -kv[1]["boot_mean"])
    top_lo = ranked[0][1]["ci_lo"] if ranked else None
    for model, v in ranked:
        overlaps_top = "  <-- overlaps top" if top_lo is not None and v["ci_hi"] >= top_lo and model != ranked[0][0] else ""
        print(f"  {model:15s} boot_mean={v['boot_mean']:+.4f}  [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]{overlaps_top}")


if __name__ == "__main__":
    main()
