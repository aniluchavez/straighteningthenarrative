"""
pc5_layer_analysis.py
========================
Fixed top-5 PCA (matched, both sides), dur80/40 window, spike rate.
Computes word-level curvature/displacement/position correlations for every
layer of every model, then aggregates into low/mid/high thirds.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_geometry_analysis_pca20 import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, MIN_PAIRS,
)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

K = 5
MODELS = ["bert-base", "roberta-base", "gpt2", "gpt2-medium", "gpt2-xl",
          "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]

def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2: return None
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2: return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)

def safe_r(x, y, min_n=15):
    fin = np.isfinite(x) & np.isfinite(y)
    if fin.sum() < min_n: return float("nan")
    r, _ = pearsonr(x[fin], y[fin]); return float(r)

print("Building per-patient neural features (dur80/40, rate)...", flush=True)
neu_cache = {}
for pid in PODCAST_PATIENTS:
    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        continue
    data0 = load_podcast_patient(pid, region="hippocampus", model_tag=MODELS[0], context_words=200)
    if data0 is None: continue
    n0 = min(data0["n_words"], len(onset_ms))
    preg = _convo_regions(pid)
    neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n0], offset_ms[:n0], 80, 40, rate=False)
    if neu_vecs is None: continue
    neu_cache[pid] = dict(neu_vecs=neu_vecs, n0=n0)
print(f"  {len(neu_cache)} patients", flush=True)

all_results = {}
for model in MODELS:
    print(f"=== {model} ===", flush=True)
    per_patient_curv, per_patient_disp, per_patient_pos = [], [], []
    for pid in PODCAST_PATIENTS:
        if pid not in neu_cache: continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=model, context_words=200)
        if data is None: continue
        n = min(data["n_words"], neu_cache[pid]["n0"])
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        if valid.sum() < MIN_PAIRS: continue

        Y_neu = pca_fixed(neu_cache[pid]["neu_vecs"][:n])
        if Y_neu is None: continue
        c_neu_full = trajectory_curvature(Y_neu)
        d_neu_full = trajectory_displacement(Y_neu)
        p_neu_full = Y_neu[:, 0]
        m_c = min(len(c_neu_full), len(valid))
        c_neu = c_neu_full[:m_c][valid[:m_c]]
        m_d = min(len(d_neu_full), len(valid))
        d_neu = d_neu_full[:m_d][valid[:m_d]]
        m_p = min(len(p_neu_full), len(valid))
        p_neu = p_neu_full[:m_p][valid[:m_p]]

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        n_layers = emb_mm.shape[0]
        curv_L, disp_L, pos_L = [], [], []
        for li in range(n_layers):
            emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
            Y_sem = pca_fixed(emb_l)
            if Y_sem is None:
                curv_L.append(np.nan); disp_L.append(np.nan); pos_L.append(np.nan); continue
            c_sem_full = trajectory_curvature(Y_sem)
            d_sem_full = trajectory_displacement(Y_sem)
            p_sem_full = Y_sem[:, 0]

            mc = min(len(c_sem_full), len(valid))
            c_sem = c_sem_full[:mc][valid[:mc]]
            md = min(len(d_sem_full), len(valid))
            d_sem = d_sem_full[:md][valid[:md]]
            mp = min(len(p_sem_full), len(valid))
            p_sem = p_sem_full[:mp][valid[:mp]]

            n_c = min(len(c_sem), len(c_neu))
            curv_L.append(safe_r(c_sem[:n_c], c_neu[:n_c]))
            n_d = min(len(d_sem), len(d_neu))
            disp_L.append(safe_r(d_sem[:n_d], d_neu[:n_d]))
            n_p = min(len(p_sem), len(p_neu))
            pos_L.append(safe_r(p_sem[:n_p], p_neu[:n_p]))
        del emb_mm
        per_patient_curv.append(curv_L)
        per_patient_disp.append(disp_L)
        per_patient_pos.append(pos_L)
        print(f"  {pid}: n_layers={n_layers}", flush=True)

    all_results[model] = dict(curv=per_patient_curv, disp=per_patient_disp, pos=per_patient_pos)

os.makedirs("results", exist_ok=True)
json.dump(all_results, open("results/pc5_layer_analysis.json", "w"))
print("\nSaved results/pc5_layer_analysis.json")

print("\n\n=== SUMMARY: all-layers profile + low/mid/high thirds ===")
print(f"{'model':14s} {'metric':6s} {'peak':>8s} {'@layer':>8s}  {'low-third':>10s} {'mid-third':>10s} {'high-third':>10s}")
for model in MODELS:
    for metric in ["curv", "disp", "pos"]:
        rows = [r for r in all_results[model][metric] if len(r)]
        if not rows: continue
        mat = np.array(rows, dtype=float)
        mu = np.nanmean(mat, axis=0)
        n_l = len(mu)
        peak_layer = int(np.nanargmax(np.abs(mu[1:]))) + 1
        peak_val = mu[peak_layer]
        low = np.nanmean(mu[0:max(1,int(n_l*0.33))])
        mid = np.nanmean(mu[int(n_l*0.33):int(n_l*0.67)+1])
        high = np.nanmean(mu[int(n_l*0.67):])
        print(f"{model:14s} {metric:6s} {peak_val:+8.3f} {peak_layer:>4d}/{n_l:<3d}  {low:+10.3f} {mid:+10.3f} {high:+10.3f}")
