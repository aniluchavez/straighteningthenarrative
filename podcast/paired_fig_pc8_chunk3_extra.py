"""
paired_fig_pc8_chunk3.py
===========================
Rebuild of fig_cross_space_paired_pca8_podcast_all13_final, with CHUNK_SIZE=3
instead of 15. Fixed top-8 PCA (matched, both sides), dur80/40 window, raw
spike counts, all 13 podcast patients. Averages within-patient across
layers first, then reports both the cross-model bar chart AND the
per-model breakdown (to show whether position's sign flips by model).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_geometry_analysis_pca20 import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, ACCEPTED_QUAL,
)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs, EXTRA_SOURCES,
)

K = 8
CHUNK_SIZE = 3
MODELS = ["gpt2-large", "gpt2-xl"]
ALL13 = list(PODCAST_PATIENTS) + list(EXTRA_SOURCES.keys())


def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2: return None
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2: return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def chunk_r(x, y, chunk_size=CHUNK_SIZE, min_chunks=3):
    fin = np.isfinite(x) & np.isfinite(y)
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        f = fin[sl]
        if f.sum() < chunk_size // 2: continue
        xa.append(np.mean(x[sl][f])); ya.append(np.mean(y[sl][f]))
    if len(xa) < min_chunks: return float("nan")
    r, _ = pearsonr(xa, ya); return float(r)


print("Building per-patient neural features (dur80/40, count)...", flush=True)
neu_cache = {}
for pid in ALL13:
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

# per model, per patient: mean-across-layers curv/disp/pos (chunk=3)
per_model_patient = {m: {} for m in MODELS}
for model in MODELS:
    print(f"=== {model} ===", flush=True)
    for pid in ALL13:
        if pid not in neu_cache: continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=model, context_words=200)
        if data is None: continue
        n = min(data["n_words"], neu_cache[pid]["n0"])
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        if valid.sum() < 20: continue

        Y_neu = pca_fixed(neu_cache[pid]["neu_vecs"][:n])
        if Y_neu is None: continue
        c_neu_full = trajectory_curvature(Y_neu)
        d_neu_full = trajectory_displacement(Y_neu)
        p_neu_full = Y_neu[:, 0]
        mc = min(len(c_neu_full), len(valid)); c_neu = c_neu_full[:mc][valid[:mc]]
        md = min(len(d_neu_full), len(valid)); d_neu = d_neu_full[:md][valid[:md]]
        mp = min(len(p_neu_full), len(valid)); p_neu = p_neu_full[:mp][valid[:mp]]

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        n_layers = emb_mm.shape[0]
        curv_L, disp_L, pos_L = [], [], []
        for li in range(n_layers):
            emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
            Y_sem = pca_fixed(emb_l)
            if Y_sem is None: continue
            c_sem_full = trajectory_curvature(Y_sem)
            d_sem_full = trajectory_displacement(Y_sem)
            p_sem_full = Y_sem[:, 0]

            mcc = min(len(c_sem_full), len(valid)); c_sem = c_sem_full[:mcc][valid[:mcc]]
            mdd = min(len(d_sem_full), len(valid)); d_sem = d_sem_full[:mdd][valid[:mdd]]
            mpp = min(len(p_sem_full), len(valid)); p_sem = p_sem_full[:mpp][valid[:mpp]]

            nc = min(len(c_sem), len(c_neu)); curv_L.append(chunk_r(c_neu[:nc], c_sem[:nc]))
            nd = min(len(d_sem), len(d_neu)); disp_L.append(chunk_r(d_neu[:nd], d_sem[:nd]))
            npz = min(len(p_sem), len(p_neu)); pos_L.append(chunk_r(p_neu[:npz], p_sem[:npz]))
        del emb_mm
        if curv_L:
            per_model_patient[model][pid] = dict(
                curv=float(np.nanmean(curv_L)), disp=float(np.nanmean(disp_L)), pos=float(np.nanmean(pos_L)))

json.dump(per_model_patient, open("results/paired_fig_pc8_chunk3_extra.json", "w"))

print("\n\n=== Per-model mean (averaged across patients, chunk=3, PC8) ===")
model_means = {"curv": [], "disp": [], "pos": []}
for model in MODELS:
    vals = per_model_patient[model]
    if not vals: continue
    curv = np.array([v["curv"] for v in vals.values()])
    disp = np.array([v["disp"] for v in vals.values()])
    pos = np.array([v["pos"] for v in vals.values()])
    print(f"  {model:14s} curv={np.nanmean(curv):+.4f}  disp={np.nanmean(disp):+.4f}  pos={np.nanmean(pos):+.4f}  (n={len(vals)})")
    model_means["curv"].append(np.nanmean(curv))
    model_means["disp"].append(np.nanmean(disp))
    model_means["pos"].append(np.nanmean(pos))

print("\n=== CROSS-MODEL AVERAGE (what the bar chart would show), chunk=3, PC8 ===")
for metric in ["curv", "disp", "pos"]:
    vals = np.array(model_means[metric])
    t, p = ttest_1samp(vals, 0.0)
    print(f"  {metric}: mean={vals.mean():+.4f}  sd_across_models={vals.std():.4f}  t-test-vs-0 p={p:.4f}  (n_models={len(vals)})")

print("\n=== Compare to CHUNK=15 for reference (rerun with CHUNK_SIZE=15 if needed) ===")
