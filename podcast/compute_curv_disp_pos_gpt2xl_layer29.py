"""
compute_curv_disp_pos_gpt2xl_layer29.py
==========================================
Curvature vs. displacement vs. position ablation (PC8-matched, dur80/40
window, raw spike counts, chunk=3), restricted to gpt2-xl LAYER 29 --
the peak layer found in the pairwise (RSA-style) semantic-neural distance
sweep (sweep_pairwise_distance_corr_gpt2xl_pseudopop_layers.py).

The earlier ablation table (paired_fig_pc8_chunk3_merged.json, used for
the tent_contentfig1 left panel) averaged curv/disp/pos correlations
ACROSS ALL LAYERS per patient before reporting a single gpt2-xl number
(r=0.0948 for curvature) -- it was never computed at a single fixed
layer. This script isolates layer 29 specifically to see whether the
same curvature-dominance pattern holds there, not just in the
all-layers average.

Same design as podcast/paired_fig_pc8_chunk3_extra.py, single layer only.

Saves: results/curv_disp_pos_gpt2xl_layer29.json

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u compute_curv_disp_pos_gpt2xl_layer29.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import trajectory_curvature, trajectory_displacement, build_valid_mask
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

K = 8
CHUNK_SIZE = 3
MODEL = "gpt2-xl"
LAYER = 29


def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2:
        return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def chunk_r(x, y, chunk_size=CHUNK_SIZE, min_chunks=3):
    fin = np.isfinite(x) & np.isfinite(y)
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        f = fin[sl]
        if f.sum() < chunk_size // 2:
            continue
        xa.append(np.mean(x[sl][f])); ya.append(np.mean(y[sl][f]))
    if len(xa) < min_chunks:
        return float("nan")
    r, _ = pearsonr(xa, ya)
    return float(r)


def main():
    per_patient = {}
    for pid in PODCAST_PATIENTS:
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n0 = min(data["n_words"], len(onset_ms))
        preg = _convo_regions(pid)
        neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n0], offset_ms[:n0], 80, 40, rate=False)
        if neu_vecs is None:
            continue
        n = min(data["n_words"], n0)
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        if valid.sum() < 20:
            continue

        Y_neu = pca_fixed(neu_vecs[:n])
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)
        d_neu_full = trajectory_displacement(Y_neu)
        p_neu_full = Y_neu[:, 0]
        mc = min(len(c_neu_full), len(valid)); c_neu = c_neu_full[:mc][valid[:mc]]
        md = min(len(d_neu_full), len(valid)); d_neu = d_neu_full[:md][valid[:md]]
        mp = min(len(p_neu_full), len(valid)); p_neu = p_neu_full[:mp][valid[:mp]]

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_l = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        Y_sem = pca_fixed(emb_l)
        if Y_sem is None:
            continue
        c_sem_full = trajectory_curvature(Y_sem)
        d_sem_full = trajectory_displacement(Y_sem)
        p_sem_full = Y_sem[:, 0]

        mcc = min(len(c_sem_full), len(valid)); c_sem = c_sem_full[:mcc][valid[:mcc]]
        mdd = min(len(d_sem_full), len(valid)); d_sem = d_sem_full[:mdd][valid[:mdd]]
        mpp = min(len(p_sem_full), len(valid)); p_sem = p_sem_full[:mpp][valid[:mpp]]

        nc = min(len(c_sem), len(c_neu)); r_curv = chunk_r(c_neu[:nc], c_sem[:nc])
        nd = min(len(d_sem), len(d_neu)); r_disp = chunk_r(d_neu[:nd], d_sem[:nd])
        npz = min(len(p_sem), len(p_neu)); r_pos = chunk_r(p_neu[:npz], p_sem[:npz])

        per_patient[pid] = dict(curv=r_curv, disp=r_disp, pos=r_pos)
        print(f"  {pid}: curv={r_curv:+.4f}  disp={r_disp:+.4f}  pos={r_pos:+.4f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(per_patient, open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29.json"), "w"), indent=2)

    print(f"\n=== gpt2-xl layer {LAYER}, PC8-matched, chunk=3, dur80/40, n={len(per_patient)} patients ===")
    for metric, label in [("curv", "Curvature"), ("disp", "Displacement"), ("pos", "Position")]:
        vals = np.array([v[metric] for v in per_patient.values() if np.isfinite(v[metric])])
        n = len(vals)
        t, p = ttest_1samp(vals, 0.0) if n >= 2 else (np.nan, np.nan)
        sem = vals.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan
        n_pos = int((vals > 0).sum())
        print(f"  {label:14s} mean={vals.mean():+.4f}  sem={sem:.4f}  t={t:.2f}  p={p:.2e}  n_pos={n_pos}/{n}")


if __name__ == "__main__":
    main()
