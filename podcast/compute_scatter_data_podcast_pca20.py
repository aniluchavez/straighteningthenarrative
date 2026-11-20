"""
compute_scatter_data_podcast_pca20.py
=======================
Podcast equivalent of ../compute_scatter_data_pca20.py: single model
(llama-3.1-8b), PCA(20)-reduced-trajectory version. Chunk-level curvature
arrays for scatter panels and per-layer r(neural, LLM) profiles split by
surprisal quartile, plus the same for displacement.

Saves results/scatter_data_podcast_pca20_llama-3.1-8b_hippocampus.json

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 compute_scatter_data_podcast_pca20.py
"""

import json, os, sys
import numpy as np
from scipy.stats import zscore, pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    MIN_PAIRS, find_best_sentence_traj, sem_pca_reduce,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

RESULTS    = os.path.join(BASE_DIR, "results")
MODEL      = "llama-3.1-8b"
REGION     = "hippocampus"
CHUNK_SIZE = 15
N_LAYERS   = 33
SURP_HI    = 75
SURP_LO    = 25


def _chunk_averages(arr_sem, arr_neu, arr_surp, chunk_size):
    n = len(arr_sem)
    sem_out, neu_out, surp_out = [], [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        s, n_, sp = arr_sem[sl], arr_neu[sl], arr_surp[sl]
        fin = np.isfinite(s) & np.isfinite(n_) & np.isfinite(sp)
        if fin.sum() < chunk_size // 2:
            continue
        sem_out.append(float(np.mean(s[fin])))
        neu_out.append(float(np.mean(n_[fin])))
        surp_out.append(float(np.mean(sp[fin])))
    return sem_out, neu_out, surp_out


def _safe_r(x, y, mask, min_n=15):
    m = mask & np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan")
    return float(pearsonr(x[m], y[m])[0])


def main():
    json_path = os.path.join(RESULTS, f"geometry_podcast_pca20_{MODEL}_ctx200_{REGION}.json")
    records   = json.load(open(json_path))

    out              = []
    traj_candidates  = []

    for rec in records:
        pid  = rec["pid"]
        print(f"  {pid}")

        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None:
            continue

        n       = data["n_words"]
        surp    = data["surp"]
        sp      = data["sent_pos"]
        is_turn = data["is_turn"]

        valid  = build_valid_mask(n, surp, sp, is_turn)
        if valid.sum() < MIN_PAIRS:
            continue

        centre_idx = np.arange(1, n - 1)[valid]
        surp_y     = surp[centre_idx + 1]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        d_neu = trajectory_displacement(Y_neu)[1:][valid]

        fin_base = np.isfinite(c_neu) & np.isfinite(d_neu) & np.isfinite(surp_y)
        surp_fin = surp_y[fin_base]
        hi_thr   = np.percentile(surp_fin, SURP_HI)
        lo_thr   = np.percentile(surp_fin, SURP_LO)

        emb_mm = np.load(data["emb_path"], mmap_mode="r")

        r_all_layers     = []
        r_hi_layers      = []
        r_lo_layers      = []
        r_disp_all_layers = []
        r_disp_hi_layers  = []
        r_disp_lo_layers  = []
        all_layer_chunks = {}
        all_layer_disp_chunks = {}

        for L in range(N_LAYERS):
            emb_L  = np.array(emb_mm[L, :n], dtype=np.float64)
            emb_L  = sem_pca_reduce(emb_L)
            c_semL = trajectory_curvature(emb_L)[valid]
            d_semL = trajectory_displacement(emb_L)[1:][valid]
            fin_L  = np.isfinite(c_semL) & np.isfinite(c_neu) & np.isfinite(surp_y)
            fin_dL = np.isfinite(d_semL) & np.isfinite(d_neu) & np.isfinite(surp_y)

            mask_hi = fin_L & (surp_y >= hi_thr)
            mask_lo = fin_L & (surp_y <= lo_thr)
            mask_disp_hi = fin_dL & (surp_y >= hi_thr)
            mask_disp_lo = fin_dL & (surp_y <= lo_thr)

            r_all_layers.append(_safe_r(c_semL, c_neu, fin_L))
            r_hi_layers.append(_safe_r(c_semL,  c_neu, mask_hi))
            r_lo_layers.append(_safe_r(c_semL,  c_neu, mask_lo))
            r_disp_all_layers.append(_safe_r(d_semL, d_neu, fin_dL))
            r_disp_hi_layers.append(_safe_r(d_semL, d_neu, mask_disp_hi))
            r_disp_lo_layers.append(_safe_r(d_semL, d_neu, mask_disp_lo))

            if fin_L.sum() >= CHUNK_SIZE:
                c_semL_z = zscore(c_semL[fin_L])
                c_neuL_z = zscore(c_neu[fin_L])
                surpL    = surp_y[fin_L]
                ch_sem, ch_neu, ch_surp = _chunk_averages(
                    c_semL_z, c_neuL_z, surpL, CHUNK_SIZE)
                all_layer_chunks[str(L)] = dict(sem=ch_sem, neu=ch_neu, surp=ch_surp)
            if fin_dL.sum() >= CHUNK_SIZE:
                d_semL_z = zscore(d_semL[fin_dL])
                d_neuL_z = zscore(d_neu[fin_dL])
                surpD    = surp_y[fin_dL]
                ch_sem, ch_neu, ch_surp = _chunk_averages(
                    d_semL_z, d_neuL_z, surpD, CHUNK_SIZE)
                all_layer_disp_chunks[str(L)] = dict(sem=ch_sem, neu=ch_neu, surp=ch_surp)

        del emb_mm

        traj_surp_range = (rec.get("traj_viz") or {}).get("surp_range", -1.0)
        traj_candidates.append(dict(pid=pid, surp_range=float(traj_surp_range), data=data))

        out.append(dict(
            pid              = pid,
            all_layer_chunks = all_layer_chunks,
            r_all_layers     = r_all_layers,
            r_hi_layers      = r_hi_layers,
            r_lo_layers      = r_lo_layers,
            all_layer_disp_chunks = all_layer_disp_chunks,
            r_disp_all_layers     = r_disp_all_layers,
            r_disp_hi_layers      = r_disp_hi_layers,
            r_disp_lo_layers      = r_disp_lo_layers,
        ))

    r_hi_mat = np.array([r["r_hi_layers"] for r in out], dtype=float)
    r_lo_mat = np.array([r["r_lo_layers"] for r in out], dtype=float)
    delta    = np.nanmean(r_hi_mat, axis=0) - np.nanmean(r_lo_mat, axis=0)
    best_layer = int(np.argmax(delta))
    print(f"\n  best_layer = {best_layer}  (Δr = {delta[best_layer]:.4f})")

    r_disp_hi_mat = np.array([r["r_disp_hi_layers"] for r in out], dtype=float)
    r_disp_lo_mat = np.array([r["r_disp_lo_layers"] for r in out], dtype=float)
    delta_disp    = np.nanmean(r_disp_hi_mat, axis=0) - np.nanmean(r_disp_lo_mat, axis=0)
    best_disp_layer = int(np.argmax(delta_disp))
    print(f"  best_disp_layer = {best_disp_layer}  (Δr = {delta_disp[best_disp_layer]:.4f})")

    # ── Word-level (unchunked) curvature at best_layer, per patient ────────────
    # `out` and `traj_candidates` are appended in lockstep for surviving patients,
    # so zip them to recover each patient's already-loaded data without reloading.
    print(f"  Computing word-level curvature at best_layer={best_layer} for all patients...")
    for rec, cand in zip(out, traj_candidates):
        data = cand["data"]
        n = data["n_words"]
        surp, sp, is_turn = data["surp"], data["sent_pos"], data["is_turn"]
        valid = build_valid_mask(n, surp, sp, is_turn)
        centre_idx = np.arange(1, n - 1)[valid]
        surp_y = surp[centre_idx + 1]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            rec["word_level_at_best_layer"] = None
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        emb_L = np.array(emb_mm[best_layer, :n], dtype=np.float64)
        del emb_mm
        emb_L = sem_pca_reduce(emb_L)
        c_sem = trajectory_curvature(emb_L)[valid]

        fin = np.isfinite(c_sem) & np.isfinite(c_neu) & np.isfinite(surp_y)
        c_sem_f, c_neu_f, surp_f = c_sem[fin], c_neu[fin], surp_y[fin]
        if fin.sum() < 20:
            rec["word_level_at_best_layer"] = None
            continue
        rec["word_level_at_best_layer"] = dict(
            sem=zscore(c_sem_f).tolist(),
            neu=zscore(c_neu_f).tolist(),
            surp=surp_f.tolist(),
        )

    best_cand = max(traj_candidates, key=lambda x: x["surp_range"])
    print(f"  Best traj patient: {best_cand['pid']}  surp_range={best_cand['surp_range']:.2f}")
    print(f"  Computing traj_viz at layer {best_layer}...")
    traj_viz_best = find_best_sentence_traj(best_cand["data"], best_layer)
    print(f"  Computing displacement traj_viz at layer {best_disp_layer}...")
    traj_viz_disp_best = find_best_sentence_traj(best_cand["data"], best_disp_layer)

    payload = dict(
        best_layer          = best_layer,
        best_disp_layer     = best_disp_layer,
        traj_viz_best_layer = traj_viz_best,
        traj_viz_best_disp_layer = traj_viz_disp_best,
        patients            = out,
    )
    out_path = os.path.join(RESULTS, f"scatter_data_podcast_pca20_{MODEL}_{REGION}.json")
    json.dump(payload, open(out_path, "w"))

    n_ch = sum(
        len((r["all_layer_chunks"].get(str(best_layer)) or {}).get("sem", []))
        for r in out
    )
    print(f"\nSaved {len(out)} patients")
    print(f"  best_layer = {best_layer}, Panel C chunks at best layer: {n_ch}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
