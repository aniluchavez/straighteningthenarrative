"""
compute_scatter_data_pca20.py
=======================
PCA(20)-reduced-trajectory variant of compute_scatter_data.py: each
layer's LLM embedding PCA-reduced (SEM_PCA_K=20) before curvature/
displacement -- everything else identical.

Compute chunk-level curvature arrays for scatter panels and per-layer
r(neural, LLM) profiles split by surprisal quartile. Also computes the
same summaries for displacement, aligned to the step arriving at the
surprising word.

Saves results/scatter_data_pca20_llama-3.1-8b_hippocampus.json (distinct
from the raw scatter_data_llama-3.1-8b_hippocampus.json -- never
overwrites that):
  {
    "best_layer":          int   — layer maximising r_hi − r_lo delta
    "best_disp_layer":     int   — same selection for displacement
    "traj_viz_best_layer": dict  — best-sentence traj_viz at best_layer (Panels A/B)
    "traj_viz_best_disp_layer": dict — same sentence trajectory at best_disp_layer
    "patients": [
      {
        "pid":               str
        "all_layer_chunks":  {"0": {sem, neu, surp}, ..., "32": {...}}
        "r_all_layers":      list[float]  (Panel D)
        "r_hi_layers":       list[float]
        "r_lo_layers":       list[float]
        "all_layer_disp_chunks": {...}
        "r_disp_all_layers":  list[float]
        "r_disp_hi_layers":   list[float]
        "r_disp_lo_layers":   list[float]
      }, ...
    ]
  }

Run:
    conda run -n gpt2_embed python3 compute_scatter_data.py
"""

import json, os, sys
import numpy as np
from scipy.stats import zscore, pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient, PATIENTS, MIN_PAIRS, find_best_sentence_traj, sem_pca_reduce,
)

RESULTS    = os.path.join(BASE_DIR, "results")
MODEL      = "llama-3.1-8b"
REGION     = "hippocampus"
CHUNK_SIZE = 15
N_LAYERS   = 33
SURP_HI    = 75   # percentile threshold for high surprisal
SURP_LO    = 25   # percentile threshold for low surprisal


def _chunk_averages(arr_sem, arr_neu, arr_surp, chunk_size):
    """Non-overlapping chunk averages; returns three lists."""
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
    json_path = os.path.join(RESULTS, f"geometry_pca20_{MODEL}_{REGION}.json")
    records   = json.load(open(json_path))

    out              = []
    traj_candidates  = []   # (surp_range, pid, data) for post-loop traj_viz

    for rec in records:
        pid  = rec["pid"]
        print(f"  {pid}")

        preg = PATIENTS.get(pid, {})
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, MODEL)
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
        # trajectory_curvature()[i] is the angle between x[i+1]-x[i]
        # and x[i+2]-x[i+1]. Pair each curvature with surprisal at x[i+2],
        # the endpoint of the second displacement, so no word after the
        # surprising word enters the curvature estimate.
        surp_y     = surp[centre_idx + 1]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        # Displacement paired to the same surprisal word: step arriving at x[i+2].
        d_neu = trajectory_displacement(Y_neu)[1:][valid]

        # per-patient surprisal thresholds for hi/lo split
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
        all_layer_chunks = {}   # str(L) → {sem, neu, surp}
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

            # Scatter chunks at this layer (Panel C selects one layer post-hoc)
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

        # Record traj candidate: use surp_range from geometry JSON if available
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

    # ── Find best_layer: layer with maximum mean(r_hi − r_lo) across patients ──
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

    # ── Compute traj_viz at best_layer for the patient with widest surp range ──
    best_cand = max(traj_candidates, key=lambda x: x["surp_range"])
    print(f"  Best traj patient: {best_cand['pid']}  surp_range={best_cand['surp_range']:.2f}")
    print(f"  Computing traj_viz at layer {best_layer}...")
    traj_viz_best = find_best_sentence_traj(best_cand["data"], best_layer)
    print(f"  Computing displacement traj_viz at layer {best_disp_layer}...")
    traj_viz_disp_best = find_best_sentence_traj(best_cand["data"], best_disp_layer)

    # ── Save ──
    payload = dict(
        best_layer          = best_layer,
        best_disp_layer     = best_disp_layer,
        traj_viz_best_layer = traj_viz_best,
        traj_viz_best_disp_layer = traj_viz_disp_best,
        patients            = out,
    )
    out_path = os.path.join(RESULTS, f"scatter_data_pca20_{MODEL}_{REGION}.json")
    json.dump(payload, open(out_path, "w"))

    n_ch = sum(
        len((r["all_layer_chunks"].get(str(best_layer)) or {}).get("sem", []))
        for r in out
    )
    print(f"\nSaved {len(out)} patients")
    print(f"  best_layer = {best_layer}, Panel C chunks at best layer: {n_ch}")
    n_ch_disp = sum(
        len((r["all_layer_disp_chunks"].get(str(best_disp_layer)) or {}).get("sem", []))
        for r in out
    )
    print(f"  best_disp_layer = {best_disp_layer}, displacement chunks at best layer: {n_ch_disp}")
    print(f"  traj_viz_best_layer: {traj_viz_best is not None}")
    print(f"  traj_viz_best_disp_layer: {traj_viz_disp_best is not None}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
