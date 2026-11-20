"""
compute_cross_space_scatter.py
==============================
Compute chunk-level scatter data for three geometric quantities compared
across LLM (semantic) and neural (hippocampus) spaces:
  - displacement  : step-to-step Euclidean distance along the trajectory
  - curvature     : turning angle (degrees) at each interior vertex
  - position      : first-PC coordinate (where you are in the space)

For each patient at the LOO layer (layer 5 for LLaMA-3.1-8B hippocampus),
computes z-scored arrays in both spaces, 10-word chunk averages, and
per-patient Pearson r for each quantity.

Saves: results/cross_space_scatter_llama-3.1-8b_hippocampus.json

Run:
    cd /scratch/aniluchavez/straighteningthenarrative
    conda run -n gpt2_embed python3 compute_cross_space_scatter.py
"""

import json, os, sys
import numpy as np
from scipy.stats import zscore, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement,
    build_valid_mask, fit_neural_pca, load_patient,
    PATIENTS, MIN_PAIRS
)

RESULTS    = os.path.join(BASE_DIR, "results")
MODEL      = "llama-3.1-8b"
REGION     = "hippocampus"
CHUNK_SIZE = 10
LOO_LAYER  = 5   # from geometry results: all patients use layer 5


def _pc1_series(X):
    """Return z-scored first-PC time series of matrix X (n_words × d)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return np.full(X.shape[0], np.nan)
    Xs = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=1).fit_transform(Xs)[:, 0]
    s = np.std(pc1)
    if s < 1e-12:
        return np.full(len(pc1), np.nan)
    return (pc1 - pc1.mean()) / s


def _chunk_avg(a, b, surp, chunk_size, min_frac=0.5):
    """Non-overlapping chunk averages of aligned arrays a, b, surp."""
    n = len(a)
    a_out, b_out, s_out = [], [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        fa, fb, fs = a[sl], b[sl], surp[sl]
        fin = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(fs)
        if fin.sum() < int(chunk_size * min_frac):
            continue
        a_out.append(float(np.mean(fa[fin])))
        b_out.append(float(np.mean(fb[fin])))
        s_out.append(float(np.mean(fs[fin])))
    return a_out, b_out, s_out


def _safe_r(x, y, min_n=15):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan"), float("nan")
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p)


def main():
    json_path = os.path.join(RESULTS, f"geometry_{MODEL}_{REGION}.json")
    records   = json.load(open(json_path))

    out = []
    for rec in records:
        pid  = rec["pid"]
        print(f"  {pid}")

        preg = PATIENTS.get(pid, {})
        if REGION not in preg:
            print(f"    skipped (no region)")
            continue
        data = load_patient(pid, preg, REGION, MODEL)
        if data is None:
            print(f"    skipped (load failed)")
            continue

        n       = data["n_words"]
        surp    = data["surp"]
        sp      = data["sent_pos"]
        is_turn = data["is_turn"]

        valid = build_valid_mask(n, surp, sp, is_turn)
        if valid.sum() < MIN_PAIRS:
            print(f"    skipped (too few valid)")
            continue

        # interior indices: curvature / displacement both live at interior points
        interior  = np.arange(1, n - 1)
        surp_y    = surp[interior + 1][valid]   # surprisal of next word

        # ── Neural PCA trajectory ──────────────────────────────────────────────
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            print(f"    skipped (neural PCA failed)")
            continue

        # ── Load LLM embeddings at LOO layer ──────────────────────────────────
        emb_mm  = np.load(data["emb_path"], mmap_mode="r")
        emb_opt = np.array(emb_mm[LOO_LAYER, :n], dtype=np.float64)
        del emb_mm

        # ── Curvature ─────────────────────────────────────────────────────────
        c_sem = trajectory_curvature(emb_opt)[valid]
        c_neu = trajectory_curvature(Y_neu)[valid]
        fin_c = np.isfinite(c_sem) & np.isfinite(c_neu) & np.isfinite(surp_y)
        c_sem_z = zscore(c_sem[fin_c])
        c_neu_z = zscore(c_neu[fin_c])
        surp_c  = surp_y[fin_c]
        r_curv, p_curv = _safe_r(c_sem_z, c_neu_z)

        # ── Displacement ───────────────────────────────────────────────────────
        # displacement at interior vertex i = step size from i to i+1
        d_sem_full = trajectory_displacement(emb_opt)[:n - 2][valid]
        d_neu_full = trajectory_displacement(Y_neu)[:n - 2][valid]
        fin_d = np.isfinite(d_sem_full) & np.isfinite(d_neu_full) & np.isfinite(surp_y)
        d_sem_z = zscore(d_sem_full[fin_d])
        d_neu_z = zscore(d_neu_full[fin_d])
        surp_d  = surp_y[fin_d]
        r_disp, p_disp = _safe_r(d_sem_z, d_neu_z)

        # ── Position (PC1 time series) ─────────────────────────────────────────
        # PC1 evaluated at each interior point (same valid mask)
        p_sem_full = _pc1_series(emb_opt)[interior][valid]
        p_neu_full = _pc1_series(Y_neu)[interior][valid]
        fin_p = np.isfinite(p_sem_full) & np.isfinite(p_neu_full) & np.isfinite(surp_y)
        p_sem_z = p_sem_full[fin_p]   # already z-scored from _pc1_series
        p_neu_z = p_neu_full[fin_p]
        surp_p  = surp_y[fin_p]
        r_pos, p_pos = _safe_r(p_sem_z, p_neu_z)

        # ── Chunk averages ─────────────────────────────────────────────────────
        chunk_curvature_sem, chunk_curvature_neu, chunk_surp_c = \
            _chunk_avg(c_sem_z, c_neu_z, surp_c, CHUNK_SIZE)
        chunk_disp_sem, chunk_disp_neu, chunk_surp_d = \
            _chunk_avg(d_sem_z, d_neu_z, surp_d, CHUNK_SIZE)
        chunk_pos_sem, chunk_pos_neu, chunk_surp_p = \
            _chunk_avg(p_sem_z, p_neu_z, surp_p, CHUNK_SIZE)

        print(f"    r_curv={r_curv:+.4f}  r_disp={r_disp:+.4f}  r_pos={r_pos:+.4f}")

        out.append(dict(
            pid              = pid,
            r_curv=r_curv,   p_curv=p_curv,
            r_disp=r_disp,   p_disp=p_disp,
            r_pos=r_pos,     p_pos=p_pos,
            # chunk scatter data
            chunk_curvature_sem=chunk_curvature_sem,
            chunk_curvature_neu=chunk_curvature_neu,
            chunk_surp_c=chunk_surp_c,
            chunk_disp_sem=chunk_disp_sem,
            chunk_disp_neu=chunk_disp_neu,
            chunk_surp_d=chunk_surp_d,
            chunk_pos_sem=chunk_pos_sem,
            chunk_pos_neu=chunk_pos_neu,
            chunk_surp_p=chunk_surp_p,
        ))

    out_path = os.path.join(RESULTS, f"cross_space_scatter_{MODEL}_{REGION}.json")
    json.dump(out, open(out_path, "w"))
    print(f"\nSaved {len(out)} patients → {out_path}")


if __name__ == "__main__":
    main()