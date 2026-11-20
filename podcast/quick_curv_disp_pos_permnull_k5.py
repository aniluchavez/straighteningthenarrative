"""
quick_curv_disp_pos_permnull_v2.py
====================================
Fix over quick_curv_disp_pos_permnull.py: that version used
load_podcast_patient's DEFAULT neural windowing (fixed 500ms raw-count
window from word onset, WIN_MS/LISTEN_OFFSET_MS in run_geometry_analysis.py)
-- not the validated duration-locked, rate-normalized windowing
(windowed_neu_vecs with ONSET_PAD_MS=80, OFFSET_PAD_MS=40, rate=True) this
session established as the correct fix for the duration-window artifact.
This version uses that windowing. Also switches PCA_K from 20 to 50 to test
a different dimensionality.

Podcast, llama-3.1-8b / gpt2-large / bert-base, matched K=50 PCA. Reports
curvature/displacement/position vs neural, as permutation-null z-scores.

position(X) = ||X_t - mean(X)||, sliced [1:-1] to match curvature's length
(same convention as compute_podcast_randomdim_bootstrap_dur8040.py).
"""
from __future__ import annotations
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
REGION = "hippocampus"
PCA_K = 5
N_PERM = 200
SEED = 0
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, load_words_onset_offset, windowed_neu_vecs, _convo_regions,
)

MODELS = {"llama-3.1-8b": 5, "gpt2-large": 19, "bert-base": 12}


def position(X):
    X = np.asarray(X, dtype=np.float64)
    c = X.mean(axis=0, keepdims=True)
    return np.linalg.norm(X - c, axis=1)


def pca_reduce(emb_L, k=PCA_K):
    Xs = StandardScaler().fit_transform(emb_L)
    kk = min(k, Xs.shape[1], Xs.shape[0] - 1)
    if kk < 2:
        return None
    return PCA(n_components=kk, svd_solver="randomized", random_state=0).fit_transform(Xs)


def perm_z(x, y, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    r_obs, _ = pearsonr(x, y)
    null = np.empty(n_perm)
    y_shuf = y.copy()
    for i in range(n_perm):
        rng.shuffle(y_shuf)
        null[i] = pearsonr(x, y_shuf)[0]
    z = (r_obs - null.mean()) / null.std()
    return r_obs, float(z), float(null.mean()), float(null.std())


def main():
    for model_tag, layer in MODELS.items():
        sem_curv, neu_curv = [], []
        sem_disp, neu_disp = [], []
        sem_pos,  neu_pos  = [], []

        for pid in PODCAST_PATIENTS:
            data0 = load_podcast_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
            if data0 is None:
                continue
            try:
                _, onset_ms, offset_ms = load_words_onset_offset(pid)
            except (AssertionError, FileNotFoundError, OSError):
                continue
            preg = _convo_regions(pid)
            n = min(data0["n_words"], len(onset_ms))
            surp, sp, is_turn = data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n]
            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < 20:
                continue

            neu_vecs = windowed_neu_vecs(pid, REGION, preg, onset_ms[:n], offset_ms[:n],
                                          ONSET_PAD_MS, OFFSET_PAD_MS, rate=True)
            if neu_vecs is None:
                continue
            Y_neu, _ = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
            p_neu = position(Y_neu)[1:-1][valid]

            emb_mm = np.load(data0["emb_path"], mmap_mode="r")
            if layer >= emb_mm.shape[0]:
                del emb_mm
                continue
            emb_L = np.array(emb_mm[layer, :n], dtype=np.float64)
            del emb_mm
            Xs = pca_reduce(emb_L)
            if Xs is None:
                continue
            c_sem = trajectory_curvature(Xs)[valid]
            d_sem = trajectory_displacement(Xs)[:n - 2][valid]
            p_sem = position(Xs)[1:-1][valid]

            sem_curv.append(c_sem); neu_curv.append(c_neu)
            sem_disp.append(d_sem); neu_disp.append(d_neu)
            sem_pos.append(p_sem);  neu_pos.append(p_neu)

        c_s, c_n = np.concatenate(sem_curv), np.concatenate(neu_curv)
        d_s, d_n = np.concatenate(sem_disp), np.concatenate(neu_disp)
        p_s, p_n = np.concatenate(sem_pos),  np.concatenate(neu_pos)

        fin_c = np.isfinite(c_s) & np.isfinite(c_n)
        fin_d = np.isfinite(d_s) & np.isfinite(d_n)
        fin_p = np.isfinite(p_s) & np.isfinite(p_n)

        r_c, z_c, mu_c, sd_c = perm_z(c_s[fin_c], c_n[fin_c])
        r_d, z_d, mu_d, sd_d = perm_z(d_s[fin_d], d_n[fin_d])
        r_p, z_p, mu_p, sd_p = perm_z(p_s[fin_p], p_n[fin_p])

        print(f"\n=== {model_tag} (layer {layer}, PCA-{PCA_K}, dur8040 rate-normalized neural window) ===")
        print(f"  curvature:    r={r_c:+.4f}  null(mu={mu_c:+.4f}, sd={sd_c:.4f})  z={z_c:+.2f}   n={fin_c.sum()}")
        print(f"  displacement: r={r_d:+.4f}  null(mu={mu_d:+.4f}, sd={sd_d:.4f})  z={z_d:+.2f}   n={fin_d.sum()}")
        print(f"  position:     r={r_p:+.4f}  null(mu={mu_p:+.4f}, sd={sd_p:.4f})  z={z_p:+.2f}   n={fin_p.sum()}")
        best = max([("curvature", z_c), ("displacement", z_d), ("position", z_p)], key=lambda t: t[1])
        print(f"  -> strongest (by |z| vs permutation null): {best[0]} (z={best[1]:+.2f})")


if __name__ == "__main__":
    main()
