"""
quick_curv_disp_pos_permnull.py
====================================
Quick check (podcast, llama-3.1-8b / gpt2-large / bert-base only): at
MATCHED K=20 PCA, which geometric feature (curvature, displacement,
position) correlates best with the analogous neural feature -- reported as
a permutation-null z-score, not raw r, so the comparison isn't an artifact
of each feature's different intrinsic noise floor at that K.

position(X) = ||X_t - mean(X)||, the rotation/translation-invariant
0th-derivative counterpart to curvature/displacement (same definition used
in compute_podcast_randomdim_bootstrap_dur8040.py).

Best layer per model taken from the already-computed PCA-20 curvature sweep
(results/multimodel_layersweep_surprisal_curvature_pca20_podcast.json):
llama-3.1-8b=5, gpt2-large=19, bert-base=12.

Permutation null: shuffle the neural array (global shuffle, pooled across
all valid words/patients) 200x, recompute r each time.
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
PCA_K = 20
N_PERM = 200
SEED = 0

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

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
            data = load_podcast_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
            if data is None:
                continue
            n = data["n_words"]
            surp, sp, is_turn = data["surp"], data["sent_pos"], data["is_turn"]
            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < 20:
                continue

            Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
            if Y_neu is None:
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
            p_neu = position(Y_neu)[2:n][valid]

            emb_mm = np.load(data["emb_path"], mmap_mode="r")
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
            p_sem = position(Xs)[2:n][valid]

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

        print(f"\n=== {model_tag} (layer {layer}, PCA-{PCA_K}) ===")
        print(f"  curvature:    r={r_c:+.4f}  null(mu={mu_c:+.4f}, sd={sd_c:.4f})  z={z_c:+.2f}   n={fin_c.sum()}")
        print(f"  displacement: r={r_d:+.4f}  null(mu={mu_d:+.4f}, sd={sd_d:.4f})  z={z_d:+.2f}   n={fin_d.sum()}")
        print(f"  position:     r={r_p:+.4f}  null(mu={mu_p:+.4f}, sd={sd_p:.4f})  z={z_p:+.2f}   n={fin_p.sum()}")
        best = max([("curvature", z_c), ("displacement", z_d), ("position", z_p)], key=lambda t: t[1])
        print(f"  -> strongest (by |z| vs permutation null): {best[0]} (z={best[1]:+.2f})")


if __name__ == "__main__":
    main()
