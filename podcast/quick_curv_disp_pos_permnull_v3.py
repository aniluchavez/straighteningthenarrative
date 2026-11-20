"""
quick_curv_disp_pos_permnull_v3.py
====================================
Fix over v2: position(X) = ||X_t - mean(X)|| is a SLOW, smoothly-drifting
quantity (nearby words sit close to the same session-wide centroid),
unlike curvature/displacement which are local, high-frequency derivative
measures. A full random shuffle destroys ALL temporal structure, which is
a fine null for a noisy/low-autocorrelation signal but badly
UNDERESTIMATES the true null variance for an autocorrelated signal like
position -- inflating its apparent z-score for reasons having nothing to
do with genuine word-level coupling.

Fix: circular-shift null instead of full shuffle. Each permutation rotates
each PATIENT's own neural sequence by an independent random offset
(preserving that patient's own autocorrelation/smoothness), rather than
scrambling word order entirely. Same corrected dur8040 rate-normalized
window as v2. K=50.
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
PCA_K = 50
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


def circshift_perm_z(x_list, y_list, n_perm=N_PERM, seed=SEED):
    """x_list, y_list: per-patient arrays (same length pairs, may contain NaN).
    Null = circularly shift each patient's y by an independent random offset,
    preserving within-patient autocorrelation, then pool and correlate."""
    rng = np.random.default_rng(seed)

    def _pool(xs, ys):
        x_cat = np.concatenate(xs)
        y_cat = np.concatenate(ys)
        fin = np.isfinite(x_cat) & np.isfinite(y_cat)
        return x_cat[fin], y_cat[fin]

    x_obs, y_obs = _pool(x_list, y_list)
    r_obs, _ = pearsonr(x_obs, y_obs)

    x_cat = np.concatenate(x_list)
    null = np.empty(n_perm)
    for i in range(n_perm):
        y_shift = [np.roll(y, rng.integers(1, len(y))) if len(y) > 1 else y for y in y_list]
        y_cat = np.concatenate(y_shift)
        fin = np.isfinite(x_cat) & np.isfinite(y_cat)
        null[i] = pearsonr(x_cat[fin], y_cat[fin])[0]

    z = (r_obs - null.mean()) / null.std()
    return float(r_obs), float(z), float(null.mean()), float(null.std())


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

        r_c, z_c, mu_c, sd_c = circshift_perm_z(sem_curv, neu_curv)
        r_d, z_d, mu_d, sd_d = circshift_perm_z(sem_disp, neu_disp)
        r_p, z_p, mu_p, sd_p = circshift_perm_z(sem_pos, neu_pos)

        n_total = sum(len(a) for a in sem_curv)
        print(f"\n=== {model_tag} (layer {layer}, PCA-{PCA_K}, dur8040 window, CIRCULAR-SHIFT null) ===")
        print(f"  curvature:    r={r_c:+.4f}  null(mu={mu_c:+.4f}, sd={sd_c:.4f})  z={z_c:+.2f}")
        print(f"  displacement: r={r_d:+.4f}  null(mu={mu_d:+.4f}, sd={sd_d:.4f})  z={z_d:+.2f}")
        print(f"  position:     r={r_p:+.4f}  null(mu={mu_p:+.4f}, sd={sd_p:.4f})  z={z_p:+.2f}")
        best = max([("curvature", z_c), ("displacement", z_d), ("position", z_p)], key=lambda t: abs(t[1]))
        print(f"  -> strongest (by |z| vs circular-shift null): {best[0]} (z={best[1]:+.2f})")


if __name__ == "__main__":
    main()
