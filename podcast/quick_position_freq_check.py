"""
quick_position_freq_check.py
====================================
Checks whether semantic "position" (distance from trajectory centroid) is
just a proxy for word frequency -- a well-known confound (rare words sit
further from an embedding space's centroid than common/function words).
If r(semantic position, log_freq) is large, "position correlates with
neural activity" may just mean "word frequency correlates with neural
activity" through two independent routes, not genuine trajectory-geometry
alignment.

Podcast, llama-3.1-8b (layer 5) / gpt2-large (layer 19) / bert-base (layer
12), K=50 PCA (matching the circular-shift-null test), pooled across all 13
patients.
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

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import trajectory_curvature, build_valid_mask
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


def main():
    for model_tag, layer in MODELS.items():
        pos_all, freq_all, surp_all = [], [], []
        for pid in PODCAST_PATIENTS:
            data = load_podcast_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
            if data is None:
                continue
            n = data["n_words"]
            surp, sp, is_turn = data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n]
            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < 20:
                continue
            log_freq = data["log_freq"][:n]

            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            if layer >= emb_mm.shape[0]:
                del emb_mm
                continue
            emb_L = np.array(emb_mm[layer, :n], dtype=np.float64)
            del emb_mm
            Xs = pca_reduce(emb_L)
            if Xs is None:
                continue
            p_sem = position(Xs)[1:-1][valid]
            freq_y = log_freq[1:-1][valid]
            surp_y = surp[1:-1][valid]

            pos_all.append(p_sem); freq_all.append(freq_y); surp_all.append(surp_y)

        p = np.concatenate(pos_all); f = np.concatenate(freq_all); s = np.concatenate(surp_all)
        fin_f = np.isfinite(p) & np.isfinite(f)
        fin_s = np.isfinite(p) & np.isfinite(s)
        r_f, pv_f = pearsonr(p[fin_f], f[fin_f])
        r_s, pv_s = pearsonr(p[fin_s], s[fin_s])
        print(f"{model_tag} (layer {layer}, PCA-{PCA_K}):  "
              f"r(position, log_freq)={r_f:+.4f} (p={pv_f:.2e}, n={fin_f.sum()})   "
              f"r(position, surprisal)={r_s:+.4f} (p={pv_s:.2e}, n={fin_s.sum()})")


if __name__ == "__main__":
    main()
