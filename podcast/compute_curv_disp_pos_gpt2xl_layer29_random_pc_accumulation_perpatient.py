"""
compute_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient.py
============================================================================
Per-patient random PC accumulation, gpt2-xl layer 29, neural = duration
window [onset+80, offset+40] raw spike counts (panel A of
fig_curv_disp_pos_gpt2xl_layer29_factorial_AB). PC pools are full rank per
patient: semantic 1600 (standardized PCA of the layer-29 embedding at that
patient's words), neural = n_units (standardized PCA of the counts).

For each of N_ORDER random orderings, PCs are added one at a time and the
chunked (chunk=3, valid words) semantic-neural r of curvature /
displacement / position is computed at each K in KS (K <= n_units; position
= coordinate on the first PC in the ordering). Schemes:
  both : random order on both sides (independent)
  sem  : random semantic order, neural top-K
  neu  : random neural order, semantic top-K
  top  : top-K on both (reference)
Stores where semantic PC1 / neural PC1 fall in each ordering (1-based).
All 13 patients (YFX with the +574.9 s clock fix).

Saves: results/curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient.json
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import trajectory_curvature, trajectory_displacement, build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs

MODEL, LAYER, CHUNK = "gpt2-xl", 29, 3
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
N_ORDER = 200
SEED = 0
METRICS = ["curv", "disp", "pos"]
OUT_PATH = os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient.json")


def full_pca(X):
    Xs = np.nan_to_num(StandardScaler().fit_transform(np.asarray(X, dtype=np.float64)))
    return PCA(n_components=min(Xs.shape), svd_solver="full").fit_transform(Xs)


def chunked(x, valid):
    m = min(len(x), len(valid))
    x = x[:m][valid[:m]]
    n = len(x) // CHUNK * CHUNK
    return x[:n].reshape(-1, CHUNK).mean(axis=1)


def feats(Y, valid):
    return dict(curv=chunked(trajectory_curvature(Y), valid), disp=chunked(trajectory_displacement(Y), valid),
                pos=chunked(Y[:, 0], valid))


def corr3(fs, fn):
    out = []
    for m in METRICS:
        L = min(len(fs[m]), len(fn[m]))
        out.append(float(np.corrcoef(fs[m][:L], fn[m][:L])[0, 1]))
    return out


def run_patient(i_pid, pid):
    rng = np.random.default_rng([SEED, i_pid])
    if True:
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            return pid, None
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            return pid, None
        n = min(data["n_words"], len(on))
        neu = windowed_neu_vecs(pid, "hippocampus", _convo_regions(pid), on[:n], off[:n], 80, 40, rate=False)
        if neu is None:
            return pid, None
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        emb = np.array(np.load(data["emb_path"], mmap_mode="r")[LAYER, :n], dtype=np.float64)
        Ys, Yn = full_pca(emb), full_pca(neu[:n])
        ns, nn = Ys.shape[1], Yn.shape[1]
        ks = [K for K in KS if K <= nn]
        if nn not in ks:
            ks.append(nn)
        top = [corr3(feats(Ys[:, :K], valid), feats(Yn[:, :K], valid)) for K in ks]
        rec = dict(n_units=nn, ks=ks, top=top, random={})
        for scheme in ("both", "sem", "neu"):
            arr = np.full((N_ORDER, len(ks), len(METRICS)), np.nan)
            p1s, p1n = [], []
            for i in range(N_ORDER):
                os_ = rng.permutation(ns) if scheme in ("both", "sem") else np.arange(ns)
                on_ = rng.permutation(nn) if scheme in ("both", "neu") else np.arange(nn)
                p1s.append(int(np.where(os_ == 0)[0][0]) + 1)
                p1n.append(int(np.where(on_ == 0)[0][0]) + 1)
                for j, K in enumerate(ks):
                    arr[i, j] = corr3(feats(Ys[:, os_[:K]], valid), feats(Yn[:, on_[:K]], valid))
            rec["random"][scheme] = dict(r=arr.tolist(), pc1_pos_sem=p1s, pc1_pos_neu=p1n)
        return pid, rec


def main():
    out = dict(config=dict(KS=KS, N_ORDER=N_ORDER, SEED=SEED, LAYER=LAYER, CHUNK=CHUNK, window="dur80/40 counts"),
               patients={})
    res = Parallel(n_jobs=len(PODCAST_PATIENTS))(delayed(run_patient)(i, pid) for i, pid in enumerate(PODCAST_PATIENTS))
    for pid, rec in res:
        if rec is None:
            continue
        out["patients"][pid] = rec
        jK = rec["ks"].index(8)
        top = rec["top"]
        med = {s: np.median(np.array(rec["random"][s]["r"])[:, jK], axis=0) for s in rec["random"]}
        print(f"  {pid}: units={rec['n_units']}  K=8 top c/d/p {top[jK][0]:+.3f}/{top[jK][1]:+.3f}/{top[jK][2]:+.3f}  "
              + "  ".join(f"{s} med {m[0]:+.3f}/{m[1]:+.3f}/{m[2]:+.3f}" for s, m in med.items()), flush=True)
    json.dump(out, open(OUT_PATH, "w"))
    print(f"  -> {OUT_PATH}  (metrics order {METRICS})")


if __name__ == "__main__":
    main()
