"""
compute_neural_pc_sweep.py
====================================
How does NEURAL curvature/displacement (magnitude, and the word-order
shuffle null) depend on the number of neural PCs kept? fit_neural_pca
normally auto-selects n_pcs per patient via a 90%-variance threshold
(varies 7-42 across patients this session) -- this sweep instead fixes
K_neural to specific values, refitting PCA at each, for both jabberwocky
and podcast.

For each K: mean curvature, mean displacement, and a word-order-shuffle
null mean curvature (200 permutations), pooled across patients.
"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "podcast"))

from run_geometry_analysis import trajectory_curvature, trajectory_displacement, build_valid_mask
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient
from compute_pca_sweep_jabberwocky import build_neu_vecs_dur as build_neu_vecs_jab

KS = [2, 3, 5, 10, 20, 30, 50, "max"]
N_PERM = 200


def fit_neural_pca_fixed(neu_vecs, n, k):
    nv = neu_vecs[:n]
    max_pc = min(nv.shape[1], n - 1)
    if nv.std() == 0 or max_pc < 2:
        return None
    kk = max_pc if k == "max" else min(k, max_pc)
    nv_sc = StandardScaler().fit_transform(nv)
    pca = PCA(n_components=kk, svd_solver="randomized", random_state=0).fit(nv_sc)
    return pca.transform(nv_sc)


def run_dataset(name, patients, get_data_fn):
    print(f"\n=== {name} ===")
    rng = np.random.default_rng(0)
    results = {k: {"curv": [], "disp": [], "null": []} for k in KS}

    for pid in patients:
        data0, neu_vecs, n, valid = get_data_fn(pid)
        if data0 is None:
            continue
        for k in KS:
            Y = fit_neural_pca_fixed(neu_vecs, n, k)
            if Y is None:
                continue
            c = trajectory_curvature(Y)[valid]
            d = trajectory_displacement(Y)[:n - 2][valid]
            results[k]["curv"].append(np.mean(c))
            results[k]["disp"].append(np.mean(d))

            perm_means = []
            for _ in range(N_PERM):
                Y_shuf = Y[rng.permutation(n)]
                perm_means.append(np.mean(trajectory_curvature(Y_shuf)[valid]))
            results[k]["null"].append(np.mean(perm_means))
        print(f"  {pid}: done", flush=True)

    print(f"\n{name} results (mean across patients):")
    for k in KS:
        if not results[k]["curv"]:
            continue
        cm = np.mean(results[k]["curv"])
        dm = np.mean(results[k]["disp"])
        nm = np.mean(results[k]["null"])
        print(f"  K={k!s:>5}  curv={cm:6.2f}  disp={dm:7.3f}  shuffle_null={nm:6.2f}  "
              f"curv-null={cm-nm:+6.2f}")
    return results


def jabberwocky_data(pid):
    data0 = load_jabberwocky_patient(pid, region="hippocampus", model_tag="gpt2", context_words=200)
    if data0 is None:
        return None, None, None, None
    neu_vecs, n = build_neu_vecs_jab(pid, data0)
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 30:
        return None, None, None, None
    return data0, neu_vecs, n, valid


if __name__ == "__main__":
    run_dataset("JABBERWOCKY", JABBERWOCKY_PATIENTS, jabberwocky_data)
