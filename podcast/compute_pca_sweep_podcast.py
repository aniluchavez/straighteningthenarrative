"""
compute_pca_sweep_podcast.py
====================================
PODCAST (9 patients), win=500ms default, chunk=15, all-9-models-averaged.

Sweeps the number of semantic-side PCA components K over
[3, 5, 10, 20, 50, 100, 300] PLUS a per-patient "matched" variant where K
equals that patient's own neural n_pcs (from fit_neural_pca, ~90% variance
threshold -- typically small, e.g. 5-15).

Efficiency trick: PCA components are hierarchical/nested, so fitting ONE
randomized-SVD PCA per layer at K_MAX=300 and then slicing to the first K
columns for each smaller K gives the same result as fitting separately at
each K. This turns an 8x-cost sweep into roughly a 1x cost (one 300-dim fit
per layer, then cheap slicing) instead of 8 separate expensive PCA runs.

Saves: results/pca_sweep_podcast.json
Structure: { K: {"rc":..., "rd":..., "diff":..., "p":..., "n_chunks":...} }
plus a "matched" key with the same fields (patient-specific K).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

CHUNK = 15
K_MAX = 300
KS = [3, 5, 10, 20, 50, 100, 300]  # + "matched" computed per patient
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"

CACHE_DIR = os.path.join(BASE_DIR, "cache_pca_sweep_podcast")
os.makedirs(CACHE_DIR, exist_ok=True)
RESULTS = os.path.join(BASE_DIR, "results")


def fit_pca_master(X, k_max):
    X = np.asarray(X, dtype=np.float64)
    max_pc = min(X.shape[1], X.shape[0] - 1, k_max)
    if X.std() == 0 or max_pc < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
    return pca.transform(Xs)


def _chunk_pairs(x, y, cs, min_frac=0.5):
    n = len(x)
    out = []
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < cs * min_frac:
            continue
        out.append((np.mean(x[sl][fin]), np.mean(y[sl][fin])))
    return out


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def get_neural_data(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None or data0["neu_vecs"] is None:
        return None
    n = data0["n_words"]
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, n_pcs = fit_neural_pca(data0["neu_vecs"], n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    return n, valid, c_neu, d_neu, n_pcs


def get_semantic_data_allK(pid, model, n, valid, matched_k):
    cache_path = os.path.join(CACHE_DIR, f"{pid}_{model}_allK.npz")
    all_ks = KS + ["matched"]
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cached_matched_k = int(z["matched_k"]) if "matched_k" in z else None
        if cached_matched_k == matched_k:
            return {k: (z[f"c_{k}"], z[f"d_{k}"]) for k in all_ks}

    data_m = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
    if data_m is None:
        return None
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]

    c_by_k = {k: [] for k in all_ks}
    d_by_k = {k: [] for k in all_ks}

    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        emb_master = fit_pca_master(emb_L, K_MAX)
        if emb_master is None:
            continue
        max_avail = emb_master.shape[1]
        for k in KS:
            kk = min(k, max_avail)
            c_by_k[k].append(trajectory_curvature(emb_master[:, :kk])[valid])
            d_by_k[k].append(trajectory_displacement(emb_master[:, :kk])[:n - 2][valid])
        kk_m = min(matched_k, max_avail)
        c_by_k["matched"].append(trajectory_curvature(emb_master[:, :kk_m])[valid])
        d_by_k["matched"].append(trajectory_displacement(emb_master[:, :kk_m])[:n - 2][valid])
    del emb_mm

    out = {}
    save_dict = {"matched_k": matched_k}
    for k in all_ks:
        if not c_by_k[k]:
            return None
        c_sem = np.mean(c_by_k[k], axis=0)
        d_sem = np.mean(d_by_k[k], axis=0)
        out[k] = (c_sem, d_sem)
        save_dict[f"c_{k}"] = c_sem
        save_dict[f"d_{k}"] = d_sem
    np.savez(cache_path, **save_dict)
    return out


def bootstrap_test(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    patients = df.patient.unique()
    rc, _ = pearsonr(df[df.metric == "curv"].sem_z, df[df.metric == "curv"].neu_z)
    rd, _ = pearsonr(df[df.metric == "disp"].sem_z, df[df.metric == "disp"].neu_z)
    obs_diff = rc - rd
    boot_diffs = []
    for _ in range(n_boot):
        samp_pids = rng.choice(patients, size=len(patients), replace=True)
        boot_df = pd.concat([df[df.patient == p] for p in samp_pids])
        bc = boot_df[boot_df.metric == "curv"]; bd = boot_df[boot_df.metric == "disp"]
        if len(bc) < 3 or len(bd) < 3:
            continue
        rc_b, _ = pearsonr(bc.sem_z, bc.neu_z)
        rd_b, _ = pearsonr(bd.sem_z, bd.neu_z)
        boot_diffs.append(rc_b - rd_b)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boot_diffs > 0), np.mean(boot_diffs < 0))
    return rc, rd, obs_diff, ci_lo, ci_hi, p_boot


def build_df(c_pairs, d_pairs, pid):
    rows = []
    c_sem_c = zscore([p[0] for p in c_pairs]); c_neu_c = zscore([p[1] for p in c_pairs])
    d_sem_c = zscore([p[0] for p in d_pairs]); d_neu_c = zscore([p[1] for p in d_pairs])
    for i in range(len(c_pairs)):
        rows.append((pid, "curv", c_sem_c[i], c_neu_c[i]))
    for i in range(len(d_pairs)):
        rows.append((pid, "disp", d_sem_c[i], d_neu_c[i]))
    return rows


def main():
    out_path = os.path.join(RESULTS, "pca_sweep_podcast.json")
    neural = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        print(f"  {pid}: n_words={res[0]}  neural_n_pcs={res[4]}", flush=True)

    all_ks = KS + ["matched"]
    rows_by_k = {k: [] for k in all_ks}

    for pid, (n, valid, c_neu, d_neu, matched_k) in neural.items():
        print(f"\n  {pid} (matched_k={matched_k})", flush=True)
        for model in MODELS:
            sem = get_semantic_data_allK(pid, model, n, valid, matched_k)
            if sem is None:
                print(f"    {model}: SKIP")
                continue
            for k in all_ks:
                c_sem, d_sem = sem[k]
                c_pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
                d_pairs = _chunk_pairs(d_sem, d_neu, CHUNK)
                if len(c_pairs) < 3 or len(d_pairs) < 3:
                    continue
                rows_by_k[k] += build_df(c_pairs, d_pairs, pid)
            print(f"    {model}: done", flush=True)

        results = {}
        for k in all_ks:
            df = pd.DataFrame(rows_by_k[k], columns=["patient", "metric", "sem_z", "neu_z"])
            if len(df) < 10:
                results[str(k)] = None
                continue
            rc, rd, diff, lo, hi, p = bootstrap_test(df)
            n_c = (df.metric == "curv").sum()
            results[str(k)] = dict(rc=rc, rd=rd, diff=diff, ci_lo=lo, ci_hi=hi, p=p, n_chunks=int(n_c))
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  checkpoint saved -> {out_path}", flush=True)

    print(f"\nSaved -> {out_path}")
    print("\nFinal summary:")
    for k in all_ks:
        r = results.get(str(k))
        if r is None:
            print(f"  K={k}: insufficient data")
            continue
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(f"  K={k:>8}  r_curv={r['rc']:+.4f}  r_disp={r['rd']:+.4f}  diff={r['diff']:+.4f}  p={r['p']:.4f} {sig}")


if __name__ == "__main__":
    main()
