"""
compute_pca_sweep_jabberwocky.py
====================================
Full K-sweep + significance suite for JABBERWOCKY (7 patients), matching
today's podcast investigation methodology exactly:
  - True-duration neural window [onset+80ms, offset+40ms] (jabberwocky is
    pure passive listening, same as podcast -- established as the best
    convention this session, stronger and more complete than any fixed
    window, including the WIN_MS=500 the existing jabberwocky pipeline
    silently inherited from convo).
  - Semantic-side PCA sweep K in {3,5,10,20,50,100,300} + "matched" (each
    patient's own neural PCA n_pcs), via the fit-once-at-300-then-slice
    efficiency trick.
  - Per-chunk pooled + patient-cluster bootstrap test: curv vs disp diff,
    AND each metric individually vs 0 (not just vs each other).

Saves: results/pca_sweep_jabberwocky.json
"""
import json, os, sys
import numpy as np
import h5py, scipy.sparse, pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    ACCEPTED_QUAL, trajectory_curvature, trajectory_displacement,
    build_valid_mask, fit_neural_pca,
)
from jabberwocky_loader import (
    JABBERWOCKY_PATIENTS, load_jabberwocky_patient, _convo_regions,
    find_transcript, find_spikes,
)

ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40
CHUNK = 15
K_MAX = 300
KS = [3, 5, 10, 20, 50, 100, 300]
ALL_KS = KS + ["matched"]
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"

CACHE_DIR = os.path.join(BASE_DIR, "cache_pca_sweep_jabberwocky")
os.makedirs(CACHE_DIR, exist_ok=True)
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "pca_sweep_jabberwocky.json")


def fit_pca_master(X, k_max):
    X = np.asarray(X, dtype=np.float64)
    max_pc = min(X.shape[1], X.shape[0] - 1, k_max)
    if X.std() == 0 or max_pc < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def build_neu_vecs_dur(pid, data0):
    preg = _convo_regions(pid)
    with h5py.File(find_spikes(pid), "r") as f:
        qual_key = "qual" if "qual" in f else "quality"
        chan_key = "chan" if "chan" in f else "channelIds"
        spikes = scipy.sparse.csr_matrix((
            f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
        )).toarray()
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[REGION]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    reg_sp = spikes[:, neuron_idx]

    df = pd.read_excel(find_transcript(pid), sheet_name="Sheet1")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    n = min(data0["n_words"], len(df))

    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]
        offset_ms = df.iloc[wi]["offset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        if np.isfinite(offset_ms) and offset_ms > onset_ms:
            offset_ms = int(offset_ms)
        else:
            offset_ms = onset_ms + 100
        t0 = max(0, onset_ms + ONSET_PAD_MS)
        t1 = min(T, offset_ms + OFFSET_PAD_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n


def get_neural_data(pid):
    data0 = load_jabberwocky_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    neu_vecs, n = build_neu_vecs_dur(pid, data0)
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 30:
        return None
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
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

    data_m = load_jabberwocky_patient(pid, region=REGION, model_tag=model, context_words=200)
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


def build_df(c_pairs, d_pairs, pid):
    rows = []
    c_sem_c = zscore([p[0] for p in c_pairs]); c_neu_c = zscore([p[1] for p in c_pairs])
    d_sem_c = zscore([p[0] for p in d_pairs]); d_neu_c = zscore([p[1] for p in d_pairs])
    for i in range(len(c_pairs)):
        rows.append((pid, "curv", c_sem_c[i], c_neu_c[i]))
    for i in range(len(d_pairs)):
        rows.append((pid, "disp", d_sem_c[i], d_neu_c[i]))
    return rows


def bootstrap_diff(df, n_boot=2000, seed=0):
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


def bootstrap_vs_zero(df, metric, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sub = df[df.metric == metric]
    patients = sub.patient.unique()
    r_obs, _ = pearsonr(sub.sem_z, sub.neu_z)
    boot_r = []
    for _ in range(n_boot):
        samp_pids = rng.choice(patients, size=len(patients), replace=True)
        boot_df = pd.concat([sub[sub.patient == p] for p in samp_pids])
        if len(boot_df) < 3:
            continue
        r_b, _ = pearsonr(boot_df.sem_z, boot_df.neu_z)
        boot_r.append(r_b)
    boot_r = np.array(boot_r)
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boot_r > 0), np.mean(boot_r < 0))
    return r_obs, ci_lo, ci_hi, p_boot


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    neural = {}
    for pid in JABBERWOCKY_PATIENTS:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        n, valid, c_neu, d_neu, n_pcs = res
        print(f"  {pid}: n_words={n}  neural_n_pcs={n_pcs}", flush=True)

    rows_by_k = {k: [] for k in ALL_KS}
    for pid, (n, valid, c_neu, d_neu, matched_k) in neural.items():
        for model in MODELS:
            sem = get_semantic_data_allK(pid, model, n, valid, matched_k)
            if sem is None:
                print(f"  {pid}/{model}: SKIP")
                continue
            for k in ALL_KS:
                c_sem, d_sem = sem[k]
                c_pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
                d_pairs = _chunk_pairs(d_sem, d_neu, CHUNK)
                if len(c_pairs) < 3 or len(d_pairs) < 3:
                    continue
                rows_by_k[k] += build_df(c_pairs, d_pairs, pid)
            print(f"    {model}: done", flush=True)
        print(f"  {pid}: rows built", flush=True)

    results = {}
    for k in ALL_KS:
        df = pd.DataFrame(rows_by_k[k], columns=["patient", "metric", "sem_z", "neu_z"])
        if len(df) < 10:
            results[str(k)] = None
            continue
        rc, rd, diff, lo, hi, p_diff = bootstrap_diff(df)
        rc2, rc_lo, rc_hi, p_c = bootstrap_vs_zero(df, "curv")
        rd2, rd_lo, rd_hi, p_d = bootstrap_vs_zero(df, "disp")
        results[str(k)] = dict(
            rc=rc, rd=rd, diff=diff, ci_lo=lo, ci_hi=hi, p_diff=p_diff,
            rc_ci=[rc_lo, rc_hi], p_c_vs0=p_c,
            rd_ci=[rd_lo, rd_hi], p_d_vs0=p_d,
        )
        print(f"  K={k!s:>8}  curv r={rc:+.4f} vs0 p={p_c:.4f} {stars(p_c)}   "
              f"disp r={rd:+.4f} vs0 p={p_d:.4f} {stars(p_d)}   "
              f"diff={diff:+.4f} p={p_diff:.4f} {stars(p_diff)}", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
