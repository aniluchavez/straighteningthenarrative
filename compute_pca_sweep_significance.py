"""
compute_pca_sweep_significance.py
====================================
For the already-completed PC-count sweep (compute_pca_sweep_convo.py /
compute_pca_sweep_podcast.py), test whether r_curv and r_disp are each
individually significantly different from 0 (above chance), not just
whether they differ from each other.

Reuses the cached per-(patient,model) semantic embeddings
(cache_pca_sweep_convo/*.npz, podcast/cache_pca_sweep_podcast/*.npz) --
no expensive PCA refitting needed, only neural data reload (fast) and a
patient-cluster bootstrap (2000x) of each r vs 0.

Saves: results/pca_sweep_significance.json  (both datasets)
"""
import json, os, sys
import numpy as np
import h5py, scipy.sparse, pandas as pd
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "podcast"))

from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

CHUNK = 15
WIN_MS = 500
KS = [3, 5, 10, 20, 50, 100, 300]
ALL_KS = KS + ["matched"]
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"

ORIG10 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
          "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40"]
NEW4 = ["PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"]
CONVO14 = ORIG10 + NEW4

CACHE_DIR_CONVO = os.path.join(BASE_DIR, "cache_pca_sweep_convo")
CACHE_DIR_PODCAST = os.path.join(BASE_DIR, "podcast", "cache_pca_sweep_podcast")
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "pca_sweep_significance.json")


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


# ---------------- convo ----------------

def build_neu_vecs_convo(pid, win_ms):
    paths = build_paths(pid, "gpt2")
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, "gpt2")
    if data0 is None:
        return None, None, None
    n = data0["n_words"]
    with h5py.File(paths["spikes"], "r") as f:
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

    xl = pd.ExcelFile(paths["xlsx"])
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(paths["xlsx"], sheet_name=sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    n = min(n, len(df))
    is_spk1 = df["Speaker1"].astype(str).str.strip() != ""

    vecs = []
    for wi in range(n):
        onset_ms = int(df.iloc[wi]["onset"])
        t0 = max(0, onset_ms - SPEAK_OFFSET_MS) if is_spk1.iloc[wi] else max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + win_ms)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, data0


def get_neural_data_convo(pid):
    neu_vecs, n, data0 = build_neu_vecs_convo(pid, WIN_MS)
    if neu_vecs is None:
        return None
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    return n, valid, c_neu, d_neu, n_pcs


def get_neural_data_podcast(pid):
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


def load_cached_sem(cache_dir, pid, model, matched_k):
    cache_path = os.path.join(cache_dir, f"{pid}_{model}_allK.npz")
    if not os.path.exists(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    cached_matched_k = int(z["matched_k"]) if "matched_k" in z else None
    if cached_matched_k != matched_k:
        return None
    return {k: (z[f"c_{k}"], z[f"d_{k}"]) for k in ALL_KS}


def run_dataset(name, patients, get_neural_fn, cache_dir):
    print(f"\n=== {name} ===", flush=True)
    neural = {}
    for pid in patients:
        res = get_neural_fn(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        print(f"  {pid}: n_words={res[0]}  neural_n_pcs={res[4]}", flush=True)

    rows_by_k = {k: [] for k in ALL_KS}
    for pid, (n, valid, c_neu, d_neu, matched_k) in neural.items():
        for model in MODELS:
            sem = load_cached_sem(cache_dir, pid, model, matched_k)
            if sem is None:
                print(f"  {pid}/{model}: cache MISS")
                continue
            for k in ALL_KS:
                c_sem, d_sem = sem[k]
                c_pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
                d_pairs = _chunk_pairs(d_sem, d_neu, CHUNK)
                if len(c_pairs) < 3 or len(d_pairs) < 3:
                    continue
                rows_by_k[k] += build_df(c_pairs, d_pairs, pid)
        print(f"  {pid}: rows built", flush=True)

    results = {}
    for k in ALL_KS:
        df = pd.DataFrame(rows_by_k[k], columns=["patient", "metric", "sem_z", "neu_z"])
        if len(df) < 10:
            results[str(k)] = None
            continue
        rc, rc_lo, rc_hi, p_c = bootstrap_vs_zero(df, "curv")
        rd, rd_lo, rd_hi, p_d = bootstrap_vs_zero(df, "disp")
        results[str(k)] = dict(
            rc=rc, rc_ci=[rc_lo, rc_hi], p_c_vs0=p_c,
            rd=rd, rd_ci=[rd_lo, rd_hi], p_d_vs0=p_d,
        )
        sig_c = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "ns"
        sig_d = "***" if p_d < 0.001 else "**" if p_d < 0.01 else "*" if p_d < 0.05 else "ns"
        print(f"  K={k!s:>8}  curv r={rc:+.4f} [{rc_lo:+.4f},{rc_hi:+.4f}] p={p_c:.4f} {sig_c}   "
              f"disp r={rd:+.4f} [{rd_lo:+.4f},{rd_hi:+.4f}] p={p_d:.4f} {sig_d}", flush=True)
    return results


def main():
    out = {}
    if os.path.exists(OUT_PATH):
        out = json.load(open(OUT_PATH))

    out["convo"] = run_dataset("CONVO14", CONVO14, get_neural_data_convo, CACHE_DIR_CONVO)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"checkpoint saved -> {OUT_PATH}", flush=True)

    out["podcast"] = run_dataset("PODCAST", PODCAST_PATIENTS, get_neural_data_podcast, CACHE_DIR_PODCAST)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
