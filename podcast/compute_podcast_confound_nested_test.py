"""
compute_podcast_confound_nested_test.py
====================================
Tests whether displacement's K-dependence (significant at low K, loses
significance/flips at K=300) is a genuine CONFOUND -- i.e. whether the
extra PCs kept at K=300 vs K=20 carry nuisance information (word
frequency, word duration, position/context length) that's driving the
raw correlation -- vs just noise dilution.

Adapts [[feedback_nested_deviance_principle]] (nested model comparison:
does adding block B beyond block A significantly reduce unexplained
variance?) to this continuous-outcome setting (displacement/curvature are
derived geometric quantities, not spike counts, so this uses nested OLS
R^2 comparison rather than Poisson deviance -- same principle, adapted
machinery):

  Model A (nuisance only):        neu_z ~ freq_z + dur_z + pos_z
  Model B (nuisance + semantic):  neu_z ~ freq_z + dur_z + pos_z + sem_z

  unique_sem = R^2_B - R^2_A  (semantic's contribution BEYOND nuisance)

Computed at K=20 (clean, low-K result) and K=300 (raw-like, high-K
result), for both curvature and displacement. If unique_sem stays
significant at K=300 comparable to K=20, the K=300 pattern isn't a
nuisance confound. If unique_sem drops to ~0/n.s. at K=300 while the RAW
(no-nuisance-control) correlation also drops, that's consistent with --
but doesn't distinguish confound from noise dilution (both predict this).
The distinguishing evidence is whether nuisance ALONE (model A) explains
MORE variance at K=300's sem_z than at K=20's sem_z (i.e., does the sem_z
variable itself correlate with nuisance more at high K) -- reported
separately via R^2(sem_z ~ nuisance).

Patient-cluster bootstrap (2000x) throughout, matching the established
per-chunk-pooled methodology.

Saves: results/podcast_confound_nested_test.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    ACCEPTED_QUAL, trajectory_curvature, trajectory_displacement,
    build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions
import glob, h5py, scipy.sparse

ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40
CHUNK = 15
KS_TO_TEST = [20, 300]
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"

CACHE_DIR = os.path.join(BASE_DIR, "cache_pca_sweep_podcast")
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "podcast_confound_nested_test.json")


def find_spikes(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "spikes", "*_spikes.mat"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def find_transcript(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "transcript", "*.xlsx"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def build_neu_vecs(pid, data0):
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

    word_dur = np.full(n, np.nan)
    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]
        offset_ms = df.iloc[wi]["offset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        if np.isfinite(offset_ms) and offset_ms > onset_ms:
            offset_ms = int(offset_ms)
        else:
            offset_ms = onset_ms + 100
        word_dur[wi] = offset_ms - onset_ms
        t0 = max(0, onset_ms + ONSET_PAD_MS)
        t1 = min(T, offset_ms + OFFSET_PAD_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, word_dur


def get_patient_data(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    neu_vecs, n, word_dur = build_neu_vecs(pid, data0)
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

    # nuisance covariates, aligned to same "current word" (i+2) indexing as curv/disp
    freq = data0["log_freq"][:n][2:n][valid]
    dur = word_dur[2:n][valid]
    pos = data0["sent_pos"][:n][2:n][valid]

    return dict(n=n, valid=valid, c_neu=c_neu, d_neu=d_neu,
                freq=freq, dur=dur, pos=pos, n_pcs=n_pcs)


def load_cached_sem_k(pid, model, k):
    cache_path = os.path.join(CACHE_DIR, f"{pid}_{model}_allK.npz")
    if not os.path.exists(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    return z[f"c_{k}"], z[f"d_{k}"]


def _chunk_mean(x, cs, min_frac=0.5):
    n = len(x)
    idx_keep = []
    out = []
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl])
        if fin.sum() < cs * min_frac:
            continue
        out.append(np.mean(x[sl][fin]))
        idx_keep.append(i)
    return np.array(out), idx_keep


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def r2_ols(y, X):
    """R^2 of OLS y ~ X (X includes intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def bootstrap_unique_r2(df, sem_col, n_boot=2000, seed=0):
    """Patient-cluster bootstrap on unique_sem = R2(nuisance+sem) - R2(nuisance)."""
    rng = np.random.default_rng(seed)
    patients = df.patient.unique()

    def _compute(sub):
        y = sub.neu_z.values
        X_a = np.column_stack([np.ones(len(sub)), sub.freq_z, sub.dur_z, sub.pos_z])
        X_b = np.column_stack([X_a, sub[sem_col].values])
        r2_a = r2_ols(y, X_a)
        r2_b = r2_ols(y, X_b)
        return r2_a, r2_b, r2_b - r2_a

    r2_a_obs, r2_b_obs, unique_obs = _compute(df)

    # also: does sem_z correlate with nuisance itself?
    X_nuis = np.column_stack([np.ones(len(df)), df.freq_z, df.dur_z, df.pos_z])
    r2_sem_vs_nuisance = r2_ols(df[sem_col].values, X_nuis)

    boot_unique = []
    for _ in range(n_boot):
        samp_pids = rng.choice(patients, size=len(patients), replace=True)
        boot_df = pd.concat([df[df.patient == p] for p in samp_pids])
        if len(boot_df) < 10:
            continue
        _, _, u = _compute(boot_df)
        boot_unique.append(u)
    boot_unique = np.array(boot_unique)
    ci_lo, ci_hi = np.percentile(boot_unique, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boot_unique > 0), np.mean(boot_unique < 0))
    return dict(r2_nuisance_only=r2_a_obs, r2_full=r2_b_obs, unique_sem_r2=unique_obs,
                ci_lo=ci_lo, ci_hi=ci_hi, p=p_boot, r2_sem_vs_nuisance=r2_sem_vs_nuisance)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    patient_data = {}
    for pid in PODCAST_PATIENTS:
        res = get_patient_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        patient_data[pid] = res
        print(f"  {pid}: n_pcs={res['n_pcs']}", flush=True)

    rows_by_k_metric = {(k, m): [] for k in KS_TO_TEST for m in ("curv", "disp")}

    for pid, pdata in patient_data.items():
        c_neu, d_neu = pdata["c_neu"], pdata["d_neu"]
        freq, dur, pos = pdata["freq"], pdata["dur"], pdata["pos"]

        c_neu_chunk, idx_c = _chunk_mean(c_neu, CHUNK)
        d_neu_chunk, idx_d = _chunk_mean(d_neu, CHUNK)
        freq_c_chunk = np.array([np.nanmean(freq[i:i + CHUNK]) for i in idx_c])
        dur_c_chunk = np.array([np.nanmean(dur[i:i + CHUNK]) for i in idx_c])
        pos_c_chunk = np.array([np.nanmean(pos[i:i + CHUNK]) for i in idx_c])
        freq_d_chunk = np.array([np.nanmean(freq[i:i + CHUNK]) for i in idx_d])
        dur_d_chunk = np.array([np.nanmean(dur[i:i + CHUNK]) for i in idx_d])
        pos_d_chunk = np.array([np.nanmean(pos[i:i + CHUNK]) for i in idx_d])

        for model in MODELS:
            for k in KS_TO_TEST:
                sem = load_cached_sem_k(pid, model, k)
                if sem is None:
                    continue
                c_sem, d_sem = sem

                c_sem_chunk = np.array([np.nanmean(c_sem[i:i + CHUNK]) for i in idx_c])
                d_sem_chunk = np.array([np.nanmean(d_sem[i:i + CHUNK]) for i in idx_d])

                rows_by_k_metric[(k, "curv")] += [
                    (pid, cs, cn, f, d, p) for cs, cn, f, d, p in
                    zip(zscore(c_sem_chunk), zscore(c_neu_chunk),
                        zscore(freq_c_chunk), zscore(dur_c_chunk), zscore(pos_c_chunk))
                ]
                rows_by_k_metric[(k, "disp")] += [
                    (pid, ds, dn, f, d, p) for ds, dn, f, d, p in
                    zip(zscore(d_sem_chunk), zscore(d_neu_chunk),
                        zscore(freq_d_chunk), zscore(dur_d_chunk), zscore(pos_d_chunk))
                ]
        print(f"  {pid}: rows built", flush=True)

    results = {}
    for k in KS_TO_TEST:
        for metric in ("curv", "disp"):
            rows = rows_by_k_metric[(k, metric)]
            df = pd.DataFrame(rows, columns=["patient", "sem_z", "neu_z", "freq_z", "dur_z", "pos_z"])
            if len(df) < 20:
                results[f"{metric}_K{k}"] = None
                continue
            out = bootstrap_unique_r2(df, "sem_z")
            raw_r, raw_p = pearsonr(df.sem_z, df.neu_z)
            out["raw_r"] = raw_r
            out["raw_p"] = raw_p
            results[f"{metric}_K{k}"] = out
            print(f"  {metric} K={k}: raw_r={raw_r:+.4f} (p={raw_p:.4g})   "
                  f"unique_sem_R2={out['unique_sem_r2']:+.5f} p={out['p']:.4f} {stars(out['p'])}   "
                  f"R2(sem~nuisance)={out['r2_sem_vs_nuisance']:.5f}", flush=True)
            json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
