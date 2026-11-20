"""
compute_pca_sweep_podcast_dur80_40_rate.py
====================================
Rate-normalized control for compute_pca_sweep_podcast_dur80_40.py: same
variable per-word window [onset+80ms, offset+40ms], but neural vectors are
spike RATE (count / window duration in seconds) instead of raw count.
dur80_40's raw-count version showed a much stronger, convo-like curv/disp
signal than the fixed-window versions -- this checks whether that's a real
effect or the same window-length-mechanically-inflates-count artifact
already found and fixed in the hippo-speaker-semantics project (missing
Poisson exposure offset for duration -- see project memory). If the signal
survives rate-normalization, it's not that artifact.

Only the NEURAL side depends on window -- the semantic-side PCA embeddings
are window-independent, so this reuses the cached cache_pca_sweep_podcast/*.npz
files (no expensive PCA refitting) and only rebuilds neural vectors at the
onset+80/offset+40 window.

For each K, reports BOTH:
  - curv vs disp diff test (patient-cluster bootstrap)
  - curv vs 0 and disp vs 0 individually (patient-cluster bootstrap)

Saves: results/pca_sweep_podcast_dur80_40_rate.json
"""
import json, os, sys, glob
import numpy as np
import h5py, scipy.sparse, pandas as pd
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    ACCEPTED_QUAL, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions

ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40
CHUNK = 15
KS = [3, 5, 10, 20, 50, 100, 300]
ALL_KS = KS + ["matched"]
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"

CACHE_DIR = os.path.join(BASE_DIR, "cache_pca_sweep_podcast")
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "pca_sweep_podcast_dur80_40_rate.json")


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

    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]
        offset_ms = df.iloc[wi]["offset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        if np.isfinite(offset_ms) and offset_ms > onset_ms:
            offset_ms = int(offset_ms)
        else:
            offset_ms = onset_ms + 100   # fallback if offset missing/invalid
        t0 = max(0, onset_ms + ONSET_PAD_MS)
        t1 = min(T, offset_ms + OFFSET_PAD_MS)
        if t1 > t0:
            dur_s = (t1 - t0) / 1000.0
            vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) / dur_s)
        else:
            vecs.append(np.zeros(len(neuron_idx)))
    return np.array(vecs), n


def get_neural_data(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    try:
        neu_vecs, n = build_neu_vecs(pid, data0)
    except IndexError:
        # pid not under PODCAST_ROOT (one of the 4 extra-source patients);
        # this script's onset/offset-xlsx window rebuild only supports the
        # original 9 podcast_comparison patients, matching the dur80_40
        # raw-count baseline it's being compared against.
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


def load_cached_sem(pid, model, matched_k):
    """Fixed K's (3..300) don't depend on matched_k (that's a neural-side
    quantity) so they're always valid to reuse regardless of window. Only
    the "matched" column needs cached_matched_k == matched_k (win=100's
    neural n_pcs differs from win=500's) -- if it doesn't match, "matched"
    is simply omitted from the returned dict (caller must handle absence)."""
    cache_path = os.path.join(CACHE_DIR, f"{pid}_{model}_allK.npz")
    if not os.path.exists(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    out = {k: (z[f"c_{k}"], z[f"d_{k}"]) for k in KS}
    cached_matched_k = int(z["matched_k"]) if "matched_k" in z else None
    if cached_matched_k == matched_k:
        out["matched"] = (z["c_matched"], z["d_matched"])
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


def main():
    neural = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        print(f"  {pid} (dur onset+{ONSET_PAD_MS}/offset+{OFFSET_PAD_MS}): n_words={res[0]}  neural_n_pcs={res[4]}", flush=True)

    rows_by_k = {k: [] for k in ALL_KS}
    for pid, (n, valid, c_neu, d_neu, matched_k) in neural.items():
        for model in MODELS:
            sem = load_cached_sem(pid, model, matched_k)
            if sem is None:
                print(f"  {pid}/{model}: cache MISS")
                continue
            for k, (c_sem, d_sem) in sem.items():
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
        rc, rd, diff, lo, hi, p_diff = bootstrap_diff(df)
        rc2, rc_lo, rc_hi, p_c = bootstrap_vs_zero(df, "curv")
        rd2, rd_lo, rd_hi, p_d = bootstrap_vs_zero(df, "disp")
        results[str(k)] = dict(
            rc=rc, rd=rd, diff=diff, ci_lo=lo, ci_hi=hi, p_diff=p_diff,
            rc_ci=[rc_lo, rc_hi], p_c_vs0=p_c,
            rd_ci=[rd_lo, rd_hi], p_d_vs0=p_d,
        )
        def stars(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  K={k!s:>8}  curv r={rc:+.4f} vs0 p={p_c:.4f} {stars(p_c)}   "
              f"disp r={rd:+.4f} vs0 p={p_d:.4f} {stars(p_d)}   "
              f"diff={diff:+.4f} p={p_diff:.4f} {stars(p_diff)}", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
