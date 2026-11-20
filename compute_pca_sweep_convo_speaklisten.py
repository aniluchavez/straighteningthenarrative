"""
compute_pca_sweep_convo_speaklisten.py
====================================
CONVO14 cross-space K-sweep, split into SPEAKING vs LISTENING words, each
using its own respective true-duration window:
  - listening: [onset+80ms, offset+40ms]   (same as podcast's dur80_40)
  - speaking:  [onset-80ms, offset]         (pre-onset start, roughly
                                              comparable length, ends at
                                              offset with no post-offset pad)

Only the NEURAL side depends on window -- the semantic-side PCA embeddings
are window-independent, so this reuses the cached cache_pca_sweep_convo/*.npz
files (no expensive PCA refitting) and only rebuilds neural vectors with
the new per-word duration-based windows (mixed sequence, each word gets its
own condition's window, exactly as the original pipeline's SPEAK/LISTEN
branch did -- just with true-duration formulas now).

Each 15-word CHUNK is then labeled speaking/listening by majority vote of
its (valid-filtered) words and routed to the corresponding condition's
pooled bootstrap analysis. This stratifies the SAME mixed trajectory by
condition rather than building disjoint filtered subsequences.

For each condition x K, reports:
  - curv vs disp diff test (patient-cluster bootstrap)
  - curv vs 0 and disp vs 0 individually (patient-cluster bootstrap)

Saves: results/pca_sweep_convo_speaklisten.json
"""
import json, os, sys
import numpy as np
import h5py, scipy.sparse, pandas as pd
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

CHUNK = 15
ONSET_PAD_SPEAK_MS = -80   # speak window: [onset - 80, offset]
ONSET_PAD_LISTEN_MS = 80   # listen window: [onset + 80, offset + 40]
OFFSET_PAD_LISTEN_MS = 40
KS = [3, 5, 10, 20, 50, 100, 300]
ALL_KS = KS + ["matched"]
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"

ORIG10 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
          "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40"]
NEW4 = ["PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"]
CONVO14 = ORIG10 + NEW4

CACHE_DIR = os.path.join(BASE_DIR, "cache_pca_sweep_convo")
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "pca_sweep_convo_speaklisten.json")


def build_neu_vecs_convo_dur(pid):
    paths = build_paths(pid, "gpt2")
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, "gpt2")
    if data0 is None:
        return None, None, None, None
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
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    n = min(n, len(df))
    is_spk1 = (df["Speaker1"].astype(str).str.strip() != "").values[:n]

    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]
        offset_ms = df.iloc[wi]["offset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        if np.isfinite(offset_ms) and offset_ms > onset_ms:
            offset_ms = int(offset_ms)
        else:
            offset_ms = onset_ms + 100
        if is_spk1[wi]:
            t0 = max(0, onset_ms + ONSET_PAD_SPEAK_MS)
            t1 = min(T, offset_ms)
        else:
            t0 = max(0, onset_ms + ONSET_PAD_LISTEN_MS)
            t1 = min(T, offset_ms + OFFSET_PAD_LISTEN_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, data0, is_spk1


def get_neural_data(pid):
    neu_vecs, n, data0, is_spk1 = build_neu_vecs_convo_dur(pid)
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
    # condition label per curvature/displacement index: the "current"/endpoint
    # word of triple (i, i+1, i+2) is word i+2 -- matches build_valid_mask's
    # own indexing convention (checks is_turn[i+2] etc.)
    cond_is_spk1 = is_spk1[2:n][valid]
    return n, valid, c_neu, d_neu, n_pcs, cond_is_spk1


def load_cached_sem(pid, model, matched_k):
    cache_path = os.path.join(CACHE_DIR, f"{pid}_{model}_allK.npz")
    if not os.path.exists(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    out = {k: (z[f"c_{k}"], z[f"d_{k}"]) for k in KS}
    cached_matched_k = int(z["matched_k"]) if "matched_k" in z else None
    if cached_matched_k == matched_k:
        out["matched"] = (z["c_matched"], z["d_matched"])
    return out


def _chunk_pairs_by_cond(x, y, cond, cs, min_frac=0.5):
    n = len(x)
    out = {"speak": [], "listen": []}
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < cs * min_frac:
            continue
        pair = (np.mean(x[sl][fin]), np.mean(y[sl][fin]))
        frac_speak = np.mean(cond[sl][fin])
        out["speak" if frac_speak >= 0.5 else "listen"].append(pair)
    return out


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def build_df(c_pairs, d_pairs, pid):
    rows = []
    if len(c_pairs) > 0:
        c_sem_c = zscore([p[0] for p in c_pairs]); c_neu_c = zscore([p[1] for p in c_pairs])
        for i in range(len(c_pairs)):
            rows.append((pid, "curv", c_sem_c[i], c_neu_c[i]))
    if len(d_pairs) > 0:
        d_sem_c = zscore([p[0] for p in d_pairs]); d_neu_c = zscore([p[1] for p in d_pairs])
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
    for pid in CONVO14:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        n, valid, c_neu, d_neu, n_pcs, cond = res
        print(f"  {pid}: n_words={n}  neural_n_pcs={n_pcs}  "
              f"speak_chunks_frac={cond.mean():.2f}", flush=True)

    rows_by_cond_k = {cond: {k: [] for k in ALL_KS} for cond in ("speak", "listen")}
    for pid, (n, valid, c_neu, d_neu, matched_k, cond) in neural.items():
        for model in MODELS:
            sem = load_cached_sem(pid, model, matched_k)
            if sem is None:
                print(f"  {pid}/{model}: cache MISS")
                continue
            for k, (c_sem, d_sem) in sem.items():
                c_out = _chunk_pairs_by_cond(c_sem, c_neu, cond, CHUNK)
                d_out = _chunk_pairs_by_cond(d_sem, d_neu, cond, CHUNK)
                for condition in ("speak", "listen"):
                    if len(c_out[condition]) < 3 or len(d_out[condition]) < 3:
                        continue
                    rows_by_cond_k[condition][k] += build_df(
                        c_out[condition], d_out[condition], pid)
        print(f"  {pid}: rows built", flush=True)

    results = {"speak": {}, "listen": {}}
    for condition in ("speak", "listen"):
        print(f"\n=== {condition.upper()} ===", flush=True)
        for k in ALL_KS:
            df = pd.DataFrame(rows_by_cond_k[condition][k],
                               columns=["patient", "metric", "sem_z", "neu_z"])
            if len(df) < 10 or df.metric.nunique() < 2:
                results[condition][str(k)] = None
                continue
            rc, rd, diff, lo, hi, p_diff = bootstrap_diff(df)
            rc2, rc_lo, rc_hi, p_c = bootstrap_vs_zero(df, "curv")
            rd2, rd_lo, rd_hi, p_d = bootstrap_vs_zero(df, "disp")
            results[condition][str(k)] = dict(
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
