"""
compute_freq_surprisal_confound_test_gpt2xl_semantic.py
====================================
compute_freq_surprisal_confound_test.py, but the outcome is SEMANTIC
(gpt2-xl) curvature instead of neural curvature -- layer 20, raw
(no-PCA) standardized embedding, the layer/representation validated
earlier this session as the one whose event-triggered shape best matches
neural (shape_r=+0.559 vs neural, peak_lag=+1).

  Model A (freq + nuisance):      curv_z ~ freq_z + dur_z + pos_z
  Model B (freq + nuisance + surp): curv_z ~ freq_z + dur_z + pos_z + surp_z
  unique_surp_R2 = R2_B - R2_A   (surprisal's contribution BEYOND frequency)

  Model A' (surp + nuisance):     curv_z ~ surp_z + dur_z + pos_z
  Model B' (surp + nuisance + freq): curv_z ~ surp_z + dur_z + pos_z + freq_z
  unique_freq_R2 = R2_B' - R2_A'  (frequency's contribution BEYOND surprisal)

Podcast, all 13 patients (gpt2-xl embeddings + duration/position exist for
all 13, unlike the neural version which needed podcast_comparison-only
spikes). Patient-cluster bootstrap (2000x), 15-word chunks.

Saves: results/freq_surprisal_confound_test_gpt2xl_semantic.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import trajectory_curvature, build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, load_words_onset_offset

MODEL = "gpt2-xl"
LAYER = 20
REGION = "hippocampus"
CHUNK = 15
N_BOOT = 2000
SEED = 0

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "freq_surprisal_confound_test_gpt2xl_semantic.json")


def _chunk_mean(x, cs, min_frac=0.5):
    n = len(x)
    idx_keep, out = [], []
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
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def get_patient_rows(pid):
    data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    if data is None:
        return None
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
    if valid.sum() < 50:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    if LAYER >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
    del emb_mm
    if emb_L.std() == 0:
        return None
    Xs = StandardScaler().fit_transform(emb_L)  # raw dims, NO PCA
    c_sem = trajectory_curvature(Xs)[valid]

    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    onset_ms, offset_ms = onset_ms[:n], offset_ms[:n]
    word_dur = offset_ms - onset_ms

    freq = data["log_freq"][:n][2:n][valid]
    surp = data["surp"][:n][2:n][valid]
    dur = word_dur[2:n][valid]
    pos = data["sent_pos"][:n][2:n][valid]

    c_chunk, idx = _chunk_mean(c_sem, CHUNK)
    freq_c = np.array([np.nanmean(freq[i:i + CHUNK]) for i in idx])
    surp_c = np.array([np.nanmean(surp[i:i + CHUNK]) for i in idx])
    dur_c = np.array([np.nanmean(dur[i:i + CHUNK]) for i in idx])
    pos_c = np.array([np.nanmean(pos[i:i + CHUNK]) for i in idx])

    rows = list(zip(zscore(c_chunk), zscore(freq_c), zscore(surp_c), zscore(dur_c), zscore(pos_c)))
    return [(pid, *r) for r in rows]


def bootstrap_unique_r2(df, base_cols, add_col, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    patients = df.patient.unique()

    def _compute(sub):
        y = sub.curv_z.values
        X_a = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in base_cols])
        X_b = np.column_stack([X_a, sub[add_col].values])
        return r2_ols(y, X_a), r2_ols(y, X_b)

    r2_a_obs, r2_b_obs = _compute(df)
    unique_obs = r2_b_obs - r2_a_obs

    boot_unique = []
    for _ in range(n_boot):
        samp = rng.choice(patients, size=len(patients), replace=True)
        bd = pd.concat([df[df.patient == p] for p in samp])
        if len(bd) < 10:
            continue
        ra, rb = _compute(bd)
        boot_unique.append(rb - ra)
    boot_unique = np.array(boot_unique)
    ci_lo, ci_hi = np.percentile(boot_unique, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boot_unique > 0), np.mean(boot_unique < 0))
    return dict(r2_base=float(r2_a_obs), r2_full=float(r2_b_obs), unique_r2=float(unique_obs),
                ci_lo=float(ci_lo), ci_hi=float(ci_hi), p=float(p_boot))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    rows = []
    for pid in PODCAST_PATIENTS:
        r = get_patient_rows(pid)
        if r is None:
            print(f"  {pid}: SKIP")
            continue
        rows += r
        print(f"  {pid}: {len(r)} chunks", flush=True)

    df = pd.DataFrame(rows, columns=["patient", "curv_z", "freq_z", "surp_z", "dur_z", "pos_z"])
    print(f"\nTotal chunks: {len(df)}  patients: {df.patient.nunique()}")

    raw_r_freq, _ = pearsonr(df.freq_z, df.curv_z)
    raw_r_surp, _ = pearsonr(df.surp_z, df.curv_z)
    raw_r_freq_surp, _ = pearsonr(df.freq_z, df.surp_z)
    print(f"\nraw r(freq, curv)  = {raw_r_freq:+.4f}")
    print(f"raw r(surp, curv)  = {raw_r_surp:+.4f}")
    print(f"raw r(freq, surp)  = {raw_r_freq_surp:+.4f}  (collinearity check)")

    out_surp = bootstrap_unique_r2(df, ["freq_z", "dur_z", "pos_z"], "surp_z")
    print(f"\nunique_surp_R2 (surprisal beyond freq+dur+pos): {out_surp['unique_r2']:+.5f}  "
          f"[{out_surp['ci_lo']:+.5f}, {out_surp['ci_hi']:+.5f}]  p={out_surp['p']:.4f} {stars(out_surp['p'])}")

    out_freq = bootstrap_unique_r2(df, ["surp_z", "dur_z", "pos_z"], "freq_z")
    print(f"unique_freq_R2 (frequency beyond surp+dur+pos): {out_freq['unique_r2']:+.5f}  "
          f"[{out_freq['ci_lo']:+.5f}, {out_freq['ci_hi']:+.5f}]  p={out_freq['p']:.4f} {stars(out_freq['p'])}")

    out = dict(raw_r_freq=raw_r_freq, raw_r_surp=raw_r_surp, raw_r_freq_surp=raw_r_freq_surp,
               unique_surp_beyond_freq=out_surp, unique_freq_beyond_surp=out_freq,
               n_chunks=len(df), n_patients=int(df.patient.nunique()))
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
