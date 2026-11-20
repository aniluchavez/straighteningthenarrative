"""
compute_freq_surprisal_full_grid_test.py
====================================
Completes the 2x2x2 grid of {word-level, chunk-level(15)} x
{semantic(gpt2-xl L20), neural(spikes->PCA)} x {no POS control, with
function-word control} for frequency vs surprisal's unique contribution to
curvature -- filling in the two cells not yet run
(compute_freq_surprisal_confound_test.py did chunk-level/neural/no-POS;
compute_freq_surprisal_pos_control_test.py did word-level/semantic/both).

Same nested unique-R2 test throughout:
  Model A (X + nuisance):        curv_z ~ X_z + dur_z + pos_z [+ isfunc_z]
  Model B (A + other predictor):  + other_z
  unique_R2 = R2_B - R2_A

Podcast, all 13 patients, patient-cluster bootstrap (2000x).

Saves: results/freq_surprisal_full_grid_test.json
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

from run_geometry_analysis import trajectory_curvature, build_valid_mask, fit_neural_pca
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, load_words_onset_offset

REGION = "hippocampus"
SEM_MODEL = "gpt2-xl"
SEM_LAYER = 20
CHUNKS = [1, 15]
N_BOOT = 2000
SEED = 0

FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "freq_surprisal_full_grid_test.json")

import spacy
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def get_is_function(words):
    from spacy.tokens import Doc
    doc = Doc(_NLP.vocab, words=[str(w) for w in words])
    for name, proc in _NLP.pipeline:
        doc = proc(doc)
    return np.array([1.0 if t.pos_ in FUNCTION_POS else 0.0 for t in doc])


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


def get_patient_base(pid):
    """Load once per patient: neural curvature, semantic curvature, and all
    covariates (word-level, pre-chunking) aligned to the same [2:n][valid]
    indexing used throughout this project."""
    data = load_podcast_patient(pid, region=REGION, model_tag=SEM_MODEL, context_words=200)
    if data is None:
        return None
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
    if valid.sum() < 50:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    if SEM_LAYER >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_L = np.array(emb_mm[SEM_LAYER, :n], dtype=np.float64)
    del emb_mm
    if emb_L.std() == 0:
        return None
    Xs = StandardScaler().fit_transform(emb_L)
    c_sem = trajectory_curvature(Xs)[valid]

    Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]

    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    onset_ms, offset_ms = onset_ms[:n], offset_ms[:n]
    word_dur = offset_ms - onset_ms

    words = np.asarray(data["words"][:n], dtype=object)
    is_func_full = get_is_function(words)

    freq = data["log_freq"][:n][2:n][valid]
    surp = data["surp"][:n][2:n][valid]
    dur = word_dur[2:n][valid]
    pos = data["sent_pos"][:n][2:n][valid]
    isfunc = is_func_full[2:n][valid]

    return dict(c_sem=c_sem, c_neu=c_neu, freq=freq, surp=surp, dur=dur, pos=pos, isfunc=isfunc)


def build_chunked_df(patient_data, curv_key, chunk):
    rows = []
    for pid, d in patient_data.items():
        curv = d[curv_key]
        c_chunk, idx = _chunk_mean(curv, chunk)
        freq_c = np.array([np.nanmean(d["freq"][i:i + chunk]) for i in idx])
        surp_c = np.array([np.nanmean(d["surp"][i:i + chunk]) for i in idx])
        dur_c = np.array([np.nanmean(d["dur"][i:i + chunk]) for i in idx])
        pos_c = np.array([np.nanmean(d["pos"][i:i + chunk]) for i in idx])
        isfunc_c = np.array([np.nanmean(d["isfunc"][i:i + chunk]) for i in idx])

        keep = np.isfinite(c_chunk) & np.isfinite(freq_c) & np.isfinite(surp_c) & np.isfinite(dur_c) & \
               np.isfinite(pos_c) & np.isfinite(isfunc_c)
        c_chunk, freq_c, surp_c, dur_c, pos_c, isfunc_c = (
            c_chunk[keep], freq_c[keep], surp_c[keep], dur_c[keep], pos_c[keep], isfunc_c[keep])
        rows += [(pid, *r) for r in zip(zscore(c_chunk), zscore(freq_c), zscore(surp_c),
                                         zscore(dur_c), zscore(pos_c), zscore(isfunc_c))]
    return pd.DataFrame(rows, columns=["patient", "curv_z", "freq_z", "surp_z", "dur_z", "pos_z", "isfunc_z"])


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
    patient_data = {}
    for pid in PODCAST_PATIENTS:
        d = get_patient_base(pid)
        if d is None:
            print(f"  {pid}: SKIP")
            continue
        patient_data[pid] = d
        print(f"  {pid}: loaded", flush=True)

    results = {}
    for curv_key, curv_label in [("c_sem", "semantic"), ("c_neu", "neural")]:
        for chunk in CHUNKS:
            df = build_chunked_df(patient_data, curv_key, chunk)
            tag = f"{curv_label}_chunk{chunk}"
            print(f"\n=== {tag} ===  n_rows={len(df)}  n_patients={df.patient.nunique()}")

            raw_r_freq, _ = pearsonr(df.freq_z, df.curv_z)
            raw_r_surp, _ = pearsonr(df.surp_z, df.curv_z)

            out_freq_noPOS = bootstrap_unique_r2(df, ["surp_z", "dur_z", "pos_z"], "freq_z")
            out_surp_noPOS = bootstrap_unique_r2(df, ["freq_z", "dur_z", "pos_z"], "surp_z")
            out_freq_POS = bootstrap_unique_r2(df, ["surp_z", "dur_z", "pos_z", "isfunc_z"], "freq_z")
            out_surp_POS = bootstrap_unique_r2(df, ["freq_z", "dur_z", "pos_z", "isfunc_z"], "surp_z")

            ratio_noPOS = out_freq_noPOS["unique_r2"] / max(out_surp_noPOS["unique_r2"], 1e-9)
            ratio_POS = out_freq_POS["unique_r2"] / max(out_surp_POS["unique_r2"], 1e-9)

            print(f"  raw r(freq,curv)={raw_r_freq:+.4f}  raw r(surp,curv)={raw_r_surp:+.4f}")
            print(f"  [no POS]   freq_R2={out_freq_noPOS['unique_r2']:+.5f} {stars(out_freq_noPOS['p'])}   "
                  f"surp_R2={out_surp_noPOS['unique_r2']:+.5f} {stars(out_surp_noPOS['p'])}   ratio(freq/surp)={ratio_noPOS:.2f}x")
            print(f"  [with POS] freq_R2={out_freq_POS['unique_r2']:+.5f} {stars(out_freq_POS['p'])}   "
                  f"surp_R2={out_surp_POS['unique_r2']:+.5f} {stars(out_surp_POS['p'])}   ratio(freq/surp)={ratio_POS:.2f}x")

            results[tag] = dict(
                n_rows=len(df), n_patients=int(df.patient.nunique()),
                raw_r_freq=raw_r_freq, raw_r_surp=raw_r_surp,
                no_pos_control=dict(unique_freq=out_freq_noPOS, unique_surp=out_surp_noPOS, ratio_freq_over_surp=ratio_noPOS),
                with_pos_control=dict(unique_freq=out_freq_POS, unique_surp=out_surp_POS, ratio_freq_over_surp=ratio_POS),
            )
            json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
