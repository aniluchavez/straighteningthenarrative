"""
compute_freq_surprisal_pos_control_test.py
====================================
Tests the hypothesis that frequency's word-level dominance over surprisal
(unique_R2 ~3-30x surprisal's, per
compute_freq_surprisal_confound_test_multimodel_semantic_wordlevel.py) is
driven by frequency's near-definitional link to syntactic category --
function words (the/and/of/...) are both very frequent AND have
stereotyped, low-variance embeddings, so "is this a function word" acts as
a shortcut that swamps surprisal's more diffuse, context-dependent signal
in a noisy per-word regression.

Adds an is_function (function-word, via spaCy POS: DET/ADP/PRON/CCONJ/
SCONJ/AUX/PART) covariate to the nuisance block and re-runs the SAME nested
unique-R2 test both without and with this control:

  Model A   (surp + dur + pos):                curv_z ~ surp_z + dur_z + pos_z
  Model B   (A + freq):                        + freq_z
  unique_freq_beyond_surp = R2_B - R2_A                       [baseline, no POS control]

  Model A+F (surp + dur + pos + is_function):  curv_z ~ surp_z + dur_z + pos_z + isfunc_z
  Model B+F (A+F + freq):                      + freq_z
  unique_freq_beyond_surp_and_pos = R2_B+F - R2_A+F           [with POS control]

If unique_freq_beyond_surp_and_pos is substantially SMALLER than
unique_freq_beyond_surp, frequency's word-level dominance is largely (or
partly) explained by the function/content-word split, not by a genuine
frequency effect beyond lexical category. Symmetric surprisal-side test
included too, for completeness.

gpt2-xl, layer 20 (the one validated event-triggered-shape layer), all 13
podcast patients, word-level (no chunk-averaging), patient-cluster
bootstrap (2000x).

Saves: results/freq_surprisal_pos_control_test.json
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

REGION = "hippocampus"
MODEL = "gpt2-xl"
LAYER = 20
N_BOOT = 2000
SEED = 0

FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "freq_surprisal_pos_control_test.json")

import spacy
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def get_is_function(words):
    """1.0 if function word (POS in FUNCTION_POS), 0.0 otherwise. Uses a
    pre-tokenized spaCy Doc so tokens stay 1:1 aligned with `words`."""
    from spacy.tokens import Doc
    doc = Doc(_NLP.vocab, words=[str(w) for w in words])
    for name, proc in _NLP.pipeline:
        doc = proc(doc)
    return np.array([1.0 if t.pos_ in FUNCTION_POS else 0.0 for t in doc])


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
    Xs = StandardScaler().fit_transform(emb_L)
    c_sem = trajectory_curvature(Xs)[valid]

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

    keep = np.isfinite(freq) & np.isfinite(surp) & np.isfinite(dur) & np.isfinite(pos) & np.isfinite(isfunc)
    freq, surp, dur, pos, isfunc = freq[keep], surp[keep], dur[keep], pos[keep], isfunc[keep]
    c_sem = c_sem[keep]

    rows = list(zip(zscore(c_sem), zscore(freq), zscore(surp), zscore(dur), zscore(pos), zscore(isfunc)))
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
        print(f"  {pid}: {len(r)} words  done", flush=True)

    df = pd.DataFrame(rows, columns=["patient", "curv_z", "freq_z", "surp_z", "dur_z", "pos_z", "isfunc_z"])
    print(f"\nTotal words: {len(df)}  patients: {df.patient.nunique()}")
    print(f"fraction function-word: {(df.isfunc_z > 0).mean():.3f}  (z-scored, >0 means above-mean function-word-ness)")

    raw_r_freq, _ = pearsonr(df.freq_z, df.curv_z)
    raw_r_surp, _ = pearsonr(df.surp_z, df.curv_z)
    raw_r_freq_isfunc, _ = pearsonr(df.freq_z, df.isfunc_z)

    print(f"\nraw r(freq,curv)={raw_r_freq:+.4f}  raw r(surp,curv)={raw_r_surp:+.4f}  "
          f"raw r(freq,is_function)={raw_r_freq_isfunc:+.4f}")

    out_freq_noPOS = bootstrap_unique_r2(df, ["surp_z", "dur_z", "pos_z"], "freq_z")
    out_surp_noPOS = bootstrap_unique_r2(df, ["freq_z", "dur_z", "pos_z"], "surp_z")
    out_freq_POS = bootstrap_unique_r2(df, ["surp_z", "dur_z", "pos_z", "isfunc_z"], "freq_z")
    out_surp_POS = bootstrap_unique_r2(df, ["freq_z", "dur_z", "pos_z", "isfunc_z"], "surp_z")

    print(f"\n[no POS control]  unique_freq_R2={out_freq_noPOS['unique_r2']:+.5f} "
          f"p={out_freq_noPOS['p']:.4f} {stars(out_freq_noPOS['p'])}   "
          f"unique_surp_R2={out_surp_noPOS['unique_r2']:+.5f} p={out_surp_noPOS['p']:.4f} {stars(out_surp_noPOS['p'])}")
    print(f"[with POS control] unique_freq_R2={out_freq_POS['unique_r2']:+.5f} "
          f"p={out_freq_POS['p']:.4f} {stars(out_freq_POS['p'])}   "
          f"unique_surp_R2={out_surp_POS['unique_r2']:+.5f} p={out_surp_POS['p']:.4f} {stars(out_surp_POS['p'])}")
    print(f"\nfreq dominance ratio (freq/surp unique R2):  no-POS={out_freq_noPOS['unique_r2']/max(out_surp_noPOS['unique_r2'],1e-9):.2f}x   "
          f"with-POS={out_freq_POS['unique_r2']/max(out_surp_POS['unique_r2'],1e-9):.2f}x")

    results = dict(
        model=MODEL, layer=LAYER, n_words=len(df), n_patients=int(df.patient.nunique()),
        raw_r_freq=raw_r_freq, raw_r_surp=raw_r_surp, raw_r_freq_isfunc=raw_r_freq_isfunc,
        no_pos_control=dict(unique_freq_beyond_surp=out_freq_noPOS, unique_surp_beyond_freq=out_surp_noPOS),
        with_pos_control=dict(unique_freq_beyond_surp=out_freq_POS, unique_surp_beyond_freq=out_surp_POS),
    )
    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
