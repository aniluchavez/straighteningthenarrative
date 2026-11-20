"""
definitive_curv_disp_pos_test.py
====================================
The "definitive" version of the curvature vs displacement vs position
comparison, addressing every objection raised so far in this investigation:

1. No PCA/K choice on the semantic side -- native (full) embedding
   dimensionality, standardized only. Removes the K-dependence confound
   entirely rather than trying to "match" it.
2. Real ENGLISH CORPUS frequency (wordfreq.zipf_frequency), not the
   within-document frequency get_log_freq() computes elsewhere in this
   project.
3. Confound-controlled: each semantic feature's contribution is measured
   as UNIQUE R^2 beyond a nuisance block (Zipf frequency, word duration,
   sentence position) -- same nested-deviance logic already validated
   this session for frequency vs surprisal.
4. Cross-validated: leave-one-patient-out. z-scoring for each fold is
   fit on the TRAIN patients only, applied to the held-out patient, to
   avoid leakage. Reports out-of-sample R^2, which can go negative --
   that's expected and informative (a feature that's just fitting noise
   in-sample will not generalize).
5. Raw spike counts (not rate-normalized), dur8040 duration-locked window
   ([onset-80ms, offset+40ms]).
6. Matched pairs only (semantic curvature -> neural curvature, etc.) --
   each feature competes against nuisance alone, not against the other
   two features for a shared outcome, keeping the design simple and
   directly interpretable.

llama-3.1-8b (layer 5), gpt2-large (layer 19). Podcast, all 13 patients.

Saves: results/definitive_curv_disp_pos_test.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import wordfreq
from sklearn.decomposition import PCA

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS    = os.path.join(PARENT_DIR, "results")
REGION = "hippocampus"
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, load_words_onset_offset, windowed_neu_vecs, _convo_regions,
)

MODELS = {"llama-3.1-8b": 5, "gpt2-large": 19}


def position(X):
    X = np.asarray(X, dtype=np.float64)
    c = X.mean(axis=0, keepdims=True)
    return np.linalg.norm(X - c, axis=1)


def get_zipf(words):
    return np.array([wordfreq.zipf_frequency(w, "en") if w else 0.0 for w in words], dtype=float)


def zscore_fit(arr):
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    sd = sd if sd > 1e-9 else 1.0
    return mu, sd


def zscore_apply(arr, mu, sd):
    return (arr - mu) / sd


def r2_ols_fit(X, y):
    """Fit OLS (X includes intercept col) on train data, return beta."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def r2_oos(X_test, y_test, beta):
    pred = X_test @ beta
    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def r2_in_sample(X, y, beta):
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def load_patient_features(pid, model_tag, layer):
    data0 = load_podcast_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
    if data0 is None:
        return None
    try:
        words_raw, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    preg = _convo_regions(pid)
    n = min(data0["n_words"], len(onset_ms))
    surp, sp, is_turn = data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n]
    valid = build_valid_mask(n, surp, sp, is_turn)
    if valid.sum() < 20:
        return None

    neu_vecs = windowed_neu_vecs(pid, REGION, preg, onset_ms[:n], offset_ms[:n],
                                  ONSET_PAD_MS, OFFSET_PAD_MS, rate=False)
    if neu_vecs is None:
        return None
    Y_neu, n_pcs_neu = fit_neural_pca(neu_vecs, n)
    if Y_neu is None or n_pcs_neu is None or n_pcs_neu < 2:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    p_neu = position(Y_neu)[1:-1][valid]

    emb_mm = np.load(data0["emb_path"], mmap_mode="r")
    if layer >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_L = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    emb_std = (emb_L - emb_L.mean(axis=0)) / (emb_L.std(axis=0) + 1e-9)
    kk = min(n_pcs_neu, emb_std.shape[1], emb_std.shape[0] - 1)
    Xs = PCA(n_components=kk, svd_solver="randomized", random_state=0).fit_transform(emb_std)
    print(f"    {pid}: neural n_pcs={n_pcs_neu}  (semantic matched to same K)", flush=True)
    c_sem = trajectory_curvature(Xs)[valid]
    d_sem = trajectory_displacement(Xs)[:n - 2][valid]
    p_sem = position(Xs)[1:-1][valid]

    words = data0["words"][:n]
    zipf = get_zipf(words)[1:-1][valid]
    word_dur = (offset_ms[:n] - onset_ms[:n])[1:-1][valid]
    sent_pos_y = sp[1:-1][valid]

    keep = (np.isfinite(c_neu) & np.isfinite(d_neu) & np.isfinite(p_neu) &
            np.isfinite(c_sem) & np.isfinite(d_sem) & np.isfinite(p_sem) &
            np.isfinite(zipf) & np.isfinite(word_dur) & np.isfinite(sent_pos_y))

    return dict(c_neu=c_neu[keep], d_neu=d_neu[keep], p_neu=p_neu[keep],
                c_sem=c_sem[keep], d_sem=d_sem[keep], p_sem=p_sem[keep],
                zipf=zipf[keep], dur=word_dur[keep], sentpos=sent_pos_y[keep],
                n=int(keep.sum()))


def run_lopo(patient_data, neu_key, sem_key):
    """Leave-one-patient-out CV: Base = zipf+dur+sentpos, Full = Base + sem_key."""
    pids = list(patient_data.keys())
    oos_base, oos_full = [], []
    r2_base_pooled_num, r2_base_pooled_den = 0.0, 0.0
    r2_full_pooled_num, r2_full_pooled_den = 0.0, 0.0
    in_sample_unique = []

    for held_pid in pids:
        train_pids = [p for p in pids if p != held_pid]

        def stack(cols, pid_list):
            return {c: np.concatenate([patient_data[p][c] for p in pid_list]) for c in cols}

        cols = ["zipf", "dur", "sentpos", sem_key, neu_key]
        train = stack(cols, train_pids)
        test = stack(cols, [held_pid])

        # fit zscore params on train only
        params = {c: zscore_fit(train[c]) for c in ["zipf", "dur", "sentpos", sem_key]}

        def build_X(d, cols_x):
            arrs = [np.ones(len(d[cols_x[0]]))]
            for c in cols_x:
                mu, sd = params[c]
                arrs.append(zscore_apply(d[c], mu, sd))
            return np.column_stack(arrs)

        y_train = train[neu_key]
        y_test = test[neu_key]

        X_base_train = build_X(train, ["zipf", "dur", "sentpos"])
        X_full_train = build_X(train, ["zipf", "dur", "sentpos", sem_key])
        X_base_test = build_X(test, ["zipf", "dur", "sentpos"])
        X_full_test = build_X(test, ["zipf", "dur", "sentpos", sem_key])

        beta_base = r2_ols_fit(X_base_train, y_train)
        beta_full = r2_ols_fit(X_full_train, y_train)

        oos_base.append(r2_oos(X_base_test, y_test, beta_base))
        oos_full.append(r2_oos(X_full_test, y_test, beta_full))

        pred_base = X_base_test @ beta_base
        pred_full = X_full_test @ beta_full
        r2_base_pooled_num += np.sum((y_test - pred_base) ** 2)
        r2_base_pooled_den += np.sum((y_test - y_test.mean()) ** 2)
        r2_full_pooled_num += np.sum((y_test - pred_full) ** 2)
        r2_full_pooled_den += np.sum((y_test - y_test.mean()) ** 2)

        r2b_in = r2_in_sample(X_base_train, y_train, beta_base)
        r2f_in = r2_in_sample(X_full_train, y_train, beta_full)
        in_sample_unique.append(r2f_in - r2b_in)

    r2_base_pooled = 1 - r2_base_pooled_num / r2_base_pooled_den
    r2_full_pooled = 1 - r2_full_pooled_num / r2_full_pooled_den

    return dict(
        oos_r2_base_mean=float(np.mean(oos_base)), oos_r2_full_mean=float(np.mean(oos_full)),
        oos_r2_base_pooled=float(r2_base_pooled), oos_r2_full_pooled=float(r2_full_pooled),
        oos_unique_mean=float(np.mean(oos_full) - np.mean(oos_base)),
        oos_unique_pooled=float(r2_full_pooled - r2_base_pooled),
        in_sample_unique_mean=float(np.mean(in_sample_unique)),
    )


def main():
    results = {}
    for model_tag, layer in MODELS.items():
        print(f"\n=== {model_tag} (layer {layer}, MATCHED per-patient K=neural n_pcs, English Zipf freq, raw counts, LOPO) ===")
        patient_data = {}
        for pid in PODCAST_PATIENTS:
            d = load_patient_features(pid, model_tag, layer)
            if d is None:
                print(f"  {pid}: SKIP")
                continue
            patient_data[pid] = d
            print(f"  {pid}: n={d['n']}  loaded", flush=True)

        model_results = {}
        for neu_key, sem_key, label in [("c_neu", "c_sem", "curvature"),
                                          ("d_neu", "d_sem", "displacement"),
                                          ("p_neu", "p_sem", "position")]:
            out = run_lopo(patient_data, neu_key, sem_key)
            model_results[label] = out
            print(f"  {label:14s}  in-sample unique R2={out['in_sample_unique_mean']:+.5f}   "
                  f"OOS R2: base={out['oos_r2_base_pooled']:+.5f} full={out['oos_r2_full_pooled']:+.5f}  "
                  f"OOS unique (pooled)={out['oos_unique_pooled']:+.5f}  OOS unique (fold-mean)={out['oos_unique_mean']:+.5f}")

        results[model_tag] = model_results
        best = max(model_results.items(), key=lambda kv: kv[1]["oos_unique_pooled"])
        print(f"  -> strongest by OOS unique R2 (pooled): {best[0]} ({best[1]['oos_unique_pooled']:+.5f})")

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "definitive_curv_disp_pos_test_matchedK.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
