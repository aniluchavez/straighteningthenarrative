"""
curvature_freq_surprisal_test.py
====================================
Tests whether curvature (neural and semantic) is driven by word frequency
vs. surprisal, using the "vertex" alignment (curvature[i] pairs with word
i+1, the geometric vertex). Reports:
  1. Raw r(curvature, freq) and r(curvature, surprisal) for both sides
  2. Partial r(curvature, surprisal | freq) and r(curvature, freq | surprisal)
  3. Whether the neural<->semantic curvature correlation itself survives
     controlling for freq and surprisal (shared confound check)
PC8, dur80/40 window, raw counts, bert-base (highest-curvature model).
"""
import os, sys, math
from collections import Counter
import numpy as np
from wordfreq import zipf_frequency
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import trajectory_curvature, build_valid_mask, MIN_PAIRS
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs

K = 8
MODEL = "bert-base"


_zipf_cache = {}
def get_log_freq(words):
    """SUBTLEX-derived English corpus frequency (wordfreq zipf_frequency), NOT
    within-transcript repetition frequency."""
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = w.lower()
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out


def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2: return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def partial_r(x, y, z):
    fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if fin.sum() < 20: return float("nan"), int(fin.sum())
    x, y, z = x[fin], y[fin], z[fin]
    rxy, _ = pearsonr(x, y); rxz, _ = pearsonr(x, z); ryz, _ = pearsonr(y, z)
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom > 1e-8 else float("nan"), int(fin.sum())


def partial_r_2cov(x, y, z1, z2):
    fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(z1) & np.isfinite(z2)
    if fin.sum() < 20: return float("nan")
    x, y, z1, z2 = x[fin], y[fin], z1[fin], z2[fin]
    Z = np.column_stack([z1, z2, np.ones_like(z1)])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None); by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx; ry = y - Z @ by
    r, _ = pearsonr(rx, ry); return float(r)


rows = []
for pid in PODCAST_PATIENTS:
    try:
        words_raw, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        continue
    data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
    if data is None: continue
    n0 = min(data["n_words"], len(onset_ms), len(words_raw))
    preg = _convo_regions(pid)
    neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n0], offset_ms[:n0], 80, 40, rate=False)
    if neu_vecs is None: continue
    n = min(n0, neu_vecs.shape[0])
    valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
    if valid.sum() < MIN_PAIRS: continue

    Y_neu = pca_fixed(neu_vecs[:n])
    if Y_neu is None: continue
    c_neu = trajectory_curvature(Y_neu)  # c_neu[i] = vertex at word i+1

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_l = np.array(emb_mm[7, :n], dtype=np.float64)  # mid-ish layer
    del emb_mm
    Y_sem = pca_fixed(emb_l)
    if Y_sem is None: continue
    c_sem = trajectory_curvature(Y_sem)

    import re as _re
    cleaned_words = [_re.sub(r"[^a-z0-9]", "", str(w).lower()) for w in words_raw[:n]]
    log_freq = get_log_freq(cleaned_words)
    surp = data["surp"][:n]

    # vertex alignment: curvature[i] <-> word i+1's freq/surprisal
    m = min(len(c_neu), len(c_sem), n - 1)
    freq_vertex = log_freq[1:m+1]
    surp_vertex = surp[1:m+1]
    v = valid[1:m+1]  # valid mask shifted to match vertex word

    c_neu_v = c_neu[:m][v[:len(c_neu[:m])]] if False else None
    # simpler: just build parallel arrays and mask together
    mm = min(len(c_neu), len(c_sem), len(freq_vertex), len(surp_vertex), len(valid) - 1)
    cn = c_neu[:mm]; cs = c_sem[:mm]; fr = freq_vertex[:mm]; sp = surp_vertex[:mm]; vv = valid[1:mm+1]

    rows.append(dict(pid=pid, c_neu=cn[vv], c_sem=cs[vv], freq=fr[vv], surp=sp[vv]))
    print(f"  {pid}: n_valid={vv.sum()}", flush=True)

# pool everything, patient-standardized (z-score within patient before pooling, to remove between-patient offset)
def zs(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)

all_c_neu, all_c_sem, all_freq, all_surp = [], [], [], []
for r in rows:
    all_c_neu.append(zs(r["c_neu"])); all_c_sem.append(zs(r["c_sem"]))
    all_freq.append(zs(r["freq"])); all_surp.append(zs(r["surp"]))
all_c_neu = np.concatenate(all_c_neu); all_c_sem = np.concatenate(all_c_sem)
all_freq = np.concatenate(all_freq); all_surp = np.concatenate(all_surp)

print(f"\n=== Pooled, patient-standardized, N={len(all_c_neu)} ===")
print(f"\n--- Raw correlations ---")
for name, arr in [("c_neu", all_c_neu), ("c_sem", all_c_sem)]:
    r_freq, _ = pearsonr(arr[np.isfinite(arr) & np.isfinite(all_freq)], all_freq[np.isfinite(arr) & np.isfinite(all_freq)])
    r_surp, _ = pearsonr(arr[np.isfinite(arr) & np.isfinite(all_surp)], all_surp[np.isfinite(arr) & np.isfinite(all_surp)])
    print(f"  r({name}, freq) = {r_freq:+.4f}   r({name}, surprisal) = {r_surp:+.4f}")

print(f"\n--- Partial correlations (does surprisal survive controlling for freq, and vice versa?) ---")
for name, arr in [("c_neu", all_c_neu), ("c_sem", all_c_sem)]:
    r_surp_given_freq, n1 = partial_r(arr, all_surp, all_freq)
    r_freq_given_surp, n2 = partial_r(arr, all_freq, all_surp)
    print(f"  {name}: r(curv, surprisal | freq) = {r_surp_given_freq:+.4f}  (n={n1})")
    print(f"  {name}: r(curv, freq | surprisal) = {r_freq_given_surp:+.4f}  (n={n2})")

print(f"\n--- Does the neural<->semantic curvature link survive controlling for freq & surprisal? ---")
r_raw, _ = pearsonr(all_c_neu[np.isfinite(all_c_neu) & np.isfinite(all_c_sem)],
                     all_c_sem[np.isfinite(all_c_neu) & np.isfinite(all_c_sem)])
r_partial = partial_r_2cov(all_c_neu, all_c_sem, all_freq, all_surp)
print(f"  raw r(c_neu, c_sem) = {r_raw:+.4f}")
print(f"  partial r(c_neu, c_sem | freq, surprisal) = {r_partial:+.4f}")
