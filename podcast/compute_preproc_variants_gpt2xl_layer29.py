"""
compute_preproc_variants_gpt2xl_layer29.py
==========================================
How does the preprocessing of each space change curvature / displacement and
their neural-semantic alignment?

Semantic (GPT-2 XL layer 29, 1600-D):
  zscore  : z-score each hidden unit (current pipeline)
  raw     : raw activations (Hosseini & Fedorenko 2023)
  l2      : raw activations, each word vector scaled to unit length

Neural (hippocampus spike counts, dur80/40 window, N neurons):
  zscore  : z-score each neuron (current pipeline)
  sqrt    : 2*sqrt(count)  (Henaff/Goris 2021 Eq. 8 with sigma->0, pure Poisson)
  log     : log(1 + count)
  l2      : raw counts, each word vector scaled to unit length
  sqrt_l2 : 2*sqrt(count), each word vector scaled to unit length

Dims: full (no PCA) and pc8 (centered, unscaled PCA on the transformed data;
      for zscore this equals the current pipeline).

Zero-length steps (identical consecutive vectors) give NaN curvature here
(trajectory_curvature in run_geometry_analysis_pca20 returns 0 deg); their
fraction is reported per patient.

Alignment metric: chunk_r (3-word chunks) on valid words, as in
compute_curv_disp_pos_gpt2xl_layer29.py. Displacement is symmetrised
(d_{t-1}+d_t)/2 to align with curvature (methods_section.tex).
Patients with NaN word blocks (YEX) use their longest finite contiguous run.

Saves: results/preproc_variants_gpt2xl_layer29.json

    conda run -n gpt2_embed --no-capture-output python3 -u compute_preproc_variants_gpt2xl_layer29.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

MODEL, LAYER, K, CHUNK = "gpt2-xl", 29, 8, 3
RESULTS = os.path.join(BASE_DIR, "results")
OUT = os.path.join(RESULTS, "preproc_variants_gpt2xl_layer29.json")


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    out = np.full_like(X, np.nan)
    nz = n[:, 0] > 0
    out[nz] = X[nz] / n[nz]
    return out


SEM = {
    "zscore": lambda X: StandardScaler().fit_transform(X),
    "raw": lambda X: X,
    "l2": unit,
}
NEU = {
    "zscore": lambda X: np.nan_to_num(StandardScaler().fit_transform(X)),
    "sqrt": lambda X: 2.0 * np.sqrt(X),
    "log": np.log1p,
    "l2": unit,
    "sqrt_l2": lambda X: unit(2.0 * np.sqrt(X)),
}


def reduce(X, dims):
    if dims == "full":
        return X
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.nan_to_num(Xc)
    return PCA(n_components=min(K, X.shape[1]), svd_solver="full").fit_transform(Xc)


def curv_disp(X):
    """Per interior word: turning angle (deg, NaN if a step has zero length) and
    symmetrised step length. Both length T-2."""
    D = np.diff(X, axis=0)
    L = np.linalg.norm(D, axis=1)
    num = np.einsum("ij,ij->i", D[:-1], D[1:])
    den = L[:-1] * L[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    c[~(den > 0)] = np.nan
    d = 0.5 * (L[:-1] + L[1:])
    return c, d, float(np.mean(L == 0))


def chunk_r(x, y, cs=CHUNK, min_chunks=3):
    fin = np.isfinite(x) & np.isfinite(y)
    xa, ya = [], []
    for i in range(0, len(x) - cs + 1, cs):
        f = fin[i:i + cs]
        if f.sum() < cs // 2:
            continue
        xa.append(np.mean(x[i:i + cs][f])); ya.append(np.mean(y[i:i + cs][f]))
    return float(pearsonr(xa, ya)[0]) if len(xa) >= min_chunks else float("nan")


def longest_finite_run(ok):
    best, cur, start, bstart = 0, 0, 0, 0
    for i, v in enumerate(ok):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, bstart = cur, start
        else:
            cur = 0
    return bstart, bstart + best


def main():
    out = {}
    for pid in PODCAST_PATIENTS:
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n0 = min(data["n_words"], len(on))
        neu = windowed_neu_vecs(pid, "hippocampus", _convo_regions(pid), on[:n0], off[:n0], 80, 40, rate=False)
        if neu is None:
            continue
        emb = np.load(data["emb_path"], mmap_mode="r")
        n = min(n0, emb.shape[1], len(neu))
        sem = np.array(emb[LAYER, :n], dtype=np.float64); del emb
        neu = np.asarray(neu[:n], dtype=np.float64)
        a, b = longest_finite_run(np.isfinite(sem).all(1) & np.isfinite(neu).all(1))
        sem, neu = sem[a:b], neu[a:b]
        surp = np.asarray(data["surp"])[a:b]; sp = np.asarray(data["sent_pos"])[a:b]
        turn = np.asarray(data["is_turn"])[a:b]
        valid = build_valid_mask(len(sem), surp, sp, turn)

        feats = {"sem": {}, "neu": {}}
        zero_frac = {}
        for space, X, variants in (("sem", sem, SEM), ("neu", neu, NEU)):
            for v, f in variants.items():
                Xt = f(X)
                for dims in ("full", "pc8"):
                    c, d, z = curv_disp(reduce(Xt, dims))
                    feats[space][(v, dims)] = (c[valid], d[valid])
                    zero_frac[f"{space}/{v}/{dims}"] = z

        res = {"n_words": int(len(sem)), "n_neurons": int(neu.shape[1]),
               "words_used": [int(a), int(b)], "n_valid": int(valid.sum()),
               "zero_step_frac": zero_frac, "mean_curv": {}, "r": {}}
        for space in ("sem", "neu"):
            for (v, dims), (c, _) in feats[space].items():
                res["mean_curv"][f"{space}/{v}/{dims}"] = float(np.nanmean(c))
        for (sv, sd), (cs, ds) in feats["sem"].items():
            for (nv, nd), (cn, dn) in feats["neu"].items():
                if sd != nd:
                    continue
                res["r"][f"{sv}|{nv}|{sd}"] = {"curv": chunk_r(cn, cs), "disp": chunk_r(dn, ds)}
        out[pid] = res
        print(f"{pid}: words={len(sem)} neurons={neu.shape[1]} valid={valid.sum()} "
              f"neu zero-step frac (raw counts)={zero_frac['neu/sqrt/full']:.3f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    # summary across patients
    keys = sorted(next(iter(out.values()))["r"].keys())
    print("\nsem|neu|dims            curv r: mean (t, p)            disp r: mean (t, p)")
    for k in keys:
        row = []
        for m in ("curv", "disp"):
            v = np.array([out[p]["r"][k][m] for p in out if np.isfinite(out[p]["r"][k][m])])
            t, pv = ttest_1samp(v, 0)
            row.append(f"{v.mean():+.4f} (t={t:+.2f}, p={pv:.3g}, n={len(v)})")
        print(f"{k:<22} {row[0]:<38} {row[1]}")
    print("\nmean curvature (deg), averaged over patients")
    for k in sorted(next(iter(out.values()))["mean_curv"].keys()):
        print(f"  {k:<22} {np.mean([out[p]['mean_curv'][k] for p in out]):.1f}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
