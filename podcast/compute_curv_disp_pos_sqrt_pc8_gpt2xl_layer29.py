"""
compute_curv_disp_pos_sqrt_pc8_gpt2xl_layer29.py
================================================
Adopted definitions (2026-11-23):

  semantic : GPT-2 XL layer 29 raw activations (Hosseini & Fedorenko 2023)
  neural   : hippocampus 2*sqrt(spike count), dur80/40 (Henaff/Goris 2021, sigma->0)
  space    : top-8 PCs of mean-subtracted (centered), UNSCALED data, per space
             y_t = PCA_8(x_t - mean)
  position     p_t = ||y_t||  (= distance from the session centroid in the 8 PCs)
  displacement d_t = (||y_t - y_{t-1}|| + ||y_{t+1} - y_t||) / 2
  curvature    c_t = angle(y_t - y_{t-1}, y_{t+1} - y_t), NaN for zero-length steps

Alignment: chunk_r (3-word chunks), valid words (build_valid_mask).
  perpatient : r per patient, t-test across 13 patients (YEX: longest finite run)
  pseudopop  : 12 patients / 436 units (results/pseudopop_counts_dur8040_true.npz),
               circular-shift null on the neural chunk-mean series

Saves: results/curv_disp_pos_sqrt_pc8_gpt2xl_layer29.json

    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_pos_sqrt_pc8_gpt2xl_layer29.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
                            load_words_onset_offset, windowed_neu_vecs)
from compute_preproc_variants_gpt2xl_layer29 import chunk_r, longest_finite_run
from compute_curv_disp_pos_sqrt_full_gpt2xl_layer29 import quantities
from compute_curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop import load_pseudopop, chunk_means, zs

MODEL, LAYER, K = "gpt2-xl", 29, 8
METRICS = ("curv", "disp", "pos")
N_SHIFT, SEED = 5000, 0
RESULTS = os.path.join(BASE_DIR, "results")
OUT = os.path.join(RESULTS, "curv_disp_pos_sqrt_pc8_gpt2xl_layer29.json")


def pc8(X):
    """Top-K PCs of centered, unscaled X (PCA centers internally)."""
    return PCA(n_components=min(K, X.shape[1]), svd_solver="full").fit_transform(X)


def feats(X):
    q, z = quantities(pc8(X))
    return {"curv": q["curv"], "disp": q["disp"], "pos": q["mag"]}, z   # |y| of centered PCs


def run_perpatient():
    per = {}
    for pid in PODCAST_PATIENTS:
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n0 = min(data["n_words"], len(on))
        Kc = windowed_neu_vecs(pid, "hippocampus", _convo_regions(pid), on[:n0], off[:n0], 80, 40, rate=False)
        if Kc is None:
            continue
        emb = np.load(data["emb_path"], mmap_mode="r")
        n = min(n0, emb.shape[1], len(Kc))
        sem = np.array(emb[LAYER, :n], dtype=np.float64); del emb
        neu = 2.0 * np.sqrt(np.asarray(Kc[:n], dtype=np.float64))
        a, b = longest_finite_run(np.isfinite(sem).all(1) & np.isfinite(neu).all(1))
        sem, neu = sem[a:b], neu[a:b]
        valid = build_valid_mask(len(sem), np.asarray(data["surp"])[a:b],
                                 np.asarray(data["sent_pos"])[a:b], np.asarray(data["is_turn"])[a:b])
        qs, _ = feats(sem); qn, zn = feats(neu)
        r = {m: chunk_r(qn[m][valid], qs[m][valid]) for m in METRICS}
        per[pid] = dict(n_neurons=int(neu.shape[1]), n_valid=int(valid.sum()), zero_step_frac_neu=zn, r=r)
        print(f"  {pid}: neurons={neu.shape[1]:>2} valid={valid.sum():>4}  "
              + "  ".join(f"{m}={r[m]:+.4f}" for m in METRICS), flush=True)
    summ = {}
    print(f"\n=== per patient, 8 centered PCs, n={len(per)} ===")
    for m in METRICS:
        v = np.array([per[p]["r"][m] for p in per if np.isfinite(per[p]["r"][m])])
        t, p = ttest_1samp(v, 0)
        summ[m] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))), t=float(t), p=float(p),
                       n_pos=int((v > 0).sum()), n=len(v))
        print(f"  {m:<5} mean={v.mean():+.4f}  sem={summ[m]['sem']:.4f}  t={t:+.2f}  p={p:.2e}  n_pos={(v > 0).sum()}/{len(v)}")
    return dict(per_patient=per, summary=summ)


def run_pseudopop():
    sem, Kc, common, valid, patients, _ = load_pseudopop()
    qs, _ = feats(sem); qn, _ = feats(2.0 * np.sqrt(Kc))
    rng = np.random.default_rng(SEED)
    res = {}
    print(f"\n=== pseudopop ({len(patients)} patients, {Kc.shape[1]} units), 8 centered PCs ===")
    for m in METRICS:
        xn, xs = chunk_means(qn[m][valid], qs[m][valid])
        a, b = zs(xn), zs(xs)
        r = float(np.mean(a * b)); L = len(a)
        sh = rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)
        null = np.array([np.mean(np.roll(a, s) * b) for s in sh])
        res[m] = dict(r=r, p=float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_SHIFT + 1)),
                      null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)))
        print(f"  {m:<5} r={r:+.4f}  p={res[m]['p']:.4f}  null 95%=[{res[m]['null_lo']:+.4f}, {res[m]['null_hi']:+.4f}]")
    return dict(patients=patients, n_units=int(Kc.shape[1]), n_valid=int(valid.sum()), r=res)


def main():
    out = {"perpatient": run_perpatient(), "pseudopop": run_pseudopop()}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
