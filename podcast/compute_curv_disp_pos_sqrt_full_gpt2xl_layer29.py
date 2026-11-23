"""
compute_curv_disp_pos_sqrt_full_gpt2xl_layer29.py
=================================================
Curvature / displacement / position alignment with the adopted preprocessing
(2026-11-22):

  semantic : GPT-2 XL layer 29 raw activations, 1600-D   (Hosseini & Fedorenko 2023)
  neural   : hippocampus 2*sqrt(spike count), dur80/40 window, D = n neurons
             (Henaff/Bai/Goris 2021, sigma -> 0; distance ~ discriminability)
  space    : full (no z-score, no PCA)

Per interior word t (length T-2, aligned with build_valid_mask):
  curvature    c_t = angle between x_t - x_{t-1} and x_{t+1} - x_t (deg);
                     NaN if either step has zero length
  displacement d_t = (|x_t - x_{t-1}| + |x_{t+1} - x_t|) / 2
  position     p_t = |x_t - session centroid|     ("pos")
               m_t = |x_t|                         ("mag"; neural = 2*sqrt(total count))

Alignment: chunk_r (3-word chunks) between neural and semantic, valid words only.
Patients with NaN word blocks (YEX) use their longest finite contiguous run.

Saves: results/curv_disp_pos_sqrt_full_gpt2xl_layer29.json

    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_pos_sqrt_full_gpt2xl_layer29.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_1samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)
from compute_preproc_variants_gpt2xl_layer29 import chunk_r, longest_finite_run

MODEL, LAYER = "gpt2-xl", 29
RESULTS = os.path.join(BASE_DIR, "results")
OUT = os.path.join(RESULTS, "curv_disp_pos_sqrt_full_gpt2xl_layer29.json")
METRICS = ("curv", "disp", "pos", "mag")


def sem_transform(X):
    return X


def neu_transform(counts):
    return 2.0 * np.sqrt(counts)


def quantities(X):
    D = np.diff(X, axis=0)
    L = np.linalg.norm(D, axis=1)
    num = np.einsum("ij,ij->i", D[:-1], D[1:])
    den = L[:-1] * L[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    c[~(den > 0)] = np.nan
    inner = X[1:-1]
    return {
        "curv": c,
        "disp": 0.5 * (L[:-1] + L[1:]),
        "pos": np.linalg.norm(inner - X.mean(0), axis=1),
        "mag": np.linalg.norm(inner, axis=1),
    }, float(np.mean(L == 0))


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
        valid = build_valid_mask(len(sem), np.asarray(data["surp"])[a:b],
                                 np.asarray(data["sent_pos"])[a:b], np.asarray(data["is_turn"])[a:b])

        qs, zs = quantities(sem_transform(sem))
        qn, zn = quantities(neu_transform(neu))
        r = {m: chunk_r(qn[m][valid], qs[m][valid]) for m in METRICS}
        out[pid] = {"n_words": int(len(sem)), "n_neurons": int(neu.shape[1]),
                    "words_used": [int(a), int(b)], "n_valid": int(valid.sum()),
                    "zero_step_frac": {"sem": zs, "neu": zn},
                    "mean_curv": {"sem": float(np.nanmean(qs["curv"][valid])),
                                  "neu": float(np.nanmean(qn["curv"][valid]))},
                    "r": r}
        print(f"{pid}: neurons={neu.shape[1]:>2} valid={valid.sum():>4}  "
              + "  ".join(f"{m}={r[m]:+.4f}" for m in METRICS), flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"\n=== raw sem x 2sqrt(count) neu, full space, L{LAYER}, chunk=3, n={len(out)} ===")
    for m in METRICS:
        v = np.array([out[p]["r"][m] for p in out if np.isfinite(out[p]["r"][m])])
        t, pv = ttest_1samp(v, 0)
        print(f"  {m:<5} mean={v.mean():+.4f}  sem={v.std(ddof=1)/np.sqrt(len(v)):.4f}  "
              f"t={t:+.2f}  p={pv:.2e}  n_pos={(v > 0).sum()}/{len(v)}")
    for s in ("sem", "neu"):
        print(f"  mean curvature {s}: {np.mean([out[p]['mean_curv'][s] for p in out]):.1f} deg")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
