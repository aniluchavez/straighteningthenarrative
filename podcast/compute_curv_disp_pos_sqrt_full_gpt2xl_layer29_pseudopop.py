"""
compute_curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop.py
===========================================================
Pseudopopulation version of compute_curv_disp_pos_sqrt_full_gpt2xl_layer29.py.

Neural: results/pseudopop_counts_dur8040_true.npz (built by
        compute_poisson_normalized_geometry.py pseudopop, 2026-11-11):
        12 patients (YEX excluded), 436 hippocampal units, dur80/40 spike
        counts, words aligned to YEY (difflib), common words only.
        -> 2*sqrt(count), full 436-D.
Semantic: GPT-2 XL layer 29 raw activations at the same YEY word indices, 1600-D.

Quantities per interior word (curvature, symmetrised displacement,
|x - centroid|, |x|) as in the per-patient script. Valid mask from YEY,
plus triplets that bridge a gap in the common-word index are dropped.

Inference (one pseudo-subject): circular shift of the neural chunk-mean
series (shift >= 10% of length), N_SHIFT shifts, two-sided p.

Saves: results/curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop.json

    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop.py
"""
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import load_podcast_patient
from compute_curv_disp_pos_sqrt_full_gpt2xl_layer29 import quantities, neu_transform, sem_transform, METRICS

MODEL, LAYER, REF_PID = "gpt2-xl", 29, "YEY"
CHUNK, N_SHIFT, SEED = 3, 5000, 0
RESULTS = os.path.join(BASE_DIR, "results")
CACHE = os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz")
OUT = os.path.join(RESULTS, "curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop.json")


def load_pseudopop():
    c = np.load(CACHE)
    K, common = c["K"].astype(np.float64), c["common"]
    ref = load_podcast_patient(REF_PID, region="hippocampus", model_tag=MODEL, context_words=200)
    sem = np.array(np.load(ref["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    valid = build_valid_mask(len(common), np.asarray(ref["surp"])[common],
                             np.asarray(ref["sent_pos"])[common], np.asarray(ref["is_turn"])[common])
    valid &= (common[2:] - common[:-2]) == 2          # no triplet across a gap
    return sem, K, common, valid, list(c["patients"]), c["block_sizes"]


def chunk_means(x, y, cs=CHUNK):
    fin = np.isfinite(x) & np.isfinite(y)
    xa, ya = [], []
    for i in range(0, len(x) - cs + 1, cs):
        f = fin[i:i + cs]
        if f.sum() < cs // 2:
            continue
        xa.append(np.mean(x[i:i + cs][f])); ya.append(np.mean(y[i:i + cs][f]))
    return np.array(xa), np.array(ya)


def zs(x):
    return (x - x.mean()) / x.std()


def main():
    sem, K, common, valid, patients, blocks = load_pseudopop()
    print(f"pseudopop: {len(patients)} patients, {K.shape[1]} units, {len(common)} words, valid={valid.sum()}",
          flush=True)
    qs, zsem = quantities(sem_transform(sem))
    qn, zneu = quantities(neu_transform(K))
    rng = np.random.default_rng(SEED)
    res = {}
    for m in METRICS:
        xn, xs = chunk_means(qn[m][valid], qs[m][valid])
        a, b = zs(xn), zs(xs)
        r = float(np.mean(a * b))
        L = len(a)
        sh = rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)
        null = np.array([np.mean(np.roll(a, s) * b) for s in sh])
        res[m] = dict(r=r, p=float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_SHIFT + 1)),
                      null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
                      n_chunks=int(L))
        print(f"  {m:<5} r={r:+.4f}  p={res[m]['p']:.4f}  null 95%=[{res[m]['null_lo']:+.4f}, {res[m]['null_hi']:+.4f}]"
              f"  chunks={L}", flush=True)
    out = dict(patients=patients, block_sizes=[int(b) for b in blocks], n_units=int(K.shape[1]),
               n_words=int(len(common)), n_valid=int(valid.sum()),
               zero_step_frac={"sem": zsem, "neu": zneu},
               mean_curv={"sem": float(np.nanmean(qs["curv"][valid])), "neu": float(np.nanmean(qn["curv"][valid]))},
               r=res)
    print(f"  mean curvature sem={out['mean_curv']['sem']:.1f}  neu={out['mean_curv']['neu']:.1f} deg;"
          f"  zero-step frac neu={zneu:.4f}")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
