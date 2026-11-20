"""
compute_curvature_surprisal_frequency_regression_pseudopop.py
================================================================
Variance partition: turning angle AT each word (event-word vertex, lag +1
convention) ~ surprisal + log SUBTLEX frequency + log word duration
(z-scored), valid words common to all 13 patients.

Curvature series:
  neural   pseudopop 475 units, fixed [onset+80, onset+330] counts (all units)
  sem475   gpt2-xl layer 29, top 475 standardized PCs
  semfull  gpt2-xl layer 29, raw 1600-d embedding

Reports standardized betas (full model), simple r with curvature, and
unique R^2 (full-model R^2 minus R^2 without that predictor), with
p-values from N_SHIFT circular shifts of the curvature series (>= 10% of
length), which keeps its autocorrelation.

Saves: results/curvature_surprisal_frequency_regression_pseudopop.json
"""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

sys.argv = sys.argv[:1]
from run_geometry_analysis_pca20 import trajectory_curvature
from podcast_loader import load_podcast_patient, load_words_onset_offset
from make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 import subtlex_log_freq
import compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop as PP

PP.RATE = False
PP.ONSET_PAD_MS, PP.OFFSET_PAD_MS = 80, 330
_lwoo = PP.load_words_onset_offset
PP.load_words_onset_offset = lambda pid: (lambda w, on, off: (w, on, on))(*_lwoo(pid))
PP.CACHE_PATH = os.path.join(RESULTS, "pseudopop_full_sequence_fixed250_count.npz")

N_SHIFT = 1000
PRED = ["surprisal", "log_freq", "log_dur"]


def r2(X, y):
    X1 = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    return 1 - res @ res / ((y - y.mean()) @ (y - y.mean())), b[1:]


def r2_many(X, Y):
    """R^2 of each column of Y on X."""
    X1 = np.c_[np.ones(len(X)), X]
    B = np.linalg.lstsq(X1, Y, rcond=None)[0]
    R = Y - X1 @ B
    Yc = Y - Y.mean(axis=0)
    return 1 - (R ** 2).sum(axis=0) / (Yc ** 2).sum(axis=0)


def main():
    neu_z, common, valid = PP.build_pseudopop()
    n = len(common)
    _, on, off = load_words_onset_offset(PP.REF_PID)
    _, _, off_true = _lwoo(PP.REF_PID)
    d = load_podcast_patient(PP.REF_PID, region="hippocampus", model_tag="gpt2-xl", context_words=200)
    surp = np.asarray(d["surp"], dtype=float)[common]
    lf = subtlex_log_freq(np.asarray(d["words"])[common])
    dur = (off_true - on)[common]
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
    emb = np.array(np.load(d["emb_path"], mmap_mode="r")[29, common], dtype=np.float64)
    series = {
        "neural": np.nan_to_num(neu_z),
        "sem475": PCA(n_components=475, svd_solver="full").fit_transform(StandardScaler().fit_transform(emb)),
        "semfull": emb,
    }
    vfull = np.zeros(n, bool)
    vfull[:len(valid)] = valid[:n]
    curv = {k: np.r_[np.nan, trajectory_curvature(v), np.nan][:n] for k, v in series.items()}
    m = vfull & np.isfinite(surp) & np.isfinite(lf) & np.all([np.isfinite(c) for c in curv.values()], axis=0)
    X = StandardScaler().fit_transform(np.c_[surp[m], lf[m], np.log(dur[m])])
    print(f"words={int(m.sum())}  predictor correlations: surp-freq {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:+.3f}  "
          f"surp-dur {np.corrcoef(X[:, 0], X[:, 2])[0, 1]:+.3f}  freq-dur {np.corrcoef(X[:, 1], X[:, 2])[0, 1]:+.3f}")
    rng = np.random.default_rng(0)
    L = int(m.sum())
    shifts = rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)

    out = dict(n_words=L, predictors=PRED, results={})
    for name, c in curv.items():
        y = (c[m] - c[m].mean()) / c[m].std()
        R2_full, betas = r2(X, y)
        Ysh = np.stack([np.roll(y, s) for s in shifts], axis=1)
        full_null = r2_many(X, Ysh)
        res = dict(R2_full=float(R2_full), p_full=float((np.sum(full_null >= R2_full) + 1) / (N_SHIFT + 1)), preds={})
        for j, p in enumerate(PRED):
            keep = [k for k in range(len(PRED)) if k != j]
            uniq = R2_full - r2(X[:, keep], y)[0]
            uniq_null = full_null - r2_many(X[:, keep], Ysh)
            res["preds"][p] = dict(beta=float(betas[j]), r=float(np.corrcoef(X[:, j], y)[0, 1]),
                                   unique_R2=float(uniq),
                                   p_unique=float((np.sum(uniq_null >= uniq) + 1) / (N_SHIFT + 1)))
        # surprisal + frequency only (no duration), for comparison
        R2_sf, b_sf = r2(X[:, :2], y)
        res["no_duration"] = dict(R2=float(R2_sf), unique_surprisal=float(R2_sf - r2(X[:, [1]], y)[0]),
                                  unique_freq=float(R2_sf - r2(X[:, [0]], y)[0]),
                                  beta_surprisal=float(b_sf[0]), beta_freq=float(b_sf[1]))
        out["results"][name] = res
        print(f"\n== {name}: full R2 = {R2_full * 100:.2f}% (p={res['p_full']:.3f})")
        for p in PRED:
            q = res["preds"][p]
            print(f"   {p:10s} r={q['r']:+.3f}  beta={q['beta']:+.3f}  unique R2={q['unique_R2'] * 100:.3f}%  p={q['p_unique']:.3f}")
        nd = res["no_duration"]
        print(f"   [surprisal + freq only] R2={nd['R2'] * 100:.2f}%  unique surp={nd['unique_surprisal'] * 100:.3f}%  "
              f"unique freq={nd['unique_freq'] * 100:.3f}%  beta surp={nd['beta_surprisal']:+.3f} freq={nd['beta_freq']:+.3f}")
    json.dump(out, open(os.path.join(RESULTS, "curvature_surprisal_frequency_regression_pseudopop.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
