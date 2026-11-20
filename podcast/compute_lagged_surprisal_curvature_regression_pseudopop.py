"""
compute_lagged_surprisal_curvature_regression_pseudopop.py
=============================================================
Does curvature increase AFTER a surprising word? Lagged regression.

For word j and lag L (event-triggered convention: lag +1 = vertex at word j),
the outcome is the turning angle at vertex v = j + L - 1, which is built from
words (v-1, v, v+1). Word j is inside that triplet only for L = 0, +1, +2;
for L >= +3 (and L <= -1) word j is NOT one of the three points.

  curv(v) ~ surprisal(j)
            + [surprisal, log SUBTLEX freq, log duration] of v-1, v, v+1
              (excluding word j itself when j is in the triplet -- its
               surprisal is the predictor of interest)

All predictors z-scored; beta for surprisal(j) per lag, p from N_SHIFT
circular shifts of surprisal(j) relative to the curvature/controls. Rows j
restricted to valid words with all terms finite.

Curvature: neural pseudopop (475 units, fixed [onset+80, onset+330] counts,
all units) and semantic gpt2-xl layer 29 (raw full space).

Saves: results/lagged_surprisal_curvature_regression_pseudopop.json
       figures/fig_lagged_surprisal_curvature_regression_pseudopop.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(BASE_DIR, "figures")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

sys.argv = sys.argv[:1]
from run_geometry_analysis_pca20 import trajectory_curvature
from podcast_loader import load_podcast_patient
from make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 import subtlex_log_freq
import compute_curvature_surprisal_frequency_regression_pseudopop as RG

PP = RG.PP
LAGS = list(range(-3, 8))
N_SHIFT = 1000

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def z(x):
    return (x - x.mean()) / x.std()


def main():
    neu_z, common, valid = PP.build_pseudopop()
    n = len(common)
    _, on, _ = RG.load_words_onset_offset(PP.REF_PID)
    _, _, off = RG._lwoo(PP.REF_PID)
    d = load_podcast_patient(PP.REF_PID, region="hippocampus", model_tag="gpt2-xl", context_words=200)
    surp = np.asarray(d["surp"], float)[common]
    lf = subtlex_log_freq(np.asarray(d["words"])[common])
    dur = (off - on)[common]
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
    ld = np.log(dur)
    emb = np.array(np.load(d["emb_path"], mmap_mode="r")[29, common], float)
    curv = dict(neural=np.r_[np.nan, trajectory_curvature(np.nan_to_num(neu_z)), np.nan][:n],
                semantic=np.r_[np.nan, trajectory_curvature(emb), np.nan][:n])
    vfull = np.zeros(n, bool)
    vfull[:len(valid)] = valid[:n]
    rng = np.random.default_rng(0)

    out = dict(lags=LAGS, results={})
    for name, c in curv.items():
        res = []
        for L in LAGS:
            v = np.arange(n) + L - 1                      # vertex index for each j
            ok = vfull & (v - 1 >= 0) & (v + 1 < n)
            j = np.where(ok)[0]
            vv = v[j]
            cols, labs = [surp[j]], ["surp_j"]
            for off_ in (-1, 0, 1):
                t = vv + off_
                inside = t == j
                for arr, nm in ((surp, "surp"), (lf, "freq"), (ld, "dur")):
                    if nm == "surp" and inside.all():
                        continue                          # word j is this triplet member
                    cols.append(arr[t]); labs.append(f"{nm}_{off_:+d}")
            X = np.column_stack(cols)
            y = c[vv]
            f = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X, y = np.column_stack([z(col) for col in X[f].T]), z(y[f])
            X1 = np.c_[np.ones(len(y)), X]
            beta = np.linalg.lstsq(X1, y, rcond=None)[0][1]
            Lh = len(y)
            shifts = rng.integers(int(0.1 * Lh), Lh - int(0.1 * Lh), size=N_SHIFT)
            null = np.empty(N_SHIFT)
            for i, s in enumerate(shifts):
                Xs = X1.copy(); Xs[:, 1] = np.roll(X[:, 0], s)
                null[i] = np.linalg.lstsq(Xs, y, rcond=None)[0][1]
            p = (np.sum(np.abs(null) >= abs(beta)) + 1) / (N_SHIFT + 1)
            in_trip = L in (0, 1, 2)
            res.append(dict(lag=L, beta=float(beta), null_lo=float(np.percentile(null, 2.5)),
                            null_hi=float(np.percentile(null, 97.5)), p=float(p), n=int(Lh), j_in_triplet=in_trip))
            print(f"  {name:8s} lag {L:+d} (vertex j{L - 1:+d}; j {'IN' if in_trip else 'not in'} triplet)  "
                  f"beta={beta:+.4f}  null95=[{np.percentile(null, 2.5):+.4f},{np.percentile(null, 97.5):+.4f}]  p={p:.3f}",
                  flush=True)
        out["results"][name] = res

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, (name, col) in zip(axes, (("semantic", "#5B8FD4"), ("neural", "#E67E22"))):
        R = out["results"][name]
        x = np.array([r["lag"] for r in R])
        ax.axvspan(-0.5, 2.5, color="0.92", lw=0, label="word j is one of the 3 points in the angle")
        ax.fill_between(x, [r["null_lo"] for r in R], [r["null_hi"] for r in R], color="0.75", alpha=0.6, lw=0,
                        label="circular-shift null 95%")
        ax.plot(x, [r["beta"] for r in R], "-o", color=col, lw=2, ms=5, label="β for surprisal of word j")
        for r in R:
            if r["p"] < 0.05:
                ax.text(r["lag"], r["beta"], " *", color=col, fontsize=13, va="bottom", ha="left")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xlabel("Lag (+1 = angle at word j; ≥ +3 = vertices after j, not containing j)")
        ax.set_ylabel("Standardized β (curvature ~ surprisal of word j)")
        ttl = "A  Semantic (gpt2-xl layer 29, full space)" if name == "semantic" else \
              "B  Neural pseudopop (475 units, fixed 250 ms window)"
        ax.set_title(ttl, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Does word j's surprisal predict curvature at later vertices? Controls: surprisal, frequency and "
                 "duration of the 3 words forming each angle", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_lagged_surprisal_curvature_regression_pseudopop"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    json.dump(out, open(os.path.join(RESULTS, "lagged_surprisal_curvature_regression_pseudopop.json"), "w"), indent=1)
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
