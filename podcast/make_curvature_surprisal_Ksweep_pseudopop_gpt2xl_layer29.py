"""
make_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29.py
==============================================================
Does curvature increase with surprisal, and at how many PCs? Neural
pseudopopulation (475 units, 13 patients, duration window [onset+80,
offset+40] raw spike counts, z-scored; YFX clock-fixed;
results/pseudopop_full_sequence_dur8040_count.npz) alongside semantic
(gpt2-xl layer 29), on the words common to all patients. Surprisal, word
duration from YEY. Curvature AT word j (turning angle at j), valid words.

For each K (standardized full-rank PCA, top K):
  A  Spearman rho(surprisal, curvature at word), 95% bootstrap CI
  B  same, partial: log word duration regressed out of both ranks
  C  curvature at word by surprisal decile, full space (semantic raw
     embedding; neural all 475 units), centred on each curve's mean

argv[1] = "fixed250": neural = fixed [onset+80, onset+330] window counts
(results/pseudopop_full_sequence_fixed250_count.npz) instead of the
duration window.

Saves: figures/fig_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29{,_fixed250}.{pdf,png,svg}
       results/curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29{,_fixed250}.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import trajectory_curvature
from podcast_loader import load_podcast_patient, load_words_onset_offset
import compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop as PP

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

PP.RATE = False
FIXED = len(sys.argv) > 1 and sys.argv[1] == "fixed250"
SUF = "_fixed250" if FIXED else ""
if FIXED:
    PP.ONSET_PAD_MS, PP.OFFSET_PAD_MS = 80, 330
    _lwoo = PP.load_words_onset_offset
    PP.load_words_onset_offset = lambda pid: (lambda w, on, off: (w, on, on))(*_lwoo(pid))
    PP.CACHE_PATH = os.path.join(RESULTS, "pseudopop_full_sequence_fixed250_count.npz")
else:
    PP.CACHE_PATH = os.path.join(RESULTS, "pseudopop_full_sequence_dur8040_count.npz")
MODEL, LAYER = "gpt2-xl", 29
KS = [2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 475, 768, 1024, 1600]
N_BINS, N_BOOT = 10, 500
COL = dict(sem="#5B8FD4", neu="#E67E22")


def resid(y, x):
    X = np.c_[np.ones_like(x), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    neu_z, common, valid = PP.build_pseudopop()
    n = len(common)
    d = load_podcast_patient(PP.REF_PID, region="hippocampus", model_tag=MODEL, context_words=200)
    surp = np.asarray(d["surp"], dtype=float)[common]
    _, on, off = load_words_onset_offset(PP.REF_PID)
    dur = (off - on)[common]
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
    emb = np.array(np.load(d["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    pcs = dict(sem=PCA(svd_solver="full").fit_transform(StandardScaler().fit_transform(emb)),
               neu=PCA(svd_solver="full").fit_transform(np.nan_to_num(neu_z)))
    full = dict(sem=emb, neu=np.nan_to_num(neu_z))

    def at_word(Z):
        return np.r_[np.nan, trajectory_curvature(Z), np.nan][:n]

    vfull = np.zeros(n, bool)
    vfull[:len(valid)] = valid[:n]
    m = vfull & np.isfinite(surp) & np.isfinite(at_word(emb))
    s, ld = surp[m], np.log(dur[m])
    rs, rld = rankdata(s), rankdata(ld)
    rs_p = resid(rs, rld)
    edges = np.percentile(s, np.linspace(0, 100, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, N_BINS - 1)
    boot = np.random.default_rng(0).integers(0, len(s), size=(N_BOOT, len(s)))
    print(f"words={n} valid={int(m.sum())}  rho(surp, log dur)={np.corrcoef(rs, rld)[0, 1]:+.3f}")

    res = {}
    for side in ("sem", "neu"):
        res[side] = {}
        for K in KS + ["full"]:
            if K != "full" and K > pcs[side].shape[1]:
                continue
            c = at_word(full[side] if K == "full" else pcs[side][:, :K])[m]
            rc = rankdata(c)
            rho = np.corrcoef(rs, rc)[0, 1]
            bs = np.array([np.corrcoef(rs[i], rc[i])[0, 1] for i in boot])
            rho_p = np.corrcoef(rs_p, resid(rc, rld))[0, 1]
            res[side][str(K)] = dict(rho=float(rho), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                     rho_partial=float(rho_p), deciles=[float(c[b == i].mean()) for i in range(N_BINS)])
            r = res[side][str(K)]
            print(f"  {side} K={str(K):>4}  rho={r['rho']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  "
                  f"partial(dur)={r['rho_partial']:+.3f}  dec1={r['deciles'][0]:.1f} dec10={r['deciles'][-1]:.1f}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for side, lab in (("sem", f"Semantic ({MODEL} L{LAYER})"), ("neu", f"Neural pseudopop ({neu_z.shape[1]} units)")):
        ks = [K for K in KS if str(K) in res[side]]
        x = np.array(ks)
        R = res[side]
        axes[0].fill_between(x, [R[str(K)]["lo"] for K in ks], [R[str(K)]["hi"] for K in ks], color=COL[side], alpha=0.2, lw=0)
        axes[0].plot(x, [R[str(K)]["rho"] for K in ks], "-o", color=COL[side], ms=4, label=lab)
        axes[1].plot(x, [R[str(K)]["rho_partial"] for K in ks], "-o", color=COL[side], ms=4, label=lab)
        for a, key in ((axes[0], "rho"), (axes[1], "rho_partial")):
            a.axhline(R["full"][key], color=COL[side], ls="--", lw=1)
        dd = np.array(R["full"]["deciles"])
        axes[2].plot(np.arange(1, N_BINS + 1), dd, "-o", color=COL[side], ms=4,
                     label=f"{lab}, full space")
    for a, t in ((axes[0], "A  Spearman ρ(surprisal, curvature at word)"),
                 (axes[1], "B  Same, log word duration regressed out")):
        a.axhline(0, color="k", lw=0.5)
        a.set_xscale("log", base=2)
        a.set_xticks([2, 8, 32, 128, 475, 1600])
        a.set_xticklabels([2, 8, 32, 128, 475, 1600])
        a.set_xlabel("PCs kept (K); dashed = full space")
        a.set_ylabel("Spearman ρ")
        a.set_title(t, loc="left")
    axes[0].legend(frameon=False, loc="lower right")
    axes[2].set_xticks(np.arange(1, N_BINS + 1))
    axes[2].set_xlabel("Surprisal decile (1 = least surprising)")
    axes[2].set_ylabel("Curvature at the word (°)")
    axes[2].set_title("C  Curvature by surprisal decile, full space", loc="left")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Does curvature increase with surprisal? Pseudopopulation ({'fixed 250 ms' if FIXED else 'duration'}-window spike counts) and "
                 f"{MODEL} layer {LAYER}, {int(m.sum())} valid common words", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29" + SUF
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    json.dump(res, open(os.path.join(RESULTS, f"curvature_surprisal_Ksweep_pseudopop_gpt2xl_layer29{SUF}.json"), "w"), indent=1)
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
