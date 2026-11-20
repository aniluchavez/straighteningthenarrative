"""
make_semantic_curvature_surprisal_Ksweep_fasttext.py
======================================================
fastText (static wiki-news-300d) copy of
make_semantic_curvature_surprisal_Ksweep_gpt2xl_layer29.py. OOV words (zero
vector, 0.14%) -> curvature at angles touching them = NaN (excluded).

At how many PCs does semantic curvature start to increase with surprisal?
gpt2-xl layer 29, YEY transcript (semantic curvature is the same stimulus
for every patient). Standardized full-rank PCA once; for each K the
trajectory is the top-K PCs ("raw" = un-standardized 1600-d embedding).
Curvature AT word j (turning angle at j, = lag +1 in the event-triggered
plots), valid words.

  A  Spearman rho(surprisal, curvature at word) vs K, with 95% bootstrap CI
  B  high - low surprisal (top/bottom 25%) mean curvature at the word vs K
  C  curvature at word by surprisal decile for selected K

Saves: figures/fig_semantic_curvature_surprisal_Ksweep_gpt2xl_layer29.{pdf,png,svg}
       results/semantic_curvature_surprisal_Ksweep_gpt2xl_layer29.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
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

from run_geometry_analysis_pca20 import trajectory_curvature, build_valid_mask
from podcast_loader import load_podcast_patient

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

PID, MODEL, LAYER = "YEY", "fasttext-wiki", 0
KS = [2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 300]
SHOW = [8, 32, 64, 128, 300]
N_BINS, N_BOOT = 10, 500


def main():
    d = load_podcast_patient(PID, region="hippocampus", model_tag=MODEL, context_words=200)
    n = d["n_words"]
    surp = np.asarray(d["surp"][:n], dtype=float)
    valid = np.zeros(n, bool)
    v0 = build_valid_mask(n, d["surp"][:n], d["sent_pos"][:n], d["is_turn"][:n])
    valid[:min(n, len(v0))] = v0[:n]
    emb = np.array(np.load(d["emb_path"], mmap_mode="r")[LAYER, :n], dtype=np.float64)
    Y = PCA(svd_solver="full").fit_transform(StandardScaler().fit_transform(emb))

    zero = np.linalg.norm(emb, axis=1) == 0
    bad = np.r_[True, zero[:-2] | zero[1:-1] | zero[2:], True][:n]
    print(f"OOV rows: {zero.mean() * 100:.2f}%")

    def at_word(Z):
        return np.where(bad, np.nan, np.r_[np.nan, trajectory_curvature(Z), np.nan][:n])

    m = valid & np.isfinite(surp)
    c_raw = at_word(emb)
    m &= np.isfinite(c_raw)
    s = surp[m]
    hi, lo = s >= np.percentile(s, 75), s <= np.percentile(s, 25)
    edges = np.percentile(s, np.linspace(0, 100, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, N_BINS - 1)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(s), size=(N_BOOT, len(s)))

    res = {}
    for K in KS + ["raw"]:
        c = (c_raw if K == "raw" else at_word(Y[:, :K]))[m]
        rho = spearmanr(s, c)[0]
        # rank-based bootstrap CI
        rs, rc = np.argsort(np.argsort(s)), np.argsort(np.argsort(c))
        bs = np.array([np.corrcoef(rs[i], rc[i])[0, 1] for i in boot])
        res[str(K)] = dict(rho=float(rho), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                           hi_minus_lo=float(c[hi].mean() - c[lo].mean()),
                           deciles=[float(c[b == i].mean()) for i in range(N_BINS)])
        r = res[str(K)]
        print(f"  K={str(K):>4}  rho={r['rho']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  hi-lo={r['hi_minus_lo']:+.2f} deg  "
              f"dec1={r['deciles'][0]:.1f} dec10={r['deciles'][-1]:.1f}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.array(KS)
    ax = axes[0]
    ax.fill_between(x, [res[str(K)]["lo"] for K in KS], [res[str(K)]["hi"] for K in KS], color="#5B8FD4", alpha=0.25, lw=0)
    ax.plot(x, [res[str(K)]["rho"] for K in KS], "-o", color="#5B8FD4", ms=4)
    ax.axhline(res["raw"]["rho"], color="k", ls="--", lw=1, label=f"raw embedding (ρ={res['raw']['rho']:+.3f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Spearman ρ(surprisal, curvature at word)")
    ax.set_title("A  Curvature–surprisal correlation vs. PCs kept", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    ax.plot(x, [res[str(K)]["hi_minus_lo"] for K in KS], "-o", color="#C0392B", ms=4)
    ax.axhline(res["raw"]["hi_minus_lo"], color="k", ls="--", lw=1, label="raw embedding")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("High − low surprisal curvature (°)")
    ax.set_title("B  Top vs. bottom 25% surprisal, at the word", loc="left")
    ax.legend(frameon=False, loc="upper left")
    for a in axes[:2]:
        a.set_xscale("log", base=2)
        a.set_xticks([2, 8, 32, 128, 300])
        a.set_xticklabels([2, 8, 32, 128, 300])
        a.set_xlabel("Semantic PCs kept (K)")
    ax = axes[2]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(SHOW)))
    for col, K in zip(cmap, SHOW):
        dd = np.array(res[str(K)]["deciles"])
        ax.plot(np.arange(1, N_BINS + 1), dd - dd.mean(), "-o", color=col, ms=3.5, label=f"K = {K}")
    dd = np.array(res["raw"]["deciles"])
    ax.plot(np.arange(1, N_BINS + 1), dd - dd.mean(), "--", color="k", lw=1.2, label="raw")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(1, N_BINS + 1))
    ax.set_xlabel("Surprisal decile (1 = least surprising)")
    ax.set_ylabel("Curvature at word, centred (°)")
    ax.set_title("C  Curvature by surprisal decile", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Semantic curvature vs. surprisal as PCs are added — fastText (static, 300-d) ({PID} transcript, "
                 f"{int(m.sum())} valid words)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_semantic_curvature_surprisal_Ksweep_fasttext"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    json.dump(res, open(os.path.join(RESULTS, "semantic_curvature_surprisal_Ksweep_fasttext.json"), "w"), indent=1)
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
