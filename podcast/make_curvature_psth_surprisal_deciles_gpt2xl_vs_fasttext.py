"""
make_curvature_psth_surprisal_deciles_gpt2xl_vs_fasttext.py
=============================================================
Event-triggered semantic curvature, most- vs least-surprising decile
(top / bottom 10% surprisal over valid words), gpt2-xl layer 29 (contextual)
vs. fastText wiki-news-300d (static). Both reduced to the top K = 100
standardized PCs (argv[1] to change K).

Words: pseudopop common word set (YEY transcript), valid mask as elsewhere.
Lag convention as the other curvature PSTHs: lag k = turning angle at word
j+k-1, so lag +1 = angle AT the event word, lag 0 / +2 = angles that include
the event word as an end point, lag >= +3 = angles not containing it.
fastText OOV words (zero vector) -> angles touching them = NaN.

Bands = SEM across events. Panel C: high - low difference, 95% CI
(Welch), * = Welch p < 0.05.

Saves: figures/fig_curvature_psth_surprisal_deciles_gpt2xl_vs_fasttext_{pc{K}|full}.{pdf,png,svg}
       results/curvature_psth_surprisal_deciles_gpt2xl_vs_fasttext_{pc{K}|full}.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES, RESULTS = os.path.join(BASE_DIR, "figures"), os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)
ARG = sys.argv[1] if len(sys.argv) > 1 else "100"
K = None if ARG == "full" else int(ARG)   # "full" = raw embedding, no PCA
KTAG = "full" if K is None else f"pc{K}"
sys.argv = sys.argv[:1]

from run_geometry_analysis_pca20 import trajectory_curvature, build_valid_mask   # noqa: E402
from podcast_loader import load_podcast_patient   # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
LAGS = np.arange(-3, 8)
HI, LO = "#C0392B", "#5B8FD4"
MODELS = [("gpt2xl", "gpt2-xl layer 29 (contextual)", "#5B8FD4"),
          ("fasttext", "fastText (static, 300-d)", "#2E9E6C")]


def curv_at_word(E, k):
    zero = np.linalg.norm(E, axis=1) == 0
    Y = E if k is None else PCA(n_components=k, svd_solver="full").fit_transform(StandardScaler().fit_transform(E))
    c = np.r_[np.nan, trajectory_curvature(Y), np.nan]
    bad = np.r_[True, zero[:-2] | zero[1:-1] | zero[2:], True]
    return np.where(bad, np.nan, c)


def main():
    common = np.load(os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz"))["common"]
    n = len(common)
    g = load_podcast_patient("YEY", region="hippocampus", model_tag="gpt2-xl", context_words=200)
    f = load_podcast_patient("YEY", region="hippocampus", model_tag="fasttext-wiki", context_words=200)
    valid = build_valid_mask(n, g["surp"][common], g["sent_pos"][common], g["is_turn"][common])
    vfull = np.zeros(n, bool); vfull[:len(valid)] = valid
    surp = np.asarray(g["surp"], float)[common]
    E = dict(gpt2xl=np.array(np.load(g["emb_path"], mmap_mode="r")[29, common], float),
             fasttext=np.array(np.load(f["emb_path"], mmap_mode="r")[0, common], float))
    curv = {m: curv_at_word(E[m], K) for m in E}

    ev = vfull & np.isfinite(surp)
    hi_t, lo_t = np.percentile(surp[ev], [90, 10])
    groups = dict(hi=np.where(ev & (surp >= hi_t))[0], lo=np.where(ev & (surp <= lo_t))[0])
    print(f"K={KTAG}  events: hi={len(groups['hi'])}  lo={len(groups['lo'])}")

    out = dict(K=K, lags=LAGS.tolist(), n_hi=int(len(groups["hi"])), n_lo=int(len(groups["lo"])), models={})
    for m in curv:
        c = curv[m]
        rows = {}
        for gname, idx in groups.items():
            M = np.full((len(idx), len(LAGS)), np.nan)
            for li, L in enumerate(LAGS):
                t = idx + L - 1
                ok = (t >= 0) & (t < n)
                M[ok, li] = c[t[ok]]
            rows[gname] = M
        res = dict(hi_mean=[], hi_sem=[], lo_mean=[], lo_sem=[], diff=[], diff_lo=[], diff_hi=[], p=[])
        for li in range(len(LAGS)):
            a = rows["hi"][:, li]; a = a[np.isfinite(a)]
            b = rows["lo"][:, li]; b = b[np.isfinite(b)]
            sa, sb = a.std(ddof=1) / np.sqrt(len(a)), b.std(ddof=1) / np.sqrt(len(b))
            d, sd = a.mean() - b.mean(), np.sqrt(sa ** 2 + sb ** 2)
            res["hi_mean"].append(float(a.mean())); res["hi_sem"].append(float(sa))
            res["lo_mean"].append(float(b.mean())); res["lo_sem"].append(float(sb))
            res["diff"].append(float(d)); res["diff_lo"].append(float(d - 1.96 * sd)); res["diff_hi"].append(float(d + 1.96 * sd))
            res["p"].append(float(ttest_ind(a, b, equal_var=False)[1]))
        out["models"][m] = res
        print(f"  {m:8s} hi-lo (deg): " + "  ".join(
            f"{L:+d}:{dd:+.2f}{'*' if p < 0.05 else ' '}" for L, dd, p in zip(LAGS, res["diff"], res["p"])), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (m, lab, _) in zip(axes[:2], MODELS):
        r = out["models"][m]
        for key, col, nm in (("hi", HI, f"most surprising 10% (n={out['n_hi']})"),
                             ("lo", LO, f"least surprising 10% (n={out['n_lo']})")):
            mu, se = np.array(r[f"{key}_mean"]), np.array(r[f"{key}_sem"])
            ax.fill_between(LAGS, mu - se, mu + se, color=col, alpha=0.2, lw=0)
            ax.plot(LAGS, mu, "-o", color=col, ms=4, lw=1.8, label=nm)
        ax.axvline(1, color="k", lw=0.6, ls=":")
        ax.set_ylabel("Curvature (°)")
        ax.set_title(f"{'A' if m == 'gpt2xl' else 'B'}  {lab}", loc="left")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax = axes[2]
    for m, lab, col in MODELS:
        r = out["models"][m]
        d, p = np.array(r["diff"]), np.array(r["p"])
        ax.fill_between(LAGS, r["diff_lo"], r["diff_hi"], color=col, alpha=0.2, lw=0)
        ax.plot(LAGS, d, color=col, lw=1.8, label=lab)
        ax.scatter(LAGS[p < 0.05], d[p < 0.05], color=col, s=28, zorder=3)
        ax.scatter(LAGS[p >= 0.05], d[p >= 0.05], facecolor="white", edgecolor=col, s=28, lw=1.4, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(1, color="k", lw=0.6, ls=":")
    ax.set_ylabel("Most − least surprising curvature (°)")
    ax.set_title("C  Difference (95% CI; filled = Welch p < 0.05)", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    for a in axes:
        a.set_xticks(LAGS)
        a.set_xlabel("Words relative to event (lag +1 = angle at the event word)")
    fig.suptitle(f"Event-triggered semantic curvature, most vs. least surprising decile — {'full space (raw embedding)' if K is None else f'top {K} standardized PCs'} "
                 f"(podcast, {int(ev.sum())} valid words; bands = SEM across events)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    stem = f"fig_curvature_psth_surprisal_deciles_gpt2xl_vs_fasttext_{KTAG}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    json.dump(out, open(os.path.join(RESULTS, f"curvature_psth_surprisal_deciles_gpt2xl_vs_fasttext_{KTAG}.json"), "w"), indent=1)
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
