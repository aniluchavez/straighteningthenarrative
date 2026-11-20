"""
compute_static_embedding_curvature_surprisal.py
=================================================
Curvature vs. surprisal for a STATIC embedding (fastText wiki-news-300d-1M,
one vector per word type, no context) next to gpt2-xl layer 29.

Words: the pseudopop word set (common to all 13 patients, YEY transcript),
valid mask as in the other pseudopop analyses. OOV words (zero fastText
vector) -> curvature at vertices touching them = NaN.

Geometry (both embeddings): raw Euclidean, and L2-normalized with tangent
curvature (sphere_tan), each at top-K PCs (8..128) and full space -- same
functions as compute_poisson_normalized_geometry.py.

Analyses (circular-shift p, 1000 shifts):
  (b) unique R^2 of surprisal / log SUBTLEX freq / log duration for curvature
  (c) lagged regression: beta of surprisal(j) on curvature at vertex j+L-1,
      controls = surprisal / freq / duration of the 3 triplet words

Saves: results/static_embedding_curvature_surprisal.json
       figures/fig_static_embedding_curvature_surprisal.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = sys.argv[:1] + ["pseudopop", "dur8040", "full"]
import compute_poisson_normalized_geometry as G   # noqa: E402
from podcast_loader import load_podcast_patient, load_words_onset_offset   # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(BASE_DIR, "results"), os.path.join(BASE_DIR, "figures")
KS = [8, 16, 32, 64, 128, None]
N_SHIFT = 1000
PRED = ["surprisal", "log_freq", "log_dur"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def curv_series(E, K, kind, zero):
    G.KPC = K
    if kind == "raw":
        c = G.euclid(E, True)["curv"]
    else:
        nb = np.linalg.norm(E, axis=1, keepdims=True)
        X = E / np.where(nb > 0, nb, 1.0)
        X[zero] = 1.0 / np.sqrt(E.shape[1])
        c = G.sphere_tan(X, [np.arange(E.shape[1])])["curv"]
    c = np.where(zero[:-2] | zero[1:-1] | zero[2:], np.nan, c)
    return np.r_[np.nan, c, np.nan]                       # angle AT word i


def partition(y0, surp, lf, ld, vfull, rng):
    m = vfull & np.isfinite(surp) & np.isfinite(lf) & np.isfinite(y0)
    X = StandardScaler().fit_transform(np.c_[surp[m], lf[m], ld[m]])
    y = G.zs(y0[m])
    L = len(y)
    Ysh = np.stack([np.roll(y, s) for s in rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)], axis=1)
    R2f, b = G.r2(X, y)
    nullf = G.r2_many(X, Ysh)
    res = dict(n=int(L), R2_full=float(R2f), preds={})
    for j, pn in enumerate(PRED):
        keep = [k for k in range(3) if k != j]
        u = R2f - G.r2(X[:, keep], y)[0]
        un = nullf - G.r2_many(X[:, keep], Ysh)
        res["preds"][pn] = dict(beta=float(b[j]), r=float(np.corrcoef(X[:, j], y)[0, 1]), unique_R2=float(u),
                                p_unique=float((np.sum(un >= u) + 1) / (N_SHIFT + 1)))
    return res


def lagged(c, surp, lf, ld, vfull, rng, n):
    rows = []
    for L_ in G.LAGS:
        v = np.arange(n) + L_ - 1
        ok = vfull & (v - 1 >= 0) & (v + 1 < n)
        j = np.where(ok)[0]
        vv = v[j]
        colz = [surp[j]]
        for o in (-1, 0, 1):
            t = vv + o
            for arr, nm in ((surp, "surp"), (lf, "freq"), (ld, "dur")):
                if nm == "surp" and (t == j).all():
                    continue
                colz.append(arr[t])
        X = np.column_stack(colz)
        y = c[vv]
        f = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X = np.column_stack([G.zs(col) for col in X[f].T])
        y = G.zs(y[f])
        X1 = np.c_[np.ones(len(y)), X]
        beta = np.linalg.lstsq(X1, y, rcond=None)[0][1]
        Lh = len(y)
        null = np.empty(N_SHIFT)
        for i, s in enumerate(rng.integers(int(0.1 * Lh), Lh - int(0.1 * Lh), size=N_SHIFT)):
            Xs = X1.copy(); Xs[:, 1] = np.roll(X[:, 0], s)
            null[i] = np.linalg.lstsq(Xs, y, rcond=None)[0][1]
        rows.append(dict(lag=L_, beta=float(beta), null_lo=float(np.percentile(null, 2.5)),
                         null_hi=float(np.percentile(null, 97.5)),
                         p=float((np.sum(np.abs(null) >= abs(beta)) + 1) / (N_SHIFT + 1)), n=int(Lh)))
    return rows


def main():
    rng = np.random.default_rng(0)
    common = np.load(os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz"))["common"]
    n = len(common)
    ref = load_podcast_patient("YEY", region="hippocampus", model_tag="gpt2-xl", context_words=200)
    reff = load_podcast_patient("YEY", region="hippocampus", model_tag="fasttext-wiki", context_words=200)
    valid = G.build_valid_mask(n, ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    vfull = np.zeros(n, bool); vfull[:len(valid)] = valid
    surp = np.asarray(ref["surp"], float)[common]
    lf = G.subtlex_log_freq(np.asarray(ref["words"])[common])
    _, on, off = load_words_onset_offset("YEY")
    dur = (off - on)[common]
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
    ld = np.log(dur)
    embs = {"fasttext": np.array(np.load(reff["emb_path"], mmap_mode="r")[0, common], float),
            "gpt2xl_L29": np.array(np.load(ref["emb_path"], mmap_mode="r")[29, common], float)}

    out = dict(n_words=int(n), partition={}, lagged={})
    print(f"words={n}  valid={int(vfull.sum())}")
    for en, E in embs.items():
        zero = np.linalg.norm(E, axis=1) == 0
        print(f"\n== {en}: dim={E.shape[1]}  zero/OOV rows={zero.mean() * 100:.2f}%")
        for kind in ("raw", "sphere_tan"):
            for K in KS:
                key = f"{en}|{kind}|{'full' if K is None else f'pc{K}'}"
                c = curv_series(E, K, kind, zero)
                res = partition(c, surp, lf, ld, vfull, rng)
                out["partition"][key] = res
                q = res["preds"]
                print(f"  {kind:10s} K={'full' if K is None else K:>4}  n={res['n']}  R2={res['R2_full'] * 100:.2f}%  " +
                      "  ".join(f"{p}: b={q[p]['beta']:+.3f} uR2={q[p]['unique_R2'] * 100:.3f}% p={q[p]['p_unique']:.3f}"
                                for p in PRED), flush=True)
                if K is None:
                    rows = lagged(c, surp, lf, ld, vfull, rng, n)
                    out["lagged"][f"{en}|{kind}"] = rows
                    print("     lagged: " + " ".join(f"{r['lag']:+d}:{r['beta']:+.3f}{'*' if r['p'] < 0.05 else ' '}"
                                                     for r in rows), flush=True)
    json.dump(out, open(os.path.join(RESULTS, "static_embedding_curvature_surprisal.json"), "w"), indent=1)

    # ---------------- figure
    COL = {"fasttext": "#2E9E6C", "gpt2xl_L29": "#5B8FD4"}
    LAB = {"fasttext": "fastText (static, 300-d)", "gpt2xl_L29": "gpt2-xl layer 29 (contextual)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ksl = ["top 8", "top 16", "top 32", "top 64", "top 128", "full"]
    x = np.arange(len(KS))
    ax = axes[0]
    for en in embs:
        for kind, ls in (("raw", "-"), ("sphere_tan", "--")):
            R = [out["partition"][f"{en}|{kind}|{'full' if K is None else f'pc{K}'}"]["preds"]["surprisal"] for K in KS]
            u = np.array([r["unique_R2"] for r in R]) * 100
            p = np.array([r["p_unique"] for r in R])
            ax.plot(x, u, ls, color=COL[en], lw=1.8, label=f"{LAB[en]}, {'Euclidean' if kind == 'raw' else 'L2 + tangent'}")
            ax.scatter(x[p < 0.05], u[p < 0.05], color=COL[en], s=28, zorder=3)
            ax.scatter(x[p >= 0.05], u[p >= 0.05], facecolor="white", edgecolor=COL[en], s=28, lw=1.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(ksl)
    ax.set_xlabel("Dimensions (top-K PCs, or full space)")
    ax.set_ylabel("Unique R² of surprisal for curvature (%)")
    ax.set_title("A  Surprisal → curvature at the same word\n(controls: log freq, log duration)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax, kind, t in ((axes[1], "raw", "B  Lagged, Euclidean (full space)"),
                        (axes[2], "sphere_tan", "C  Lagged, L2 + tangent (full space)")):
        ax.axvspan(-0.5, 2.5, color="0.93", lw=0)
        for en in embs:
            R = out["lagged"][f"{en}|{kind}"]
            lx = np.array([r["lag"] for r in R]); b = np.array([r["beta"] for r in R])
            p = np.array([r["p"] for r in R])
            ax.fill_between(lx, [r["null_lo"] for r in R], [r["null_hi"] for r in R], color=COL[en], alpha=0.12, lw=0)
            ax.plot(lx, b, color=COL[en], lw=1.8, label=LAB[en])
            ax.scatter(lx[p < 0.05], b[p < 0.05], color=COL[en], s=28, zorder=3)
            ax.scatter(lx[p >= 0.05], b[p >= 0.05], facecolor="white", edgecolor=COL[en], s=28, lw=1.4, zorder=3)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(G.LAGS)
        ax.set_xlabel("Lag (+1 = angle at word j; shaded = j is one of the 3 points)")
        ax.set_ylabel("Standardized β, surprisal of word j")
        ax.set_title(t, loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Static vs. contextual embedding: curvature ~ surprisal (podcast, {int(vfull.sum())} valid words); "
                 "filled = circular-shift p < 0.05, open = n.s., band = null 95%", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    stem = "fig_static_embedding_curvature_surprisal"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
