"""
compare_fake_vs_real_200_direct.py
====================================
Fake neurons compared DIRECTLY with real neurons: r between the fake neurons'
geometry and the real neurons' geometry (both raw counts, same top-K / full),
instead of each vs the clean LLM. Reference line: real vs clean LLM.
Fake / real construction identical to compare_fake_vs_real_200.py (below).

Original description:
200 fake LLM neurons vs 200 real neurons, same pipeline, same clean target.

Per seed (N_SEED):
  real : 200 units drawn at random from the 12-patient pseudopop (436 units,
         YEX excluded), raw [onset+80, offset+40] spike counts
  fake : 200 random gpt2-xl layer 29 dims; rate = c * (activation - min),
         c = mean count of the matching sampled real unit; counts ~ Poisson
  fake noise-free : the same rates without Poisson draws
Each -> raw-count geometry (centered PCA top-K, or full 200-d) -> chunk(3) r
with the CLEAN gpt2-xl embedding processed as in all real analyses
(standardized top-K PCs, raw full). Position as |r| (PC1 sign arbitrary).

Saves: results/compare_fake_vs_real_200_direct.json
       figures/fig_compare_fake_vs_real_200.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ARGS = sys.argv[1:]
# "noscale" anywhere in args: lambda = activation - min (no scaling to real mean counts)
NOSCALE = "noscale" in _ARGS
TAG = "_noscale" if NOSCALE else ""
sys.argv = sys.argv[:1] + ["pseudopop", "dur8040", "full", "four"]
import compute_poisson_normalized_geometry as G   # noqa: E402
from podcast_loader import load_podcast_patient   # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(BASE_DIR, "results"), os.path.join(BASE_DIR, "figures")
N_NEUR, N_SEED = 200, 10
N_SHIFT = 1000   # circular shifts for significance of the mean r over subsamples
KS = [8, 16, 32, 64, 128, None]
METRICS = [("curv", "Curvature"), ("disp", "Displacement"), ("pos", "Position |r|")]
SETS = [("fake_real", "fake LLM neurons (Poisson) vs real neurons", "#8E44AD", "-"),
        ("real_llm", "real neurons vs clean LLM (reference)", "#C0392B", "-")]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def main():
    if _ARGS[:1] == ["plot"]:   # re-plot from saved JSON
        J = json.load(open(os.path.join(RESULTS, f"compare_fake_vs_real_200_direct{TAG}.json")))
        R = {s_: {m: np.array(v) for m, v in dv.items()} for s_, dv in J["results"].items()}
        P = {s_: {m: np.array(v) for m, v in dv.items()} for s_, dv in J["p"].items()} if "p" in J else None
        return plot(R, J.get("n_words", 7346), P)
    d = np.load(os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz"))
    assert "YEX" not in list(d["patients"])
    Kreal, common = d["K"], d["common"]
    n = len(common)
    ref = load_podcast_patient("YEY", region="hippocampus", model_tag="gpt2-xl", context_words=200)
    valid = G.build_valid_mask(n, ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    emb = np.array(np.load(ref["emb_path"], mmap_mode="r")[29, common], float)
    sem = {}
    for K in KS:
        G.KPC = K
        sem[K] = G.semantic_variants(emb)["raw"]
    live = np.where(np.nanmean(Kreal, axis=0) > 0)[0]

    R = {s: {m: np.zeros((N_SEED, len(KS))) for m, _ in METRICS} for s, *_ in SETS}
    NULL = {s: {m: np.zeros((N_SEED, len(KS), N_SHIFT)) for m, _ in METRICS} for s, *_ in SETS}
    shift_frac = np.random.default_rng(12345).uniform(0.1, 0.9, N_SHIFT)   # same shifts for every subsample
    for seed in range(N_SEED):
        rng = np.random.default_rng(seed)
        units = rng.choice(live, N_NEUR, replace=False)
        real = Kreal[:, units]
        c = real.mean(axis=0)
        dims = rng.choice(emb.shape[1], N_NEUR, replace=False)
        A = emb[:, dims]
        E = A - A.min(0)
        lam = E if NOSCALE else E * (c / E.mean(0))
        fake = rng.poisson(lam).astype(float)
        for ki, K in enumerate(KS):
            G.KPC = K
            fr, ff, fn = G.euclid(real, False), G.euclid(fake, False), G.euclid(lam, False)
            for s, a, b in (("fake_real", ff, fr), ("real_llm", fr, sem[K])):
                for m, _ in METRICS:
                    r, xa, ya = G.chunk_r(G.masked(a[m], valid), G.masked(b[m], valid), return_chunks=True)
                    R[s][m][seed, ki] = abs(r) if m == "pos" else r
                    xz, yz = (xa - xa.mean()) / xa.std(), (ya - ya.mean()) / ya.std()
                    L = len(xz)
                    nr = np.array([np.mean(np.roll(xz, int(f * L)) * yz) for f in shift_frac])
                    NULL[s][m][seed, ki] = np.abs(nr) if m == "pos" else nr
    for s, lab, *_ in SETS:
        print(f"{lab:36s} " + "  ".join(
            f"{'full' if K is None else K}: c {R[s]['curv'][:, i].mean():+.3f} d {R[s]['disp'][:, i].mean():+.3f} "
            f"|p| {R[s]['pos'][:, i].mean():.3f}" for i, K in enumerate(KS)), flush=True)
    # p for the mean r over subsamples: null mean = mean over subsamples of the same shift
    P = {}
    for s, *_ in SETS:
        P[s] = {}
        for m, _ in METRICS:
            obs = R[s][m].mean(0)
            nm = NULL[s][m].mean(0)                                       # (K, N_SHIFT)
            if m == "pos":
                P[s][m] = ((nm >= obs[:, None]).sum(1) + 1) / (N_SHIFT + 1)
            else:
                P[s][m] = ((np.abs(nm) >= np.abs(obs)[:, None]).sum(1) + 1) / (N_SHIFT + 1)
        print(f"  p {s:10s} " + "  ".join(f"{m}:" + ",".join(f"{p:.3f}" for p in P[s][m]) for m, _ in METRICS))
    json.dump(dict(Ks=["full" if K is None else K for K in KS], n_neurons=N_NEUR, n_seed=N_SEED, n_words=int(n),
                   n_shift=N_SHIFT, results={s: {m: v.tolist() for m, v in dv.items()} for s, dv in R.items()},
                   p={s: {m: v.tolist() for m, v in dv.items()} for s, dv in P.items()}),
              open(os.path.join(RESULTS, f"compare_fake_vs_real_200_direct{TAG}.json"), "w"))
    plot(R, n, P)


def plot(R, n, P=None):
    x = np.arange(len(KS))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    for ax, (m, mlab), letter in zip(axes, METRICS, "ABC"):
        for s, lab, col, ls in SETS:
            M = R[s][m]
            mu = M.mean(0)
            ax.errorbar(x, mu, yerr=M.std(0, ddof=1), color=col, ls=ls, marker=None,
                        lw=1.8 if ls == "-" else 1.3, capsize=3, label=lab)
            if P is not None:
                sig = np.asarray(P[s][m]) < 0.05
                ax.scatter(x[sig], mu[sig], color=col, s=34, zorder=3)
                ax.scatter(x[~sig], mu[~sig], facecolor="white", edgecolor=col, s=34, lw=1.4, zorder=3)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"top {K}" if K else f"full ({N_NEUR})" for K in KS])
        ax.set_xlabel("Dimensions (top-K PCs, or full space)")
        ax.set_title(f"{letter}  {mlab}", loc="left")
    axes[0].set_ylabel("r (geometry vs geometry)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=10)
    fig.suptitle("Fake LLM neurons vs real neurons, compared directly (200 each)\n"
                 + ("Fake: rate = activation − min (NO scaling) → Poisson. " if NOSCALE else
                    "Fake: rate = c × (activation − min) → Poisson, c = matched real unit mean. ") +
                 f"Raw counts, [onset+80, offset+40]; {n} words, chunk 3\n"
                 f"Mean ± SD over {N_SEED} subsamples; filled = p < 0.05 (circular shift), open = n.s.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    stem = os.path.join(FIGURES, f"fig_compare_fake_vs_real_200_direct{TAG}")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}")
    print(f"  -> {stem}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
