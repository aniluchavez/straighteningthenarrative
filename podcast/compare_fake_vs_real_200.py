"""
compare_fake_vs_real_200.py
=============================
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

Saves: results/compare_fake_vs_real_200.json
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
sys.argv = sys.argv[:1] + ["pseudopop", "dur8040", "full", "four"]
import compute_poisson_normalized_geometry as G   # noqa: E402
from podcast_loader import load_podcast_patient   # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(BASE_DIR, "results"), os.path.join(BASE_DIR, "figures")
N_NEUR, N_SEED = 200, 10
KS = [8, 16, 32, 64, 128, None]
METRICS = [("curv", "Curvature"), ("disp", "Displacement"), ("pos", "Position |r|")]
SETS = [("real", "200 real neurons", "#C0392B", "-"),
        ("fake", "200 fake LLM neurons (Poisson)", "#5B8FD4", "-"),
        ("fake_nf", "200 fake LLM neurons, noise-free", "#5B8FD4", "--")]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def main():
    if _ARGS[:1] == ["plot"]:   # re-plot from saved JSON
        J = json.load(open(os.path.join(RESULTS, "compare_fake_vs_real_200.json")))
        R = {s_: {m: np.array(v) for m, v in dv.items()} for s_, dv in J["results"].items()}
        return plot(R, J.get("n_words", 7346))
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
    for seed in range(N_SEED):
        rng = np.random.default_rng(seed)
        units = rng.choice(live, N_NEUR, replace=False)
        real = Kreal[:, units]
        c = real.mean(axis=0)
        dims = rng.choice(emb.shape[1], N_NEUR, replace=False)
        A = emb[:, dims]
        E = A - A.min(0)
        lam = E * (c / E.mean(0))
        fake = rng.poisson(lam).astype(float)
        for ki, K in enumerate(KS):
            G.KPC = K
            for s, Y in (("real", real), ("fake", fake), ("fake_nf", lam)):
                f = G.euclid(Y, False)
                for m, _ in METRICS:
                    r = G.chunk_r(G.masked(f[m], valid), G.masked(sem[K][m], valid))
                    R[s][m][seed, ki] = abs(r) if m == "pos" else r
    for s, lab, *_ in SETS:
        print(f"{lab:36s} " + "  ".join(
            f"{'full' if K is None else K}: c {R[s]['curv'][:, i].mean():+.3f} d {R[s]['disp'][:, i].mean():+.3f} "
            f"|p| {R[s]['pos'][:, i].mean():.3f}" for i, K in enumerate(KS)), flush=True)
    json.dump(dict(Ks=["full" if K is None else K for K in KS], n_neurons=N_NEUR, n_seed=N_SEED, n_words=int(n),
                   results={s: {m: v.tolist() for m, v in dv.items()} for s, dv in R.items()}),
              open(os.path.join(RESULTS, "compare_fake_vs_real_200.json"), "w"))
    plot(R, n)


def plot(R, n):
    x = np.arange(len(KS))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    for ax, (m, mlab), letter in zip(axes, METRICS, "ABC"):
        for s, lab, col, ls in SETS:
            M = R[s][m]
            ax.errorbar(x, M.mean(0), yerr=M.std(0, ddof=1), color=col, ls=ls, marker="o" if ls == "-" else None,
                        ms=4, lw=1.8 if ls == "-" else 1.3, capsize=3, label=lab)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"top {K}" if K else f"full ({N_NEUR})" for K in KS])
        ax.set_xlabel("Dimensions (top-K PCs, or full space)")
        ax.set_title(f"{letter}  {mlab}", loc="left")
    axes[0].set_ylabel("r with clean gpt2-xl layer 29")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=10)
    fig.suptitle(f"200 real vs 200 fake LLM neurons (rate = c × (activation − min) → Poisson; c = matched real unit "
                 f"mean); raw counts, [onset+80, offset+40]; {n} words, chunk 3; mean ± SD over {N_SEED} subsamples",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    stem = os.path.join(FIGURES, "fig_compare_fake_vs_real_200")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}")
    print(f"  -> {stem}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
