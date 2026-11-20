"""
simulate_poisson_llm_shift100.py
==================================
SIMPLE version (no beta): rate = c_d * (activation_dt - min_t activation_dt),
c_d sets the mean to a real unit's mean count; k ~ Poisson(rate).
Otherwise identical to simulate_poisson_llm_direct100.py (below).

Original description:
Poisson noise added DIRECTLY to gpt2-xl layer 29 activations (no mixing):
each simulated neuron = one embedding dimension.

Per seed: pick N_NEUR = 100 of the 1600 dimensions at random;
  z_dt      = activation of dim d at word t, z-scored over words
  lambda_dt = c_d * exp(BETA * z_dt), c_d set so mean_t lambda_dt equals the
              mean count of a randomly drawn real pseudopop unit
              ([onset+80, offset+40] counts, 12 patients, YEX excluded)
  k_dt      ~ Poisson(lambda_dt)
Raw counts only: curvature / displacement / position of k (top-K PCs of
centered counts, or full 100-d) correlated (chunk = 3) with the CLEAN full
gpt2-xl embedding (standardized top-K PCs / raw, as in the real analysis).
"noise-free" = same pipeline on lambda (same dims). Position as |r|.

Saves: results/simulate_poisson_llm_shift100.json
       figures/fig_simulate_poisson_llm_shift100.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = sys.argv[:1] + ["pseudopop", "dur8040", "full", "four"]
import compute_poisson_normalized_geometry as G   # noqa: E402
from podcast_loader import load_podcast_patient   # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(BASE_DIR, "results"), os.path.join(BASE_DIR, "figures")
N_NEUR, N_SEED = 100, 10
KS = [8, 16, 32, 64, None]
BETAS = ["shift"]
METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position |r|", "#2E9E6C")]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 10, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def raw_r(Y, sem, valid):
    nv = G.euclid(Y, False)
    out = {m: G.chunk_r(G.masked(nv[m], valid), G.masked(sem[m], valid)) for m, _, _ in METRICS}
    out["pos"] = abs(out["pos"])
    return out


def main():
    d = np.load(os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz"))
    assert "YEX" not in list(d["patients"])
    real_means = np.nanmean(d["K"], axis=0)
    common = d["common"]
    n = len(common)
    ref = load_podcast_patient("YEY", region="hippocampus", model_tag="gpt2-xl", context_words=200)
    valid = G.build_valid_mask(n, ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    emb = np.array(np.load(ref["emb_path"], mmap_mode="r")[29, common], float)
    Z = StandardScaler().fit_transform(emb)
    sem = {}
    for K in KS:
        G.KPC = K
        sem[K] = G.semantic_variants(emb)["raw"]

    out = dict(n_words=int(n), n_neurons=N_NEUR, n_seed=N_SEED, betas=BETAS,
               Ks=["full" if K is None else K for K in KS], results={})
    for beta in BETAS:
        noisy = {m: np.zeros((N_SEED, len(KS))) for m, _, _ in METRICS}
        clean = {m: np.zeros((N_SEED, len(KS))) for m, _, _ in METRICS}
        for seed in range(N_SEED):
            rng = np.random.default_rng(seed)
            dims = rng.choice(Z.shape[1], N_NEUR, replace=False)
            c_target = rng.choice(real_means[real_means > 0], N_NEUR, replace=False)
            A = emb[:, dims]
            E = A - A.min(0)
            lam = E * (c_target / E.mean(0))
            k = rng.poisson(lam).astype(float)
            for ki, K in enumerate(KS):
                G.KPC = K
                a, b = raw_r(k, sem[K], valid), raw_r(lam, sem[K], valid)
                for m, _, _ in METRICS:
                    noisy[m][seed, ki], clean[m][seed, ki] = a[m], b[m]
        out["results"][str(beta)] = dict(noisy={m: v.tolist() for m, v in noisy.items()},
                                         noisefree={m: v.tolist() for m, v in clean.items()})
        print(f"{beta}: mean count {k.mean():.2f}  " + "  ".join(
            f"{'full' if K is None else K}: c {noisy['curv'][:, i].mean():+.2f}[{clean['curv'][:, i].mean():+.2f}] "
            f"d {noisy['disp'][:, i].mean():+.2f}[{clean['disp'][:, i].mean():+.2f}] "
            f"|p| {noisy['pos'][:, i].mean():.2f}[{clean['pos'][:, i].mean():.2f}]" for i, K in enumerate(KS)),
            flush=True)
    json.dump(out, open(os.path.join(RESULTS, "simulate_poisson_llm_shift100.json"), "w"))

    x = np.arange(len(KS))
    fig, ax0 = plt.subplots(1, 1, figsize=(8.4, 4.6))
    axes = [ax0]
    for ax, beta, letter in zip(axes, BETAS, "ABC"):
        R = out["results"][str(beta)]
        for m, lab, col in METRICS:
            A, B = np.array(R["noisy"][m]), np.array(R["noisefree"][m])
            ax.errorbar(x, A.mean(0), yerr=A.std(0, ddof=1), color=col, marker="o", ms=4, lw=1.8, capsize=3,
                        label=f"{lab}, Poisson counts")
            ax.plot(x, B.mean(0), "--", color=col, lw=1.2, label=f"{lab}, noise-free")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"top {K}" if K else f"full ({N_NEUR})" for K in KS])
        ax.set_xlabel("Dimensions (top-K PCs, or full space)")
        ax.set_title("rate = c × (activation − min), then Poisson", loc="left")
    axes[0].set_ylabel("Noisy activations – clean gpt2-xl r")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    fig.suptitle(f"Poisson noise on gpt2-xl layer 29 activations\n{N_NEUR} random dims = {N_NEUR} neurons, real mean counts, "
                 f"raw counts; {n} words, chunk 3; mean ± SD, {N_SEED} seeds",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    stem = os.path.join(FIGURES, "fig_simulate_poisson_llm_shift100")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}")
    print(f"  -> {stem}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
