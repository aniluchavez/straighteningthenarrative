"""
simulate_poisson_llm_shift100_samedims.py
===========================================
Noise-only reference for simulate_poisson_llm_shift100.py: noisy Poisson counts
of 100 random gpt2-xl layer 29 dims (rate = c * (activation - min), c = real
unit mean count) vs the SAME 100 dims' noise-free rates, processed identically
(raw, centered PCA, same K). Noise-free r = 1 by construction, so the drop is
due to Poisson noise only. Position as |r| (PC1 sign arbitrary).

Saves: results/simulate_poisson_llm_shift100_samedims.json
       figures/fig_simulate_poisson_llm_shift100_samedims.{pdf,png,svg}
"""
import json
import os
import sys

import numpy as np
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
METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position |r|", "#2E9E6C")]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 10, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def main():
    d = np.load(os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz"))
    real_means = np.nanmean(d["K"], axis=0)
    common = d["common"]
    n = len(common)
    ref = load_podcast_patient("YEY", region="hippocampus", model_tag="gpt2-xl", context_words=200)
    valid = G.build_valid_mask(n, ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    emb = np.array(np.load(ref["emb_path"], mmap_mode="r")[29, common], float)
    R = {m: np.zeros((N_SEED, len(KS))) for m, _, _ in METRICS}
    for seed in range(N_SEED):
        rng = np.random.default_rng(seed)                    # same seeds/dims/means as shift100
        dims = rng.choice(emb.shape[1], N_NEUR, replace=False)
        c = rng.choice(real_means[real_means > 0], N_NEUR, replace=False)
        A = emb[:, dims]
        E = A - A.min(0)
        lam = E * (c / E.mean(0))
        k = rng.poisson(lam).astype(float)
        for ki, K in enumerate(KS):
            G.KPC = K
            a, b = G.euclid(k, False), G.euclid(lam, False)
            for m, _, _ in METRICS:
                r = G.chunk_r(G.masked(a[m], valid), G.masked(b[m], valid))
                R[m][seed, ki] = abs(r) if m == "pos" else r
    print("noisy vs same-dims noise-free: " + "  ".join(
        f"{'full' if K is None else K}: c {R['curv'][:, i].mean():+.2f} d {R['disp'][:, i].mean():+.2f} "
        f"|p| {R['pos'][:, i].mean():.2f}" for i, K in enumerate(KS)))
    json.dump(dict(Ks=["full" if K is None else K for K in KS], results={m: v.tolist() for m, v in R.items()}),
              open(os.path.join(RESULTS, "simulate_poisson_llm_shift100_samedims.json"), "w"))
    x = np.arange(len(KS))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for m, lab, col in METRICS:
        ax.errorbar(x, R[m].mean(0), yerr=R[m].std(0, ddof=1), color=col, marker="o", ms=4, lw=1.8, capsize=3,
                    label=lab)
    ax.axhline(1, color="k", lw=0.8, ls="--", label="noise-free (= 1 by construction)")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {K}" if K else f"full ({N_NEUR})" for K in KS])
    ax.set_xlabel("Dimensions (top-K PCs, or full space)")
    ax.set_ylabel("r(noisy counts, same dims noise-free)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Noise only: same 100 dims, identical processing", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Poisson noise on gpt2-xl layer 29 activations\n{N_NEUR} random dims, rate = c × (activation − min), "
                 f"real mean counts; raw counts; {n} words, chunk 3; mean ± SD, {N_SEED} seeds",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    stem = os.path.join(FIGURES, "fig_simulate_poisson_llm_shift100_samedims")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}")
    print(f"  -> {stem}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
