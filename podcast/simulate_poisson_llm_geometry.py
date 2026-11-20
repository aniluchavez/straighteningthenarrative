"""
simulate_poisson_llm_geometry.py
==================================
Positive control: Poisson "neurons" that encode gpt2-xl layer 29 perfectly,
run through the same 4-step pipeline (raw / sqrt / L2 / sqrt+L2, top-K PCs
and full space) and correlated with the clean embedding's curvature /
displacement / position (chunk = 3), exactly as the real pseudopop.

Simulation (Poisson noise only -- no window-length scaling, no shared gain):
  z_t     = standardized gpt2-xl layer 29 embedding (1600-d), pseudopop words
  w_i     = random unit-norm direction in R^1600 (one per model unit)
  s_it    = w_i . z_t, z-scored over words
  lambda_it = c_i * exp(BETA * s_it);  c_i set so mean_t lambda_it equals the
            mean count of a real pseudopop unit (436 units, 12 patients,
            [onset+80, offset+40] counts; real means matched one-to-one)
  k_it    ~ Poisson(lambda_it)
BETA = SD of log-rate across words (tuning depth): 0.25, 0.5, 1.0.
N_SEED seeds (new w and new Poisson draws each). "noise-free" = lambda itself
(same pipeline), seed 0's w.

Saves: results/simulate_poisson_llm_geometry.json
       figures/fig_simulate_poisson_llm_geometry_beta{B}.{pdf,png,svg}
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
KS = [8, 16, 32, 64, 128, None]
BETAS = [0.25, 0.5, 1.0]
N_SEED = 10
V = ["raw", "sqrt", "l2", "sqrt_l2"]
METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position", "#2E9E6C")]
TITLES = {"raw": "A  Raw counts", "sqrt": "B  Square root  √k", "l2": "C  L2-normalized  k / ‖k‖",
          "sqrt_l2": "D  Square root + L2  √k / ‖√k‖"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def couple(nv, sv, valid):
    return {v: {m: G.chunk_r(G.masked(nv[v][m], valid), G.masked(sv["raw"][m], valid)) for m, _, _ in METRICS}
            for v in V}


def main():
    d = np.load(os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz"))
    Kreal, common = d["K"], d["common"]
    assert "YEX" not in list(d["patients"]), "rebuild pseudopop cache without YEX first"
    unit_means = np.nanmean(Kreal, axis=0)
    n, N = len(common), Kreal.shape[1]
    ref = load_podcast_patient("YEY", region="hippocampus", model_tag="gpt2-xl", context_words=200)
    valid = G.build_valid_mask(n, ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    emb = np.array(np.load(ref["emb_path"], mmap_mode="r")[29, common], float)
    Z = StandardScaler().fit_transform(emb)
    print(f"words={n} units={N} real mean count/unit: median {np.median(unit_means):.2f} "
          f"[{unit_means.min():.2f}, {unit_means.max():.2f}]", flush=True)

    sem = {}
    for K in KS:
        G.KPC = K
        sem[K] = G.semantic_variants(emb)
    out = dict(n_words=int(n), n_units=int(N), betas=BETAS, n_seed=N_SEED,
               Ks=["full" if K is None else K for K in KS], results={})
    for beta in BETAS:
        res = dict(noisy={v: {m: np.zeros((N_SEED, len(KS))) for m, _, _ in METRICS} for v in V},
                   noisefree={v: {m: np.zeros(len(KS)) for m, _, _ in METRICS} for v in V})
        for seed in range(N_SEED):
            rng = np.random.default_rng(1000 * int(beta * 100) + seed)
            W = rng.standard_normal((Z.shape[1], N))
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            S = Z @ W
            S = (S - S.mean(0)) / S.std(0)
            E = np.exp(beta * S)
            lam = E * (unit_means / E.mean(0))
            Ksim = rng.poisson(lam).astype(float)
            for ki, K in enumerate(KS):
                G.KPC = K
                r = couple(G.neural_variants4(Ksim, [np.arange(N)], rng), sem[K], valid)
                for v in V:
                    for m, _, _ in METRICS:
                        res["noisy"][v][m][seed, ki] = r[v][m]
                if seed == 0:
                    r0 = couple(G.neural_variants4(lam, [np.arange(N)], rng), sem[K], valid)
                    for v in V:
                        for m, _, _ in METRICS:
                            res["noisefree"][v][m][ki] = r0[v][m]
            if seed == 0:
                print(f"  beta={beta}: zero-spike words {np.mean(Ksim.sum(1) == 0) * 100:.2f}%  "
                      f"mean count {Ksim.mean():.2f}", flush=True)
        out["results"][str(beta)] = {
            k: {v: {m: np.asarray(a).tolist() for m, a in dv.items()} for v, dv in res[k].items()} for k in res}
        for v in V:
            line = "  ".join(
                f"K={'full' if K is None else K}: c{res['noisy'][v]['curv'][:, ki].mean():+.3f}"
                f"[nf {res['noisefree'][v]['curv'][ki]:+.3f}] d{res['noisy'][v]['disp'][:, ki].mean():+.3f}"
                f"[nf {res['noisefree'][v]['disp'][ki]:+.3f}]" for ki, K in enumerate(KS))
            print(f"  beta={beta} {v:8s} {line}", flush=True)

        # figure, one per beta
        x = np.arange(len(KS))
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
        for ax, v in zip(axes, V):
            for m, lab, col in METRICS:
                M = res["noisy"][v][m]
                ax.errorbar(x, np.nanmean(M, 0), yerr=np.nanstd(M, 0, ddof=1), color=col, marker="o", ms=4,
                            lw=1.8, capsize=3, label=f"{lab} (Poisson)")
                ax.plot(x, res["noisefree"][v][m], "--", color=col, lw=1.2, alpha=0.8,
                        label=f"{lab} (noise-free λ)")
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xticks(x)
            ax.set_xticklabels([f"top {K}" if K else "full" for K in KS])
            ax.set_xlabel("Dimensions (top-K PCs, or full space)")
            ax.set_title(TITLES[v], loc="left")
        axes[0].set_ylabel("Simulated-neural – semantic r")
        axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
        fig.suptitle(f"Simulated Poisson neurons encoding gpt2-xl layer 29 ({N} units, real per-unit mean counts, "
                     f"log-rate SD β = {beta}); {n} words, chunk = 3; mean ± SD over {N_SEED} seeds",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        stem = f"fig_simulate_poisson_llm_geometry_beta{beta}"
        for ext in ("pdf", "png", "svg"):
            fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
        plt.close(fig)
        print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "simulate_poisson_llm_geometry.json"), "w"))


if __name__ == "__main__":
    main()
