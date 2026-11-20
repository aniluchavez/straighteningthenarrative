"""
compute_curvature_dist_all_models_all_layers.py
=================================================
Panel-C-style curvature distribution (like fig1_layer_profiles_pca20.pdf's
panel C: hippocampal neural vs LLM curvature, histograms overlaid), but
instead of picking one "winning" (model, layer), this pools:

  - Neural: every valid word, every one of the 13 podcast patients
            (model-independent -- hippocampal geometry doesn't depend on
            which LLM you compare it to).
  - Semantic: every valid word, every layer, every one of the 10 contextual
              models, every patient -- i.e. no cherry-picking a best layer
              or best model, just "what does curvature look like in
              general across the whole network and every model tried."

No permutation nulls, no chunking -- just raw word-level curvature values,
so this is comparatively fast despite covering ~10x more (model, layer)
combinations than the original panel C.

Saves:
  results/curvature_dist_all_models_all_layers.json (bin counts, not raw
    arrays, to keep the file small)
  figures/appendix/fig_curvature_dist_all_models_all_layers.{pdf,png}

Run (submitted via sbatch, see run_curvature_dist_all_models.sbatch):
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u compute_curvature_dist_all_models_all_layers.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_geometry_analysis_pca20 import (
    trajectory_curvature, build_valid_mask, fit_neural_pca, sem_pca_reduce,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(PARENT_DIR, "figures")
REGION = "hippocampus"

N_LAYERS = {
    "gpt2": 13, "gpt2-medium": 25, "gpt2-xl": 49,
    "bert-base": 13, "roberta-base": 13, "opt-350m": 24,
    "llama-2-7b": 33, "llama-3.1-8b": 33, "mistral-7b": 33, "gemma-2-9b": 43,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(N_LAYERS.keys()),
                     help="subset of models to include (default: all)")
    ap.add_argument("--suffix", default="", help="filename suffix, e.g. _llama2")
    args = ap.parse_args()
    models_to_run = {m: N_LAYERS[m] for m in args.models}
    print(f"Running models: {list(models_to_run.keys())}")

    # ── Neural: pooled across all 13 patients, model-independent ───────────────
    neural_vals = []
    for pid in PODCAST_PATIENTS:
        # model_tag only needed to satisfy the loader's emb-cache existence
        # check; neu_vecs themselves don't depend on which model is named here.
        data = load_podcast_patient(pid, region=REGION, model_tag="llama-3.1-8b", context_words=200)
        if data is None or data.get("neu_vecs") is None:
            print(f"  neural {pid}: SKIP")
            continue
        n = data["n_words"]
        valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c = trajectory_curvature(Y_neu)[valid]
        c = c[np.isfinite(c)]
        neural_vals.append(c)
        print(f"  neural {pid}: n={len(c)}")
    neural_vals = np.concatenate(neural_vals)
    print(f"\nNeural pooled: N={len(neural_vals)}  mean={neural_vals.mean():.2f}")

    # ── Semantic: pooled across all models, all layers, all patients ───────────
    semantic_vals = []
    for model, n_layers in models_to_run.items():
        print(f"\nModel: {model} ({n_layers} layers)")
        for pid in PODCAST_PATIENTS:
            data = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
            if data is None or data.get("emb_path") is None:
                continue
            n = data["n_words"]
            valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            for L in range(n_layers):
                emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                emb_L = sem_pca_reduce(emb_L)
                c = trajectory_curvature(emb_L)[valid]
                c = c[np.isfinite(c)]
                semantic_vals.append(c)
            del emb_mm
        n_so_far = sum(len(a) for a in semantic_vals)
        print(f"  running total N={n_so_far}")

    semantic_vals = np.concatenate(semantic_vals)
    print(f"\nSemantic pooled (all models, all layers): N={len(semantic_vals)}  mean={semantic_vals.mean():.2f}")

    # ── Save histogram bin counts (not raw arrays) ──────────────────────────────
    bins = np.linspace(0.0, 180.0, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def density_hist(values):
        counts, _ = np.histogram(values, bins=bins)
        widths = np.diff(bins)
        total = counts.sum()
        return (counts / (total * widths)).tolist() if total else [0.0] * len(widths)

    payload = dict(
        bin_centers=centers.tolist(),
        neural_density=density_hist(neural_vals),
        semantic_density=density_hist(semantic_vals),
        n_neural=int(len(neural_vals)),
        n_semantic=int(len(semantic_vals)),
        neural_mean=float(neural_vals.mean()),
        semantic_mean=float(semantic_vals.mean()),
        models=list(models_to_run.keys()),
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, f"curvature_dist_all_models_all_layers{args.suffix}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved -> {out_json}")

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    neural_color = "#148F77"
    llm_color = "#5DADE2"

    d_neu = np.array(payload["neural_density"])
    d_llm = np.array(payload["semantic_density"])
    ax.fill_between(centers, d_neu, step="mid", color=neural_color, alpha=0.20)
    ax.plot(centers, d_neu, color=neural_color, lw=1.6,
            label=f"Hippocampal neural ($N={len(neural_vals):,}$)")
    ax.fill_between(centers, d_llm, step="mid", color=llm_color, alpha=0.18)
    ax.plot(centers, d_llm, color=llm_color, lw=1.6,
            label=f"{'+'.join(models_to_run)} (all layers, $N={len(semantic_vals):,}$)")

    for vals, col in [(neural_vals, neural_color), (semantic_vals, llm_color)]:
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        ax.axvspan(mu - sd, mu + sd, color=col, alpha=0.07, lw=0)
        ax.axvline(mu, color=col, lw=1.0)

    ax.set_xlim(0, 180)
    ax.set_xlabel("Curvature $\\kappa$ (deg)")
    ax.set_ylabel("Density")
    ax.set_title(f"Curvature distribution\naggregated across {'+'.join(models_to_run)}, all layers (podcast, n=13)",
                  loc="left", pad=8)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.85, edgecolor="#DDDDDD")
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"fig_curvature_dist_all_models_all_layers{args.suffix}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
