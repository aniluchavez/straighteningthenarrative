"""
compute_curvature_hilo_all_models_all_layers.py
=================================================
Companion to compute_curvature_dist_all_models_all_layers.py: instead of
(or in addition to) the marginal curvature distributions, this computes the
neural<->semantic curvature correlation split by surprisal (high vs. low
quartile, like fig2's Panel A/B), but pooled across ALL 10 contextual
models x ALL their layers x ALL 13 patients -- no cherry-picked "best
layer" or "best model", just the aggregate answer to "is curvature more
coupled to surprisal at high vs low surprisal, in general."

For each (model, layer, patient): compute word-level c_sem, c_neu, and the
surprisal of the target word, z-score c_sem/c_neu within that (model,
layer, patient) triple (so different models/layers/patients don't
dominate via different curvature scales), then pool ALL triples' hi-surprisal
words together and ALL lo-surprisal words together (surprisal split done
per patient, since surprisal itself is patient/session-specific).

Reports:
  - pooled r(c_sem, c_neu) for hi-surprisal words (all models/layers/patients)
  - pooled r(c_sem, c_neu) for lo-surprisal words (same)
  - the same two numbers computed per model (averaged across that model's
    layers/patients) as a sanity check / breakdown
  - patient-level test: per (model, patient) r at each layer, averaged,
    one-sample t-test hi vs 0 and lo vs 0

Saves:
  results/curvature_hilo_all_models_all_layers.json
  figures/appendix/fig_curvature_hilo_all_models_all_layers.{pdf,png}

Run (submitted via sbatch, see run_curvature_hilo_all_models.sbatch):
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u compute_curvature_hilo_all_models_all_layers.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, ttest_1samp, zscore
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
SURP_HI, SURP_LO = 75, 25

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

    all_hi_sem, all_hi_neu = [], []
    all_lo_sem, all_lo_neu = [], []
    per_model_layer_r = {"hi": {}, "lo": {}}   # model -> list of per-(layer,patient) r
    per_triple_r_hi, per_triple_r_lo = [], []  # for the patient-level t-test

    # cache neural PCA trajectories per patient (model-independent, computed once)
    neu_cache = {}
    for pid in PODCAST_PATIENTS:
        data0 = load_podcast_patient(pid, region=REGION, model_tag="llama-3.1-8b", context_words=200)
        if data0 is None or data0.get("neu_vecs") is None:
            continue
        n = data0["n_words"]
        valid = build_valid_mask(n, data0["surp"], data0["sent_pos"], data0["is_turn"])
        Y_neu, _ = fit_neural_pca(data0["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        centre_idx = np.arange(1, n - 1)[valid]
        surp_y = data0["surp"][centre_idx + 1]
        neu_cache[pid] = dict(c_neu=c_neu, surp_y=surp_y, n=n, valid=valid)
        print(f"  cached neural {pid}: n_valid={valid.sum()}")

    for model, n_layers in models_to_run.items():
        per_model_layer_r["hi"].setdefault(model, [])
        per_model_layer_r["lo"].setdefault(model, [])
        print(f"\nModel: {model} ({n_layers} layers)")
        for pid in PODCAST_PATIENTS:
            if pid not in neu_cache:
                continue
            data = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
            if data is None or data.get("emb_path") is None:
                continue
            nc = neu_cache[pid]
            c_neu, surp_y, n = nc["c_neu"], nc["surp_y"], nc["n"]

            fin_surp = np.isfinite(surp_y)
            hi_thr = np.percentile(surp_y[fin_surp], SURP_HI)
            lo_thr = np.percentile(surp_y[fin_surp], SURP_LO)

            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            for L in range(n_layers):
                emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                emb_L = sem_pca_reduce(emb_L)
                c_sem = trajectory_curvature(emb_L)[nc["valid"]]

                fin = np.isfinite(c_sem) & np.isfinite(c_neu) & np.isfinite(surp_y)
                if fin.sum() < 20:
                    continue
                c_sem_f, c_neu_f, surp_f = c_sem[fin], c_neu[fin], surp_y[fin]
                c_sem_z, c_neu_z = zscore(c_sem_f), zscore(c_neu_f)

                mask_hi = surp_f >= hi_thr
                mask_lo = surp_f <= lo_thr
                if mask_hi.sum() >= 10:
                    all_hi_sem.append(c_sem_z[mask_hi]); all_hi_neu.append(c_neu_z[mask_hi])
                    r_hi, _ = pearsonr(c_sem_z[mask_hi], c_neu_z[mask_hi])
                    per_model_layer_r["hi"][model].append(r_hi)
                    per_triple_r_hi.append(r_hi)
                if mask_lo.sum() >= 10:
                    all_lo_sem.append(c_sem_z[mask_lo]); all_lo_neu.append(c_neu_z[mask_lo])
                    r_lo, _ = pearsonr(c_sem_z[mask_lo], c_neu_z[mask_lo])
                    per_model_layer_r["lo"][model].append(r_lo)
                    per_triple_r_lo.append(r_lo)
            del emb_mm
        print(f"  hi pairs so far: {sum(len(a) for a in all_hi_sem)}   "
              f"lo pairs so far: {sum(len(a) for a in all_lo_sem)}")

    all_hi_sem = np.concatenate(all_hi_sem); all_hi_neu = np.concatenate(all_hi_neu)
    all_lo_sem = np.concatenate(all_lo_sem); all_lo_neu = np.concatenate(all_lo_neu)

    r_hi_pooled, p_hi_pooled = pearsonr(all_hi_sem, all_hi_neu)
    r_lo_pooled, p_lo_pooled = pearsonr(all_lo_sem, all_lo_neu)
    print(f"\nPOOLED (all models, all layers, all patients):")
    print(f"  high surprisal: r={r_hi_pooled:+.4f}  p={p_hi_pooled:.4g}  N={len(all_hi_sem)}")
    print(f"  low  surprisal: r={r_lo_pooled:+.4f}  p={p_lo_pooled:.4g}  N={len(all_lo_sem)}")

    r_hi_arr = np.array(per_triple_r_hi)
    r_lo_arr = np.array(per_triple_r_lo)
    t_hi, pv_hi = ttest_1samp(r_hi_arr, 0.0)
    t_lo, pv_lo = ttest_1samp(r_lo_arr, 0.0)
    print(f"\n(model,layer,patient)-level test (each triple's own r, n={len(r_hi_arr)} triples):")
    print(f"  high surprisal: mean r={r_hi_arr.mean():+.4f}  t={t_hi:.2f}  p={pv_hi:.4g}")
    print(f"  low  surprisal: mean r={r_lo_arr.mean():+.4f}  t={t_lo:.2f}  p={pv_lo:.4g}")

    print("\nPer-model breakdown (mean r across that model's layers x patients):")
    for model in models_to_run:
        hi_m = np.array(per_model_layer_r["hi"][model])
        lo_m = np.array(per_model_layer_r["lo"][model])
        print(f"  {model:14s}  hi mean r={hi_m.mean():+.4f} (n={len(hi_m)})   "
              f"lo mean r={lo_m.mean():+.4f} (n={len(lo_m)})")

    payload = dict(
        r_hi_pooled=float(r_hi_pooled), p_hi_pooled=float(p_hi_pooled), n_hi_pooled=int(len(all_hi_sem)),
        r_lo_pooled=float(r_lo_pooled), p_lo_pooled=float(p_lo_pooled), n_lo_pooled=int(len(all_lo_sem)),
        triple_level_hi_mean=float(r_hi_arr.mean()), triple_level_hi_t=float(t_hi), triple_level_hi_p=float(pv_hi),
        triple_level_lo_mean=float(r_lo_arr.mean()), triple_level_lo_t=float(t_lo), triple_level_lo_p=float(pv_lo),
        n_triples=int(len(r_hi_arr)),
        per_model_hi={m: np.array(v).tolist() for m, v in per_model_layer_r["hi"].items()},
        per_model_lo={m: np.array(v).tolist() for m, v in per_model_layer_r["lo"].items()},
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, f"curvature_hilo_all_models_all_layers{args.suffix}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved -> {out_json}")

    # ── Figure: bar of pooled r (hi vs lo) + per-model dot overlay ─────────────
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
    C_HI, C_LO = "#C0392B", "#2980B9"
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.bar(["High surprisal\n(>=75th pct)", "Low surprisal\n(<=25th pct)"],
           [r_hi_arr.mean(), r_lo_arr.mean()], color=[C_HI, C_LO], alpha=0.85,
           yerr=[r_hi_arr.std(ddof=1)/np.sqrt(len(r_hi_arr)), r_lo_arr.std(ddof=1)/np.sqrt(len(r_lo_arr))],
           capsize=6)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_ylabel("r(neural curvature, semantic curvature)")
    ax.set_title(f"Curvature-surprisal coupling, aggregated\n{'+'.join(models_to_run)}, all layers x all patients "
                 f"(n={len(r_hi_arr)} triples)", loc="left", pad=8, fontsize=10.5)
    ax.text(0.02, 0.02, f"pooled hi: r={r_hi_pooled:+.3f}, p={p_hi_pooled:.2e}\n"
                        f"pooled lo: r={r_lo_pooled:+.3f}, p={p_lo_pooled:.2e}",
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"fig_curvature_hilo_all_models_all_layers{args.suffix}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
