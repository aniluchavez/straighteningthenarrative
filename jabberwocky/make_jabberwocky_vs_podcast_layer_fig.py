"""
make_jabberwocky_vs_podcast_layer_fig.py
=========================================
Layer-by-layer overlay of Jabberwocky vs. Podcast geometry profiles
(llama-3.1-8b, hippocampus), from the two per-layer JSONs produced by
run_jabberwocky_geometry_analysis_pca20.py and
../podcast/run_podcast_geometry_analysis_pca20.py.

Panel A: mean semantic (LLM) curvature by normalized layer position.
Panel B: neural-LLM curvature alignment (Pearson r) by normalized layer
         position -- this is where "jabberwocky is more curvy" gets a
         layer-resolved picture, alongside whether the neural<->semantic
         coupling itself differs by condition.

Podcast patients are subsampled (without replacement, matched to
Jabberwocky's n) inside a bootstrap so both curves reflect the same
patient count, consistent with make_jabberwocky_vs_podcast_neural_curvature.py.

Run:
    cd .../geometry_paper/jabberwocky
    conda run -n gpt2_embed python3 -u make_jabberwocky_vs_podcast_layer_fig.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL = "llama-3.1-8b"
REGION = "hippocampus"

JAB_PATH = os.path.join(BASE_DIR, "results", f"geometry_jabberwocky_pca20_{MODEL}_ctx200_{REGION}.json")
POD_PATH = os.path.join(PARENT_DIR, "podcast", "results", f"geometry_podcast_pca20_{MODEL}_ctx200_{REGION}.json")


def load_records(path):
    with open(path) as f:
        return json.load(f)


def stack_field(records, field):
    """Pad/trim to common n_layers (all should match for one model) -> (n_patients, n_layers)."""
    arrs = [np.asarray(r[field], dtype=float) for r in records]
    n_layers = min(len(a) for a in arrs)
    return np.stack([a[:n_layers] for a in arrs])


def normalized_x(n_layers):
    return np.linspace(0, 1, n_layers)


def bootstrap_mean_band(mat, n_match=None, n_boot=2000, seed=0):
    """mat: (n_patients, n_layers). If n_match given, subsample (no replacement)
    n_match patients per bootstrap draw so group sizes are matched; else standard
    patient-resampling bootstrap at full n."""
    rng = np.random.default_rng(seed)
    n_pat = mat.shape[0]
    k = n_match if n_match is not None else n_pat
    draws = np.empty((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.choice(n_pat, size=k, replace=(n_match is None))
        draws[b] = mat[idx].mean(axis=0)
    return mat.mean(axis=0) if n_match is None else draws.mean(axis=0), \
        np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def main():
    jab_records = load_records(JAB_PATH)
    pod_records_raw = load_records(POD_PATH)
    pod_records = [r["profile"] for r in pod_records_raw]
    print(f"Jabberwocky: {len(jab_records)} patients")
    print(f"Podcast:     {len(pod_records)} patients")

    n_match = len(jab_records)

    jab_curv = stack_field(jab_records, "mean_curv")
    pod_curv = stack_field(pod_records, "mean_curv")
    jab_rneu = stack_field(jab_records, "r_neural")
    pod_rneu = stack_field(pod_records, "r_neural")

    n_layers = min(jab_curv.shape[1], pod_curv.shape[1])
    jab_curv, pod_curv = jab_curv[:, :n_layers], pod_curv[:, :n_layers]
    jab_rneu, pod_rneu = jab_rneu[:, :n_layers], pod_rneu[:, :n_layers]
    x = normalized_x(n_layers)

    jab_curv_mean, jab_curv_lo, jab_curv_hi = bootstrap_mean_band(jab_curv)
    pod_curv_mean, pod_curv_lo, pod_curv_hi = bootstrap_mean_band(pod_curv, n_match=n_match)
    jab_rneu_mean, jab_rneu_lo, jab_rneu_hi = bootstrap_mean_band(jab_rneu)
    pod_rneu_mean, pod_rneu_lo, pod_rneu_hi = bootstrap_mean_band(pod_rneu, n_match=n_match)

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
    C_JAB = "#8E44AD"
    C_POD = "#2980B9"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle(f"Jabberwocky vs. Podcast, layer-by-layer ({MODEL}, hippocampus, "
                 f"n={n_match} vs n={n_match} of {len(pod_records)})", y=1.03)

    ax = axes[0]
    ax.plot(x, jab_curv_mean, color=C_JAB, label="Jabberwocky", linewidth=2)
    ax.fill_between(x, jab_curv_lo, jab_curv_hi, color=C_JAB, alpha=0.2)
    ax.plot(x, pod_curv_mean, color=C_POD, label="Podcast (matched n)", linewidth=2)
    ax.fill_between(x, pod_curv_lo, pod_curv_hi, color=C_POD, alpha=0.2)
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Mean semantic curvature (deg)")
    ax.set_title("A  Semantic (LLM) curvature by layer")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, jab_rneu_mean, color=C_JAB, label="Jabberwocky", linewidth=2)
    ax.fill_between(x, jab_rneu_lo, jab_rneu_hi, color=C_JAB, alpha=0.2)
    ax.plot(x, pod_rneu_mean, color=C_POD, label="Podcast (matched n)", linewidth=2)
    ax.fill_between(x, pod_rneu_lo, pod_rneu_hi, color=C_POD, alpha=0.2)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("r(neural curvature, semantic curvature)")
    ax.set_title("B  Neural-LLM curvature alignment by layer")
    ax.legend(frameon=False)

    out_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"fig_jabberwocky_vs_podcast_layer_profile.{ext}")
        fig.savefig(out_path)
        print(f"\nSaved -> {out_path}")

    # quick numeric summary
    peak_jab_layer = int(np.argmax(jab_rneu_mean))
    peak_pod_layer = int(np.argmax(pod_rneu_mean))
    print(f"\nPeak neural-alignment layer: jabberwocky={peak_jab_layer}/{n_layers-1} "
          f"(r={jab_rneu_mean[peak_jab_layer]:.4f}), podcast={peak_pod_layer}/{n_layers-1} "
          f"(r={pod_rneu_mean[peak_pod_layer]:.4f})")
    mean_curv_diff = jab_curv_mean.mean() - pod_curv_mean.mean()
    print(f"Mean semantic curvature averaged across all layers: jabberwocky - podcast = {mean_curv_diff:+.3f} deg")


if __name__ == "__main__":
    main()
