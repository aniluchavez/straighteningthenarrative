"""
make_curvature_histograms.py
============================
Standalone diagnostic histograms for curvature values.

This does not integrate into Figure 1.  It uses the same valid curvature mask
as fig1_layer_profiles and compares:

  A. Hippocampal neural curvature vs LLaMA-3.1-8B curvature at the winning
     LOO-neural layer.
  B. LLaMA-3.1-8B curvature histograms across all layers as a heatmap.

Run:
    conda run -n gpt2_embed python3 make_curvature_histograms.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (  # noqa: E402
    MODELS,
    PATIENTS,
    build_valid_mask,
    fit_neural_pca,
    load_patient,
    trajectory_curvature,
)

RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(BASE_DIR, "figures", "appendix")

MODEL = "llama-3.1-8b"
REGION = "hippocampus"
N_BINS = 120

C_NEU = "#1B7F4A"
C_LLM = "#E91E63"
C_Q = "#333333"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PANEL_LABEL_KW = dict(
    fontsize=28, fontweight="bold", fontfamily="sans-serif",
    va="top", ha="left",
)


def add_panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, **PANEL_LABEL_KW)


def density_hist(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    widths = np.diff(bins)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths, dtype=float)
    return counts / (total * widths)


def collect_curvatures():
    json_path = os.path.join(RESULTS, f"geometry_{MODEL}_{REGION}.json")
    records = json.load(open(json_path))
    n_layers = MODELS[MODEL]

    neural_vals = []
    winning_vals = []
    all_layer_vals = [[] for _ in range(n_layers)]
    layer_counts = np.zeros(n_layers, dtype=int)
    patient_rows = []

    for rec in records:
        pid = rec["pid"]
        preg = PATIENTS.get(pid, {})
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, MODEL)
        if data is None:
            continue

        n = data["n_words"]
        valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
        if valid.sum() == 0:
            continue

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        c_neu = c_neu[np.isfinite(c_neu)]
        neural_vals.append(c_neu)

        win_layer = int(rec["loo_layer_neu"])
        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        for layer in range(n_layers):
            emb_layer = np.asarray(emb_mm[layer, :n], dtype=np.float64)
            c_sem = trajectory_curvature(emb_layer)[valid]
            c_sem = c_sem[np.isfinite(c_sem)]
            all_layer_vals[layer].append(c_sem)
            layer_counts[layer] += len(c_sem)
            if layer == win_layer:
                winning_vals.append(c_sem)
        del emb_mm

        patient_rows.append(dict(
            pid=pid,
            n_valid=int(valid.sum()),
            loo_layer_neu=win_layer,
            neural_mean=float(np.mean(c_neu)),
            neural_median=float(np.median(c_neu)),
        ))
        print(f"  {pid}: n={int(valid.sum())}, winning_layer={win_layer}")

    neural = np.concatenate(neural_vals)
    winning = np.concatenate(winning_vals)
    layers = [np.concatenate(vals) if vals else np.array([]) for vals in all_layer_vals]
    return neural, winning, layers, layer_counts, patient_rows


def robust_xlim(*arrays):
    joined = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)])
    lo, hi = np.percentile(joined, [0.5, 99.5])
    pad = 0.06 * (hi - lo)
    return max(0.0, lo - pad), min(180.0, hi + pad)


def draw_overlay(ax, neural, winning, bins, winning_layer):
    centers = 0.5 * (bins[:-1] + bins[1:])
    d_neu = density_hist(neural, bins)
    d_win = density_hist(winning, bins)

    ax.fill_between(centers, d_neu, step="mid", color=C_NEU, alpha=0.22)
    ax.plot(centers, d_neu, color=C_NEU, lw=1.9,
            label=f"Neural hippocampus (N={len(neural):,})")
    ax.fill_between(centers, d_win, step="mid", color=C_LLM, alpha=0.18)
    ax.plot(centers, d_win, color=C_LLM, lw=1.9,
            label=f"LLaMA layer {winning_layer} (N={len(winning):,})")

    for vals, color in [(neural, C_NEU), (winning, C_LLM)]:
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        ax.axvline(mu, color=color, lw=1.1, ls="-")
        ax.axvspan(mu - sd, mu + sd, color=color, alpha=0.08, lw=0)

    ax.set_xlabel("Curvature κ (degrees)")
    ax.set_ylabel("Density")
    ax.set_title("Curvature distribution at winning layer", loc="left")
    add_panel_label(ax, "A")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#DDDDDD")
    ax.text(
        0.98,
        0.96,
        "vertical line = mean\nshaded band = mean ± SD",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=C_Q,
    )


def draw_layer_heatmap(ax, layers, bins):
    dens = np.vstack([density_hist(vals, bins) for vals in layers])
    dens = dens / np.maximum(dens.max(axis=1, keepdims=True), 1e-12)

    im = ax.imshow(
        dens,
        aspect="auto",
        origin="lower",
        extent=[bins[0], bins[-1], -0.5, len(layers) - 0.5],
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    means = np.array([np.mean(vals) for vals in layers])
    ax.plot(means, np.arange(len(layers)), color="white", lw=1.1,
            marker="o", ms=2.0, label="Layer mean")
    ax.set_xlabel("Curvature κ (degrees)")
    ax.set_ylabel("LLaMA-3.1-8B layer")
    ax.set_title("Per-layer curvature histogram heatmap", loc="left")
    add_panel_label(ax, "B")
    ax.set_yticks([0, 8, 16, 24, 32])
    ax.legend(loc="lower right", frameon=True, framealpha=0.85,
              edgecolor="#DDDDDD", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Density within layer (row-normalized)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def save_summary(neural, winning, layers, layer_counts, patient_rows, winning_layer):
    def summary(vals):
        return dict(
            n=int(len(vals)),
            mean=float(np.mean(vals)),
            median=float(np.median(vals)),
            q25=float(np.percentile(vals, 25)),
            q75=float(np.percentile(vals, 75)),
            q95=float(np.percentile(vals, 95)),
        )

    payload = dict(
        model=MODEL,
        region=REGION,
        winning_layer=int(winning_layer),
        neural=summary(neural),
        winning_llm_layer=summary(winning),
        layer_counts=layer_counts.astype(int).tolist(),
        layer_means=[float(np.mean(vals)) for vals in layers],
        layer_medians=[float(np.median(vals)) for vals in layers],
        patients=patient_rows,
    )
    path = os.path.join(RESULTS, f"curvature_histograms_{MODEL}_{REGION}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  -> {path}")


def main():
    neural, winning, layers, layer_counts, patient_rows = collect_curvatures()
    winning_layers = sorted(set(row["loo_layer_neu"] for row in patient_rows))
    winning_layer_label = winning_layers[0] if len(winning_layers) == 1 else "LOO"

    xlo, xhi = robust_xlim(neural, winning, *layers)
    bins = np.linspace(0.0, 180.0, N_BINS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.16, top=0.84, wspace=0.32)
    draw_overlay(axes[0], neural, winning, bins, winning_layer_label)
    draw_layer_heatmap(axes[1], layers, bins)
    for ax in axes:
        ax.set_xlim(xlo, xhi)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_histograms"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        if ext == "svg":
            with open(path, "r") as fh:
                txt = fh.read().replace("TeX Gyre Termes", "Times New Roman")
            with open(path, "w") as fh:
                fh.write(txt)
        print(f"  -> {path}")
    plt.close(fig)

    save_summary(neural, winning, layers, layer_counts, patient_rows,
                 winning_layer_label)


if __name__ == "__main__":
    main()
