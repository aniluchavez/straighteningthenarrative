"""
make_cross_space_chunk_sweep_fig.py
=====================================
Shows mean r(LLM, neural) vs chunk size for displacement, curvature, and
position, averaged across all models and patients.  Justifies the 10-word
chunk choice by showing the pattern is stable across a range of sizes.

Each panel = one metric.  Each model shown as a thin line; thick line = mean
across models ± SE.  Vertical dashed line marks chunk_size = 10.

Run:
    conda run -n gpt2_embed python3 make_cross_space_chunk_sweep_fig.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         13,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

MODEL_STYLE = {
    "bert-base":    dict(color="#E67E22", label="BERT-base"),
    "roberta-base": dict(color="#F39C12", label="RoBERTa-base"),
    "gpt2":         dict(color="#2980B9", label="GPT-2"),
    "gpt2-medium":  dict(color="#1F618D", label="GPT-2 medium"),
    "opt-350m":     dict(color="#884EA0", label="OPT-350m"),
    "llama-2-7b":   dict(color="#C0392B", label="LLaMA-2 7B"),
    "llama-3.1-8b": dict(color="#E91E63", label="LLaMA-3.1 8B"),
    "gemma-2-9b":   dict(color="#148F77", label="Gemma-2 9B"),
    "mistral-7b":   dict(color="#D35400", label="Mistral-7B"),
}

METRICS = [
    dict(key="disp", label="Displacement",  panel="A"),
    dict(key="curv", label="Curvature κ",   panel="B"),
    dict(key="pos",  label="Position (PC1)", panel="C"),
]


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  → {p}")
    plt.close(fig)


def main():
    sweep_path = os.path.join(RESULTS, "cross_space_chunk_sweep.json")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f"Run compute_cross_space_chunk_sweep.py first.\n  Missing: {sweep_path}")

    data = json.load(open(sweep_path))
    chunk_sizes = sorted(int(k) for k in data.keys())
    xs = np.array(chunk_sizes)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.80,
                        wspace=0.42)

    for ax, metric in zip(axes, METRICS):
        mk = metric["key"]

        # Per-model curves (thin)
        model_means = []
        for model, st in MODEL_STYLE.items():
            ys = []
            for cs in chunk_sizes:
                vals = data[str(cs)].get(model, {}).get(mk, [])
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                ys.append(np.mean(vals) if vals else np.nan)
            ys = np.array(ys)
            ax.plot(xs, ys, color=st["color"], lw=0.9, alpha=0.5,
                    label=st["label"])
            model_means.append(ys)

        # Mean ± SE across models (thick)
        model_means = np.array(model_means)
        mu = np.nanmean(model_means, axis=0)
        se = np.array([scipy_sem(model_means[:, i][np.isfinite(model_means[:, i])])
                       for i in range(len(xs))])
        ax.plot(xs, mu, color="black", lw=2.2, zorder=5, label="Mean")
        ax.fill_between(xs, mu - se, mu + se, color="black", alpha=0.15, zorder=4)

        ax.axvline(15, color="#555555", lw=1.0, ls="--", zorder=3,
                   label="15-word elbow" if ax is axes[0] else "_")
        ax.axhline(0,  color="k", lw=0.4, ls=":")

        ax.set_xlabel("Chunk size (words)")
        ax.set_ylabel("Mean r  (LLM ↔ Neural)")
        ax.set_title(f"{metric['panel']}  {metric['label']}\n"
                     "all models · hippocampus · avg layers",
                     loc="left", fontsize=10)
        ax.set_xticks(chunk_sizes)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=11,
               frameon=False, bbox_to_anchor=(0.5, -0.06),
               handlelength=1.4, columnspacing=0.8, fontsize=8)

    fig.suptitle(
        "Cross-space coupling vs chunk size  ·  Hippocampus",
        fontsize=10, fontweight="bold", y=0.93)

    _save(fig, "fig_cross_space_chunk_sweep")
    print("Done.")


if __name__ == "__main__":
    main()
