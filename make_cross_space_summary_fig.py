"""
make_cross_space_summary_fig.py
================================
Summarise cross-space geometric coupling as mean r averaged across ALL layers,
for each model × metric combination.  No layer breakdown.

Two side-by-side heatmaps:
  Left  — word-by-word correlations
  Right — 10-word-chunk correlations

Rows = models, Columns = Displacement / Curvature / Position
Cell colour = mean r across layers and patients; annotated with value.

Run:
    cd /scratch/aniluchavez/straighteningthenarrative
    conda run -n gpt2_embed python3 make_cross_space_summary_fig.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import sem as scipy_sem, ttest_1samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION   = "hippocampus"

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         12,
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

MODEL_ORDER = [
    "bert-base", "roberta-base",
    "gpt2", "gpt2-medium", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]
MODEL_LABELS = {
    "bert-base":    "BERT-base",
    "roberta-base": "RoBERTa-base",
    "gpt2":         "GPT-2",
    "gpt2-medium":  "GPT-2 medium",
    "opt-350m":     "OPT-350m",
    "llama-2-7b":   "LLaMA-2 7B",
    "llama-3.1-8b": "LLaMA-3.1 8B",
    "gemma-2-9b":   "Gemma-2 9B",
    "mistral-7b":   "Mistral-7B",
}

METRICS = [
    dict(key="disp", label="Displacement", r_key="r_disp_layers",    r_key_ch="r_disp_layers_ch"),
    dict(key="curv", label="Curvature κ",  r_key="r_curv_layers",    r_key_ch="r_curv_layers_ch"),
    dict(key="pos",  label="Position",     r_key="r_pos_layers",     r_key_ch="r_pos_layers_ch"),
]


def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def build_matrix(r_key):
    """
    Returns (mean_mat, sig_mat) each shape (n_models, n_metrics).
    mean_mat[i, j] = mean r across layers and patients for model i, metric j.
    sig_mat[i, j]  = significance stars string.
    """
    n_models  = len(MODEL_ORDER)
    n_metrics = len(METRICS)
    mean_mat = np.full((n_models, n_metrics), np.nan)
    sig_mat  = np.full((n_models, n_metrics), "", dtype=object)

    for mi, model in enumerate(MODEL_ORDER):
        path = os.path.join(RESULTS, f"cross_space_layers_{model}_{REGION}.json")
        if not os.path.exists(path):
            continue
        recs = json.load(open(path))

        for qi, metric in enumerate(METRICS):
            key = metric[r_key]
            # per-patient mean across layers
            pat_means = []
            for rec in recs:
                if key not in rec:
                    continue
                arr = np.array(rec[key], dtype=float)
                if np.isfinite(arr).any():
                    pat_means.append(np.nanmean(arr))
            if not pat_means:
                continue
            pat_means = np.array(pat_means)
            mean_mat[mi, qi] = np.mean(pat_means)
            if len(pat_means) > 1:
                _, p = ttest_1samp(pat_means, 0)
                sig_mat[mi, qi] = _stars(p)

    return mean_mat, sig_mat


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
    mat_c, sig_c = build_matrix("r_key_ch")   # chunk-averaged only

    all_vals = mat_c.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    vmax = np.percentile(np.abs(all_vals), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    metric_labels = [m["label"] for m in METRICS]
    model_labels  = [MODEL_LABELS[m] for m in MODEL_ORDER]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 5.2))
    fig.subplots_adjust(left=0.28, right=0.88, bottom=0.10, top=0.88)

    im = ax.imshow(mat_c, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(model_labels, fontsize=10)

    # Separator between encoder and decoder groups
    ax.axhline(1.5, color="white", lw=2.0)

    for mi in range(len(MODEL_ORDER)):
        for qi in range(len(METRICS)):
            val = mat_c[mi, qi]
            if np.isnan(val):
                continue
            star = sig_c[mi, qi]
            txt  = f"{val:+.3f}{star}"
            bg = norm(val)
            fg = "white" if abs(bg - 0.5) > 0.28 else "#222222"
            ax.text(qi, mi, txt, ha="center", va="center",
                    fontsize=7.5, color=fg, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("Mean r (LLM ↔ Neural)", fontsize=9)

    fig.suptitle(
        "Cross-space geometric coupling  ·  Hippocampus\n"
        "15-word chunks, mean r across all layers and patients",
        fontsize=9.5, fontweight="bold", y=0.97)

    _save(fig, "fig_cross_space_summary_heatmap")
    print("Done.")


if __name__ == "__main__":
    main()
