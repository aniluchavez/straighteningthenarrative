"""
make_convo_podcast_curvature_comparison_surp_deciles.py
=============================================
Same settings/data as make_convo_podcast_curvature_comparison_freq.py
(convo: llama-3.1-8b, 15 patients, each at cached loo_layer_neu; podcast:
llama-3.1-8b semantic single-patient YEY + 9-patient neural, all raw/no-PCA
embeddings), but instead of a hi/lo (top/bottom 25%) word-frequency split,
bins words into 10 SURPRISAL DECILES (decile 0 = lowest/most predictable
10%, decile 9 = highest/most surprising 10%) and plots all 10
panel, colored by a continuous colormap instead of two discrete colors.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem as scipy_sem

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
FIGURES     = os.path.join(BASE_DIR, "figures")
WINDOW      = 5
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"
N_DECILES   = 10

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, EVENT_WINDOW, build_paths,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

PODCAST_SEM_MODEL = "llama-3.1-8b"
PODCAST_RESULTS_DIR = os.path.join(BASE_DIR, "results")
PODCAST_SEM_SINGLE_PID = "YEY"

CMAP = cm.get_cmap("RdYlGn_r")  # low decile (predictable, green-ish) -> high decile (surprising, red-ish)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def decile_event_windows(values, split_var, n_words, n_bins=N_DECILES):
    """Returns list of n_bins arrays, each [n_events, len(lags)], for words
    whose split_var falls in that decile bin (bin 0 = lowest values)."""
    finite = np.isfinite(split_var)
    if finite.sum() < n_bins * 5:
        return None
    edges = np.percentile(split_var[finite], np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9  # inclusive bounds
    bin_idx = np.digitize(split_var, edges) - 1  # 0..n_bins-1, or out-of-range

    lags = list(range(-WINDOW, WINDOW + 1))
    rows_by_bin = [[[] for _ in lags] for _ in range(n_bins)]
    m = len(values)
    for j in range(n_words):
        if not np.isfinite(split_var[j]):
            continue
        b = bin_idx[j]
        if not (0 <= b < n_bins):
            continue
        for ki, k in enumerate(lags):
            ci = (j + k) - 2
            if 0 <= ci < m:
                rows_by_bin[b][ki].append(values[ci])

    out = []
    for b in range(n_bins):
        row_means = [np.mean(x) if x else np.nan for x in rows_by_bin[b]]
        out.append(row_means)
    return np.array(out)  # (n_bins, n_lags)


def load_convo_surp_deciles():
    conv_path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    cached = {r["pid"]: r["loo_layer_neu"] for r in json.load(open(conv_path))}

    pat_sem, pat_neu = [], []
    for pid, preg in PATIENTS.items():
        if pid not in cached or REGION not in preg:
            continue
        layer = cached[pid]
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)

        paths = build_paths(pid, CONVO_MODEL)
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        c_sem_full = trajectory_curvature(emb_opt)

        dec_s = decile_event_windows(c_sem_full, surp, n)
        dec_n = decile_event_windows(c_neu_full, surp, n)
        if dec_s is None or dec_n is None:
            continue
        pat_sem.append(dec_s); pat_neu.append(dec_n)
        print(f"  convo {pid}: layer={layer}  done", flush=True)

    return np.array(pat_sem), np.array(pat_neu)  # (n_pat, n_bins, n_lags)


def load_podcast_neural_surp_deciles():
    pat = []
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, model_tag="gpt2", context_words=200)  # model_tag="gpt2" (not default None) so the 4 extra-source patients are included too
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        X, _ = fit_neural_pca(data["neu_vecs"], n)
        if X is None:
            continue
        c = trajectory_curvature(X)
        dec = decile_event_windows(c, surp, n)
        if dec is None:
            continue
        pat.append(dec)
    return np.array(pat)


def load_single_patient_sem_surp_deciles(pid, model_tag, layer, context_words=200):
    data = load_podcast_patient(pid, model_tag=model_tag, context_words=context_words)
    n = data["n_words"]
    surp = data["surp"][:n]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    c_sem_full = trajectory_curvature(emb_opt)
    dec = decile_event_windows(c_sem_full, surp, n)
    return dec, n  # (n_bins, n_lags), single-patient so no cross-patient SEM


def _panel(ax, mat, title, n_label="patients", n_count=None):
    lags = np.arange(-WINDOW, WINDOW + 1)
    if mat.ndim == 3:
        n_count = mat.shape[0] if n_count is None else n_count
        mu = np.nanmean(mat, axis=0)   # (n_bins, n_lags)
        se = scipy_sem(mat, axis=0, nan_policy="omit")
    else:
        mu = mat
        se = np.zeros_like(mu)

    for b in range(N_DECILES):
        color = CMAP(b / (N_DECILES - 1))
        ax.plot(lags, mu[b], color=color, lw=1.3, alpha=0.9)
        ax.fill_between(lags, mu[b] - se[b], mu[b] + se[b], color=color, alpha=0.10)

    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} {n_label}", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    convo_sem, convo_neu = load_convo_surp_deciles()
    pod_neu = load_podcast_neural_surp_deciles()

    ctx200_path = os.path.join(PODCAST_RESULTS_DIR, f"geometry_podcast_{PODCAST_SEM_MODEL}_ctx200_{REGION}.json")
    with open(ctx200_path) as f:
        pod_records = json.load(f)
    pod_rec = next(r for r in pod_records if r["pid"] == PODCAST_SEM_SINGLE_PID)
    pod_layer = pod_rec["loo_layer_neu"]
    pod_sem, pod_sem_n = load_single_patient_sem_surp_deciles(PODCAST_SEM_SINGLE_PID, PODCAST_SEM_MODEL, pod_layer)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.09, top=0.86, wspace=0.36, hspace=0.62)

    _panel(axes[0, 0], convo_sem, f"A  Convo semantic ({CONVO_MODEL})")
    _panel(axes[0, 1], convo_neu, "B  Convo neural (spikes)")
    _panel(axes[1, 0], pod_sem, f"C  Podcast semantic ({PODCAST_SEM_MODEL}, layer {pod_layer}, {PODCAST_SEM_SINGLE_PID} only)",
           n_label="words", n_count=pod_sem_n)
    _panel(axes[1, 1], pod_neu, "D  Podcast neural (spikes)")

    sm = cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=N_DECILES))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=30, ticks=[1, 5, 10])
    cbar.set_label("Surprisal decile", fontsize=9)
    cbar.ax.set_yticklabels(["1 (lowest surprisal)", "5", "10 (highest surprisal)"], fontsize=7.5)

    fig.suptitle("Event-triggered curvature, hippocampus  ·  convo vs podcast, semantic vs neural  ·  by surprisal decile",
                 fontsize=9.5, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_podcast_curvature_comparison_surp_deciles"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
