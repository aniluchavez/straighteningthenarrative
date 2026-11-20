"""
make_convo_podcast_curvature_comparison.py
=============================================
Side-by-side event-triggered neural curvature (hippocampus), hi- vs
lo-surprisal: convo (llama-3.1-8b, 15 patients) vs podcast (gpt2-large
spikes, 9 patients), same axes/style for direct visual comparison.

Convo panel: reuses the "event_triggered" field already saved in
results/geometry_llama-3.1-8b_hippocampus.json (from run_geometry_analysis.py).
Podcast panel: recomputed directly from spikes via podcast_loader.py, same
logic as make_podcast_event_triggered.py's "curvature" metric.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem as scipy_sem

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
FIGURES     = os.path.join(BASE_DIR, "figures")
WINDOW      = 5
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import fit_neural_pca, trajectory_curvature
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

# Podcast semantic panel now sources from its own LOO-layer-selected results
# (run_podcast_geometry_analysis.py --models llama-3.1-8b), same as convo.
PODCAST_SEM_MODEL = "llama-3.1-8b"
PODCAST_RESULTS_DIR = os.path.join(BASE_DIR, "results")

C_HI = "#C0392B"
C_LO = "#2980B9"

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


def load_event_triggered_from_json(path, key, only_pid=None):
    with open(path) as f:
        records = json.load(f)
    hi_rows, lo_rows = [], []
    for rec in records:
        if only_pid is not None and rec.get("pid") != only_pid:
            continue
        et = rec.get("event_triggered")
        if not et or et.get(f"hi_{key}") is None:
            continue
        hi_rows.append(et[f"hi_{key}"])
        lo_rows.append(et[f"lo_{key}"])
    return np.array(hi_rows), np.array(lo_rows)


def load_convo_curvature(key):
    path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    return load_event_triggered_from_json(path, key)


def load_podcast_sem_curvature(only_pid=None):
    path = os.path.join(PODCAST_RESULTS_DIR, f"geometry_podcast_{PODCAST_SEM_MODEL}_{REGION}.json")
    return load_event_triggered_from_json(path, "sem", only_pid=only_pid)


HI_PCTILE = 75
LO_PCTILE = 25


def event_windows(values, surp, n_words):
    finite = np.isfinite(surp)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(surp[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(surp[finite], LO_PCTILE)
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_words):
        sj = surp[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = [values[(j + k) - 2] if 0 <= (j + k) - 2 < m else np.nan for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def load_podcast_neural_curvature():
    """Neural (spikes -> PCA) curvature, recomputed directly -- no LOO layer involved."""
    pat_hi, pat_lo = [], []
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, model_tag="gpt2", context_words=200)  # model_tag="gpt2" (not default None) so the 4 extra-source patients are included too
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        X, _ = fit_neural_pca(data["neu_vecs"], n)
        if X is None:
            continue

        c = trajectory_curvature(X)
        hi_rows, lo_rows = event_windows(c, data["surp"], n)
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))
    return np.array(pat_hi), np.array(pat_lo)


def load_single_patient_sem_curvature(pid, model_tag, layer, context_words=200):
    """One patient's semantic curvature, keeping per-EVENT values so SEM is
    computed across events (not across patients -- there's only one)."""
    data = load_podcast_patient(pid, model_tag=model_tag, context_words=context_words)
    n = data["n_words"]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    c_sem_full = trajectory_curvature(emb_opt)

    from run_geometry_analysis import surprisal_quantile_event_masks, EVENT_WINDOW
    hi_mask, lo_mask, _, _ = surprisal_quantile_event_masks(data["surp"])
    lags = list(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    hi_vals = [[] for _ in lags]
    lo_vals = [[] for _ in lags]
    for j in range(n):
        is_hi, is_lo = hi_mask[j], lo_mask[j]
        if not (is_hi or is_lo):
            continue
        for ki, k in enumerate(lags):
            ci = (j + k) - 2
            if 0 <= ci < len(c_sem_full):
                (hi_vals if is_hi else lo_vals)[ki].append(c_sem_full[ci])

    hi_mu = np.array([np.mean(v) for v in hi_vals])
    hi_se = np.array([scipy_sem(v) for v in hi_vals])
    lo_mu = np.array([np.mean(v) for v in lo_vals])
    lo_se = np.array([scipy_sem(v) for v in lo_vals])
    n_events = len(hi_vals[0])
    return hi_mu, hi_se, lo_mu, lo_se, n_events


def _panel(ax, hi, lo, title, n_label="patients", precomputed=None):
    lags = np.arange(-WINDOW, WINDOW + 1)
    if precomputed is not None:
        hi_mu, hi_se, lo_mu, lo_se, n_count = precomputed
    else:
        n_count = hi.shape[0]
        hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
        lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")

    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} {n_label}", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))



PODCAST_SEM_SINGLE_PID = "YEY"


def main():
    hi_convo_neu, lo_convo_neu = load_convo_curvature("neu")
    hi_convo_sem, lo_convo_sem = load_convo_curvature("sem")
    hi_pod_neu, lo_pod_neu = load_podcast_neural_curvature()

    ctx200_path = os.path.join(PODCAST_RESULTS_DIR, f"geometry_podcast_{PODCAST_SEM_MODEL}_ctx200_{REGION}.json")
    with open(ctx200_path) as f:
        pod_records = json.load(f)
    pod_rec = next(r for r in pod_records if r["pid"] == PODCAST_SEM_SINGLE_PID)
    pod_layer = pod_rec["loo_layer_neu"]
    pod_precomputed = load_single_patient_sem_curvature(
        PODCAST_SEM_SINGLE_PID, PODCAST_SEM_MODEL, pod_layer)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.8, wspace=0.32, hspace=0.6)

    _panel(axes[0, 0], hi_convo_sem, lo_convo_sem, f"A  Convo semantic ({CONVO_MODEL})")
    _panel(axes[0, 1], hi_convo_neu, lo_convo_neu, "B  Convo neural (spikes)")
    _panel(axes[1, 0], None, None,
           f"C  Podcast semantic ({PODCAST_SEM_MODEL}, layer {pod_layer}, {PODCAST_SEM_SINGLE_PID} only)",
           n_label="events", precomputed=pod_precomputed)
    _panel(axes[1, 1], hi_pod_neu, lo_pod_neu, "D  Podcast neural (spikes)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle("Event-triggered curvature, hippocampus  ·  convo vs podcast, semantic vs neural",
                 fontsize=9, fontweight="bold", y=0.995)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_podcast_curvature_comparison"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
