"""
make_convo_podcast_curvature_comparison_freq_surp_overlay_gpt2large.py
=============================================
make_convo_podcast_curvature_comparison_freq_surp_overlay.py, but using
gpt2-large instead of llama-3.1-8b. No cached LOO-layer selection exists
for gpt2-large (that json only exists for llama-3.1-8b), so this uses a
FIXED near-input layer (5/37) -- matching llama-3.1-8b's own cached LOO
layer (5/33), which was the layer where semantic curvature's shape most
cleanly matched neural (single peak at lag+1, no oscillation) in the
layer-depth comparison (near-input/middle/late) done for llama-3.1-8b.

  orange/green solid  = most frequent / most infrequent word (top/bottom 25%)
  red/blue dashed      = high / low surprisal (top/bottom 25%)
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
CONVO_MODEL = "gpt2-large"
REGION      = "hippocampus"
FIXED_LAYER = 5  # near-input, matching llama-3.1-8b's cached LOO layer 5/33

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, EVENT_WINDOW, build_paths,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

PODCAST_SEM_MODEL = "gpt2-large"
PODCAST_RESULTS_DIR = os.path.join(BASE_DIR, "results")
PODCAST_SEM_SINGLE_PID = "YEY"

C_FREQ_HI = "#E67E22"   # orange -- most frequent word
C_FREQ_LO = "#27AE60"   # green  -- most infrequent word
C_SURP_HI = "#C0392B"   # red    -- high surprisal
C_SURP_LO = "#2980B9"   # blue   -- low surprisal

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

HI_PCTILE = 75
LO_PCTILE = 25


def event_windows(values, split_var, n_words):
    finite = np.isfinite(split_var)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_words):
        sj = split_var[j]
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


def load_convo_split_curvature(split_key, layer=FIXED_LAYER):
    """split_key: 'log_freq' or 'surp'. Returns (hi_sem, lo_sem), (hi_neu, lo_neu)."""
    pat_hi_sem, pat_lo_sem, pat_hi_neu, pat_lo_neu = [], [], [], []
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        split_var = data[split_key][:n]

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

        hi_s, lo_s = event_windows(c_sem_full, split_var, n)
        hi_n, lo_n = event_windows(c_neu_full, split_var, n)
        if hi_s is None or hi_n is None:
            continue
        pat_hi_sem.append(np.nanmean(hi_s, axis=0)); pat_lo_sem.append(np.nanmean(lo_s, axis=0))
        pat_hi_neu.append(np.nanmean(hi_n, axis=0)); pat_lo_neu.append(np.nanmean(lo_n, axis=0))
    return (np.array(pat_hi_sem), np.array(pat_lo_sem)), (np.array(pat_hi_neu), np.array(pat_lo_neu))


def load_podcast_neural_split_curvature(split_key):
    pat_hi, pat_lo = [], []
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, model_tag="gpt2", context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        split_var = data[split_key][:n]
        X, _ = fit_neural_pca(data["neu_vecs"], n)
        if X is None:
            continue
        c = trajectory_curvature(X)
        hi_rows, lo_rows = event_windows(c, split_var, n)
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))
    return np.array(pat_hi), np.array(pat_lo)


def load_single_patient_sem_split_curvature(pid, model_tag, layer, split_key, context_words=200):
    data = load_podcast_patient(pid, model_tag=model_tag, context_words=context_words)
    n = data["n_words"]
    split_var = data[split_key][:n]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    c_sem_full = trajectory_curvature(emb_opt)

    finite = np.isfinite(split_var)
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)
    hi_mask = split_var >= hi_thr
    lo_mask = split_var <= lo_thr

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


def _draw(ax, lags, hi_mu, hi_se, lo_mu, lo_se, color_hi, color_lo, ls, lbl_hi, lbl_lo):
    ax.plot(lags, hi_mu, color=color_hi, lw=1.6, ls=ls, label=lbl_hi)
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=color_hi, alpha=0.15)
    ax.plot(lags, lo_mu, color=color_lo, lw=1.6, ls=ls, label=lbl_lo)
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=color_lo, alpha=0.15)


def _panel(ax, freq_pair, surp_pair, title, n_count, n_label="patients"):
    lags = np.arange(-WINDOW, WINDOW + 1)

    def _stats(pair):
        hi, lo = pair
        hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
        lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")
        return hi_mu, hi_se, lo_mu, lo_se

    fhi_mu, fhi_se, flo_mu, flo_se = _stats(freq_pair)
    shi_mu, shi_se, slo_mu, slo_se = _stats(surp_pair)

    _draw(ax, lags, fhi_mu, fhi_se, flo_mu, flo_se, C_FREQ_HI, C_FREQ_LO, "-",
          "Most frequent word", "Most infrequent word")
    _draw(ax, lags, shi_mu, shi_se, slo_mu, slo_se, C_SURP_HI, C_SURP_LO, "--",
          "High surprisal", "Low surprisal")

    ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} {n_label}", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _panel_single(ax, freq_precomp, surp_precomp, title, n_count):
    lags = np.arange(-WINDOW, WINDOW + 1)
    fhi_mu, fhi_se, flo_mu, flo_se, _ = freq_precomp
    shi_mu, shi_se, slo_mu, slo_se, _ = surp_precomp

    _draw(ax, lags, fhi_mu, fhi_se, flo_mu, flo_se, C_FREQ_HI, C_FREQ_LO, "-",
          "Most frequent word", "Most infrequent word")
    _draw(ax, lags, shi_mu, shi_se, slo_mu, slo_se, C_SURP_HI, C_SURP_LO, "--",
          "High surprisal", "Low surprisal")

    ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} events", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    print("Convo (freq)...", flush=True)
    convo_freq_sem, convo_freq_neu = load_convo_split_curvature("log_freq")
    print("Convo (surp)...", flush=True)
    convo_surp_sem, convo_surp_neu = load_convo_split_curvature("surp")

    print("Podcast neural (freq)...", flush=True)
    pod_freq_neu = load_podcast_neural_split_curvature("log_freq")
    print("Podcast neural (surp)...", flush=True)
    pod_surp_neu = load_podcast_neural_split_curvature("surp")

    pod_layer = FIXED_LAYER
    print("Podcast semantic (single patient, freq+surp)...", flush=True)
    pod_freq_sem = load_single_patient_sem_split_curvature(
        PODCAST_SEM_SINGLE_PID, PODCAST_SEM_MODEL, pod_layer, "log_freq")
    pod_surp_sem = load_single_patient_sem_split_curvature(
        PODCAST_SEM_SINGLE_PID, PODCAST_SEM_MODEL, pod_layer, "surp")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.82, wspace=0.34, hspace=0.6)

    _panel(axes[0, 0], convo_freq_sem, convo_surp_sem, f"A  Convo semantic ({CONVO_MODEL}, layer {FIXED_LAYER})",
           convo_freq_sem[0].shape[0])
    _panel(axes[0, 1], convo_freq_neu, convo_surp_neu, "B  Convo neural (spikes)",
           convo_freq_neu[0].shape[0])
    _panel_single(axes[1, 0], pod_freq_sem, pod_surp_sem,
                  f"C  Podcast semantic ({PODCAST_SEM_MODEL}, layer {pod_layer}, {PODCAST_SEM_SINGLE_PID} only)",
                  pod_freq_sem[-1])
    _panel(axes[1, 1], pod_freq_neu, pod_surp_neu, "D  Podcast neural (spikes)",
           pod_freq_neu[0].shape[0])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle(f"Event-triggered curvature, hippocampus  ·  convo vs podcast, semantic vs neural  ·  frequency + surprisal overlay, layer {FIXED_LAYER}",
                 fontsize=9, fontweight="bold", y=0.995)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_podcast_curvature_comparison_freq_surp_overlay_gpt2large"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
