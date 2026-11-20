"""
make_convo_podcast_curvature_comparison_surp_globalthresh_freqbands.py
=============================================
Same three frequency-decile bands (low/mid/high, bottom-10%/45-55%/top-10%)
as make_convo_podcast_curvature_comparison_surp_freqbands_overlay.py, but
here the hi/lo SURPRISAL THRESHOLD is computed GLOBALLY, over the full
unrestricted surprisal distribution (top 25% / bottom 25% of ALL words) --
not within each frequency band. A word only enters the low/mid/high-freq
"hi" or "lo" bucket if (a) it clears the GLOBAL surprisal threshold AND (b)
its own frequency falls in that band. This decouples "how the surprisal
threshold is computed" from "which frequency band the event happens to fall
in" -- checking whether the earlier band-overlay result (rare-word events
peaking at lag +1, mid/high-freq events peaking at lag 0) was an artifact of
computing thresholds separately within each band, or holds up when using one
shared global threshold.

Saves: figures/fig_convo_podcast_curvature_comparison_surp_globalthresh_freqbands.{pdf,png,svg}
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
from matplotlib.lines import Line2D

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
FIGURES     = os.path.join(BASE_DIR, "figures")
WINDOW      = 5
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, EVENT_WINDOW, build_paths,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

PODCAST_SEM_MODEL = "llama-3.1-8b"
PODCAST_RESULTS_DIR = os.path.join(BASE_DIR, "results")
PODCAST_SEM_SINGLE_PID = "YEY"

HI_PCTILE = 75
LO_PCTILE = 25

# (label, lo_pctile, hi_pctile, color)
BANDS = [
    ("Low freq (bottom 10%, rarest)",  0,  10, "#8E44AD"),
    ("Mid freq (45-55%, typical)",    45,  55, "#16A085"),
    ("High freq (top 10%, frequent)", 90, 100, "#D35400"),
]

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


def event_windows_from_masks(values, hi_mask, lo_mask, n_words):
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_words):
        is_hi, is_lo = hi_mask[j], lo_mask[j]
        if not (is_hi or is_lo):
            continue
        row = [values[(j + k) - 2] if 0 <= (j + k) - 2 < m else np.nan for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def band_in_mask(log_freq, lo_pctile, hi_pctile):
    finite = np.isfinite(log_freq)
    lo_edge = np.nanpercentile(log_freq[finite], lo_pctile)
    hi_edge = np.nanpercentile(log_freq[finite], hi_pctile)
    return finite & (log_freq >= lo_edge) & (log_freq <= hi_edge)


def load_convo_all_bands():
    conv_path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    cached = {r["pid"]: r["loo_layer_neu"] for r in json.load(open(conv_path))}

    per_band_sem = {label: ([], []) for label, *_ in BANDS}
    per_band_neu = {label: ([], []) for label, *_ in BANDS}

    for pid, preg in PATIENTS.items():
        if pid not in cached or REGION not in preg:
            continue
        layer = cached[pid]
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = data["log_freq"][:n]
        finite_surp = np.isfinite(surp)
        if finite_surp.sum() < 20:
            continue
        hi_thr = np.nanpercentile(surp[finite_surp], HI_PCTILE)
        lo_thr = np.nanpercentile(surp[finite_surp], LO_PCTILE)
        global_hi = finite_surp & (surp >= hi_thr)
        global_lo = finite_surp & (surp <= lo_thr)

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

        for label, lo_p, hi_p, color in BANDS:
            in_band = band_in_mask(log_freq, lo_p, hi_p)
            hi_mask = global_hi & in_band
            lo_mask = global_lo & in_band
            hi_s, lo_s = event_windows_from_masks(c_sem_full, hi_mask, lo_mask, n)
            hi_n, lo_n = event_windows_from_masks(c_neu_full, hi_mask, lo_mask, n)
            if hi_s is None or hi_n is None:
                continue
            per_band_sem[label][0].append(np.nanmean(hi_s, axis=0))
            per_band_sem[label][1].append(np.nanmean(lo_s, axis=0))
            per_band_neu[label][0].append(np.nanmean(hi_n, axis=0))
            per_band_neu[label][1].append(np.nanmean(lo_n, axis=0))
        print(f"  convo {pid}: layer={layer}  done", flush=True)

    sem_results, neu_results = [], []
    for label, *_ in BANDS:
        hi_arr, lo_arr = per_band_sem[label]
        n_count = len(hi_arr)
        sem_results.append((np.nanmean(np.array(hi_arr), axis=0), np.nanmean(np.array(lo_arr), axis=0), n_count))
        hi_arr, lo_arr = per_band_neu[label]
        neu_results.append((np.nanmean(np.array(hi_arr), axis=0), np.nanmean(np.array(lo_arr), axis=0), n_count))
    return sem_results, neu_results


def load_podcast_neural_all_bands():
    per_band = {label: ([], []) for label, *_ in BANDS}
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, model_tag="gpt2", context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = data["log_freq"][:n]
        finite_surp = np.isfinite(surp)
        if finite_surp.sum() < 20:
            continue
        hi_thr = np.nanpercentile(surp[finite_surp], HI_PCTILE)
        lo_thr = np.nanpercentile(surp[finite_surp], LO_PCTILE)
        global_hi = finite_surp & (surp >= hi_thr)
        global_lo = finite_surp & (surp <= lo_thr)

        X, _ = fit_neural_pca(data["neu_vecs"], n)
        if X is None:
            continue
        c = trajectory_curvature(X)

        for label, lo_p, hi_p, color in BANDS:
            in_band = band_in_mask(log_freq, lo_p, hi_p)
            hi_mask = global_hi & in_band
            lo_mask = global_lo & in_band
            hi_rows, lo_rows = event_windows_from_masks(c, hi_mask, lo_mask, n)
            if hi_rows is None:
                continue
            per_band[label][0].append(np.nanmean(hi_rows, axis=0))
            per_band[label][1].append(np.nanmean(lo_rows, axis=0))

    results = []
    for label, *_ in BANDS:
        hi_arr, lo_arr = per_band[label]
        n_count = len(hi_arr)
        results.append((np.nanmean(np.array(hi_arr), axis=0), np.nanmean(np.array(lo_arr), axis=0), n_count))
    return results


def load_single_patient_sem_all_bands(pid, model_tag, layer, context_words=200):
    data = load_podcast_patient(pid, model_tag=model_tag, context_words=context_words)
    n = data["n_words"]
    surp = data["surp"][:n]
    log_freq = data["log_freq"][:n]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    c_sem_full = trajectory_curvature(emb_opt)

    finite_surp = np.isfinite(surp)
    hi_thr = np.nanpercentile(surp[finite_surp], HI_PCTILE)
    lo_thr = np.nanpercentile(surp[finite_surp], LO_PCTILE)
    global_hi = finite_surp & (surp >= hi_thr)
    global_lo = finite_surp & (surp <= lo_thr)

    lags = list(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    results = []
    for label, lo_p, hi_p, color in BANDS:
        in_band = band_in_mask(log_freq, lo_p, hi_p)
        hi_mask = global_hi & in_band
        lo_mask = global_lo & in_band
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
        hi_mu = np.array([np.mean(v) if v else np.nan for v in hi_vals])
        lo_mu = np.array([np.mean(v) if v else np.nan for v in lo_vals])
        n_events = len(hi_vals[0])
        results.append((hi_mu, lo_mu, n_events))
    return results


def _panel(ax, band_results, title, n_label="patients"):
    lags = np.arange(-WINDOW, WINDOW + 1)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    n_counts = []
    for (label, lo_p, hi_p, color), (hi, lo, n_count) in zip(BANDS, band_results):
        n_counts.append(n_count)
        ax.plot(lags, hi, color=color, lw=1.6, ls="-")
        ax.plot(lags, lo, color=color, lw=1.6, ls="--")
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    n_str = ", ".join(str(c) for c in n_counts)
    ax.set_title(f"{title}\n$n$ = {n_str} {n_label} (low/mid/high freq)", loc="left", fontsize=9.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    convo_sem_results, convo_neu_results = load_convo_all_bands()
    podcast_neu_results = load_podcast_neural_all_bands()

    ctx200_path = os.path.join(PODCAST_RESULTS_DIR, f"geometry_podcast_{PODCAST_SEM_MODEL}_ctx200_{REGION}.json")
    with open(ctx200_path) as f:
        pod_records = json.load(f)
    pod_rec = next(r for r in pod_records if r["pid"] == PODCAST_SEM_SINGLE_PID)
    pod_layer = pod_rec["loo_layer_neu"]
    podcast_sem_results = load_single_patient_sem_all_bands(PODCAST_SEM_SINGLE_PID, PODCAST_SEM_MODEL, pod_layer)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.78, wspace=0.32, hspace=0.65)

    _panel(axes[0, 0], convo_sem_results, f"A  Convo semantic ({CONVO_MODEL})")
    _panel(axes[0, 1], convo_neu_results, "B  Convo neural (spikes)")
    _panel(axes[1, 0], podcast_sem_results,
           f"C  Podcast semantic ({PODCAST_SEM_MODEL}, layer {pod_layer}, {PODCAST_SEM_SINGLE_PID} only)",
           n_label="events")
    _panel(axes[1, 1], podcast_neu_results, "D  Podcast neural (spikes)")

    handles = []
    for label, lo_p, hi_p, color in BANDS:
        handles.append(Line2D([0], [0], color=color, lw=1.6, ls="-", label=f"{label} — hi surprisal"))
        handles.append(Line2D([0], [0], color=color, lw=1.6, ls="--", label=f"{label} — lo surprisal"))
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.975),
               fontsize=8, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle("Event-triggered curvature, hippocampus  ·  convo vs podcast, semantic vs neural  ·  "
                 "GLOBAL surprisal threshold, bucketed by frequency band (low/mid/high deciles)",
                 fontsize=9, fontweight="bold", y=0.998)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_podcast_curvature_comparison_surp_globalthresh_freqbands"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
