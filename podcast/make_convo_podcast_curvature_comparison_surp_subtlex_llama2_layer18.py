"""
make_convo_podcast_curvature_comparison_surp_subtlex_llama2_layer18.py
========================================================================
Same design as make_convo_podcast_curvature_comparison_surp_midfreq_midlayer.py
(event-triggered curvature, hi/lo surprisal split restricted to words in the
middle frequency band, convo vs podcast x semantic vs neural), but:

  - Model: llama-2-7b instead of llama-3.1-8b, fixed at layer 18 (the layer
    of maximal hi/lo surprisal separation found for this model in the
    podcast word-level analysis).
  - Frequency: real SUBTLEX-derived population frequency (wordfreq's
    zipf_frequency), NOT the crude within-transcript word-count proxy the
    _midfreq_midlayer script used.
  - Surprisal: unchanged -- the standard model-based next-word surprisal
    already precomputed in {pid}_surprisal.csv (log-probability of the
    observed word under the LM), same "usual" measure used throughout the
    rest of this project.

Saves: figures/fig_convo_podcast_curvature_comparison_surp_subtlex_llama2_layer18.{pdf,png,svg}
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem as scipy_sem
from wordfreq import zipf_frequency

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
FIGURES     = os.path.join(BASE_DIR, "figures")
WINDOW      = 5
CONVO_MODEL = "llama-2-7b"
REGION      = "hippocampus"
MIDFREQ_LO_PCTILE = 45
MIDFREQ_HI_PCTILE = 55
LAYER       = 18  # llama-2-7b layer of max hi/lo surprisal separation (podcast word-level analysis)

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, EVENT_WINDOW, build_paths,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

PODCAST_SEM_MODEL = "llama-2-7b"
PODCAST_SEM_SINGLE_PID = "YEY"

C_HI = "#C0392B"   # red  -- high surprisal
C_LO = "#2980B9"   # blue -- low surprisal

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

_zipf_cache = {}
def subtlex_log_freq(words):
    """Real SUBTLEX-derived population frequency (wordfreq's zipf_frequency)."""
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = str(w).lower()
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out


def midfreq_masked_surp(surp, log_freq):
    """surp, with entries outside the middle 10% of log_freq set to NaN --
    only words with 'typical' frequency remain eligible as hi/lo-surprisal
    events."""
    finite = np.isfinite(log_freq)
    lo_edge = np.nanpercentile(log_freq[finite], MIDFREQ_LO_PCTILE)
    hi_edge = np.nanpercentile(log_freq[finite], MIDFREQ_HI_PCTILE)
    in_band = finite & (log_freq >= lo_edge) & (log_freq <= hi_edge)
    out = np.full_like(surp, np.nan, dtype=float)
    out[in_band] = surp[in_band]
    return out, int(in_band.sum())


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


def load_convo_surp_subtlex(layer=LAYER):
    pat_hi_sem, pat_lo_sem, pat_hi_neu, pat_lo_neu = [], [], [], []
    n_events_total = 0
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = subtlex_log_freq(data["words"][:n])
        surp_mf, n_band = midfreq_masked_surp(surp, log_freq)

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

        hi_s, lo_s = event_windows(c_sem_full, surp_mf, n)
        hi_n, lo_n = event_windows(c_neu_full, surp_mf, n)
        if hi_s is None or hi_n is None:
            continue
        pat_hi_sem.append(np.nanmean(hi_s, axis=0)); pat_lo_sem.append(np.nanmean(lo_s, axis=0))
        pat_hi_neu.append(np.nanmean(hi_n, axis=0)); pat_lo_neu.append(np.nanmean(lo_n, axis=0))
        n_events_total += n_band
        print(f"  convo {pid}: layer={layer}  midfreq_words={n_band}  done", flush=True)

    return (np.array(pat_hi_sem), np.array(pat_lo_sem)), (np.array(pat_hi_neu), np.array(pat_lo_neu))


def load_podcast_neural_surp_subtlex():
    pat_hi, pat_lo = [], []
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, model_tag=PODCAST_SEM_MODEL, context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = subtlex_log_freq(data["words"][:n])
        surp_mf, n_band = midfreq_masked_surp(surp, log_freq)
        X, _ = fit_neural_pca(data["neu_vecs"], n)
        if X is None:
            continue
        c = trajectory_curvature(X)
        hi_rows, lo_rows = event_windows(c, surp_mf, n)
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))
    return np.array(pat_hi), np.array(pat_lo)


def load_single_patient_sem_surp_subtlex(pid, model_tag, layer, context_words=200):
    data = load_podcast_patient(pid, model_tag=model_tag, context_words=context_words)
    n = data["n_words"]
    surp = data["surp"][:n]
    log_freq = subtlex_log_freq(data["words"][:n])
    surp_mf, n_band = midfreq_masked_surp(surp, log_freq)
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    c_sem_full = trajectory_curvature(emb_opt)

    finite = np.isfinite(surp_mf)
    hi_thr = np.nanpercentile(surp_mf[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(surp_mf[finite], LO_PCTILE)
    hi_mask = surp_mf >= hi_thr
    lo_mask = surp_mf <= lo_thr

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


def main():
    (hi_convo_sem, lo_convo_sem), (hi_convo_neu, lo_convo_neu) = load_convo_surp_subtlex()
    hi_pod_neu, lo_pod_neu = load_podcast_neural_surp_subtlex()

    pod_precomputed = load_single_patient_sem_surp_subtlex(
        PODCAST_SEM_SINGLE_PID, PODCAST_SEM_MODEL, LAYER)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.8, wspace=0.32, hspace=0.6)

    _panel(axes[0, 0], hi_convo_sem, lo_convo_sem, f"A  Convo semantic ({CONVO_MODEL}, layer {LAYER})")
    _panel(axes[0, 1], hi_convo_neu, lo_convo_neu, "B  Convo neural (spikes)")
    _panel(axes[1, 0], None, None,
           f"C  Podcast semantic ({PODCAST_SEM_MODEL}, layer {LAYER}, {PODCAST_SEM_SINGLE_PID} only)",
           n_label="events", precomputed=pod_precomputed)
    _panel(axes[1, 1], hi_pod_neu, lo_pod_neu, "D  Podcast neural (spikes)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle(f"Event-triggered curvature, hippocampus  ·  convo vs podcast, semantic vs neural  ·  "
                 f"SUBTLEX-masked surprisal split (middle {MIDFREQ_HI_PCTILE - MIDFREQ_LO_PCTILE}% freq only), "
                 f"{CONVO_MODEL} layer {LAYER}",
                 fontsize=8.5, fontweight="bold", y=0.995)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_podcast_curvature_comparison_surp_subtlex_llama2_layer18"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
