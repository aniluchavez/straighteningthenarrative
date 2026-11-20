"""
make_podcast_curvature_allwords_vs_midfreq.py
================================================
Podcast-only (all 13 patients pooled), llama-2-7b layer 18, event-triggered
curvature by surprisal (top vs bottom 25%), comparing:

  - All words (no frequency filtering)
  - Words restricted to the middle 10% of SUBTLEX population frequency
    (wordfreq's zipf_frequency), same nested-deviance style control used
    elsewhere in this project

...for BOTH semantic (LLM layer 18) and neural (hippocampal spikes) curvature.

2x2 figure: rows = {semantic, neural}, columns = {all words, mid-frequency only}.

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_podcast_curvature_allwords_vs_midfreq.py
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
WINDOW = 5
MODEL = "llama-2-7b"
LAYER = 18
REGION = "hippocampus"
MIDFREQ_LO_PCTILE = 45
MIDFREQ_HI_PCTILE = 55
HI_PCTILE = 75
LO_PCTILE = 25

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import fit_neural_pca, trajectory_curvature, build_valid_mask
from run_geometry_analysis_pca20 import sem_pca_reduce
PCA_K = 100
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

C_HI = "#C0392B"
C_LO = "#2980B9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
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

_zipf_cache = {}
def subtlex_log_freq(words):
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = str(w).lower()
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out


def midfreq_mask(log_freq):
    finite = np.isfinite(log_freq)
    lo_edge = np.nanpercentile(log_freq[finite], MIDFREQ_LO_PCTILE)
    hi_edge = np.nanpercentile(log_freq[finite], MIDFREQ_HI_PCTILE)
    return finite & (log_freq >= lo_edge) & (log_freq <= hi_edge)


def event_windows(values, split_var, eligible, n_words):
    """hi/lo split done on split_var restricted to `eligible` words only."""
    finite = np.isfinite(split_var) & eligible
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_words):
        if not eligible[j] or not np.isfinite(split_var[j]):
            continue
        sj = split_var[j]
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


def main():
    pat_hi_sem_all, pat_lo_sem_all = [], []
    pat_hi_neu_all, pat_lo_neu_all = [], []
    pat_hi_sem_mid, pat_lo_sem_mid = [], []
    pat_hi_neu_mid, pat_lo_neu_mid = [], []
    n_all_total, n_mid_total = 0, 0

    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data.get("neu_vecs") is None or data.get("emb_path") is None:
            print(f"  {pid}: SKIP (no data)")
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        words = data["words"][:n]
        log_freq = subtlex_log_freq(words)
        mid_eligible = midfreq_mask(log_freq)
        all_eligible = np.isfinite(log_freq)  # essentially all words

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            print(f"  {pid}: SKIP (no neural PCA)")
            continue
        c_neu = trajectory_curvature(Y_neu)

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        emb_L = sem_pca_reduce(emb_L, k=PCA_K)
        c_sem = trajectory_curvature(emb_L)

        # all words
        hi_s, lo_s = event_windows(c_sem, surp, all_eligible, n)
        hi_n, lo_n = event_windows(c_neu, surp, all_eligible, n)
        if hi_s is not None and hi_n is not None:
            pat_hi_sem_all.append(np.nanmean(hi_s, axis=0)); pat_lo_sem_all.append(np.nanmean(lo_s, axis=0))
            pat_hi_neu_all.append(np.nanmean(hi_n, axis=0)); pat_lo_neu_all.append(np.nanmean(lo_n, axis=0))
            n_all_total += hi_s.shape[0] + lo_s.shape[0]

        # mid-frequency only
        hi_s2, lo_s2 = event_windows(c_sem, surp, mid_eligible, n)
        hi_n2, lo_n2 = event_windows(c_neu, surp, mid_eligible, n)
        if hi_s2 is not None and hi_n2 is not None:
            pat_hi_sem_mid.append(np.nanmean(hi_s2, axis=0)); pat_lo_sem_mid.append(np.nanmean(lo_s2, axis=0))
            pat_hi_neu_mid.append(np.nanmean(hi_n2, axis=0)); pat_lo_neu_mid.append(np.nanmean(lo_n2, axis=0))
            n_mid_total += hi_s2.shape[0] + lo_s2.shape[0]

        print(f"  {pid}: n_words={n}  mid_eligible={mid_eligible.sum()}  done", flush=True)

    def to_arr(lst):
        return np.array(lst)

    pat_hi_sem_all, pat_lo_sem_all = to_arr(pat_hi_sem_all), to_arr(pat_lo_sem_all)
    pat_hi_neu_all, pat_lo_neu_all = to_arr(pat_hi_neu_all), to_arr(pat_lo_neu_all)
    pat_hi_sem_mid, pat_lo_sem_mid = to_arr(pat_hi_sem_mid), to_arr(pat_lo_sem_mid)
    pat_hi_neu_mid, pat_lo_neu_mid = to_arr(pat_hi_neu_mid), to_arr(pat_lo_neu_mid)

    print(f"\nn_patients (all-words): {len(pat_hi_sem_all)}   n_patients (mid-freq): {len(pat_hi_sem_mid)}")

    def _panel(ax, hi, lo, title, n_label="patients"):
        lags = np.arange(-WINDOW, WINDOW + 1)
        n_count = hi.shape[0]
        hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
        lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")
        ax.plot(lags, hi_mu, color=C_HI, lw=1.8, label="High surprisal (top 25%)")
        ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
        ax.plot(lags, lo_mu, color=C_LO, lw=1.8, label="Low surprisal (bottom 25%)")
        ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
        ax.set_ylabel("Mean curvature (°)", fontsize=10)
        ax.set_title(f"{title}\n$n$ = {n_count} {n_label}", loc="left", fontsize=10.5, pad=3)
        ax.tick_params(labelsize=9)
        ax.set_xticks(lags)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.8, wspace=0.32, hspace=0.6)

    _panel(axes[0, 0], pat_hi_sem_all, pat_lo_sem_all, f"A  Semantic ({MODEL}, layer {LAYER}, PCA{PCA_K})\nAll words")
    _panel(axes[0, 1], pat_hi_neu_all, pat_lo_neu_all, "B  Neural (spikes)\nAll words")
    _panel(axes[1, 0], pat_hi_sem_mid, pat_lo_sem_mid, f"C  Semantic ({MODEL}, layer {LAYER}, PCA{PCA_K})\nMiddle 10% SUBTLEX freq only")
    _panel(axes[1, 1], pat_hi_neu_mid, pat_lo_neu_mid, "D  Neural (spikes)\nMiddle 10% SUBTLEX freq only")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle(f"Podcast event-triggered curvature, hippocampus  ·  {MODEL} layer {LAYER}  ·  "
                 f"all words vs. SUBTLEX middle-frequency-only", fontsize=9, fontweight="bold", y=0.995)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_curvature_allwords_vs_midfreq_pca100"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
