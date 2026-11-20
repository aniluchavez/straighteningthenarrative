"""
make_podcast_event_triggered_freq.py
=====================================
make_podcast_event_triggered.py's curvature panel, but the hi/lo event split
is by word LOG-FREQUENCY (data["log_freq"], same get_log_freq() used
throughout the pipeline) instead of surprisal: "most frequent word" (top 25%
log_freq) vs "most infrequent word" (bottom 25% log_freq), words -5..+5
relative to the event word.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

from podcast_loader import PODCAST_PATIENTS, load_podcast_patient
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import fit_neural_pca, trajectory_curvature

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES  = os.path.join(BASE_DIR, "figures")
WINDOW   = 5
HI_PCTILE = 75
LO_PCTILE = 25

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


def event_windows(values_by_index, split_var, n_words, index_offset=0):
    """values_by_index: array whose index i corresponds to word (i+index_offset).
    Returns (hi_rows, lo_rows), each [n_events, len(lags)], using split_var at
    the word the value 'arrives at' for the event-defining condition. hi =
    top 25% of split_var (most frequent word), lo = bottom 25% (most
    infrequent word)."""
    finite = np.isfinite(split_var)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)

    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values_by_index)
    for j in range(n_words):
        sj = split_var[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = []
        for k in lags:
            idx = (j + k) - index_offset
            row.append(values_by_index[idx] if 0 <= idx < m else np.nan)
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def collect_curvature_by_freq():
    pat_hi, pat_lo = [], []
    for short in PODCAST_PATIENTS:
        # model_tag="gpt2" (not the default lab-provided raw embeddings) so the
        # 4 extra-source patients (no PODCAST_ROOT/embeddings/ dir) are included
        # too -- curvature here is neural-PCA-only, embeddings are never used.
        data = load_podcast_patient(short, model_tag="gpt2", context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        log_freq = data["log_freq"][:n]
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue

        steps = Y_neu[1:] - Y_neu[:-1]
        if len(steps) < 4:
            continue
        values = trajectory_curvature(Y_neu)   # index i -> word i+2 (arriving)
        offset = 2

        hi_rows, lo_rows = event_windows(values, log_freq, n, index_offset=offset)
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))

    return np.array(pat_hi), np.array(pat_lo)


def main():
    hi_c, lo_c = collect_curvature_by_freq()

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.8))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.82)

    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = hi_c.shape[0]
    hi_mu, hi_se = np.nanmean(hi_c, axis=0), scipy_sem(hi_c, axis=0, nan_policy="omit")
    lo_mu, lo_se = np.nanmean(lo_c, axis=0), scipy_sem(lo_c, axis=0, nan_policy="omit")

    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label="Most frequent word (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label="Most infrequent word (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=9.5)
    ax.set_title(f"Curvature\n$n$ = {n_pat} patients", loc="left", fontsize=10, pad=3)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)

    fig.suptitle("Podcast · event-triggered neural curvature by word frequency, hippocampus",
                 fontsize=9, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_event_triggered_freq"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
