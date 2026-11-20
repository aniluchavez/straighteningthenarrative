"""
make_podcast_event_triggered.py
=================================
Same event-triggered hi/lo-surprisal design as the convo geometry_paper
figures (fig3_event_triggered / _magnitude / _direction_hilo / _norm), run
on the podcast_comparison dataset instead. Neural-only (hippocampus) --
no semantic panel, since no LOO GPT2-large layer selection has been done
for this dataset yet.

Produces one 4-panel figure: curvature, displacement magnitude, direction
alignment, state norm -- each with hi (top-25% surprisal) vs lo (bottom-25%)
lines, words -5..+5 relative to the event word.
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


def unit_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-12] = np.nan
    return V / n


def event_windows(values_by_index, surp, n_words, index_offset=0):
    """values_by_index: array whose index i corresponds to word (i+index_offset).
    Returns (hi_rows, lo_rows), each [n_events, len(lags)], using surp at the word
    the value 'arrives at' for the event-defining condition."""
    finite = np.isfinite(surp)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(surp[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(surp[finite], LO_PCTILE)

    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values_by_index)
    for j in range(n_words):
        sj = surp[j]
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


def collect_all(metric):
    """metric in {'curvature','magnitude','direction','norm'}."""
    pat_hi, pat_lo = [], []
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        surp = data["surp"]
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue

        if metric == "norm":
            values = np.linalg.norm(Y_neu, axis=1)          # index i = word i
            offset = 0
        else:
            steps = Y_neu[1:] - Y_neu[:-1]                   # index k = steps[k] = Y[k+1]-Y[k]
            if len(steps) < 4:
                continue
            if metric == "magnitude":
                values = np.linalg.norm(steps[1:], axis=1)   # index i -> word i+2 (arriving)
                offset = 2
            elif metric == "curvature":
                values = trajectory_curvature(Y_neu)          # index i -> word i+2
                offset = 2
            elif metric == "direction":
                unit_steps = unit_rows(steps)
                mu = np.nanmean(unit_steps, axis=0)
                mu_norm = np.linalg.norm(mu)
                if mu_norm < 1e-12:
                    continue
                mu_hat = mu / mu_norm
                values = unit_steps[1:] @ mu_hat               # index i -> word i+2
                offset = 2
            else:
                raise ValueError(metric)

        hi_rows, lo_rows = event_windows(values, surp, n, index_offset=offset)
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))

    return np.array(pat_hi), np.array(pat_lo)


def _panel(ax, hi, lo, title, ylabel):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = hi.shape[0]

    hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
    lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")

    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10, pad=3)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)


def main():
    hi_c, lo_c = collect_all("curvature")
    hi_m, lo_m = collect_all("magnitude")
    hi_d, lo_d = collect_all("direction")
    hi_n, lo_n = collect_all("norm")

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.22, top=0.72, wspace=0.42)

    _panel(axes[0], hi_c, lo_c, "A  Curvature", "Mean curvature (°)")
    _panel(axes[1], hi_m, lo_m, "B  Displacement magnitude", "Mean magnitude")
    _panel(axes[2], hi_d, lo_d, "C  Direction alignment", "Mean cos(step, session mean)")
    _panel(axes[3], hi_n, lo_n, "D  State norm", r"Mean $||Y_i||$")

    fig.suptitle("Podcast (single-narrator) · event-triggered neural geometry, hippocampus, gpt2-large spikes",
                 fontsize=9.5, fontweight="bold", y=0.95)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_event_triggered"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
