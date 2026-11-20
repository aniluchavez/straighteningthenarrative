"""
make_event_triggered_magnitude_pca20.py
==================================
PCA(20)-reduced-trajectory variant of make_event_triggered_magnitude.py:
sources d_sem from the PCA20 clean_fig4 word table (run_clean_fig4_tests_pca20.py)
instead of the raw-embedding one -- everything else identical.

Companion to fig3_event_triggered.png: same event-triggered design
(words relative to event, high vs low surprisal, hippocampus), but the
y-axis is population-vector magnitude (displacement, d_sem / d_neu:
the step size ||x_t - x_{t-1}|| in representation space) instead of
curvature.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, "results", "clean_fig4_pca20")
FIGURES    = os.path.join(BASE_DIR, "figures")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_causal_word_table.csv")
REGION     = "hippocampus"

WINDOW    = 5
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


def event_triggered_patient(sub, d_col):
    sub = sub.sort_values("center_word_index").copy()
    idx = sub["center_word_index"].values
    d = sub[d_col].values.astype(float)
    surp = sub["surprisal"].values.astype(float)

    hi_thr = np.nanpercentile(surp, HI_PCTILE)
    lo_thr = np.nanpercentile(surp, LO_PCTILE)

    lut_d = dict(zip(idx, d))
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_d, lo_d = [], []

    for i, si in enumerate(surp):
        if not np.isfinite(si):
            continue
        is_hi = si >= hi_thr
        is_lo = si <= lo_thr
        if not (is_hi or is_lo):
            continue
        j = idx[i]
        row_d = [lut_d.get(j + k, np.nan) for k in lags]
        if np.sum(np.isfinite(row_d)) < WINDOW:
            continue
        (hi_d if is_hi else lo_d).append(row_d)

    return np.array(hi_d), np.array(lo_d)


def collect_all(wt, d_col):
    pat_hi, pat_lo = [], []
    for pid, sub in wt.groupby("pid"):
        sub = sub.dropna(subset=[d_col, "surprisal", "center_word_index"])
        if len(sub) < 50:
            continue
        hi_d, lo_d = event_triggered_patient(sub, d_col)
        if len(hi_d) < 3 or len(lo_d) < 3:
            continue
        pat_hi.append(np.nanmean(hi_d, axis=0))
        pat_lo.append(np.nanmean(lo_d, axis=0))
    return np.array(pat_hi), np.array(pat_lo)


def _panel(ax, hi, lo, title):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = hi.shape[0]

    hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
    lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")

    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 magnitude ends at event word", fontsize=9)
    ax.set_ylabel("Mean magnitude (displacement)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    hi_sem, lo_sem = collect_all(wt, "d_sem")
    hi_neu, lo_neu = collect_all(wt, "d_neu")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.76, wspace=0.38)

    _panel(axes[0], hi_sem, lo_sem, "A  Semantic magnitude")
    _panel(axes[1], hi_neu, lo_neu, "B  Neural magnitude (hippocampus)")

    fig.suptitle(
        "llama-3.1-8b  ·  event-triggered magnitude (displacement)  ·  PCA(20) semantic trajectory  ·  lag 0 uses words t-2..t",
        fontsize=9, fontweight="bold", y=0.96)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig3_event_triggered_magnitude_pca20"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        if ext == "svg":
            txt = open(path).read().replace("TeX Gyre Termes", "Times New Roman")
            open(path, "w").write(txt)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
