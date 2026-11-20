"""
make_event_triggered_norm.py
=============================
Event-triggered population-vector NORM (not displacement): ||Y_neu[i]|| at
each word i, as a function of words relative to event, hi- vs lo-surprisal.
This is distinct from fig3_event_triggered_magnitude.py, which plots the
step size ||Y_neu[i]-Y_neu[i-1]|| (change between consecutive words).

Y_neu comes out of fit_neural_pca mean-centered across the patient's full
session (PCA centers by default), so ||Y_neu[i]|| is the distance of word
i's population state from that patient's session-average state -- i.e. how
far from "typical" the response is at that word, not an absolute firing
rate. Same for the semantic panel with the raw embedding E (not
mean-centered by construction, so semantic panel is norm around the origin
of the embedding space, not session-mean -- interpret accordingly).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

from run_geometry_analysis import PATIENTS, load_patient, fit_neural_pca

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION   = "hippocampus"
MODEL    = "llama-3.1-8b"
LAYER    = 5  # LOO-selected neural layer, same for all 15 patients
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


def event_triggered_patient(norm, surp, n_words):
    finite = np.isfinite(surp)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(surp[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(surp[finite], LO_PCTILE)

    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    for j in range(n_words):
        sj = surp[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = [norm[j + k] if 0 <= j + k < n_words else np.nan for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def collect_all(space):
    pat_hi, pat_lo = [], []
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"]

        if space == "neu":
            if data["neu_vecs"] is None:
                continue
            Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
            if Y_neu is None:
                continue
            norm = np.linalg.norm(Y_neu, axis=1)
        else:
            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            E = np.asarray(emb_mm[LAYER, :n], dtype=np.float64)
            del emb_mm
            norm = np.linalg.norm(E, axis=1)

        hi_rows, lo_rows = event_triggered_patient(norm, surp, n)
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))

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
    ax.set_xlabel("Words relative to event", fontsize=9)
    ax.set_ylabel("Mean $||$state$||$", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)


def main():
    hi_sem, lo_sem = collect_all("sem")
    hi_neu, lo_neu = collect_all("neu")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.76, wspace=0.38)

    _panel(axes[0], hi_sem, lo_sem, "A  Semantic state norm")
    _panel(axes[1], hi_neu, lo_neu, "B  Neural state norm (hippocampus)")

    fig.suptitle(
        "llama-3.1-8b  ·  event-triggered population-vector norm $||Y_i||$ (not step size)  ·  hippocampus",
        fontsize=9, fontweight="bold", y=0.96)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig3_event_triggered_norm"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
