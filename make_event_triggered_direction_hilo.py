"""
make_event_triggered_direction_hilo.py
=======================================
Direction companion to fig3_event_triggered{,_magnitude}: same event-triggered
design (words relative to event, hippocampus), but instead of one hi-vs-lo
"angle between" line, plots high- and low-surprisal as two separate lines
like the curvature/magnitude figures.

Per patient, per space (semantic E / neural Y_neu):
  mu           = mean direction (session-wide, unit-normalized steps, both
                 conditions pooled) -- a fixed per-patient reference axis.
  alignment[j] = cos_angle(step[j], mu)   for the step arriving at word j.

Then for each lag and condition (hi/lo), alignment is averaged over that
condition's words, giving "how aligned are this condition's steps with the
patient's own typical movement direction" -- directly comparable in the same
two-line hi/lo format as the other event-triggered panels.
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
LAYER    = 5  # matches LOO-selected layer for all 15 patients (geometry_llama-3.1-8b_hippocampus.json)
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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def patient_lag_alignment(surp, n_words, steps):
    """steps[k] = X[k+1]-X[k]. Returns dict lag -> {'hi': [...align...], 'lo': [...align...]}."""
    unit_steps = [unit(v) for v in steps]
    valid = [i for i, u in enumerate(unit_steps) if u is not None]
    if len(valid) < 5:
        return None
    mu = np.mean([unit_steps[i] for i in valid], axis=0)
    mu_norm = np.linalg.norm(mu)
    if mu_norm < 1e-12:
        return None
    mu_hat = mu / mu_norm

    hi_thr = np.nanpercentile(surp, HI_PCTILE)
    lo_thr = np.nanpercentile(surp, LO_PCTILE)

    lags = list(range(-WINDOW, WINDOW + 1))
    out = {lag: {"hi": [], "lo": []} for lag in lags}
    for j in range(n_words):
        sj = surp[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        for lag in lags:
            k = (j + lag) - 1  # step arriving at word (j+lag), same indexing as run_event_triggered_direction.py
            if 0 <= k < len(unit_steps) and unit_steps[k] is not None:
                align = float(np.dot(unit_steps[k], mu_hat))
                (out[lag]["hi"] if is_hi else out[lag]["lo"]).append(align)
    return out


def collect_all(space):
    """space: 'sem' or 'neu'. Returns (hi_mat, lo_mat) each [n_patients, n_lags]."""
    lags = list(range(-WINDOW, WINDOW + 1))
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
            steps = Y_neu[1:] - Y_neu[:-1]
        else:
            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            E = np.asarray(emb_mm[LAYER, :n], dtype=np.float64)
            del emb_mm
            steps = E[1:] - E[:-1]

        lag_align = patient_lag_alignment(surp, n, steps)
        if lag_align is None:
            continue

        hi_row = [np.mean(lag_align[lag]["hi"]) if len(lag_align[lag]["hi"]) >= 3 else np.nan for lag in lags]
        lo_row = [np.mean(lag_align[lag]["lo"]) if len(lag_align[lag]["lo"]) >= 3 else np.nan for lag in lags]
        if np.all(np.isnan(hi_row)) or np.all(np.isnan(lo_row)):
            continue
        pat_hi.append(hi_row)
        pat_lo.append(lo_row)

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
    ax.set_xlabel("Words relative to event\nlag 0 alignment ends at event word", fontsize=9)
    ax.set_ylabel("Mean alignment to session\nmean direction (cos)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)


def main():
    hi_sem, lo_sem = collect_all("sem")
    hi_neu, lo_neu = collect_all("neu")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.23, top=0.76, wspace=0.4)

    _panel(axes[0], hi_sem, lo_sem, "A  Semantic direction")
    _panel(axes[1], hi_neu, lo_neu, "B  Neural direction (hippocampus)")

    fig.suptitle(
        "llama-3.1-8b  ·  event-triggered direction (alignment to session mean)  ·  lag 0 uses words t-2..t",
        fontsize=9, fontweight="bold", y=0.96)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig3_event_triggered_direction_hilo"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
