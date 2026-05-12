"""
make_event_triggered_displacement.py
====================================
Event-triggered displacement around surprising words.

This figure is the displacement-only companion to the event-triggered
curvature plot.  High/low events are chosen from word-by-word surprisal
quantiles within patient.  Displacement at lag k is the step arriving at
word event+k:

    d[event+k] = ||x[event+k] - x[event+k-1]||

Therefore lag 0 is the movement into the surprising word itself, with no
future word entering that lag-0 estimate.

Run:
    conda run -n gpt2_embed python3 make_event_triggered_displacement.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_rel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (  # noqa: E402
    PATIENTS,
    fit_neural_pca,
    load_patient,
    trajectory_displacement,
)

RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(BASE_DIR, "figures", "appendix")

MODEL = "llama-3.1-8b"
REGION = "hippocampus"
WINDOW = 5
HI_PCTILE = 75
LO_PCTILE = 25
MIN_EVENTS = 3

C_HI = "#C0392B"
C_LO = "#2980B9"
C_ZERO = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
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

PANEL_LABEL_KW = dict(
    fontsize=28, fontweight="bold", fontfamily="sans-serif",
    va="top", ha="left",
)


def add_panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, **PANEL_LABEL_KW)


def zscore_finite(x):
    """Z-score finite entries, preserving NaNs."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    fin = np.isfinite(x)
    if fin.sum() < 2:
        return out
    mu = np.nanmean(x[fin])
    sd = np.nanstd(x[fin])
    if sd <= 1e-12:
        return out
    out[fin] = (x[fin] - mu) / sd
    return out


def pstar(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def patient_event_traces(data, layer):
    """Return high/low semantic and neural event-triggered displacement rows."""
    n = data["n_words"]
    surp = np.asarray(data["surp"], dtype=float)
    lags = np.arange(-WINDOW, WINDOW + 1)

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_layer = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm

    Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    # trajectory_displacement()[i] is ||x[i+1] - x[i]||, i.e. the step
    # arriving at word index i+1.  So word t maps to displacement index t-1.
    d_sem = zscore_finite(trajectory_displacement(emb_layer))
    d_neu = zscore_finite(trajectory_displacement(Y_neu))

    surp_fin = surp[np.isfinite(surp)]
    if len(surp_fin) < 20:
        return None
    hi_thr = float(np.nanpercentile(surp_fin, HI_PCTILE))
    lo_thr = float(np.nanpercentile(surp_fin, LO_PCTILE))

    hi_sem, lo_sem = [], []
    hi_neu, lo_neu = [], []

    for event_word, s in enumerate(surp):
        if not np.isfinite(s):
            continue
        is_hi = s >= hi_thr
        is_lo = s <= lo_thr
        if not (is_hi or is_lo):
            continue

        row_sem = []
        row_neu = []
        for lag in lags:
            target_word = event_word + lag
            disp_idx = target_word - 1
            if 0 <= disp_idx < len(d_sem):
                row_sem.append(d_sem[disp_idx])
                row_neu.append(d_neu[disp_idx])
            else:
                row_sem.append(np.nan)
                row_neu.append(np.nan)

        row_sem = np.asarray(row_sem, dtype=float)
        row_neu = np.asarray(row_neu, dtype=float)
        paired = np.isfinite(row_sem) & np.isfinite(row_neu)
        if not paired[WINDOW] or paired.sum() < WINDOW:
            continue

        if is_hi:
            hi_sem.append(row_sem)
            hi_neu.append(row_neu)
        else:
            lo_sem.append(row_sem)
            lo_neu.append(row_neu)

    if len(hi_sem) < MIN_EVENTS or len(lo_sem) < MIN_EVENTS:
        return None
    return (
        np.asarray(hi_sem),
        np.asarray(lo_sem),
        np.asarray(hi_neu),
        np.asarray(lo_neu),
        hi_thr,
        lo_thr,
    )


def collect_patient_means():
    json_path = os.path.join(RESULTS, f"geometry_{MODEL}_{REGION}.json")
    records = json.load(open(json_path))

    sem_hi, sem_lo = [], []
    neu_hi, neu_lo = [], []
    counts = []

    for rec in records:
        pid = rec["pid"]
        preg = PATIENTS.get(pid, {})
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, MODEL)
        if data is None:
            continue
        traces = patient_event_traces(data, int(rec["loo_layer_neu"]))
        if traces is None:
            continue
        hi_s, lo_s, hi_n, lo_n, hi_thr, lo_thr = traces
        sem_hi.append(np.nanmean(hi_s, axis=0))
        sem_lo.append(np.nanmean(lo_s, axis=0))
        neu_hi.append(np.nanmean(hi_n, axis=0))
        neu_lo.append(np.nanmean(lo_n, axis=0))
        counts.append(dict(
            pid=pid,
            layer=int(rec["loo_layer_neu"]),
            n_hi=int(len(hi_s)),
            n_lo=int(len(lo_s)),
            hi_thr=hi_thr,
            lo_thr=lo_thr,
        ))
        print(
            f"  {pid}: layer={int(rec['loo_layer_neu'])}, "
            f"n_hi={len(hi_s)}, n_lo={len(lo_s)}"
        )

    return (
        np.asarray(sem_hi),
        np.asarray(sem_lo),
        np.asarray(neu_hi),
        np.asarray(neu_lo),
        counts,
    )


def draw_panel(ax, hi, lo, title):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = hi.shape[0]

    for mat, color, label in [
        (hi, C_HI, "High surprisal"),
        (lo, C_LO, "Low surprisal"),
    ]:
        mu = np.nanmean(mat, axis=0)
        se = scipy_sem(mat, axis=0, nan_policy="omit")
        ax.plot(lags, mu, color=color, lw=2.1, marker="o", ms=3.7, label=label)
        ax.fill_between(lags, mu - se, mu + se, color=color, alpha=0.15)

    lag0 = WINDOW
    if n_pat >= 2:
        _, p = ttest_rel(hi[:, lag0], lo[:, lag0])
        sig = pstar(p)
        if sig and sig != "n.s.":
            ymin, ymax = ax.get_ylim()
            ax.text(
                0,
                ymax - 0.08 * (ymax - ymin),
                sig,
                ha="center",
                va="top",
                fontsize=11,
                color="#333333",
                fontweight="bold",
            )

    ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.45)
    ax.axhline(0, color=C_ZERO, lw=0.7, ls=":", alpha=0.55)
    ax.set_xticks(lags)
    ax.set_xlabel("Words relative to event (lag 0 = surprising word)")
    ax.set_ylabel("Arriving-step displacement (z-score within patient)")
    ax.set_title(title, loc="left", pad=6)
    ax.text(
        0.01,
        0.97,
        f"$N={n_pat}$ patients",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#555555",
        style="italic",
    )
    ax.legend(loc="upper left", framealpha=0.9, frameon=True, edgecolor="#CCCCCC")


def main():
    sem_hi, sem_lo, neu_hi, neu_lo, counts = collect_patient_means()
    if len(counts) == 0:
        raise RuntimeError("No patients had enough high/low displacement events.")

    fig = plt.figure(figsize=(12.0, 4.8))
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        wspace=0.36,
        left=0.07,
        right=0.98,
        bottom=0.15,
        top=0.84,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    draw_panel(ax_a, sem_hi, sem_lo, "Semantic trajectory")
    draw_panel(ax_b, neu_hi, neu_lo, "Neural trajectory  (hippocampus)")
    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")

    ymin = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0])
    ymax = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
    for ax in (ax_a, ax_b):
        ax.set_ylim(ymin, ymax)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_event_triggered_displacement"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        if ext == "svg":
            with open(path, "r") as fh:
                txt = fh.read().replace("TeX Gyre Termes", "Times New Roman")
            with open(path, "w") as fh:
                fh.write(txt)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
