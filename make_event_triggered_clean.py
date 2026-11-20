"""
make_event_triggered_clean.py
==============================
Event-triggered geometry around surprising words, built from the
backward-only lag=0 word table.

Two panels — Semantic (A) and Neural (B) — each showing curvature
AND displacement for high vs low surprisal events.

Measures are z-scored within patient within measure before averaging,
so curvature and displacement are on a common scale in each panel.

The lag=0 causal table right-aligns geometry to the event word: curvature uses
words t-2..t and displacement is the arriving step ||x_t - x_{t-1}||.

High surprisal = top 25%, low surprisal = bottom 25% (per patient).
Window = ±5 words relative to event.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_rel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, "results", "clean_fig4")
FIGURES    = os.path.join(BASE_DIR, "figures")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_causal_word_table.csv")
REGION     = "hippocampus"

WINDOW      = 5        # ± words around event
HI_PCTILE   = 75
LO_PCTILE   = 25

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_ZERO = "#888888"

LS_HI = "-"
LS_LO = "--"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
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


def pstar(p):
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def event_triggered_patient(sub, c_col, d_col):
    """
    Returns (hi_c, lo_c, hi_d, lo_d) each shape (n_events, 2*WINDOW+1),
    values z-scored within patient within measure.
    """
    sub = sub.sort_values("center_word_index").copy()
    idx = sub["center_word_index"].values
    c   = sub[c_col].values.astype(float)
    d   = sub[d_col].values.astype(float)
    surp = sub["surprisal"].values.astype(float)

    # z-score each measure across all words for this patient
    c = (c - np.nanmean(c)) / (np.nanstd(c) + 1e-12)
    d = (d - np.nanmean(d)) / (np.nanstd(d) + 1e-12)

    hi_thr = np.nanpercentile(surp, HI_PCTILE)
    lo_thr = np.nanpercentile(surp, LO_PCTILE)

    # build lookup: word_index → (c, d)
    lut_c = dict(zip(idx, c))
    lut_d = dict(zip(idx, d))

    lags = list(range(-WINDOW, WINDOW + 1))
    hi_c, lo_c, hi_d, lo_d = [], [], [], []

    for i, si in enumerate(surp):
        if not np.isfinite(si):
            continue
        is_hi = si >= hi_thr
        is_lo = si <= lo_thr
        if not (is_hi or is_lo):
            continue
        j = idx[i]
        row_c = [lut_c.get(j + k, np.nan) for k in lags]
        row_d = [lut_d.get(j + k, np.nan) for k in lags]
        # only include event if lag-0 and most neighbors exist
        if np.sum(np.isfinite(row_c)) < WINDOW:
            continue
        if is_hi:
            hi_c.append(row_c)
            hi_d.append(row_d)
        else:
            lo_c.append(row_c)
            lo_d.append(row_d)

    return (np.array(hi_c), np.array(lo_c),
            np.array(hi_d), np.array(lo_d))


def collect_all(wt, c_col, d_col):
    """Collect per-patient event-triggered traces; return patient-mean matrices."""
    pat_hi_c, pat_lo_c = [], []
    pat_hi_d, pat_lo_d = [], []

    for pid, sub in wt.groupby("pid"):
        sub = sub.dropna(subset=[c_col, d_col, "surprisal", "center_word_index"])
        if len(sub) < 50:
            continue
        hi_c, lo_c, hi_d, lo_d = event_triggered_patient(sub, c_col, d_col)
        if len(hi_c) < 3 or len(lo_c) < 3:
            continue
        # patient mean across events
        pat_hi_c.append(np.nanmean(hi_c, axis=0))
        pat_lo_c.append(np.nanmean(lo_c, axis=0))
        pat_hi_d.append(np.nanmean(hi_d, axis=0))
        pat_lo_d.append(np.nanmean(lo_d, axis=0))

    return (np.array(pat_hi_c), np.array(pat_lo_c),
            np.array(pat_hi_d), np.array(pat_lo_d))


def draw_panel(ax, hi_c, lo_c, hi_d, lo_d, title):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = hi_c.shape[0]

    for mat, col, ls, lbl in [
        (hi_c, C_CURV, LS_HI, r"Curvature ($\kappa$) — high surp."),
        (lo_c, C_CURV, LS_LO, r"Curvature ($\kappa$) — low surp."),
        (hi_d, C_DISP, LS_HI, r"Displacement ($d$) — high surp."),
        (lo_d, C_DISP, LS_LO, r"Displacement ($d$) — low surp."),
    ]:
        mu = np.nanmean(mat, axis=0)
        se = scipy_sem(mat, axis=0, nan_policy="omit")
        ax.plot(lags, mu, color=col, ls=ls, lw=2.0,
                marker="o", ms=3.5, zorder=4, label=lbl)
        ax.fill_between(lags, mu - se, mu + se,
                        color=col, alpha=0.12, zorder=3)

    # significance at lag 0: curvature hi vs lo
    lag0 = WINDOW  # index of lag=0
    _, p_c = ttest_rel(hi_c[:, lag0], lo_c[:, lag0])
    _, p_d = ttest_rel(hi_d[:, lag0], lo_d[:, lag0])

    # annotate lag-0 difference
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5
    for col, p, off in [(C_CURV, p_c, 0.08), (C_DISP, p_d, -0.08)]:
        sig = pstar(p)
        if sig and sig != "n.s.":
            ax.text(0 + off, ymax * 0.92, sig,
                    ha="center", va="top",
                    fontsize=10, color=col, fontweight="bold")

    ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.45, zorder=5,
               label="_nolegend_")
    ax.axhline(0, color=C_ZERO, lw=0.7, ls=":", alpha=0.5, zorder=1)

    ax.set_xticks(lags)
    ax.set_xlabel("Words relative to event (lag 0 = surprising word)")
    ax.set_ylabel("Geometry (z-score within patient)")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.text(0.01, 0.97, f"$N={n_pat}$ patients",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#555555", style="italic")

    ax.legend(loc="upper left", framealpha=0.88, frameon=True,
              edgecolor="#CCCCCC", handlelength=2.2, fontsize=8.5)


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    # semantic
    s_hi_c, s_lo_c, s_hi_d, s_lo_d = collect_all(wt, "c_sem", "d_sem")
    # neural
    n_hi_c, n_lo_c, n_hi_d, n_lo_d = collect_all(wt, "c_neu", "d_neu")

    fig = plt.figure(figsize=(12.0, 4.8))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40,
                            left=0.07, right=0.98,
                            bottom=0.14, top=0.84)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    draw_panel(axA, s_hi_c, s_lo_c, s_hi_d, s_lo_d,
               "A  Semantic trajectory")
    draw_panel(axB, n_hi_c, n_lo_c, n_hi_d, n_lo_d,
               "B  Neural trajectory  (hippocampus)")

    # sync y-axes
    ymin = min(axA.get_ylim()[0], axB.get_ylim()[0])
    ymax = max(axA.get_ylim()[1], axB.get_ylim()[1])
    for ax in [axA, axB]:
        ax.set_ylim(ymin, ymax)

    fig.suptitle(
        "Event-triggered geometry at surprising words  ·  llama-3.1-8B  ·  hippocampus\n"
        "High surp. = top 25%, low surp. = bottom 25% (per patient)  ·  "
        "geometry z-scored within patient",
        fontsize=9.5, fontweight="bold", y=0.97,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_event_triggered_clean"
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
