"""
make_curvature_at_surprisal_fig.py
====================================
Publication figure: curvature and displacement AT surprising words.

Layout (2 rows × 2 cols):
  A  Semantic quintile profiles     B  Neural quintile profiles
  C  Semantic Q5-Q1 patient bars    D  Neural  Q5-Q1 patient bars

Panels A/B: group mean ± SEM with per-patient light traces.
Panels C/D: paired per-patient dots + bar, with significance bracket.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import sem as scipy_sem, ttest_rel, ttest_1samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results", "clean_fig4")
FIGURES  = os.path.join(BASE_DIR, "figures")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_word_table.csv")
REGION     = "hippocampus"
N_BINS     = 5

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_ZERO = "#888888"
C_PT   = "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def pstar(p: float) -> str:
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def get_patient_quintiles(sub: pd.DataFrame):
    """Return (norm_c, norm_d) arrays of shape (N_BINS,) for one patient."""
    sub = sub.copy()
    sub["bin"] = pd.qcut(sub["surprisal"], N_BINS, labels=False, duplicates="drop")
    rows_c, rows_d = [], []
    for b in range(N_BINS):
        bsub = sub[sub["bin"] == b]
        rows_c.append(bsub["c"].mean())
        rows_d.append(bsub["d"].mean())
    arr_c = np.array(rows_c)
    arr_d = np.array(rows_d)
    if arr_c[0] == 0 or arr_d[0] == 0:
        return None, None
    return arr_c / arr_c[0], arr_d / arr_d[0]


def panel_quintile(ax, mat_c, mat_d, title, ylabel, show_xlabel=True):
    xs = np.arange(1, N_BINS + 1)
    xlabels = ["Low\n(Q1)", "Q2", "Q3", "Q4", "High\n(Q5)"]

    mu_c, se_c = mat_c.mean(0), scipy_sem(mat_c, axis=0)
    mu_d, se_d = mat_d.mean(0), scipy_sem(mat_d, axis=0)

    # per-patient traces (light)
    for row in mat_c:
        ax.plot(xs, row, color=C_CURV, lw=0.6, alpha=0.18, zorder=1)
    for row in mat_d:
        ax.plot(xs, row, color=C_DISP, lw=0.6, alpha=0.18, zorder=1)

    # group mean ± SEM
    ax.plot(xs, mu_c, color=C_CURV, lw=2.4, marker="o", ms=5.5,
            label=r"Curvature ($\kappa$)", zorder=4)
    ax.fill_between(xs, mu_c - se_c, mu_c + se_c,
                    color=C_CURV, alpha=0.18, zorder=3)

    ax.plot(xs, mu_d, color=C_DISP, lw=2.4, marker="o", ms=5.5,
            label="Displacement ($d$)", zorder=4)
    ax.fill_between(xs, mu_d - se_d, mu_d + se_d,
                    color=C_DISP, alpha=0.18, zorder=3)

    # right-hand value labels
    x_end = 5.12
    _, p_c = ttest_1samp(mat_c[:, -1], 1.0)
    _, p_d = ttest_1samp(mat_d[:, -1], 1.0)
    ax.text(x_end, mu_c[-1], f"{mu_c[-1]:.3f}× {pstar(p_c)}",
            color=C_CURV, fontsize=7.5, va="center", ha="left")
    ax.text(x_end, mu_d[-1], f"{mu_d[-1]:.3f}× {pstar(p_d)}",
            color=C_DISP, fontsize=7.5, va="center", ha="left")

    # Q5 curvature vs displacement significance bracket
    _, p_diff = ttest_rel(mat_c[:, -1], mat_d[:, -1])
    ymax = max(mu_c[-1] + se_c[-1], mu_d[-1] + se_d[-1])
    yrange = max(np.nanmax(mat_c), np.nanmax(mat_d)) - 1.0
    ytop = ymax + 0.06 * yrange
    bx = [4.85, 4.85, 5.15, 5.15]
    by = [mu_c[-1], ytop, ytop, mu_d[-1]]
    ax.plot(bx, by, color="#444444", lw=0.9, zorder=5)
    ax.text(5.0, ytop + 0.005 * yrange, pstar(p_diff),
            ha="center", va="bottom", fontsize=9, color="#222222")

    ax.axhline(1.0, color=C_ZERO, lw=0.8, ls="--", alpha=0.55, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels if show_xlabel else [])
    ax.set_xlim(0.72, 5.9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.legend(loc="upper left", framealpha=0.85, frameon=True,
              edgecolor="#CCCCCC", handlelength=2.0)


def panel_q5q1_bars(ax, vals_c, vals_d, title, ylabel):
    """Paired bar panel: Q5-Q1 per patient with individual dots."""
    rng = np.random.default_rng(7)
    mu_c, se_c = vals_c.mean(), scipy_sem(vals_c)
    mu_d, se_d = vals_d.mean(), scipy_sem(vals_d)

    _, p_c    = ttest_1samp(vals_c, 0)
    _, p_d    = ttest_1samp(vals_d, 0)
    _, p_diff = ttest_rel(vals_c, vals_d)

    xs = np.array([0.0, 0.7])
    for xi, mu, se, col, vals, pv in [
        (xs[0], mu_c, se_c, C_CURV, vals_c, p_c),
        (xs[1], mu_d, se_d, C_DISP, vals_d, p_d),
    ]:
        ax.bar(xi, mu, 0.42, color=col, alpha=0.88,
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(xi, mu, yerr=se, fmt="none",
                    color="k", capsize=3.5, lw=1.2, zorder=4)
        jit = rng.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(xi + jit, vals, s=18, color=col,
                   edgecolors="white", lw=0.4, alpha=0.78, zorder=5)
        ax.text(xi, mu + se + abs(mu) * 0.04, pstar(pv),
                ha="center", va="bottom", fontsize=9, color=col)

    # paired lines
    for vc, vd in zip(vals_c, vals_d):
        ax.plot(xs, [vc, vd], color=C_PT, lw=0.55, alpha=0.22, zorder=2)

    # significance bracket between bars
    ymax = max(vals_c.max(), vals_d.max())
    yrange = ymax - min(vals_c.min(), vals_d.min())
    ytop = ymax + 0.18 * yrange
    bx = [xs[0], xs[0], xs[1], xs[1]]
    by = [mu_c + se_c, ytop, ytop, mu_d + se_d]
    ax.plot(bx, by, color="#444444", lw=0.9, zorder=6)
    ax.text(xs.mean(), ytop + 0.02 * yrange,
            f"{pstar(p_diff)}  (p={p_diff:.3f})",
            ha="center", va="bottom", fontsize=8.5, color="#222222")

    ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.5, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"Curvature ($\kappa$)", "Displacement ($d$)"],
                       fontsize=9.5)
    ax.set_xlim(-0.38, 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    sem_mats_c, sem_mats_d = [], []
    neu_mats_c, neu_mats_d = [], []

    for pid, sub in wt.groupby("pid"):
        sub = sub.dropna(subset=["surprisal", "c_sem", "d_sem", "c_neu", "d_neu"])
        if len(sub) < 50:
            continue

        sub_sem = sub[["surprisal", "c_sem", "d_sem"]].rename(
            columns={"c_sem": "c", "d_sem": "d"})
        nc, nd = get_patient_quintiles(sub_sem)
        if nc is not None:
            sem_mats_c.append(nc)
            sem_mats_d.append(nd)

        sub_neu = sub[["surprisal", "c_neu", "d_neu"]].rename(
            columns={"c_neu": "c", "d_neu": "d"})
        nc, nd = get_patient_quintiles(sub_neu)
        if nc is not None:
            neu_mats_c.append(nc)
            neu_mats_d.append(nd)

    sem_c = np.array(sem_mats_c)
    sem_d = np.array(sem_mats_d)
    neu_c = np.array(neu_mats_c)
    neu_d = np.array(neu_mats_d)

    # Q5-Q1 per patient
    sem_diff_c = sem_c[:, -1] - sem_c[:, 0]
    sem_diff_d = sem_d[:, -1] - sem_d[:, 0]
    neu_diff_c = neu_c[:, -1] - neu_c[:, 0]
    neu_diff_d = neu_d[:, -1] - neu_d[:, 0]

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9.6, 7.8))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            wspace=0.40, hspace=0.52,
                            left=0.09, right=0.88,
                            bottom=0.10, top=0.88)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    panel_quintile(
        axA, sem_c, sem_d,
        "A  Semantic trajectory",
        "Geometry (norm. to Q1)",
    )
    panel_quintile(
        axB, neu_c, neu_d,
        "B  Neural trajectory  (hippocampus)",
        "Geometry (norm. to Q1)",
    )
    panel_q5q1_bars(
        axC, sem_diff_c, sem_diff_d,
        "C  Q5 − Q1 elevation  (semantic)",
        "High − low surprisal (norm. units)",
    )
    panel_q5q1_bars(
        axD, neu_diff_c, neu_diff_d,
        "D  Q5 − Q1 elevation  (neural)",
        "High − low surprisal (norm. units)",
    )

    for ax in [axA, axB]:
        ax.set_xlabel("Surprisal quintile")

    fig.suptitle(
        "Curvature is more elevated at surprising words than displacement\n"
        r"(geometry centered at word $t$, surprisal at $t$  ·  llama-3.1-8b  ·  hippocampus)",
        fontsize=10, fontweight="bold", y=0.955,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_at_surprisal_clean"
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
