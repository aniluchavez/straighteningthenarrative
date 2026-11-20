"""
make_ablation_forest.py  —  Option 1: forest / coefficient plot
================================================================
Two stacked panels (Semantic / Neural).
Each panel: one row per control set, two dots (κ orange, d periwinkle)
with SEM bars and a zero reference line.
Cross-space alignment as a small inset in the top-right.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_1samp, ttest_rel, t as t_dist

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, "results", "clean_fig4")
FIGURES    = os.path.join(BASE_DIR, "figures")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_word_table.csv")
ALIGN_CSV  = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag1_patient_summary.csv")
REGION     = "hippocampus"

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_ZERO = "#888888"

CONTROL_SETS = {
    "Raw\n(no controls)":       [],
    "+ sentence\nposition":     ["sent_pos", "sentence_start"],
    "+ turn &\nspeaker":        ["sent_pos", "sentence_start",
                                 "turn_boundary", "is_speaker", "dist_turn"],
    "+ length &\nfrequency":    ["sent_pos", "sentence_start",
                                 "turn_boundary", "is_speaker", "dist_turn",
                                 "word_len", "log_freq"],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
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


def ols_beta(y, sub, controls):
    cols = ["surprisal"] + controls
    available = [c for c in cols if c in sub.columns]
    X = sub[available].copy()
    mask = np.isfinite(y) & X.notna().all(axis=1)
    y  = np.asarray(y, dtype=float)[mask]
    X  = X[mask].to_numpy(float)
    n  = len(y)
    if n < len(available) + 3:
        return float("nan"), float("nan")
    y  = (y - y.mean()) / (y.std() + 1e-12)
    X  = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xm = np.column_stack([np.ones(n), X])
    b, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ b
    s2 = np.dot(resid, resid) / max(n - Xm.shape[1], 1)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
        tv = b / np.where(se > 0, se, np.nan)
        pv = 2 * t_dist.sf(np.abs(tv), df=max(n - Xm.shape[1], 1))
    except np.linalg.LinAlgError:
        pv = np.full(len(b), np.nan)
    return float(b[1]), float(pv[1])


def collect(wt, c_col, d_col):
    label_keys = list(CONTROL_SETS.keys())
    bc_all = {k: [] for k in label_keys}
    bd_all = {k: [] for k in label_keys}
    for pid, sub in wt.groupby("pid"):
        if len(sub) < 50:
            continue
        for k, ctrl in CONTROL_SETS.items():
            bc, _ = ols_beta(sub[c_col].values, sub, ctrl)
            bd, _ = ols_beta(sub[d_col].values, sub, ctrl)
            if np.isfinite(bc): bc_all[k].append(bc)
            if np.isfinite(bd): bd_all[k].append(bd)
    return bc_all, bd_all


def forest_panel(ax, bc_all, bd_all, title, show_legend=False):
    label_keys = list(CONTROL_SETS.keys())
    n = len(label_keys)
    ys = np.arange(n, dtype=float)[::-1]   # top = raw, bottom = most controls
    offset = 0.18

    for yi, k in zip(ys, label_keys):
        for vals, col, off, lbl in [
            (bc_all[k], C_CURV, +offset, r"Curvature ($\kappa$)"),
            (bd_all[k], C_DISP, -offset, "Displacement ($d$)"),
        ]:
            arr = np.array(vals)
            mu  = arr.mean()
            se  = scipy_sem(arr)
            _, p = ttest_1samp(arr, 0)

            ax.errorbar(mu, yi + off, xerr=se,
                        fmt="o", color=col, ms=7, lw=1.8,
                        capsize=3.5, capthick=1.4,
                        ecolor=col, zorder=4,
                        label=lbl if yi == ys[0] else "_nolegend_")

            # significance label to the right of the CI
            xlab = mu + se + 0.004
            sig  = pstar(p)
            if sig:
                ax.text(xlab, yi + off, sig,
                        va="center", ha="left",
                        fontsize=8, color=col, fontweight="bold")

        # paired test bracket between κ and d dots
        bc_arr = np.array(bc_all[k])
        bd_arr = np.array(bd_all[k])
        shared = min(len(bc_arr), len(bd_arr))
        if shared >= 3:
            _, pp = ttest_rel(bc_arr[:shared], bd_arr[:shared])
            mu_c = bc_arr.mean()
            mu_d = bd_arr.mean()
            pp_sig = pstar(pp)
            if pp_sig and pp_sig != "n.s.":
                xmid = (mu_c + mu_d) / 2
                ax.annotate("", xy=(mu_c, yi + offset), xytext=(mu_d, yi - offset),
                            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6))

    ax.axvline(0, color=C_ZERO, lw=0.9, ls="--", alpha=0.55, zorder=1)

    # shaded rows for readability
    for yi, k in zip(ys, label_keys):
        if int(round(yi)) % 2 == 0:
            ax.axhspan(yi - 0.45, yi + 0.45, color="#F5F5F5", zorder=0, lw=0)

    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace("\n", " ") for k in label_keys])
    ax.set_xlabel(r"Std.\ $\hat\beta$ (surprisal $\rightarrow$ geometry)")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.set_ylim(-0.6, n - 0.4)

    if show_legend:
        ax.legend(loc="lower right", framealpha=0.9,
                  edgecolor="#CCCCCC", handlelength=0.8, handletextpad=0.5)


def alignment_inset(fig, rect):
    """Small alignment bar inset [left, bottom, width, height] in fig coords."""
    ax = fig.add_axes(rect)
    align = pd.read_csv(ALIGN_CSV)
    align = align[align["region"] == REGION].dropna(
        subset=["r_csem_cneu", "r_dsem_dneu"])
    rc = align["r_csem_cneu"].values
    rd = align["r_dsem_dneu"].values

    for xi, vals, col, lbl in [
        (0, rc, C_CURV, r"$\kappa$"),
        (1, rd, C_DISP, "$d$"),
    ]:
        mu, se = vals.mean(), scipy_sem(vals)
        _, p = ttest_1samp(vals, 0)
        ax.bar(xi, mu, 0.5, color=col, alpha=0.30,
               edgecolor=col, linewidth=1.4)
        ax.errorbar(xi, mu, yerr=se, fmt="none",
                    color=col, capsize=3, lw=1.4)
        ax.text(xi, mu + se + 0.001, pstar(p),
                ha="center", va="bottom",
                fontsize=8, color=col, fontweight="bold")

    _, pp = ttest_rel(rc, rd)
    ymax = max(rc.mean(), rd.mean()) + max(scipy_sem(rc), scipy_sem(rd))
    ytop = ymax + 0.006
    ax.plot([0, 0, 1, 1], [rc.mean() + scipy_sem(rc), ytop,
                             ytop, rd.mean() + scipy_sem(rd)],
            color="#444444", lw=0.8)
    ax.text(0.5, ytop + 0.001, pstar(pp),
            ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$\kappa$", "$d$"], fontsize=9)
    ax.set_ylabel("Sem $\\leftrightarrow$ neu $r$", fontsize=7.5)
    ax.set_title("Cross-space\nalignment", fontsize=8, fontweight="bold",
                 loc="center", pad=3)
    ax.set_xlim(-0.5, 1.5)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    sem_c, sem_d = collect(wt, "c_sem", "d_sem")
    neu_c, neu_d = collect(wt, "c_neu", "d_neu")

    fig = plt.figure(figsize=(8.0, 6.2))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.52,
                            left=0.26, right=0.88,
                            bottom=0.10, top=0.88)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    forest_panel(axA, sem_c, sem_d,
                 "A  Semantic trajectory", show_legend=True)
    forest_panel(axB, neu_c, neu_d,
                 "B  Neural trajectory  (hippocampus)", show_legend=False)

    # sync x-axes
    xmin = min(axA.get_xlim()[0], axB.get_xlim()[0])
    xmax = max(axA.get_xlim()[1], axB.get_xlim()[1])
    for ax in [axA, axB]:
        ax.set_xlim(xmin, xmax)

    # inset: cross-space alignment
    alignment_inset(fig, [0.90, 0.55, 0.09, 0.30])

    fig.suptitle(
        r"Curvature vs.\ displacement as predictors of surprisal — ablation study"
        "\nllama-3.1-8B  ·  hippocampus  ·  geometry at word $t$, surprisal at $t$",
        fontsize=9.5, fontweight="bold", y=0.96,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_ablation_forest"
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
