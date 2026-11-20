"""
make_surp_geometry_regression_fig.py
======================================
Per-patient regression: surprisal → curvature vs displacement.

OLS with surprisal as predictor of c_sem / d_sem / c_neu / d_neu,
controlling for sent_pos, sentence_start, turn_boundary, is_speaker,
word_len, log_freq, dist_turn.  All variables z-scored → betas comparable.

Two panels:
  A  Semantic trajectory  (c_sem vs d_sem)
  B  Neural trajectory    (c_neu vs d_neu)

Per-patient paired dots + connecting lines; group mean ± SEM bar;
significance of each beta vs 0 and paired c vs d test.
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
REGION     = "hippocampus"

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_BOTH = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10,
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

CONTROLS = ["sent_pos", "sentence_start", "turn_boundary",
            "is_speaker", "word_len", "log_freq", "dist_turn"]


def pstar(p: float) -> str:
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def ols_beta_surp(y: np.ndarray, sub: pd.DataFrame):
    X = sub[["surprisal"] + CONTROLS].copy()
    mask = np.isfinite(y) & X.notna().all(axis=1)
    y  = np.asarray(y, dtype=float)[mask]
    X  = X[mask].to_numpy(float)
    n  = len(y)
    if n < len(CONTROLS) + 4:
        return float("nan"), float("nan")
    y  = (y  - y.mean())  / (y.std()  + 1e-12)
    X  = (X  - X.mean(0)) / (X.std(0) + 1e-12)
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


def draw_panel(ax, betas_c, betas_d, title, ylabel):
    betas_c = np.array(betas_c)
    betas_d = np.array(betas_d)

    xs = np.array([0.0, 1.0])
    mu_c, se_c = betas_c.mean(), scipy_sem(betas_c)
    mu_d, se_d = betas_d.mean(), scipy_sem(betas_d)
    _, p_c    = ttest_1samp(betas_c, 0)
    _, p_d    = ttest_1samp(betas_d, 0)
    _, p_diff = ttest_rel(betas_c, betas_d)
    n_c_wins  = int((betas_c > betas_d).sum())

    # per-patient connecting lines + dots
    for bc, bd in zip(betas_c, betas_d):
        col = C_CURV if bc >= bd else C_DISP
        ax.plot(xs, [bc, bd], color=col, lw=0.9, alpha=0.30, zorder=2)

    rng = np.random.default_rng(42)
    for xi, vals, col in [(xs[0], betas_c, C_CURV), (xs[1], betas_d, C_DISP)]:
        jit = rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(xi + jit, vals, s=38, color=col,
                   edgecolors="white", lw=0.5, alpha=0.85, zorder=4)

    # group mean bars
    bar_w = 0.32
    for xi, mu, se, col, pv in [
        (xs[0], mu_c, se_c, C_CURV, p_c),
        (xs[1], mu_d, se_d, C_DISP, p_d),
    ]:
        ax.bar(xi, mu, bar_w, color=col, alpha=0.30,
               edgecolor=col, linewidth=1.4, zorder=3)
        ax.errorbar(xi, mu, yerr=se, fmt="none",
                    color=col, capsize=4, lw=1.5, zorder=5)
        ax.text(xi, mu + se + 0.004,
                pstar(pv), ha="center", va="bottom",
                fontsize=10, color=col, fontweight="bold")

    # significance bracket between means
    yvals = np.concatenate([betas_c, betas_d])
    ymax  = np.nanmax(yvals)
    ymin  = np.nanmin(yvals)
    span  = max(ymax - ymin, 0.02)
    ytop  = ymax + 0.20 * span
    bx = [xs[0], xs[0], xs[1], xs[1]]
    by = [mu_c + se_c, ytop, ytop, mu_d + se_d]
    ax.plot(bx, by, color="#444444", lw=1.0, zorder=6)
    ax.text(xs.mean(), ytop + 0.02 * span,
            f"c vs d: {pstar(p_diff)} (p={p_diff:.3f})",
            ha="center", va="bottom", fontsize=8.5, color="#333333")

    # c > d tally
    ax.text(0.98, 0.02,
            f"κ > d in {n_c_wins}/{len(betas_c)} patients",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#555555", style="italic")

    ax.axhline(0, color="#999999", lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"Curvature ($\kappa$)", "Displacement ($d$)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION]

    sem_c, sem_d = [], []
    neu_c, neu_d = [], []

    for pid, sub in wt.groupby("pid"):
        sub = sub.dropna(subset=["surprisal","c_sem","d_sem","c_neu","d_neu"] + CONTROLS)
        if len(sub) < 50:
            continue
        bc, _ = ols_beta_surp(sub["c_sem"].values, sub)
        bd, _ = ols_beta_surp(sub["d_sem"].values, sub)
        bn, _ = ols_beta_surp(sub["c_neu"].values, sub)
        bm, _ = ols_beta_surp(sub["d_neu"].values, sub)
        if all(np.isfinite([bc, bd, bn, bm])):
            sem_c.append(bc); sem_d.append(bd)
            neu_c.append(bn); neu_d.append(bm)

    fig = plt.figure(figsize=(8.4, 4.6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42,
                            left=0.10, right=0.97,
                            bottom=0.12, top=0.84)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    draw_panel(axA, sem_c, sem_d,
               "A  Semantic trajectory",
               r"Std. $\beta$ (surprisal → geometry)")
    draw_panel(axB, neu_c, neu_d,
               "B  Neural trajectory  (hippocampus)",
               r"Std. $\beta$ (surprisal → geometry)")

    fig.suptitle(
        "Surprisal predicts curvature and displacement  ·  llama-3.1-8b\n"
        r"OLS controlling for sent. pos, turn, speaker, length, freq  ·  geometry at word $t$, surprisal at $t$",
        fontsize=9, fontweight="bold", y=0.97,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_surp_geometry_regression"
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
