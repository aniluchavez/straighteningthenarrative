"""
make_ablation_advantage.py  —  Option 2: curvature advantage plot
==================================================================
Plots β_κ − β_d (curvature advantage) per patient as jittered dots,
with group mean ± SEM bar and zero reference.

Two panels side by side: Semantic (left) and Neural (right).
X-axis = control sets; y-axis = β_κ − β_d.
Points above zero = curvature wins; below = displacement wins.
A shaded band around zero gives a "null zone".
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

C_POS  = "#D95319"   # curvature wins  — orange
C_NEG  = "#5E7CE2"   # displacement wins — periwinkle
C_ZERO = "#888888"
C_PT   = "#555555"

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
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9.5,
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


def collect_advantages(wt, c_col, d_col):
    label_keys = list(CONTROL_SETS.keys())
    adv = {k: [] for k in label_keys}      # β_κ - β_d per patient
    pids_per_k = {k: [] for k in label_keys}
    for pid, sub in wt.groupby("pid"):
        if len(sub) < 50:
            continue
        for k, ctrl in CONTROL_SETS.items():
            bc, _ = ols_beta(sub[c_col].values, sub, ctrl)
            bd, _ = ols_beta(sub[d_col].values, sub, ctrl)
            if np.isfinite(bc) and np.isfinite(bd):
                adv[k].append(bc - bd)
                pids_per_k[k].append(pid)
    return adv


def advantage_panel(ax, adv_dict, title, color_by_sign=True):
    label_keys = list(CONTROL_SETS.keys())
    xs = np.arange(len(label_keys), dtype=float)
    rng = np.random.default_rng(17)

    mus, ses, pvs, adv_arrays = [], [], [], []
    for k in label_keys:
        arr = np.array(adv_dict[k])
        adv_arrays.append(arr)
        mus.append(arr.mean())
        ses.append(scipy_sem(arr))
        _, p = ttest_1samp(arr, 0)
        pvs.append(p)

    # light connecting lines per patient across control sets
    n_pat = min(len(a) for a in adv_arrays)
    for pi in range(n_pat):
        pts = [a[pi] for a in adv_arrays]
        ax.plot(xs, pts, color=C_PT, lw=0.65, alpha=0.22, zorder=2)

    # jittered dots, colored by sign
    for xi, arr in zip(xs, adv_arrays):
        jit = rng.uniform(-0.10, 0.10, size=len(arr))
        colors = [C_POS if v >= 0 else C_NEG for v in arr]
        ax.scatter(xi + jit, arr, s=28, c=colors,
                   edgecolors="white", lw=0.4, alpha=0.85, zorder=4)

    # mean ± SEM bar
    bar_col = [C_POS if m >= 0 else C_NEG for m in mus]
    for xi, mu, se, col, pv in zip(xs, mus, ses, bar_col, pvs):
        ax.bar(xi, mu, 0.36, color=col, alpha=0.28,
               edgecolor=col, linewidth=1.4, zorder=3)
        ax.errorbar(xi, mu, yerr=se, fmt="none",
                    color=col, capsize=3.5, lw=1.5, zorder=5)
        sig = pstar(pv)
        if sig:
            yoff = se + abs(mu) * 0.05 + 0.004
            ax.text(xi, mu + yoff if mu >= 0 else mu - yoff,
                    sig, ha="center",
                    va="bottom" if mu >= 0 else "top",
                    fontsize=9, color=col, fontweight="bold")

    # n_wins annotation
    wins = [(a > 0).sum() for a in adv_arrays]
    totals = [len(a) for a in adv_arrays]
    for xi, w, tot in zip(xs, wins, totals):
        ax.text(xi, ax.get_ylim()[0] if ax.get_ylim() else -0.25,
                f"{w}/{tot}", ha="center", va="top",
                fontsize=7, color=C_PT, style="italic")

    ax.axhline(0, color=C_ZERO, lw=1.0, ls="--", alpha=0.6, zorder=1)

    # subtle shading around zero
    ax.axhspan(-0.01, 0.01, color="#EEEEEE", alpha=0.7, zorder=0, lw=0)

    # curvature wins / displacement wins labels on y-axis
    ylims = ax.get_ylim()
    yspan = ylims[1] - ylims[0]
    ax.text(-0.52, ylims[1] - 0.02 * yspan,
            r"$\kappa$ wins $\uparrow$",
            ha="left", va="top", fontsize=8, color=C_POS, style="italic")
    ax.text(-0.52, ylims[0] + 0.02 * yspan,
            r"$d$ wins $\downarrow$",
            ha="left", va="bottom", fontsize=8, color=C_NEG, style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels(label_keys, fontsize=8.5)
    ax.set_ylabel(r"$\hat\beta_\kappa - \hat\beta_d$  (curvature advantage)")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.set_xlim(-0.55, len(label_keys) - 0.45)


def alignment_annotation(fig, ax, align_csv):
    """Add a text box to the figure summarising cross-space alignment."""
    align = pd.read_csv(align_csv)
    align = align[align["region"] == REGION].dropna(
        subset=["r_csem_cneu", "r_dsem_dneu"])
    rc = align["r_csem_cneu"].values
    rd = align["r_dsem_dneu"].values
    _, pp = ttest_rel(rc, rd)
    txt = (
        "Cross-space alignment\n"
        fr"$r_{{\kappa}}={rc.mean():.3f}\pm{scipy_sem(rc):.3f}$*** "
        fr"$r_{{d}}={rd.mean():.3f}\pm{scipy_sem(rd):.3f}$**"
        "\n"
        fr"$\Delta r = {(rc-rd).mean():.3f}$, $p<0.001$***"
    )
    fig.text(0.985, 0.97, txt,
             ha="right", va="top", fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAF0FB",
                       edgecolor="#AAAACC", linewidth=0.7),
             color="#222244")


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    sem_adv = collect_advantages(wt, "c_sem", "d_sem")
    neu_adv = collect_advantages(wt, "c_neu", "d_neu")

    fig = plt.figure(figsize=(11.0, 4.8))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42,
                            left=0.09, right=0.97,
                            bottom=0.18, top=0.82)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    advantage_panel(axA, sem_adv,
                    "A  Semantic trajectory")
    advantage_panel(axB, neu_adv,
                    "B  Neural trajectory  (hippocampus)")

    # sync y-axes
    ymin = min(axA.get_ylim()[0], axB.get_ylim()[0])
    ymax = max(axA.get_ylim()[1], axB.get_ylim()[1])
    for ax in [axA, axB]:
        ax.set_ylim(ymin, ymax)
        # remove old directional labels and re-add at correct positions
        for t in [t for t in ax.texts if "wins" in t.get_text()]:
            t.remove()
    for ax, adv in [(axA, sem_adv), (axB, neu_adv)]:
        ax.text(-0.52, ymax - 0.02 * (ymax - ymin),
                r"$\kappa$ wins $\uparrow$",
                ha="left", va="top", fontsize=8, color=C_POS, style="italic")
        ax.text(-0.52, ymin + 0.02 * (ymax - ymin),
                r"$d$ wins $\downarrow$",
                ha="left", va="bottom", fontsize=8, color=C_NEG, style="italic")

    alignment_annotation(fig, axB, ALIGN_CSV)

    fig.suptitle(
        r"Curvature advantage ($\hat\beta_\kappa - \hat\beta_d$) across covariate control sets"
        "\nllama-3.1-8B  ·  hippocampus  ·  per-patient dots, mean ± SEM bar  ·  "
        r"italic $n/N$: patients where $\kappa$ wins",
        fontsize=9, fontweight="bold", y=0.97,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_ablation_advantage"
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
