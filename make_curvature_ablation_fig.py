"""
make_curvature_ablation_fig.py
================================
Three-panel figure telling the full curvature story:

  A  Cross-space alignment: r(c_sem,c_neu) vs r(d_sem,d_neu)
     — curvature is the shared geometric signal, displacement is not

  B  Semantic ablation: surprisal → c_sem vs d_sem across control models
     — curvature predicts surprisal; discourse structure explains part of it
       but a residual survives all controls; displacement is flat throughout

  C  Neural ablation: surprisal → c_neu vs d_neu across control models
     — both contribute in neural space; noisier picture
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_1samp, ttest_rel, t as t_dist, f as f_dist

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, "results", "clean_fig4")
FIGURES    = os.path.join(BASE_DIR, "figures")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_word_table.csv")
ALIGN_CSV  = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag1_patient_summary.csv")
REGION     = "hippocampus"

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_ZERO = "#888888"
C_PT   = "#555555"

CONTROL_SETS = {
    "Raw\n(no controls)":        [],
    "+ sentence\nposition":      ["sent_pos", "sentence_start"],
    "+ turn &\nspeaker":         ["sent_pos", "sentence_start",
                                  "turn_boundary", "is_speaker", "dist_turn"],
    "+ length\n& frequency":     ["sent_pos", "sentence_start",
                                  "turn_boundary", "is_speaker", "dist_turn",
                                  "word_len", "log_freq"],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 8.5,
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
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def ols_beta(y: np.ndarray, sub: pd.DataFrame, controls: list[str]):
    """Std beta of surprisal predicting y, controlling for `controls`."""
    cols = ["surprisal"] + controls
    available = [c for c in cols if c in sub.columns]
    X = sub[available].copy()
    mask = np.isfinite(y) & X.notna().all(axis=1)
    y  = np.asarray(y, dtype=float)[mask]
    X  = X[mask].to_numpy(float)
    n  = len(y)
    if n < len(available) + 3:
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


def delta_r2_and_ftest(y: np.ndarray, sub: pd.DataFrame, controls: list[str]):
    """ΔR² and incremental F-test p-value for adding surprisal to the controls-only model."""
    cols_full = ["surprisal"] + controls
    avail_full = [c for c in cols_full if c in sub.columns]
    X_full = sub[avail_full].copy()
    mask = np.isfinite(y) & X_full.notna().all(axis=1)
    y_  = np.asarray(y, dtype=float)[mask]
    X_f = X_full[mask].to_numpy(float)
    n   = len(y_)
    k   = len(avail_full)
    if n < k + 3:
        return float("nan"), float("nan")

    y_  = (y_  - y_.mean())  / (y_.std()  + 1e-12)
    X_f = (X_f - X_f.mean(0)) / (X_f.std(0) + 1e-12)
    ss_tot = np.dot(y_, y_)  # y is mean-centered

    # full model
    Xm_full      = np.column_stack([np.ones(n), X_f])
    b_full, _, _, _ = np.linalg.lstsq(Xm_full, y_, rcond=None)
    ss_res_full  = np.sum((y_ - Xm_full @ b_full) ** 2)
    r2_full      = 1 - ss_res_full / (ss_tot + 1e-12)

    # reduced model (controls only, no surprisal)
    if controls:
        avail_ctrl = [c for c in controls if c in sub.columns]
        X_c = sub[avail_ctrl][mask].to_numpy(float)
        X_c = (X_c - X_c.mean(0)) / (X_c.std(0) + 1e-12)
        Xm_ctrl = np.column_stack([np.ones(n), X_c])
    else:
        Xm_ctrl = np.ones((n, 1))

    b_ctrl, _, _, _ = np.linalg.lstsq(Xm_ctrl, y_, rcond=None)
    ss_res_ctrl = np.sum((y_ - Xm_ctrl @ b_ctrl) ** 2)
    r2_ctrl     = 1 - ss_res_ctrl / (ss_tot + 1e-12)

    delta_r2 = r2_full - r2_ctrl

    df2 = n - k - 1
    if df2 <= 0 or ss_res_full <= 0:
        return float(delta_r2), float("nan")
    F   = (delta_r2 / 1.0) / ((1 - r2_full) / df2)
    pv  = f_dist.sf(F, 1, df2)
    return float(delta_r2), float(pv)


# ── Panel A: cross-space alignment ───────────────────────────────────────────

def panel_alignment(ax):
    df = pd.read_csv(ALIGN_CSV)
    df = df[df["region"] == REGION].dropna(
        subset=["r_csem_cneu", "r_dsem_dneu"])

    vals_c = df["r_csem_cneu"].to_numpy()
    vals_d = df["r_dsem_dneu"].to_numpy()
    mu_c, se_c = vals_c.mean(), scipy_sem(vals_c)
    mu_d, se_d = vals_d.mean(), scipy_sem(vals_d)
    _, p_c    = ttest_1samp(vals_c, 0)
    _, p_d    = ttest_1samp(vals_d, 0)
    _, p_diff = ttest_rel(vals_c, vals_d)

    rng = np.random.default_rng(0)
    xs  = np.array([0.0, 1.0])

    for bc, bd in zip(vals_c, vals_d):
        ax.plot(xs, [bc, bd], color=C_PT, lw=0.7, alpha=0.22, zorder=2)

    for xi, vals, col, pv in [
        (xs[0], vals_c, C_CURV, p_c),
        (xs[1], vals_d, C_DISP, p_d),
    ]:
        jit = rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(xi + jit, vals, s=32, color=col,
                   edgecolors="white", lw=0.5, alpha=0.85, zorder=4)
        ax.bar(xi, vals.mean(), 0.32, color=col, alpha=0.28,
               edgecolor=col, linewidth=1.4, zorder=3)
        ax.errorbar(xi, vals.mean(), yerr=scipy_sem(vals),
                    fmt="none", color=col, capsize=3.5, lw=1.4, zorder=5)
        ax.text(xi, vals.mean() + scipy_sem(vals) + 0.002,
                pstar(pv), ha="center", va="bottom",
                fontsize=9, color=col, fontweight="bold")

    # bracket
    ymax  = max(vals_c.max(), vals_d.max())
    span  = ymax - min(vals_c.min(), vals_d.min())
    ytop  = ymax + 0.22 * span
    ax.plot([xs[0], xs[0], xs[1], xs[1]],
            [mu_c + se_c, ytop, ytop, mu_d + se_d],
            color="#444444", lw=0.9, zorder=6)
    ax.text(xs.mean(), ytop + 0.01 * span,
            f"{pstar(p_diff)} (p={p_diff:.3f})",
            ha="center", va="bottom", fontsize=8, color="#333333")

    ax.axhline(0, color=C_ZERO, lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"Curvature ($\kappa$)", "Displacement ($d$)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Semantic ↔ neural Pearson $r$")
    ax.set_title("A  Cross-space geometry alignment",
                 loc="left", fontweight="bold", pad=5)


# ── Panels B & C: ablation ────────────────────────────────────────────────────

def panel_ablation(ax, wt, c_col, d_col, title):
    label_keys = list(CONTROL_SETS.keys())
    xs = np.arange(len(label_keys), dtype=float)

    betas_c = {k: [] for k in label_keys}
    betas_d = {k: [] for k in label_keys}

    for pid, sub in wt.groupby("pid"):
        sub = sub.copy()
        if len(sub) < 50:
            continue
        for k, ctrl in CONTROL_SETS.items():
            bc, _ = ols_beta(sub[c_col].values, sub, ctrl)
            bd, _ = ols_beta(sub[d_col].values, sub, ctrl)
            if np.isfinite(bc): betas_c[k].append(bc)
            if np.isfinite(bd): betas_d[k].append(bd)

    offset = 0.13
    for vals_dict, col, lbl, off in [
        (betas_c, C_CURV, r"Curvature ($\kappa$)",  -offset),
        (betas_d, C_DISP, "Displacement ($d$)", +offset),
    ]:
        mus = [np.mean(vals_dict[k]) for k in label_keys]
        ses = [scipy_sem(vals_dict[k]) for k in label_keys]
        pvs = [ttest_1samp(vals_dict[k], 0)[1] for k in label_keys]

        ax.plot(xs + off, mus, color=col, lw=1.8,
                marker="o", ms=5.5, zorder=4, label=lbl)
        ax.fill_between(xs + off,
                        np.array(mus) - np.array(ses),
                        np.array(mus) + np.array(ses),
                        color=col, alpha=0.15, zorder=3)

        # significance stars above last point
        for xi, mu, se, pv in zip(xs + off, mus, ses, pvs):
            if pv < 0.05:
                ax.text(xi, mu + se + 0.003, pstar(pv),
                        ha="center", va="bottom",
                        fontsize=7.5, color=col)

    # shade the "discourse" region
    ax.axvspan(0.5, 2.5, color="#F0F0F0", alpha=0.55, zorder=1,
               label="_nolegend_")
    ax.text(1.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.001,
            "discourse controls", ha="center", va="bottom",
            fontsize=7, color="#999999", style="italic")

    ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.5, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(label_keys, fontsize=8)
    ax.set_ylabel(r"Std. $\beta$ (surprisal → geometry)")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.legend(loc="upper right", framealpha=0.85, frameon=True,
              edgecolor="#CCCCCC", handlelength=2.0)
    ax.set_xlim(-0.55, len(label_keys) - 0.45)


def panel_delta_r2(ax, wt, c_col, d_col, title):
    """ΔR² (surprisal added on top of controls) with incremental F-test stars."""
    label_keys = list(CONTROL_SETS.keys())
    xs = np.arange(len(label_keys), dtype=float)

    dr2_c  = {k: [] for k in label_keys}
    dr2_d  = {k: [] for k in label_keys}
    pval_c = {k: [] for k in label_keys}
    pval_d = {k: [] for k in label_keys}

    for pid, sub in wt.groupby("pid"):
        sub = sub.copy()
        if len(sub) < 50:
            continue
        for k, ctrl in CONTROL_SETS.items():
            dc, pc = delta_r2_and_ftest(sub[c_col].values, sub, ctrl)
            dd, pd_ = delta_r2_and_ftest(sub[d_col].values, sub, ctrl)
            if np.isfinite(dc): dr2_c[k].append(dc)
            if np.isfinite(pc): pval_c[k].append(pc)
            if np.isfinite(dd): dr2_d[k].append(dd)
            if np.isfinite(pd_): pval_d[k].append(pd_)

    offset = 0.13
    for vals_dict, pv_dict, col, lbl, off in [
        (dr2_c, pval_c, C_CURV, r"Curvature ($\kappa$)",  -offset),
        (dr2_d, pval_d, C_DISP, "Displacement ($d$)", +offset),
    ]:
        mus = [np.mean(vals_dict[k]) for k in label_keys]
        ses = [scipy_sem(vals_dict[k]) for k in label_keys]
        # median within-subject F-test p-value across participants
        pvs = [np.median(pv_dict[k]) if pv_dict[k] else float("nan")
               for k in label_keys]

        ax.plot(xs + off, mus, color=col, lw=1.8,
                marker="o", ms=5.5, zorder=4, label=lbl)
        ax.fill_between(xs + off,
                        np.array(mus) - np.array(ses),
                        np.array(mus) + np.array(ses),
                        color=col, alpha=0.15, zorder=3)

        for xi, mu, se, pv in zip(xs + off, mus, ses, pvs):
            if pv < 0.05:
                ax.text(xi, mu + se + 0.0003, pstar(pv),
                        ha="center", va="bottom",
                        fontsize=7.5, color=col)

    ax.axvspan(0.5, 2.5, color="#F0F0F0", alpha=0.55, zorder=1)
    ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.5, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(label_keys, fontsize=8)
    ax.set_ylabel(r"$\Delta R^2$ (surprisal unique contribution)")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.legend(loc="upper right", framealpha=0.85, frameon=True,
              edgecolor="#CCCCCC", handlelength=2.0)
    ax.set_xlim(-0.55, len(label_keys) - 0.45)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.42)

    panel_delta_r2(axA, wt, "c_sem", "d_sem",
                   r"A  Semantic trajectory — $\Delta R^2$")
    panel_delta_r2(axB, wt, "c_neu", "d_neu",
                   r"B  Neural trajectory — $\Delta R^2$  (hippocampus)")

    fig.suptitle(
        r"Surprisal unique contribution ($\Delta R^2$) to curvature  ·  llama-3.1-8b  ·  hippocampus",
        fontsize=10, fontweight="bold", y=0.98,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_ablation"
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
