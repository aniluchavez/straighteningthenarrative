"""
run_curvature_at_surprisal.py
==============================
Tests whether curvature or displacement is elevated AT surprising words
compared to expected words.

Uses the lag=0 word table: geometry is centered at word t (uses t-1, t, t+1),
surprisal is measured at the same word t.  No prediction — this is a concurrent
association test asking "are high-surprisal words more curved?".

Two analyses:
  1. Quintile profiles  — mean curvature / displacement per surprisal bin,
                          normalised to Q1, with paired t-test Q5 vs Q1.
  2. Regression test    — surprisal → curvature (and displacement) controlling
                          for sent_pos, sentence_start, turn_boundary,
                          is_speaker, word_len, log_freq, dist_turn.
                          Compares beta_surprisal in the c vs d model.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem, ttest_rel, ttest_1samp, t as t_dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results", "clean_fig4")
FIGURES  = os.path.join(BASE_DIR, "figures")

WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_word_table.csv")
REGION     = "hippocampus"
N_BINS     = 5

C_CURV = "#C9922A"   # gold — curvature
C_DISP = "#B5446E"   # rose  — displacement
C_PT   = "#333333"
C_ZERO = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


# ── OLS helper ────────────────────────────────────────────────────────────────

def ols_beta(y: np.ndarray, X_named: Dict[str, np.ndarray],
             standardize: bool = True) -> Dict:
    names = list(X_named)
    y  = np.asarray(y, dtype=float)
    Xs = [np.asarray(X_named[n], dtype=float) for n in names]
    mask = np.isfinite(y)
    for x in Xs:
        mask &= np.isfinite(x)
    y  = y[mask]
    Xs = [x[mask] for x in Xs]
    n  = len(y)
    k  = len(names) + 1          # +1 for intercept
    if n < k + 2:
        return {nm: float("nan") for nm in names}, {nm: float("nan") for nm in names}
    if standardize:
        y  = (y  - y.mean())  / (y.std()  + 1e-12)
        Xs = [(x - x.mean()) / (x.std() + 1e-12) for x in Xs]
    Xm = np.column_stack([np.ones(n)] + Xs)
    b, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ b
    ss_res = float(np.dot(resid, resid))
    s2 = ss_res / max(n - k, 1)
    try:
        cov = s2 * np.linalg.inv(Xm.T @ Xm)
        se  = np.sqrt(np.diag(cov))
        tv  = b / np.where(se > 0, se, np.nan)
        pv  = 2 * t_dist.sf(np.abs(tv), df=max(n - k, 1))
    except np.linalg.LinAlgError:
        pv = np.full(k, np.nan)
    betas = {nm: float(b[i+1]) for i, nm in enumerate(names)}
    pvals = {nm: float(pv[i+1]) for i, nm in enumerate(names)}
    return betas, pvals


CONTROLS = ["sent_pos", "sentence_start", "turn_boundary",
            "is_speaker", "word_len", "log_freq", "dist_turn"]


# ── per-patient analysis ───────────────────────────────────────────────────────

def analyse_patient(sub: pd.DataFrame) -> Dict:
    sub = sub.copy()

    # ── quintile profiles (normalised to Q1) ──────────────────────────────────
    sub["surp_bin"] = pd.qcut(sub["surprisal"], N_BINS,
                              labels=False, duplicates="drop")
    bin_rows = []
    for b in range(N_BINS):
        bsub = sub[sub["surp_bin"] == b]
        bin_rows.append({
            "bin": b,
            "c_sem": bsub["c_sem"].mean(),
            "d_sem": bsub["d_sem"].mean(),
            "c_neu": bsub["c_neu"].mean(),
            "d_neu": bsub["d_neu"].mean(),
        })
    bdf = pd.DataFrame(bin_rows).set_index("bin")

    # normalise to Q1
    q1 = bdf.loc[0]
    norm = bdf / q1

    # Q5 - Q1 raw difference
    diff_csem = bdf.loc[N_BINS-1, "c_sem"] - bdf.loc[0, "c_sem"]
    diff_dsem = bdf.loc[N_BINS-1, "d_sem"] - bdf.loc[0, "d_sem"]
    diff_cneu = bdf.loc[N_BINS-1, "c_neu"] - bdf.loc[0, "c_neu"]
    diff_dneu = bdf.loc[N_BINS-1, "d_neu"] - bdf.loc[0, "d_neu"]

    # ── regression: surprisal → curvature / displacement ──────────────────────
    predictors = {"surprisal": sub["surprisal"],
                  **{c: sub[c] for c in CONTROLS if c in sub.columns}}

    b_csem, p_csem = ols_beta(sub["c_sem"], predictors)
    b_dsem, p_dsem = ols_beta(sub["d_sem"], predictors)
    b_cneu, p_cneu = ols_beta(sub["c_neu"], predictors)
    b_dneu, p_dneu = ols_beta(sub["d_neu"], predictors)

    return dict(
        pid    = sub["pid"].iloc[0],
        region = sub["region"].iloc[0],
        n      = len(sub),
        # quintile profiles (normalised)
        norm_csem = norm["c_sem"].tolist(),
        norm_dsem = norm["d_sem"].tolist(),
        norm_cneu = norm["c_neu"].tolist(),
        norm_dneu = norm["d_neu"].tolist(),
        # Q5 - Q1 raw differences
        diff_csem = diff_csem,
        diff_dsem = diff_dsem,
        diff_cneu = diff_cneu,
        diff_dneu = diff_dneu,
        # regression betas: how much does surprisal predict each geometry measure?
        beta_surp_to_csem = b_csem["surprisal"],
        beta_surp_to_dsem = b_dsem["surprisal"],
        beta_surp_to_cneu = b_cneu["surprisal"],
        beta_surp_to_dneu = b_dneu["surprisal"],
        p_surp_to_csem = p_csem["surprisal"],
        p_surp_to_dsem = p_dsem["surprisal"],
        p_surp_to_cneu = p_cneu["surprisal"],
        p_surp_to_dneu = p_dneu["surprisal"],
    )


# ── group stats ───────────────────────────────────────────────────────────────

def pstar(p: float) -> str:
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def group_stats(rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)
    print("\n=== Regression: surprisal → geometry (std beta, controlling for nuisance) ===")
    for col, label in [
        ("beta_surp_to_csem", "c_sem"),
        ("beta_surp_to_dsem", "d_sem"),
        ("beta_surp_to_cneu", "c_neu"),
        ("beta_surp_to_dneu", "d_neu"),
    ]:
        vals = df[col].dropna().to_numpy()
        t1, p1 = ttest_1samp(vals, 0)
        print(f"  {label:8s}: mean β={vals.mean():.4f}  t={t1:.2f}  p={p1:.4f}  {pstar(p1)}")

    print("\n=== Paired test: c vs d (surprisal → curvature vs displacement) ===")
    sem_c = df["beta_surp_to_csem"].dropna()
    sem_d = df["beta_surp_to_dsem"].dropna()
    idx   = sem_c.index.intersection(sem_d.index)
    t2, p2 = ttest_rel(sem_c[idx], sem_d[idx])
    print(f"  semantic:  c_sem vs d_sem  t={t2:.2f}  p={p2:.4f}  {pstar(p2)}")

    neu_c = df["beta_surp_to_cneu"].dropna()
    neu_d = df["beta_surp_to_dneu"].dropna()
    idx   = neu_c.index.intersection(neu_d.index)
    t3, p3 = ttest_rel(neu_c[idx], neu_d[idx])
    print(f"  neural:    c_neu vs d_neu  t={t3:.2f}  p={p3:.4f}  {pstar(p3)}")

    print("\n=== Q5 - Q1 raw difference (is geometry elevated at surprising words?) ===")
    for col, label in [
        ("diff_csem", "c_sem"), ("diff_dsem", "d_sem"),
        ("diff_cneu", "c_neu"), ("diff_dneu", "d_neu"),
    ]:
        vals = df[col].dropna().to_numpy()
        t1, p1 = ttest_1samp(vals, 0)
        print(f"  {label:8s}: mean Δ={vals.mean():.4f}  t={t1:.2f}  p={p1:.4f}  {pstar(p1)}")

    print("\n=== Paired: Q5-Q1 curvature vs displacement ===")
    t4, p4 = ttest_rel(df["diff_csem"].dropna(), df["diff_dsem"].dropna())
    print(f"  semantic:  c_sem vs d_sem  t={t4:.2f}  p={p4:.4f}  {pstar(p4)}")
    t5, p5 = ttest_rel(df["diff_cneu"].dropna(), df["diff_dneu"].dropna())
    print(f"  neural:    c_neu vs d_neu  t={t5:.2f}  p={p5:.4f}  {pstar(p5)}")


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(10.5, 8.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.52)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    xs_q     = np.arange(1, N_BINS + 1)
    xlabels  = ["Low\n(Q1)", "Q2", "Q3", "Q4", "High\n(Q5)"]

    specs = [
        ("norm_csem", "norm_dsem",
         "A  Semantic geometry at surprising words",
         r"Curvature / displacement (norm. to Q1)", "c_sem", "d_sem"),
        ("norm_cneu", "norm_dneu",
         "B  Neural geometry at surprising words",
         r"Curvature / displacement (norm. to Q1)", "c_neu", "d_neu"),
        ("beta_surp_to_csem", "beta_surp_to_dsem",
         "C  Regression: surprisal → semantic geometry",
         "Std. β (surprisal predicting geometry)", "c_sem", "d_sem"),
        ("beta_surp_to_cneu", "beta_surp_to_dneu",
         "D  Regression: surprisal → neural geometry",
         "Std. β (surprisal predicting geometry)", "c_neu", "d_neu"),
    ]

    rng = np.random.default_rng(42)

    for ax, (key_c, key_d, title, ylabel, lbl_c, lbl_d) in zip(axes, specs):
        is_quint = key_c.startswith("norm_")

        if is_quint:
            mat_c = np.array([r[key_c] for r in rows if len(r[key_c]) == N_BINS])
            mat_d = np.array([r[key_d] for r in rows if len(r[key_d]) == N_BINS])

            mu_c, se_c = mat_c.mean(0), scipy_sem(mat_c, axis=0)
            mu_d, se_d = mat_d.mean(0), scipy_sem(mat_d, axis=0)

            ax.plot(xs_q, mu_c, color=C_CURV, lw=2.2, marker="o", ms=5,
                    label=rf"$\kappa_{{sem}}$ ({lbl_c})", zorder=3)
            ax.fill_between(xs_q, mu_c - se_c, mu_c + se_c,
                            color=C_CURV, alpha=0.15, zorder=2)
            ax.plot(xs_q, mu_d, color=C_DISP, lw=2.2, marker="o", ms=5,
                    label=rf"$d_{{sem}}$ ({lbl_d})", zorder=3)
            ax.fill_between(xs_q, mu_d - se_d, mu_d + se_d,
                            color=C_DISP, alpha=0.15, zorder=2)

            # Q5 annotation
            for mu, se, col in [(mu_c, se_c, C_CURV), (mu_d, se_d, C_DISP)]:
                t1, p1 = ttest_1samp(mat_c[:, -1] if col == C_CURV else mat_d[:, -1], 1.0)
                ax.text(5.12, mu[-1], f"{mu[-1]:.3f}x {pstar(p1)}",
                        color=col, fontsize=7, va="center", ha="left")

            ax.axhline(1.0, color=C_ZERO, lw=0.8, ls="--", alpha=0.6, zorder=1)
            ax.set_xticks(xs_q)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel("Surprisal quintile")
            ax.set_xlim(0.75, 5.9)

            # paired test label
            t_diff, p_diff = ttest_rel(mat_c[:, -1], mat_d[:, -1])
            ax.text(0.97, 0.03,
                    f"Q5: c vs d  {pstar(p_diff)} (p={p_diff:.3f})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="#444444")

        else:
            vals_c = df[key_c].dropna().to_numpy()
            vals_d = df[key_d].dropna().to_numpy()
            mu_c, se_c = vals_c.mean(), scipy_sem(vals_c)
            mu_d, se_d = vals_d.mean(), scipy_sem(vals_d)

            xs = np.array([0.0, 0.6])
            for xi, mu, se, col, vals in [
                (xs[0], mu_c, se_c, C_CURV, vals_c),
                (xs[1], mu_d, se_d, C_DISP, vals_d),
            ]:
                ax.bar(xi, mu, 0.38, color=col, alpha=0.85, zorder=3)
                ax.errorbar(xi, mu, yerr=se, fmt="none", color="k",
                            capsize=4, lw=1.2, zorder=4)
                jit = rng.uniform(-0.09, 0.09, size=len(vals))
                ax.scatter(xi + jit, vals, s=16, color=col,
                           edgecolors="white", lw=0.4, alpha=0.80, zorder=5)
                _, p1 = ttest_1samp(vals, 0)
                ax.text(xi, mu + se + 0.002, pstar(p1),
                        ha="center", va="bottom", fontsize=8, color=col)

            t_diff, p_diff = ttest_rel(vals_c, vals_d)
            ax.axhline(0, color=C_ZERO, lw=0.8, ls="--", alpha=0.5, zorder=1)
            ax.set_xticks(xs)
            ax.set_xticklabels([rf"$\kappa$ ({lbl_c})", f"$d$ ({lbl_d})"],
                               fontsize=9)
            ax.set_xlim(-0.35, 1.0)
            ax.text(0.97, 0.97,
                    f"c vs d: {pstar(p_diff)} (p={p_diff:.3f})",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="#444444")

        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel)

    fig.suptitle(
        "Curvature and displacement AT surprising words  ·  llama-3.1-8b  ·  hippocampus\n"
        "Geometry centered at t (words t−1, t, t+1), surprisal at t  [concurrent, no prediction]",
        fontsize=9, fontweight="bold", y=0.995,
    )

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_at_surprisal"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    rows = []
    for pid, sub in wt.groupby("pid"):
        sub = sub.dropna(subset=["surprisal", "c_sem", "d_sem", "c_neu", "d_neu"])
        if len(sub) < 50:
            continue
        rows.append(analyse_patient(sub))
        print(f"  {pid}  n={len(sub)}")

    group_stats(rows)
    make_figure(rows)


if __name__ == "__main__":
    main()
