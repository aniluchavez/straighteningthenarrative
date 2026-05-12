"""
make_ablation_table_fig.py
===========================
Publication table: surprisal → geometry ablation results.

Shows the regression formula and a 2-space (semantic / neural) × 4-control-set
table with std β ± SEM and significance for curvature and displacement.
Also includes the cross-space alignment row at the top.
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
from scipy.stats import sem as scipy_sem, ttest_1samp, ttest_rel, t as t_dist

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, "results", "clean_fig4")
FIGURES    = os.path.join(BASE_DIR, "figures")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_word_table.csv")
ALIGN_CSV  = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag1_patient_summary.csv")
REGION     = "hippocampus"

C_CURV  = "#D95319"
C_DISP  = "#5E7CE2"
C_HEAD  = "#2C2C2C"
C_BAND  = "#F5F5F5"
C_BAND2 = "#FFFFFF"

CONTROL_SETS = {
    "Raw\n(no controls)":          [],
    "$+$ sentence\nposition":      ["sent_pos", "sentence_start"],
    "$+$ turn\n$\\&$ speaker":     ["sent_pos", "sentence_start",
                                    "turn_boundary", "is_speaker", "dist_turn"],
    "$+$ length\n$\\&$ frequency": ["sent_pos", "sentence_start",
                                    "turn_boundary", "is_speaker", "dist_turn",
                                    "word_len", "log_freq"],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 10,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def pstar(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


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
        from scipy.stats import t as t_dist
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
        tv = b / np.where(se > 0, se, np.nan)
        pv = 2 * t_dist.sf(np.abs(tv), df=max(n - Xm.shape[1], 1))
    except np.linalg.LinAlgError:
        pv = np.full(len(b), np.nan)
    return float(b[1]), float(pv[1])


def collect_betas(wt):
    label_keys = list(CONTROL_SETS.keys())
    results = {}
    for space, c_col, d_col in [("Semantic", "c_sem", "d_sem"),
                                  ("Neural",   "c_neu", "d_neu")]:
        bc_all = {k: [] for k in label_keys}
        bd_all = {k: [] for k in label_keys}
        for pid, sub in wt.groupby("pid"):
            sub = sub.copy()
            if len(sub) < 50:
                continue
            for k, ctrl in CONTROL_SETS.items():
                bc, _ = ols_beta(sub[c_col].values, sub, ctrl)
                bd, _ = ols_beta(sub[d_col].values, sub, ctrl)
                if np.isfinite(bc): bc_all[k].append(bc)
                if np.isfinite(bd): bd_all[k].append(bd)
        results[space] = {"curv": bc_all, "disp": bd_all}
    return results


def fmt_cell(vals, paired_vals=None):
    """Return (beta_str, sig_str, p_paired_str) for a list of betas."""
    arr = np.array(vals)
    mu  = arr.mean()
    se  = scipy_sem(arr)
    _, p = ttest_1samp(arr, 0)
    cell = f"{mu:+.3f} ± {se:.3f}"
    sig  = pstar(p)
    if paired_vals is not None:
        parr = np.array(paired_vals)
        _, pp = ttest_rel(arr, parr)
        psig = pstar(pp)
    else:
        pp, psig = np.nan, ""
    return cell, sig, pp, psig


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    align = pd.read_csv(ALIGN_CSV)
    align = align[align["region"] == REGION].dropna(
        subset=["r_csem_cneu", "r_dsem_dneu"])

    betas = collect_betas(wt)
    label_keys = list(CONTROL_SETS.keys())

    # ── build table rows ──────────────────────────────────────────────────────
    # Columns: Control | Sem κ | Sem d | κ vs d (sem) | Neu κ | Neu d | κ vs d (neu)
    rows = []
    for k in label_keys:
        sc = betas["Semantic"]["curv"][k]
        sd = betas["Semantic"]["disp"][k]
        nc = betas["Neural"]["curv"][k]
        nd = betas["Neural"]["disp"][k]

        sc_str, sc_sig, pp_s, pp_s_sig = fmt_cell(sc, sd)
        sd_str, sd_sig, _,   _         = fmt_cell(sd)
        nc_str, nc_sig, pp_n, pp_n_sig = fmt_cell(nc, nd)
        nd_str, nd_sig, _,   _         = fmt_cell(nd)

        rows.append({
            "ctrl": k,
            "sc": sc_str, "sc_sig": sc_sig,
            "sd": sd_str, "sd_sig": sd_sig,
            "pp_s": pp_s, "pp_s_sig": pp_s_sig,
            "nc": nc_str, "nc_sig": nc_sig,
            "nd": nd_str, "nd_sig": nd_sig,
            "pp_n": pp_n, "pp_n_sig": pp_n_sig,
        })

    # cross-space alignment
    rc = align["r_csem_cneu"].values
    rd = align["r_dsem_dneu"].values
    _, p_rc = ttest_1samp(rc, 0)
    _, p_rd = ttest_1samp(rd, 0)
    _, p_rr = ttest_rel(rc, rd)
    align_row = {
        "rc":     f"{rc.mean():+.3f} ± {scipy_sem(rc):.3f}",
        "rc_sig": pstar(p_rc),
        "rd":     f"{rd.mean():+.3f} ± {scipy_sem(rd):.3f}",
        "rd_sig": pstar(p_rd),
        "pp":     p_rr,
        "pp_sig": pstar(p_rr),
    }

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.0, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965,
             "Geometry as a predictor of surprisal — ablation across control sets",
             ha="center", va="top", fontsize=12, fontweight="bold", color=C_HEAD)

    # ── regression formula box ────────────────────────────────────────────────
    formula_lines = [
        r"$\mathrm{Geometry}_{it} = \beta_0 + \beta_{\mathrm{surp}}\,\mathrm{surprisal}_{it} + \beta_k\,\mathrm{controls}_{it} + \varepsilon_{it}$",
        r"All variables z-scored within patient  $\Rightarrow$  $\hat\beta_{\mathrm{surp}}$ is a standardized effect size",
        r"Geometry $\in$ \{$\kappa_{\mathrm{sem}}$, $d_{\mathrm{sem}}$, $\kappa_{\mathrm{neu}}$, $d_{\mathrm{neu}}$\}   |   $N = 10$ patients, hippocampus, llama-3.1-8b, lag = 0",
    ]
    box_top = 0.915
    box_bot = 0.815
    rect = FancyBboxPatch((0.04, box_bot), 0.92, box_top - box_bot,
                          boxstyle="round,pad=0.01",
                          linewidth=0.8, edgecolor="#CCCCCC",
                          facecolor="#FAFAFA", zorder=2)
    ax.add_patch(rect)
    for i, line in enumerate(formula_lines):
        fig.text(0.5, box_top - 0.015 - i * 0.032,
                 line, ha="center", va="top", fontsize=9.5, color="#222222")

    # ── table geometry ────────────────────────────────────────────────────────
    tbl_top  = 0.790
    tbl_bot  = 0.055
    tbl_h    = tbl_top - tbl_bot
    n_data_rows = len(rows) + 1  # +1 for alignment row at bottom
    # +2 for header rows
    n_rows_total = n_data_rows + 2
    row_h = tbl_h / n_rows_total

    col_x = [0.04, 0.175, 0.345, 0.430, 0.515, 0.685, 0.770, 0.855]
    #         ctrl   sem_κ  sem_d  p(κvd) neu_κ  neu_d  p(κvd) n

    def row_y(r):
        return tbl_top - (r + 1) * row_h

    def cell(ax, r, c0, c1, txt, ha="center", color="#222222",
             fontsize=9, bold=False, va="center"):
        xc = (c0 + c1) / 2
        yc = row_y(r) + row_h / 2
        fw = "bold" if bold else "normal"
        ax.text(xc, yc, txt, ha=ha, va=va,
                fontsize=fontsize, color=color, fontweight=fw,
                transform=ax.transAxes)

    def hline(ax, r, lw=0.6, color="#CCCCCC"):
        y = row_y(r)
        ax.plot([col_x[0], col_x[-1]], [y, y],
                transform=ax.transAxes,
                color=color, lw=lw, clip_on=False)

    def vline(ax, x, r0, r1, lw=0.5, color="#DDDDDD"):
        y0 = row_y(r0)
        y1 = row_y(r1) + row_h
        ax.plot([x, x], [y0, y1],
                transform=ax.transAxes,
                color=color, lw=lw, clip_on=False)

    # ── header row 0: space labels ────────────────────────────────────────────
    # shading
    for r in range(n_rows_total + 1):
        bg = C_BAND if r % 2 == 0 else C_BAND2
        rect = FancyBboxPatch((col_x[0], row_y(r)), col_x[-1] - col_x[0], row_h,
                              boxstyle="square,pad=0",
                              linewidth=0, facecolor=bg,
                              transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)

    # header 0: "Semantic trajectory" and "Neural trajectory" spans
    for r in [0, 1]:
        rect = FancyBboxPatch((col_x[0], row_y(r)), col_x[-1] - col_x[0], row_h,
                              boxstyle="square,pad=0",
                              linewidth=0, facecolor="#EBEBEB",
                              transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)

    cell(ax, 0, col_x[1], col_x[4],
         r"Semantic trajectory  ($\kappa_\mathrm{sem}$,  $d_\mathrm{sem}$)",
         bold=True, fontsize=9.5, color=C_HEAD)
    cell(ax, 0, col_x[4], col_x[7],
         r"Neural trajectory  ($\kappa_\mathrm{neu}$,  $d_\mathrm{neu}$)  [hippocampus]",
         bold=True, fontsize=9.5, color=C_HEAD)

    # divider between semantic and neural
    vline(ax, col_x[4], 0, n_rows_total, lw=1.2, color="#AAAAAA")

    # header 1: column labels
    cell(ax, 1, col_x[0], col_x[1], "Control set", bold=True, ha="left",
         fontsize=9, color=C_HEAD)
    cell(ax, 1, col_x[1], col_x[2],
         r"$\hat\beta_{\kappa}$  (mean ± SEM)", bold=True, fontsize=8.5, color=C_CURV)
    cell(ax, 1, col_x[2], col_x[3],
         r"$\hat\beta_{d}$  (mean ± SEM)", bold=True, fontsize=8.5, color=C_DISP)
    cell(ax, 1, col_x[3], col_x[4],
         r"$p_{\kappa\,\mathrm{vs}\,d}$", bold=True, fontsize=8.5, color=C_HEAD)
    cell(ax, 1, col_x[4], col_x[5],
         r"$\hat\beta_{\kappa}$  (mean ± SEM)", bold=True, fontsize=8.5, color=C_CURV)
    cell(ax, 1, col_x[5], col_x[6],
         r"$\hat\beta_{d}$  (mean ± SEM)", bold=True, fontsize=8.5, color=C_DISP)
    cell(ax, 1, col_x[6], col_x[7],
         r"$p_{\kappa\,\mathrm{vs}\,d}$", bold=True, fontsize=8.5, color=C_HEAD)

    # ── data rows ─────────────────────────────────────────────────────────────
    for ri, row in enumerate(rows):
        r = ri + 2
        ctrl_label = row["ctrl"].replace("\n", " ").replace("$+$", "+").replace("$\\&$", "&")

        cell(ax, r, col_x[0], col_x[1], ctrl_label, ha="left", fontsize=8.2)

        # semantic κ
        cell(ax, r, col_x[1], col_x[2],
             f"{row['sc']} {row['sc_sig']}", fontsize=8, color=C_CURV)
        # semantic d
        cell(ax, r, col_x[2], col_x[3],
             f"{row['sd']} {row['sd_sig']}", fontsize=8, color=C_DISP)
        # κ vs d (sem)
        p_s_str = f"p={row['pp_s']:.3f} {row['pp_s_sig']}" if np.isfinite(row['pp_s']) else "—"
        cell(ax, r, col_x[3], col_x[4], p_s_str, fontsize=7.5)

        # neural κ
        cell(ax, r, col_x[4], col_x[5],
             f"{row['nc']} {row['nc_sig']}", fontsize=8, color=C_CURV)
        # neural d
        cell(ax, r, col_x[5], col_x[6],
             f"{row['nd']} {row['nd_sig']}", fontsize=8, color=C_DISP)
        # κ vs d (neu)
        p_n_str = f"p={row['pp_n']:.3f} {row['pp_n_sig']}" if np.isfinite(row['pp_n']) else "—"
        cell(ax, r, col_x[6], col_x[7], p_n_str, fontsize=7.5)

    # ── alignment row (bottom, shaded differently) ────────────────────────────
    r_align = len(rows) + 2
    rect = FancyBboxPatch((col_x[0], row_y(r_align)), col_x[-1] - col_x[0], row_h,
                          boxstyle="square,pad=0",
                          linewidth=0, facecolor="#EAF0FB",
                          transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)

    cell(ax, r_align, col_x[0], col_x[1],
         "Cross-space alignment\n(Pearson r)", ha="left", fontsize=8.0, bold=True)
    cell(ax, r_align, col_x[1], col_x[2],
         f"{align_row['rc']} {align_row['rc_sig']}",
         fontsize=8, color=C_CURV)
    cell(ax, r_align, col_x[2], col_x[3],
         f"{align_row['rd']} {align_row['rd_sig']}",
         fontsize=8, color=C_DISP)
    p_rr_str = f"p={align_row['pp']:.4f} {align_row['pp_sig']}"
    cell(ax, r_align, col_x[3], col_x[5], p_rr_str, fontsize=7.5)
    cell(ax, r_align, col_x[5], col_x[7],
         r"(same patients; $\kappa_\mathrm{sem}$$\leftrightarrow$$\kappa_\mathrm{neu}$"
         r" vs $d_\mathrm{sem}$$\leftrightarrow$$d_\mathrm{neu}$)",
         fontsize=7.5, color="#555555")

    # ── grid lines ────────────────────────────────────────────────────────────
    hline(ax, 0, lw=1.2, color="#888888")   # top border
    hline(ax, 1, lw=0.8, color="#888888")   # below space headers
    hline(ax, 2, lw=1.0, color="#888888")   # below col headers
    for ri in range(len(rows)):
        hline(ax, ri + 3, lw=0.4)
    hline(ax, r_align, lw=1.0, color="#888888")  # above alignment row
    # bottom border
    ax.plot([col_x[0], col_x[-1]],
            [row_y(r_align) - row_h * 0.0, row_y(r_align) - row_h * 0.0],
            transform=ax.transAxes, color="#888888", lw=1.2, clip_on=False)

    for x in col_x[1:4] + col_x[5:7]:
        vline(ax, x, 2, r_align, lw=0.4)

    # ── footnote ─────────────────────────────────────────────────────────────
    fig.text(0.04, 0.022,
             r"$\hat\beta_{\mathrm{surp}}$ = standardized OLS coefficient for surprisal.  "
             r"Mean ± SEM across $N=10$ patients.  "
             r"Significance vs. 0 by one-sample $t$-test; $\kappa$ vs $d$ by paired $t$-test.  "
             r"*** $p<0.001$, ** $p<0.01$, * $p<0.05$.",
             fontsize=7.5, color="#555555", va="bottom")

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(FIGURES, exist_ok=True)
    stem = "table_ablation_regression"
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
