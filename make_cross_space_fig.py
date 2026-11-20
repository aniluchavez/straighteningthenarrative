"""
make_cross_space_fig.py
=======================
Figure: cross-space geometric coupling between LLM (semantic) and neural
(hippocampal) trajectories for three geometric quantities:

  displacement  — step-to-step Euclidean distance (how far each word moves)
  curvature     — turning angle κ (how much the trajectory bends)
  position      — first-PC coordinate (where you are in the space)

Layout (portrait):
  Row 1: three scatter panels (LLM metric vs neural metric, coloured by
          surprisal), in the style of fig2 panel C.
  Row 2: matplotlib summary table — mean r ± SE, t, p, n_sig for each
          quantity, with significance stars.

Run:
    cd /scratch/aniluchavez/straighteningthenarrative
    conda run -n gpt2_embed python3 make_cross_space_fig.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, sem as scipy_sem, ttest_1samp

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")
MODEL    = "llama-3.1-8b"
REGION   = "hippocampus"

# ── NeurIPS style (matches make_geometry_figures.py) ──────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         14,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

C_HI  = "#C0392B"
C_LO  = "#2980B9"

QUANTITIES = [
    dict(key="disp", label="Displacement",
         r_key="r_disp", p_key="p_disp",
         x_key="chunk_disp_sem", y_key="chunk_disp_neu", surp_key="chunk_surp_d",
         xlabel="LLM displacement  (z-scored)",
         ylabel="Neural displacement  (z-scored)"),
    dict(key="curv", label="Curvature",
         r_key="r_curv", p_key="p_curv",
         x_key="chunk_curvature_sem", y_key="chunk_curvature_neu", surp_key="chunk_surp_c",
         xlabel="LLM curvature κ  (z-scored)",
         ylabel="Neural curvature κ  (z-scored)"),
    dict(key="pos",  label="Position",
         r_key="r_pos",  p_key="p_pos",
         x_key="chunk_pos_sem",  y_key="chunk_pos_neu",  surp_key="chunk_surp_p",
         xlabel="LLM PC1 position  (z-scored)",
         ylabel="Neural PC1 position  (z-scored)"),
]

PANEL_LABELS = ["A", "B", "C"]


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  → {p}")
    plt.close(fig)


def main():
    data_path = os.path.join(RESULTS, f"cross_space_scatter_{MODEL}_{REGION}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Run compute_cross_space_scatter.py first.\n  Missing: {data_path}")
    recs = json.load(open(data_path))
    print(f"Loaded {len(recs)} patients")

    # ── Figure layout ──────────────────────────────────────────────────────────
    # Row 1: scatter panels  Row 2: table
    fig = plt.figure(figsize=(10.5, 9.5))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.09, right=0.97, bottom=0.34, top=0.93,
        wspace=0.42, hspace=0.52,
        height_ratios=[1, 1],
    )
    # Scatter panels occupy top row only; table drawn manually below
    scatter_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    # ── Scatter panels ─────────────────────────────────────────────────────────
    for qi, (ax, q) in enumerate(zip(scatter_axes, QUANTITIES)):
        # Pool across patients
        xs, ys, ss = [], [], []
        for rec in recs:
            xs.extend(rec[q["x_key"]])
            ys.extend(rec[q["y_key"]])
            ss.extend(rec[q["surp_key"]])
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        ss = np.array(ss, dtype=float)

        fin = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ss)
        xs, ys, ss = xs[fin], ys[fin], ss[fin]

        if len(xs) < 20:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{PANEL_LABELS[qi]}  {q['label']}", loc="left", pad=4)
            continue

        vmin_s, vmax_s = np.percentile(ss, [5, 95])
        sc = ax.scatter(xs, ys, c=ss, cmap="RdBu_r",
                        vmin=vmin_s, vmax=vmax_s,
                        s=10, alpha=0.40, rasterized=True, linewidths=0)

        # overall regression line
        m, b = np.polyfit(xs, ys, 1)
        r_val, p_val = pearsonr(xs, ys)
        xr = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xr, m * xr + b, color="k", lw=1.6, ls="--", zorder=5)
        pstr = f"p = {p_val:.2e}" if p_val >= 0.001 else "p < 0.001"
        ax.text(0.03, 0.97, f"r = {r_val:+.3f},  {pstr}",
                transform=ax.transAxes, fontsize=7.0, va="top")

        # high / low surprisal centroids
        lo_thr = np.percentile(ss, 25)
        hi_thr = np.percentile(ss, 75)
        for mask, color, lbl in [
            (ss >= hi_thr, C_HI, "High surp."),
            (ss <= lo_thr, C_LO, "Low surp."),
        ]:
            mx, my = np.mean(xs[mask]), np.mean(ys[mask])
            ex = scipy_sem(xs[mask]); ey = scipy_sem(ys[mask])
            ax.errorbar(mx, my, xerr=ex, yerr=ey,
                        fmt="*", color=color, ms=12, lw=1.5,
                        markeredgecolor="white", markeredgewidth=0.6,
                        zorder=8, label=lbl)

        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.7,
                  edgecolor="none", markerscale=0.8)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.03, aspect=18)
        cbar.set_label("Mean chunk surprisal (nats)", fontsize=6.0)

        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_xlabel(q["xlabel"], fontsize=7.5)
        ax.set_ylabel(q["ylabel"], fontsize=7.5)
        ax.set_title(f"{PANEL_LABELS[qi]}  LLM ↔ Neural {q['label']}\n"
                     f"(layer {5}, 10-word chunks, hippocampus)",
                     loc="left", pad=4, fontsize=9)

    fig.suptitle(
        "Cross-space geometric coupling  ·  LLaMA-3.1-8B  ·  Hippocampus",
        fontsize=10, fontweight="bold", y=0.97)

    # ── Summary table ──────────────────────────────────────────────────────────
    # Drawn in figure coordinates below the scatter grid
    col_headers = ["Geometric\nquantity", "Mean r", "SE", "t", "p-value",
                   "n sig.\n(p < 0.05)", ""]
    col_xs      = [0.04, 0.24, 0.37, 0.48, 0.61, 0.76, 0.88]
    row_y_top   = 0.29
    row_h       = 0.055
    header_y    = row_y_top + row_h * 1.05

    # Column headers
    for cx, hdr in zip(col_xs, col_headers):
        fig.text(cx, header_y, hdr, transform=fig.transFigure,
                 fontsize=8.5, fontweight="bold", va="bottom", ha="left",
                 linespacing=1.2)

    # Horizontal rule below header
    line_y = row_y_top + row_h * 0.92
    fig.add_artist(plt.Line2D([0.03, 0.97], [line_y, line_y],
                              transform=fig.transFigure,
                              color="black", lw=0.9, clip_on=False))

    for ri, q in enumerate(QUANTITIES):
        r_vals = np.array([rec[q["r_key"]] for rec in recs], dtype=float)
        r_vals = r_vals[np.isfinite(r_vals)]
        if len(r_vals) == 0:
            continue
        mu  = np.mean(r_vals)
        se  = scipy_sem(r_vals)
        t, p = ttest_1samp(r_vals, 0)
        n_sig = int((np.array([rec[q["p_key"]] for rec in recs], dtype=float) < 0.05).sum())
        row_y = row_y_top - ri * row_h

        # alternating row shading
        if ri % 2 == 0:
            rect = plt.Rectangle((0.03, row_y - row_h * 0.07), 0.94, row_h * 0.95,
                                  transform=fig.transFigure,
                                  color="#F5F5F5", zorder=0, clip_on=False)
            fig.add_artist(rect)

        vals = [
            q["label"],
            f"{mu:+.4f}",
            f"{se:.4f}",
            f"{t:+.3f}",
            f"{p:.4f}",
            f"{n_sig}/{len(recs)}",
            _stars(p),
        ]
        colors = ["#111111"] * 6 + [
            C_HI if p < 0.05 else "#888888"
        ]
        for cx, val, col in zip(col_xs, vals, colors):
            fig.text(cx, row_y, val, transform=fig.transFigure,
                     fontsize=8.5, va="center", ha="left", color=col)

    # Bottom rule
    bot_y = row_y_top - len(QUANTITIES) * row_h + row_h * 0.05
    fig.add_artist(plt.Line2D([0.03, 0.97], [bot_y, bot_y],
                              transform=fig.transFigure,
                              color="black", lw=0.6, clip_on=False))

    # Table caption
    fig.text(0.03, bot_y - 0.025, transform=fig.transFigure,
             s=("Table: per-patient Pearson r between LLM and neural "
                "geometric quantities at layer 5 (LLaMA-3.1-8B, hippocampus). "
                "One-sample t-test vs. 0 across N=10 patients; *** p<0.001, ** p<0.01, * p<0.05, n.s."),
             fontsize=7.0, va="top", color="#555555", wrap=True)

    _save(fig, "fig_cross_space_geometry")
    print("Done.")


if __name__ == "__main__":
    main()
