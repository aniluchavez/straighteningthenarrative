"""
make_cross_space_paired_fig_pca20_convo.py
================================
PCA20 version of make_cross_space_paired_fig.py for CONVO (ORIG10+NEW4, 14
patients): semantic side PCA-reduced to a fixed 20 components per layer
(averaged across layers, then across all 9 models), win=500ms default.

Panel A — paired dot plot: per-patient r(LLM, neural) for curvature,
          displacement (from the PCA20 sweep) and position (PC1 -- reused
          directly from the raw/non-PCA sweep, since PC1 is mathematically
          identical whether computed on the raw embedding or on any
          PCA-reduced-to-K>=1 version of it) at 15-word chunks, averaged
          across all models. Each bar also gets a one-sample t-test
          against 0 (is this metric's mean r reliably above chance across
          patients?), shown as stars under the bar.
Panel B — chunk-size sweep for curvature, displacement, and position (no
          permutation null bands -- not computed for this PCA20 run, too
          expensive; the per-patient-vs-0 test in Panel A is the "above
          chance" evidence instead).

Run:
    cd /scratch/aniluchavez/semantic-curvature
    conda run -n gpt2_embed python3 make_cross_space_paired_fig_pca20_convo.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_rel, ttest_1samp, sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")

ALL_MODELS = [
    "gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         13,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
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

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_POS  = "#3DAA6C"


def _sig_bracket(ax, x0, x1, y, h, p, fontsize=8.5):
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=1.0, color="k", clip_on=False)
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    pstr = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text((x0 + x1) / 2, y + h * 1.15,
            f"{stars} ({pstr})", ha="center", va="bottom",
            fontsize=fontsize, color="k")


def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def _vs_zero_label(ax, x, y, vals):
    """One-sample t-test of vals against 0 -- 'is this reliably above chance'."""
    finite = vals[np.isfinite(vals)]
    _, p = ttest_1samp(finite, popmean=0.0)
    ax.text(x, y, f"{_stars(p)}\nvs 0", ha="center", va="top", fontsize=7.5, color="#444444")


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  -> {p}")
    plt.close(fig)


CHUNK_SIZE_PANEL_A = 15
PCA20_JSON_PATH = os.path.join(RESULTS, "cross_space_chunk_sweep_pca20.json")
RAW_JSON_PATH   = os.path.join(RESULTS, "cross_space_chunk_sweep.json")  # for position (PC1) only


def load_panel_a():
    sweep_pca = json.load(open(PCA20_JSON_PATH))
    sweep_raw = json.load(open(RAW_JSON_PATH))
    cs_data_pca = sweep_pca[str(CHUNK_SIZE_PANEL_A)]
    cs_data_raw = sweep_raw[str(CHUNK_SIZE_PANEL_A)]

    n_pat = max(len(v.get("curv", [])) for v in cs_data_pca.values())
    curv_by_pat = [[] for _ in range(n_pat)]
    disp_by_pat = [[] for _ in range(n_pat)]
    pos_by_pat  = [[] for _ in range(n_pat)]

    for model in ALL_MODELS:
        md = cs_data_pca.get(model, {})
        for i, (c, d) in enumerate(zip(md.get("curv", []), md.get("disp", []))):
            if c is not None and not np.isnan(c): curv_by_pat[i].append(c)
            if d is not None and not np.isnan(d): disp_by_pat[i].append(d)
        mp = cs_data_raw.get(model, {})
        for i, p in enumerate(mp.get("pos", [])):
            if p is not None and not np.isnan(p): pos_by_pat[i].append(p)

    curv_pat = np.array([np.mean(v) if v else np.nan for v in curv_by_pat])
    disp_pat = np.array([np.mean(v) if v else np.nan for v in disp_by_pat])
    pos_pat  = np.array([np.mean(v) if v else np.nan for v in pos_by_pat])
    return curv_pat, disp_pat, pos_pat


def load_panel_b():
    data_pca = json.load(open(PCA20_JSON_PATH))
    data_raw = json.load(open(RAW_JSON_PATH))
    chunk_sizes = sorted(int(k) for k in data_pca.keys())

    curv_by_cs, disp_by_cs, pos_by_cs = [], [], []
    for cs in chunk_sizes:
        c_vals, d_vals, p_vals = [], [], []
        for model_data in data_pca[str(cs)].values():
            c_vals.extend([v for v in model_data.get("curv", [])
                           if v is not None and not np.isnan(v)])
            d_vals.extend([v for v in model_data.get("disp", [])
                           if v is not None and not np.isnan(v)])
        for model_data in data_raw[str(cs)].values():
            p_vals.extend([v for v in model_data.get("pos", [])
                           if v is not None and not np.isnan(v)])
        curv_by_cs.append((np.mean(c_vals), scipy_sem(c_vals)) if c_vals else (np.nan, np.nan))
        disp_by_cs.append((np.mean(d_vals), scipy_sem(d_vals)) if d_vals else (np.nan, np.nan))
        pos_by_cs.append((np.mean(p_vals), scipy_sem(p_vals)) if p_vals else (np.nan, np.nan))

    return chunk_sizes, curv_by_cs, disp_by_cs, pos_by_cs


def main():
    curv, disp, pos = load_panel_a()
    chunk_sizes, curv_sweep, disp_sweep, pos_sweep = load_panel_b()

    fig = plt.figure(figsize=(9.8, 4.4))
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.09, right=0.97, bottom=0.16, top=0.86,
                            wspace=0.40, width_ratios=[1.1, 1.6])
    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])

    # Panel A: paired dot plot
    groups  = [curv, disp, pos]
    colors  = [C_CURV, C_DISP, C_POS]
    labels  = ["Curvature (κ)", "Displacement (d)", "Position"]
    x_pos   = [0, 1, 2]
    n_pat   = len(curv)

    for x, vals, col in zip(x_pos, groups, colors):
        mu = np.nanmean(vals)
        ax_A.bar(x, mu, width=0.55, color=col, alpha=0.35, zorder=2, edgecolor="none")
        ax_A.bar(x, mu, width=0.55, color="none", zorder=2, edgecolor=col, linewidth=1.2)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, n_pat)
    for i in range(n_pat):
        ys = [curv[i], disp[i], pos[i]]
        xs = [x + jitter[i] for x in x_pos]
        ax_A.plot(xs, ys, color="#AAAAAA", lw=0.7, zorder=3, alpha=0.6)

    for x, vals, col in zip(x_pos, groups, colors):
        xs = [x + j for j in jitter]
        ax_A.scatter(xs, vals, color=col, s=28, zorder=4, edgecolors="white", linewidths=0.5)

    for x, vals, col in zip(x_pos, groups, colors):
        mu = np.nanmean(vals)
        se = scipy_sem(vals[np.isfinite(vals)])
        ax_A.errorbar(x, mu, yerr=se, fmt="none", color=col, capsize=3, lw=1.6, zorder=5)

    ax_A.axhline(0, color="k", lw=0.5, ls=":")

    y_top = max(np.nanmax(curv), np.nanmax(disp), np.nanmax(pos))
    y_bot = min(np.nanmin(curv), np.nanmin(disp), np.nanmin(pos))
    gap   = (y_top - y_bot) * 0.13

    _, p_cd = ttest_rel(curv[np.isfinite(curv) & np.isfinite(disp)],
                        disp[np.isfinite(curv) & np.isfinite(disp)])
    _, p_cp = ttest_rel(curv[np.isfinite(curv) & np.isfinite(pos)],
                        pos[np.isfinite(curv) & np.isfinite(pos)])

    b1 = y_top + gap * 0.8
    b2 = b1 + gap * 1.8
    _sig_bracket(ax_A, 0, 1, b1, gap * 0.3, p_cd)
    _sig_bracket(ax_A, 0, 2, b2, gap * 0.3, p_cp)

    # "vs 0" one-sample t-test labels under each bar (above-chance evidence)
    for x, vals in zip(x_pos, groups):
        _vs_zero_label(ax_A, x, y_bot - gap * 0.35, vals)

    ax_A.set_xticks(x_pos)
    ax_A.set_xticklabels(labels, fontsize=10)
    ax_A.set_ylabel("Semantic ↔ neural Pearson r", fontsize=10)
    ax_A.set_title("A  Cross-space geometry alignment (PCA-20)\n"
                   "all models · hippocampus · 15-word chunks",
                   loc="left", fontsize=9.5)
    ax_A.set_ylim(y_bot - gap * 1.3, b2 + gap * 1.6)

    # Panel B: chunk sweep
    xs = np.array(chunk_sizes)
    for sweep, color, label in [(curv_sweep, C_CURV, "Curvature (κ)"),
                                 (disp_sweep, C_DISP, "Displacement (d)"),
                                 (pos_sweep,  C_POS,  "Position")]:
        mus = np.array([v[0] for v in sweep])
        ses = np.array([v[1] for v in sweep])
        ax_B.plot(xs, mus, color=color, lw=2.0, marker="o", ms=4.5, label=label)
        ax_B.fill_between(xs, mus - ses, mus + ses, color=color, alpha=0.20)

    ax_B.axvline(15, color="#555555", lw=1.1, ls="--", label="15-word elbow (Panel A)")
    ax_B.axhline(0,  color="k", lw=0.4, ls=":")

    ax_B.set_xlabel("Chunk size (words)")
    ax_B.set_ylabel("Mean r  (LLM ↔ Neural)")
    ax_B.set_title("B  Coupling vs chunk size (PCA-20)\n"
                   "all models · hippocampus · avg across layers",
                   loc="left", fontsize=9.5)
    ax_B.set_xticks(xs)
    ax_B.legend(frameon=False, fontsize=9)

    _save(fig, "fig_cross_space_paired_pca20_convo")
    print("Done.")


if __name__ == "__main__":
    main()
