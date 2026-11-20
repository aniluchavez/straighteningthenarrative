"""
make_elbow_justification_fig.py
=================================
Formally identifies the elbow in the curvature cross-space coupling curve
using piecewise linear (segmented) regression.

For each patient we have one curve: mean r across models vs chunk size.
We grid-search over breakpoints to find the one that minimises the total
residual of a two-segment piecewise linear fit (left rising, right flat).

Figure layout:
  Panel A  — individual patient curves + group mean, piecewise fit overlaid,
              elbow marked with vertical line
  Panel B  — slope before vs after the optimal elbow, per patient
              (paired bar + dots; t-tests vs zero)

The optimal elbow chunk size and its stats are printed and annotated.

Run:
    cd /scratch/aniluchavez/neural_network_similarity/Experiments/p6/geometry_paper
    conda run -n gpt2_embed python3 make_elbow_justification_fig.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, sem as scipy_sem
from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION   = "hippocampus"

ALL_MODELS = [
    "gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]

C_CURV = "#D95319"

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         13,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_patient_curves():
    """
    Returns:
      chunk_sizes : list of int
      pat_curves  : dict {pid: np.array of mean r per chunk size}
                    (averaged across all models for that patient)
    """
    sweep_path = os.path.join(RESULTS, "cross_space_chunk_sweep.json")
    data = json.load(open(sweep_path))
    chunk_sizes = sorted(int(k) for k in data.keys())

    # collect per-patient, per-chunk-size r values across models
    by_pid = {}   # pid -> {cs: [r values across models]}
    for model in ALL_MODELS:
        layer_path = os.path.join(RESULTS,
                                  f"cross_space_layers_{model}_{REGION}.json")
        if not os.path.exists(layer_path):
            continue
        recs = json.load(open(layer_path))
        pids = [r["pid"] for r in recs]

        for ci, cs in enumerate(chunk_sizes):
            model_vals = data[str(cs)].get(model, {}).get("curv", [])
            for pid, val in zip(pids, model_vals):
                by_pid.setdefault(pid, {cs: [] for cs in chunk_sizes})
                if val is not None and not np.isnan(val):
                    by_pid[pid][cs].append(val)

    # average across models per patient
    pat_curves = {}
    for pid, cs_dict in by_pid.items():
        curve = []
        for cs in chunk_sizes:
            vals = cs_dict.get(cs, [])
            curve.append(np.mean(vals) if vals else np.nan)
        pat_curves[pid] = np.array(curve)

    return chunk_sizes, pat_curves


# ── Piecewise linear fit ───────────────────────────────────────────────────────

def piecewise_fit(xs, ys):
    """
    Grid-search over breakpoints.  Left segment: linear (slope ≥ 0).
    Right segment: flat (slope = 0, value = left segment at breakpoint).
    Returns best_bp, slope_left, intercept_left, plateau.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    fin = np.isfinite(ys)

    candidates = xs[1:-1]   # exclude first and last as breakpoints
    best_sse, best_bp = np.inf, candidates[0]
    best_params = None

    for bp in candidates:
        left  = fin & (xs <= bp)
        right = fin & (xs >  bp)
        if left.sum() < 2 or right.sum() < 1:
            continue

        # fit left segment
        coeffs = np.polyfit(xs[left], ys[left], 1)
        slope, intercept = coeffs
        plateau = slope * bp + intercept

        # predict
        pred = np.where(xs <= bp, slope * xs + intercept, plateau)
        sse  = np.sum((ys[fin] - pred[fin]) ** 2)
        if sse < best_sse:
            best_sse   = sse
            best_bp    = bp
            best_params = (slope, intercept, plateau)

    return best_bp, best_params


def patient_slopes(chunk_sizes, pat_curves, elbow):
    """
    Per-patient slope before and after the elbow chunk size.
    Returns (slopes_before, slopes_after) as arrays.
    """
    xs = np.array(chunk_sizes, dtype=float)
    before, after = [], []
    for curve in pat_curves.values():
        fin = np.isfinite(curve)
        left  = fin & (xs <= elbow)
        right = fin & (xs >  elbow)
        if left.sum() >= 2:
            before.append(np.polyfit(xs[left], curve[left], 1)[0])
        else:
            before.append(np.nan)
        if right.sum() >= 2:
            after.append(np.polyfit(xs[right], curve[right], 1)[0])
        else:
            after.append(np.nan)
    return np.array(before), np.array(after)


# ── Figure ────────────────────────────────────────────────────────────────────

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
    chunk_sizes, pat_curves = load_patient_curves()
    xs = np.array(chunk_sizes, dtype=float)

    # Group-mean curve (across patients)
    mat   = np.array(list(pat_curves.values()))
    mu    = np.nanmean(mat, axis=0)
    se    = np.array([scipy_sem(mat[:, i][np.isfinite(mat[:, i])])
                      for i in range(mat.shape[1])])

    # Piecewise fit on group mean
    elbow, params = piecewise_fit(chunk_sizes, mu)
    slope_l, intercept_l, plateau = params
    elbow = int(elbow)
    print(f"Optimal elbow: {elbow} words")
    print(f"  Left slope:  {slope_l:+.5f} r/word")
    print(f"  Plateau:     {plateau:+.4f}")

    # Fit line for plotting
    xs_plot = np.linspace(xs[0], xs[-1], 300)
    fit_y   = np.where(xs_plot <= elbow,
                       slope_l * xs_plot + intercept_l,
                       plateau)

    # Per-patient slopes before / after elbow
    sl_before, sl_after = patient_slopes(chunk_sizes, pat_curves, elbow)
    _, p_before = ttest_1samp(sl_before[np.isfinite(sl_before)], 0)
    _, p_after  = ttest_1samp(sl_after[np.isfinite(sl_after)],   0)

    print(f"  Slope BEFORE elbow: mean={np.nanmean(sl_before):+.5f}  "
          f"p={p_before:.4f}")
    print(f"  Slope AFTER  elbow: mean={np.nanmean(sl_after):+.5f}  "
          f"p={p_after:.4f}")

    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.15, top=0.85,
                        wspace=0.42)

    # ── Panel A: curves + piecewise fit ───────────────────────────────────────
    for curve in pat_curves.values():
        ax_A.plot(xs, curve, color=C_CURV, lw=0.7, alpha=0.30)

    ax_A.plot(xs, mu, color=C_CURV, lw=2.2, label="Mean across patients")
    ax_A.fill_between(xs, mu - se, mu + se, color=C_CURV, alpha=0.18)

    ax_A.plot(xs_plot, fit_y, color="black", lw=1.6, ls="--",
              label="Piecewise linear fit")
    ax_A.axvline(elbow, color="#444444", lw=1.2, ls=":",
                 label=f"Elbow = {elbow} words")

    # Annotate slopes
    x_mid_l = (xs[0] + elbow) / 2
    x_mid_r = (elbow + xs[-1]) / 2
    y_ann   = plateau * 0.45
    pstr_b  = "p < 0.001" if p_before < 0.001 else f"p = {p_before:.3f}"
    pstr_a  = "n.s." if p_after >= 0.05 else f"p = {p_after:.3f}"
    ax_A.annotate(f"slope > 0\n({pstr_b})",
                  xy=(x_mid_l, y_ann), ha="center", va="center",
                  fontsize=8.5, color="#333333",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="#CCCCCC", lw=0.7))
    ax_A.annotate(f"slope ≈ 0\n({pstr_a})",
                  xy=(x_mid_r, y_ann), ha="center", va="center",
                  fontsize=8.5, color="#333333",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="#CCCCCC", lw=0.7))

    ax_A.axhline(0, color="k", lw=0.4, ls=":")
    ax_A.set_xlabel("Chunk size (words)")
    ax_A.set_ylabel("r(LLM curvature, neural curvature)")
    ax_A.set_xticks(chunk_sizes)
    ax_A.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax_A.set_title("A  Piecewise linear fit · curvature coupling\n"
                   "all models · hippocampus · avg layers",
                   loc="left", fontsize=9.5)

    # ── Panel B: per-patient slopes before vs after ────────────────────────────
    n_pat  = len(sl_before)
    jitter = np.random.default_rng(7).uniform(-0.07, 0.07, n_pat)

    for x, slopes, label, p in [
        (0, sl_before, f"Before\n(≤{elbow} words)", p_before),
        (1, sl_after,  f"After\n(>{elbow} words)",  p_after),
    ]:
        mu_s = np.nanmean(slopes)
        se_s = scipy_sem(slopes[np.isfinite(slopes)])
        col  = C_CURV if x == 0 else "#AAAAAA"

        ax_B.bar(x, mu_s, width=0.5, color=col, alpha=0.35, edgecolor="none")
        ax_B.bar(x, mu_s, width=0.5, color="none", edgecolor=col, lw=1.2)
        ax_B.errorbar(x, mu_s, yerr=se_s, fmt="none", color=col,
                      capsize=3, lw=1.6, zorder=5)
        ax_B.scatter([x + j for j in jitter], slopes,
                     color=col, s=26, zorder=4,
                     edgecolors="white", linewidths=0.5)

        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
               ("*" if p < 0.05 else "n.s."))
        ax_B.text(x, mu_s + se_s + 0.0012, f"{sig}\n({pstr})",
                  ha="center", va="bottom", fontsize=7.5)

    # connect paired patients
    for i in range(n_pat):
        ax_B.plot([0 + jitter[i], 1 + jitter[i]],
                  [sl_before[i], sl_after[i]],
                  color="#AAAAAA", lw=0.7, alpha=0.55, zorder=3)

    ax_B.axhline(0, color="k", lw=0.5, ls=":")
    ax_B.set_xticks([0, 1])
    ax_B.set_xticklabels([f"Before\n(≤{elbow} words)",
                          f"After\n(>{elbow} words)"], fontsize=10)
    ax_B.set_ylabel("Slope of r vs chunk size\n(r per word)")
    ax_B.set_title("B  Per-patient slopes before vs after elbow\n"
                   "all models · hippocampus",
                   loc="left", fontsize=9.5)

    fig.suptitle(
        f"Chunk-size elbow at {elbow} words: coupling rises then saturates  ·  Curvature κ",
        fontsize=10, fontweight="bold", y=0.97)

    _save(fig, "fig_elbow_justification")
    print("Done.")


if __name__ == "__main__":
    main()
