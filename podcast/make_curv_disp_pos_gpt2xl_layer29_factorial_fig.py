"""
make_curv_disp_pos_gpt2xl_layer29_factorial_fig.py
=====================================================
Figure for compute_curv_disp_pos_gpt2xl_layer29_factorial.py.
Columns: neural window (duration counts / duration rates / fixed 250 ms).
Rows: (1) semantic-neural r, (2) r after regressing log word duration out of
both sides, (3) r of each metric with log duration (solid = neural,
dashed = semantic). x = matched top-K PCs or full space. Mean +/- SEM across
13 patients.

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29_factorial.{pdf,png,svg}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = [("curv", "Curvature", "#E67E22"), ("disp", "Displacement", "#5B8FD4"), ("pos", "Position", "#2E9E6C")]
WNAME = {"dur_count": "Duration window, counts\n[onset+80, offset+40]",
         "dur_rate": "Duration window, rates\n(count / window length)",
         "fixed250": "Fixed 250 ms window, counts\n[onset+80, onset+330]"}

d = json.load(open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_factorial.json")))
KS, WINDOWS = d["config"]["KS"], d["config"]["WINDOWS"]
P = list(d["patients"].values())
x = np.arange(len(KS))


def stat(w, K, m, key):
    v = np.array([p[f"{w}|{K}"][m][key] for p in P])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


fig, axes = plt.subplots(3, len(WINDOWS), figsize=(15, 11), sharex=True)
for c, w in enumerate(WINDOWS):
    for r, (key, ylab) in enumerate([("r", "Semantic–neural r"),
                                     ("r_partial", "Semantic–neural r,\nlog duration regressed out")]):
        ax = axes[r, c]
        for m, lab, col in METRICS:
            mu, se = np.array([stat(w, K, m, key) for K in KS]).T
            ax.errorbar(x, mu, yerr=se, color=col, marker="o", ms=4, lw=1.8, capsize=3, label=lab)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        if c == 0:
            ax.set_ylabel(ylab)
        if r == 0:
            ax.set_title(f"{'ABC'[c]}  {WNAME[w]}", loc="left")
    ax = axes[2, c]
    for m, lab, col in METRICS:
        mu_n, se_n = np.array([stat(w, K, m, "r_dur_neu") for K in KS]).T
        mu_s, _ = np.array([stat(w, K, m, "r_dur_sem") for K in KS]).T
        ax.errorbar(x, mu_n, yerr=se_n, color=col, marker="o", ms=4, lw=1.8, capsize=3, label=f"{lab}, neural")
        ax.plot(x, mu_s, color=col, ls="--", lw=1.4, label=f"{lab}, semantic")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {K}" if K != "full" else "full" for K in KS])
    ax.set_xlabel("Dimensions (matched top-K PCs, or full space)")
    if c == 0:
        ax.set_ylabel("r(metric, log word duration)")
for r in range(2):
    lo = min(a.get_ylim()[0] for a in axes[r]); hi = max(a.get_ylim()[1] for a in axes[r])
    for a in axes[r]:
        a.set_ylim(lo, hi)
lo = min(a.get_ylim()[0] for a in axes[2]); hi = max(a.get_ylim()[1] for a in axes[2])
for a in axes[2]:
    a.set_ylim(lo, hi)
axes[0, 0].legend(frameon=False, loc="upper left")
axes[2, 0].legend(frameon=False, loc="best", ncol=2, fontsize=8)
fig.suptitle(f"What decides the winning metric? gpt2-xl layer {d['config']['LAYER']}, per patient "
             f"(n = {len(P)}), chunk = {d['config']['CHUNK']}, mean ± SEM", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_pos_gpt2xl_layer29_factorial"
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
