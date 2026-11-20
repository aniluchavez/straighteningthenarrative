"""
make_tentative_paired_pc8_chunk3_convo.py
============================================
Conversation-dataset counterpart to
podcast/make_tentative_paired_pc8_chunk3.py: same PC8/chunk3 paired bar
figure, but on convo data, all 11 models, different color scheme.
Saved to tentative_figures/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "tentative_figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

data = json.load(open(os.path.join(BASE_DIR, "results", "paired_fig_pc8_chunk3_convo_merged.json")))
MODELS = list(data.keys())

all_patients = set()
for m in MODELS:
    all_patients |= set(data[m].keys())

per_patient = {"curv": [], "disp": [], "pos": []}
for pid in sorted(all_patients):
    for metric in ["curv", "disp", "pos"]:
        vals = [data[m][pid][metric] for m in MODELS if pid in data[m]]
        if vals:
            per_patient[metric].append(np.mean(vals))

# different color scheme from the podcast version (orange/blue/green -> purple/teal/magenta)
METRICS = [("curv", "Curvature ($\\kappa$)", "#7B4FA3"),
           ("disp", "Displacement (d)", "#1E9E9E"),
           ("pos", "Position", "#C2417A")]

fig, ax = plt.subplots(figsize=(7, 6.2))
xpos = np.arange(3)
all_lo, all_hi = [], []
for i, (key, label, color) in enumerate(METRICS):
    vals = np.array(per_patient[key])
    mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals))
    t, p = ttest_1samp(vals, 0.0)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.bar(xpos[i], mu, yerr=se, color=color, alpha=0.55, width=0.6, capsize=4)
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(vals)) - 0.5) * 0.18
    ax.scatter(xpos[i] + jitter, vals, color=color, edgecolor="white", s=34, zorder=3)
    ytop = max(vals.max(), mu + se) + 0.03
    ax.text(xpos[i], ytop, f"{sig}\n(p={p:.1e})", ha="center", fontsize=11)
    all_lo.append(vals.min()); all_hi.append(ytop + 0.04)
    print(f"{key}: mean={mu:+.4f} p={p:.4e} n={len(vals)}")

ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xticks(xpos); ax.set_xticklabels([m[1] for m in METRICS])
ax.set_ylabel("Semantic $\\leftrightarrow$ neural Pearson $r$")
ax.set_ylim(min(all_lo) - 0.03, max(all_hi))
ax.set_title("Cross-space geometry alignment (PC8, matched) — CONVERSATION\ndur80/40 window, 15 patients, 3-word chunks, avg across 11 models", loc="left", pad=14, fontsize=13.5)
fig.tight_layout()

os.makedirs(FIGURES, exist_ok=True)
for ext in ("pdf", "png"):
    path = os.path.join(FIGURES, f"fig_cross_space_paired_pc8_chunk3_convo_11models.{ext}")
    fig.savefig(path)
    print(f"  -> {path}")
