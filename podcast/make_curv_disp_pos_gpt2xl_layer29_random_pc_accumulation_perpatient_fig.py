"""
make_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient_fig.py
=============================================================================
Figure for compute_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient.py
(duration-window counts, per patient). Columns: curvature / displacement /
position. Rows:
  1. top-K (red, mean +/- SEM across patients) vs. random PCs on both sides
     (group mean across patients for each random ordering; median + 50%/95%)
  2. random-both orderings split by whether SEMANTIC PC1 has been added by K
     (per-patient mean over orderings, then mean +/- SEM across patients)
  3. same split by NEURAL PC1
x = K (patients with fewer units than K drop out; n shown on top row), last
point = all units for each patient.

argv[1] = "row1": panels A-C only.

Saves: figures/fig_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient{,_ABC}.{pdf,png,svg}
"""
import json
import os
import sys

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
    "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

METRICS = [("curv", "Curvature"), ("disp", "Displacement"), ("pos", "Position (first PC in set)")]
ROW1 = len(sys.argv) > 1 and sys.argv[1] == "row1"
NR = 1 if ROW1 else 3
TOP_COL, RND_COL, IN_COL, OUT_COL = "#c0392b", "#5d6d7e", "#8e44ad", "#16a085"

d = json.load(open(os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient.json")))
P = d["patients"]
KS = d["config"]["KS"]
labels = [str(K) for K in KS] + ["all units"]


def slot(rec, j_label):
    """index into rec['ks'] for x-slot j_label (K or 'all')."""
    if j_label == "all units":
        return len(rec["ks"]) - 1
    K = int(j_label)
    return rec["ks"].index(K) if K in rec["ks"] and K < rec["n_units"] else (
        rec["ks"].index(K) if K in rec["ks"] else None)


cols = {}
for lab in labels:
    cols[lab] = [(pid, slot(rec, lab)) for pid, rec in P.items() if slot(rec, lab) is not None]
labels = [lab for lab in labels if len(cols[lab]) >= 3]
x = np.arange(len(labels))
nmin = len(next(iter(P.values()))["random"]["both"]["r"])

fig, axes = plt.subplots(NR, 3, figsize=(15, 11 if NR == 3 else 4.6), sharex=True, squeeze=False)
for c, (m, mname) in enumerate(METRICS):
    top_mu, top_se, rnd = [], [], []
    split = {"sem": ([], [], [], []), "neu": ([], [], [], [])}
    for lab in labels:
        tv = np.array([P[pid]["top"][j][c] for pid, j in cols[lab]])
        top_mu.append(tv.mean()); top_se.append(tv.std(ddof=1) / np.sqrt(len(tv)))
        R = np.array([np.array(P[pid]["random"]["both"]["r"])[:, j, c] for pid, j in cols[lab]])  # pat x order
        rnd.append(R.mean(axis=0))
        for side in ("sem", "neu"):
            ins, outs = [], []
            for (pid, j), row in zip(cols[lab], R):
                K = P[pid]["ks"][j]
                inc = np.array(P[pid]["random"]["both"][f"pc1_pos_{side}"]) <= K
                ins.append(row[inc].mean() if inc.any() else np.nan)
                outs.append(row[~inc].mean() if (~inc).any() else np.nan)
            ins, outs = np.array(ins), np.array(outs)
            for arr, mu_l, se_l in ((ins, split[side][0], split[side][1]), (outs, split[side][2], split[side][3])):
                f = np.isfinite(arr)
                mu_l.append(arr[f].mean() if f.sum() else np.nan)
                se_l.append(arr[f].std(ddof=1) / np.sqrt(f.sum()) if f.sum() > 1 else np.nan)
    rnd = np.array(rnd)  # K x order
    for r in range(NR):
        ax = axes[r, c]
        ax.errorbar(x, top_mu, yerr=top_se, color=TOP_COL, marker="o", ms=4, lw=2, capsize=3, label="top-K PCs")
        ax.axhline(0, color="k", lw=0.5)
        if r == 0:
            ax.fill_between(x, *np.percentile(rnd, [2.5, 97.5], axis=1), color=RND_COL, alpha=0.15, lw=0,
                            label="random PCs, 95% of orderings")
            ax.fill_between(x, *np.percentile(rnd, [25, 75], axis=1), color=RND_COL, alpha=0.35, lw=0,
                            label="random PCs, 50% of orderings")
            ax.plot(x, np.median(rnd, axis=1), color=RND_COL, lw=2, label="random PCs, median")
            ax.set_title(f"{'ABC'[c]}  {mname}", loc="left")
        else:
            side = "sem" if r == 1 else "neu"
            mi, si, mo, so = (np.array(v) for v in split[side])
            nm = "semantic" if side == "sem" else "neural"
            for mu, se, col, lab in ((mi, si, IN_COL, f"random, {nm} PC1 included"),
                                     (mo, so, OUT_COL, f"random, {nm} PC1 not included")):
                ax.errorbar(x, mu, yerr=se, color=col, marker="o", ms=3.5, lw=1.8, capsize=2.5, label=lab)
            ax.set_title(f"{'DEFGHI'[3 * (r - 1) + c]}  {mname} — split by {nm} PC1", loc="left")
        if c == 0:
            ax.set_ylabel("Semantic–neural r")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
for a in axes[-1]:
    a.set_xticks(x)
    a.set_xticklabels([f"{'all' if lab == 'all units' else lab}\n{len(cols[lab])}" for lab in labels], fontsize=9)
    a.set_xlabel("PCs per space (K)  /  n patients")
for r in range(NR):
    lo = min(a.get_ylim()[0] for a in axes[r]); hi = max(a.get_ylim()[1] for a in axes[r])
    for a in axes[r]:
        a.set_ylim(lo, hi)
fig.suptitle(f"Top-K vs. random PCs, per patient — gpt2-xl layer {d['config']['LAYER']}, duration-window spike counts, "
             f"chunk = {d['config']['CHUNK']}, {nmin} random orderings per patient", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
stem = "fig_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_perpatient" + ("_ABC" if ROW1 else "")
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")
