"""
make_podcast_curvature_freq_quartiles_gpt2xl_layer29_pseudopop.py
====================================================================
Event-triggered curvature by surprisal (top vs bottom 25% WITHIN each band),
stratified by SUBTLEX frequency quartile (wordfreq zipf, quartiles over the
common words): Q4 = top 25% most frequent ... Q1 = bottom 25%.
Full space (semantic = raw gpt2-xl layer 29 embedding; neural = all 475
pseudopop units, dur80/40 raw spike counts, z-scored; YFX clock-fixed).
Rows: semantic, neural. Columns: frequency quartile. Bands = SEM across events.

argv[1] = "surp": the mirror -- columns = surprisal quartiles, and within each
band the split is high vs low SUBTLEX frequency (top vs bottom 25% within band).

Saves: figures/fig_podcast_curvature_freq_quartiles_gpt2xl_layer29_pseudopop.{pdf,png,svg}
       figures/fig_podcast_curvature_surp_quartiles_gpt2xl_layer29_pseudopop.{pdf,png,svg} (surp mode)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

MODE = sys.argv[1] if len(sys.argv) > 1 else "freq"
sys.argv = sys.argv[:1]
import make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 as M
import compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop as PP
from podcast_loader import load_podcast_patient

PP.RATE = False
PP.CACHE_PATH = os.path.join(BASE_DIR, "results", "pseudopop_full_sequence_dur8040_count.npz")
WINDOW = M.WINDOW
if MODE == "freq":
    BANDS = [("Top 25% most frequent", 75, 100), ("50–75th pct frequency", 50, 75),
             ("25–50th pct frequency", 25, 50), ("Bottom 25% (least frequent)", 0, 25)]
else:
    BANDS = [("Top 25% most surprising", 75, 100), ("50–75th pct surprisal", 50, 75),
             ("25–50th pct surprisal", 25, 50), ("Bottom 25% (least surprising)", 0, 25)]


def main():
    neu_z, common, _ = PP.build_pseudopop()
    data = load_podcast_patient(PP.REF_PID, region=M.REGION, model_tag=M.MODEL, context_words=200)
    emb = np.array(np.load(data["emb_path"], mmap_mode="r")[M.LAYER, common], dtype=np.float64)
    surp = np.asarray(data["surp"], dtype=float)[common]
    lf = M.subtlex_log_freq(np.asarray(data["words"])[common])
    n = len(common)
    curv = dict(sem=M.trajectory_curvature(emb), neu=M.trajectory_curvature(neu_z))
    fin = np.isfinite(lf) & np.isfinite(surp)
    band_var, split_var = (lf, surp) if MODE == "freq" else (surp, lf)
    edges = np.percentile(band_var[fin], [0, 25, 50, 75, 100])
    unit = "Zipf" if MODE == "freq" else "surprisal"
    names = (("High surprisal (top 25% within band)", "Low surprisal (bottom 25% within band)") if MODE == "freq"
             else ("High frequency (top 25% within band)", "Low frequency (bottom 25% within band)"))

    lags = np.arange(-WINDOW, WINDOW + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharey="row")
    for c, (lab, lo_p, hi_p) in enumerate(BANDS):
        a, b = edges[lo_p // 25], edges[hi_p // 25]
        band = fin & (band_var >= a) & ((band_var <= b) if hi_p == 100 else (band_var < b))
        for r, side in enumerate(("sem", "neu")):
            ax = axes[r, c]
            hi, lo = M.event_windows(curv[side], split_var, band, n)
            for rows, col, name in ((hi, M.C_HI, names[0]), (lo, M.C_LO, names[1])):
                mu, se = np.nanmean(rows, axis=0), scipy_sem(rows, axis=0, nan_policy="omit")
                ax.plot(lags, mu, color=col, lw=1.8, label=name)
                ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.18)
            h, l = np.nanmean(hi, axis=0), np.nanmean(lo, axis=0)
            print(f"  {side} {lab:28s} n={band.sum():4d} ev={hi.shape[0]}+{lo.shape[0]}  "
                  f"split hi>= {np.nanpercentile(split_var[band], 75):.2f} lo<= {np.nanpercentile(split_var[band], 25):.2f}  "
                  f"lag0 {h[WINDOW]:.1f}/{l[WINDOW]:.1f}  lag+1 {h[WINDOW + 1]:.1f}/{l[WINDOW + 1]:.1f}  "
                  f"lag+2 {h[WINDOW + 2]:.1f}/{l[WINDOW + 2]:.1f}  base {np.r_[h[:3], l[:3]].mean():.1f}")
            ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_xticks(lags)
            ax.tick_params(labelsize=9)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            space = f"Semantic ({M.MODEL} L{M.LAYER})" if side == "sem" else f"Neural pseudopop ({neu_z.shape[1]} units)"
            ax.set_title(f"{'ABCDEFGH'[4 * r + c]}  {space}\n{lab} ({unit} {a:.2f}–{b:.2f})\n"
                         f"{hi.shape[0]}+{lo.shape[0]} events, SEM", loc="left", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("Mean curvature (°)")
            if r == 1:
                ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=9.5, frameon=False)
    ttl = ("by surprisal, stratified by SUBTLEX frequency quartile" if MODE == "freq"
           else "by SUBTLEX frequency, stratified by surprisal quartile")
    fig.suptitle(f"Event-triggered curvature {ttl} — full space, "
                 f"{M.MODEL} layer {M.LAYER}, hippocampal pseudopopulation (spike counts)",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = f"fig_podcast_curvature_{MODE}_quartiles_gpt2xl_layer29_pseudopop"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
