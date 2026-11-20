"""
make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29_pseudopop.py
=========================================================================
Pseudopopulation version of make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29.py.
Neural = all 13 patients' hippocampal units concatenated on the words common
to all patients, dur80/40 raw spike COUNTS (not rates -- dividing by window
length rescales long/rare words), z-scored per unit; YFX with the +574.9 s
clock fix; cached in results/pseudopop_full_sequence_dur8040_count.npz.
One curvature trace.
Semantic = gpt2-xl layer 29 (YEY embeddings at the common words). Surprisal
and SUBTLEX frequency from YEY at the common words.

argv[1]: "full" (semantic raw embedding, neural all 475 units) or K (matched
top-K PCs on both, standardized). Bands = SEM across events.

Saves: figures/fig_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29_pseudopop_{full|pcK}.{pdf,png,svg}
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

ARG = sys.argv[1] if len(sys.argv) > 1 else "full"
sys.argv = sys.argv[:1]
import make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 as M
from podcast_loader import load_podcast_patient
from compute_curv_disp_pos_gpt2xl_layer29 import pca_fixed
import compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop as PP
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import REF_PID

PP.RATE = False
PP.CACHE_PATH = os.path.join(BASE_DIR, "results", "pseudopop_full_sequence_dur8040_count.npz")

FULL = ARG == "full"
K = None if FULL else int(ARG)
KLAB = "full space" if FULL else f"top {K} PCs"
WINDOW = M.WINDOW


def main():
    neu_z, common, _ = PP.build_pseudopop()
    data = load_podcast_patient(REF_PID, region=M.REGION, model_tag=M.MODEL, context_words=200)
    emb = np.array(np.load(data["emb_path"], mmap_mode="r")[M.LAYER, common], dtype=np.float64)
    surp = np.asarray(data["surp"], dtype=float)[common]
    log_freq = M.subtlex_log_freq(np.asarray(data["words"])[common])
    n = len(common)

    c_sem = M.trajectory_curvature(emb if FULL else pca_fixed(emb, k=K))
    c_neu = M.trajectory_curvature(neu_z if FULL else pca_fixed(neu_z, k=K))

    elig = dict(all=np.isfinite(log_freq), mid=M.midfreq_mask(log_freq))
    ev = {}
    for name, e in elig.items():
        ev[("sem", name)] = M.event_windows(c_sem, surp, e, n)
        ev[("neu", name)] = M.event_windows(c_neu, surp, e, n)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.8, wspace=0.32, hspace=0.6)
    lags = np.arange(-WINDOW, WINDOW + 1)
    titles = {("sem", "all"): f"A  Semantic ({M.MODEL}, layer {M.LAYER}, {KLAB})\nAll words",
              ("neu", "all"): f"B  Neural pseudopop ({neu_z.shape[1]} units, counts, {KLAB})\nAll words",
              ("sem", "mid"): f"C  Semantic ({M.MODEL}, layer {M.LAYER}, {KLAB})\nMiddle 10% SUBTLEX freq only",
              ("neu", "mid"): f"D  Neural pseudopop ({neu_z.shape[1]} units, counts, {KLAB})\nMiddle 10% SUBTLEX freq only"}
    pos = {("sem", "all"): (0, 0), ("neu", "all"): (0, 1), ("sem", "mid"): (1, 0), ("neu", "mid"): (1, 1)}
    for key, (hi, lo) in ev.items():
        ax = axes[pos[key]]
        for rows, col, lab in ((hi, M.C_HI, "High surprisal (top 25%)"), (lo, M.C_LO, "Low surprisal (bottom 25%)")):
            mu, se = np.nanmean(rows, axis=0), scipy_sem(rows, axis=0, nan_policy="omit")
            ax.plot(lags, mu, color=col, lw=1.8, label=lab)
            ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.18)
        h, l = np.nanmean(hi, axis=0), np.nanmean(lo, axis=0)
        print(f"  {key[0]} {key[1]}: lag0 {h[WINDOW]:.1f}/{l[WINDOW]:.1f}  lag+1 {h[WINDOW + 1]:.1f}/{l[WINDOW + 1]:.1f}  "
              f"lag+2 {h[WINDOW + 2]:.1f}/{l[WINDOW + 2]:.1f}  baseline {np.r_[h[:3], l[:3]].mean():.1f}")
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
        ax.set_ylabel("Mean curvature (°)", fontsize=10)
        ax.set_title(f"{titles[key]}\n{hi.shape[0]}+{lo.shape[0]} events, SEM", loc="left", fontsize=9.5, pad=3)
        ax.tick_params(labelsize=9)
        ax.set_xticks(lags)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")
    fig.suptitle(f"Podcast event-triggered curvature, hippocampal pseudopopulation  ·  {M.MODEL} layer {M.LAYER}, "
                 f"{KLAB}  ·  all words vs. SUBTLEX middle-frequency-only", fontsize=9, fontweight="bold", y=0.995)
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29_pseudopop_" + ("full" if FULL else f"pc{K}")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
