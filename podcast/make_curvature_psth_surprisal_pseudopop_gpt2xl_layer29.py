"""
make_curvature_psth_surprisal_pseudopop_gpt2xl_layer29.py
============================================================
Event-triggered curvature ("curvature PSTH"), high vs low surprisal
(top / bottom 25% of surprisal over the common words). Same lag convention
as the earlier event-triggered figures: lag 0 = the angle whose second step
ENDS at the event word (vertex at word j-1); lag +1 = turning angle AT the
event word (vertex); lag k = turning angle at word j+k-1.

Semantic = gpt2-xl layer 29 (YEY embeddings at the common words), neural =
pseudopopulation (475 units, 13 patients), fixed [onset+80, onset+330]
window spike counts (results/pseudopop_full_sequence_fixed250_count.npz).
Both reduced to the top K standardized PCs (argv[1], default 475; "full" =
semantic raw embedding, neural all units). Bands = SEM across events.

Rows: all words; middle 10% SUBTLEX frequency (45-55th pct); top 25% most
frequent. The high/low surprisal split (top/bottom 25%) is taken WITHIN each
band's words.

Saves: figures/fig_curvature_psth_surprisal_pseudopop_gpt2xl_layer29_{pcK|full}_freqbands.{pdf,png,svg}
"""
import os
import sys

import numpy as np
from scipy.stats import sem as scipy_sem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

ARG = sys.argv[1] if len(sys.argv) > 1 else "475"
sys.argv = sys.argv[:1]
from run_geometry_analysis_pca20 import trajectory_curvature
from podcast_loader import load_podcast_patient
from make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 import subtlex_log_freq
import compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop as PP

PP.RATE = False
PP.ONSET_PAD_MS, PP.OFFSET_PAD_MS = 80, 330
_lwoo = PP.load_words_onset_offset
PP.load_words_onset_offset = lambda pid: (lambda w, on, off: (w, on, on))(*_lwoo(pid))
PP.CACHE_PATH = os.path.join(RESULTS, "pseudopop_full_sequence_fixed250_count.npz")

MODEL, LAYER = "gpt2-xl", 29
WINDOW = 5
HI_P, LO_P = 75, 25
C_HI, C_LO = "#C0392B", "#2980B9"
BANDS = [("All words", None, None), ("Middle 10% SUBTLEX frequency (45–55th pct)", 45, 55),
         ("Top 25% most frequent (SUBTLEX)", 75, 100)]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def topk(X, k):
    Xs = np.nan_to_num(StandardScaler().fit_transform(X))
    return PCA(n_components=min(k, *Xs.shape), svd_solver="full").fit_transform(Xs)


def main():
    neu_z, common, _ = PP.build_pseudopop()
    n = len(common)
    d = load_podcast_patient(PP.REF_PID, region="hippocampus", model_tag=MODEL, context_words=200)
    surp = np.asarray(d["surp"], dtype=float)[common]
    emb = np.array(np.load(d["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    if ARG == "full":
        Y = dict(sem=emb, neu=np.nan_to_num(neu_z))
        klab = "full space"
    else:
        K = int(ARG)
        Y = dict(sem=topk(emb, K), neu=topk(neu_z, K))
        klab = f"top {K} PCs"
    # curvature AT word j (vertex) -> index j-1 of trajectory_curvature
    at = {s: np.r_[np.nan, trajectory_curvature(y), np.nan][:n] for s, y in Y.items()}

    lf = subtlex_log_freq(np.asarray(d["words"])[common])
    lags = np.arange(-WINDOW, WINDOW + 1)
    fig, axes = plt.subplots(len(BANDS), 2, figsize=(11, 4.2 * len(BANDS)))
    for r, (blab, lo_p, hi_p) in enumerate(BANDS):
        f = np.isfinite(surp) & np.isfinite(lf)
        if lo_p is not None:
            a, b = np.percentile(lf[f], [lo_p, hi_p])
            f &= (lf >= a) & (lf <= b)
        hi_thr, lo_thr = np.percentile(surp[f], HI_P), np.percentile(surp[f], LO_P)
        ev = {}
        for name, mask in (("hi", f & (surp >= hi_thr)), ("lo", f & (surp <= lo_thr))):
            js = np.where(mask)[0]
            ev[name] = js[(js - WINDOW - 1 >= 1) & (js + WINDOW - 1 < n - 1)]
        for c, (side, sname) in enumerate((("sem", f"Semantic ({MODEL} layer {LAYER}, {klab})"),
                                           ("neu", f"Neural pseudopop ({neu_z.shape[1]} units, {klab})"))):
            ax = axes[r, c]
            out = {}
            for name, col, lab in (("hi", C_HI, "High surprisal (top 25% within band)"),
                                   ("lo", C_LO, "Low surprisal (bottom 25% within band)")):
                rows = at[side][ev[name][:, None] + lags[None, :] - 1]
                mu, se = np.nanmean(rows, axis=0), scipy_sem(rows, axis=0, nan_policy="omit")
                out[name] = mu
                ax.plot(lags, mu, color=col, lw=1.8, marker="o", ms=3.5, label=lab)
                ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.18)
            ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.axvline(1, color="0.6", lw=0.7, ls=":")
            ax.set_xticks(lags)
            ax.set_ylabel("Turning angle at that word (°)")
            ax.set_title(f"{'ABCDEF'[2 * r + c]}  {sname}\n{blab}; {len(ev['hi'])}+{len(ev['lo'])} events",
                         loc="left", fontsize=10)
            if r == len(BANDS) - 1:
                ax.set_xlabel("Words relative to event\nlag 0: curvature ends at the event word\n"
                              "lag +1: angle at the event word (dotted)")
            base = np.r_[out["hi"][:3], out["lo"][:3]].mean()
            print(f"  {blab[:22]:22s} {side}: " + "  ".join(f"lag{k:+d} {out['hi'][i]:.1f}/{out['lo'][i]:.1f}"
                                             for i, k in enumerate(lags) if -1 <= k <= 2) + f"   base {base:.1f}")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle(f"Curvature around high- vs low-surprisal words — neural: fixed 250 ms window [onset+80, onset+330] "
                 f"spike counts; mean ± SEM across events", fontsize=11, y=1.035)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_psth_surprisal_pseudopop_gpt2xl_layer29_" + ("full" if ARG == "full" else f"pc{ARG}") + "_freqbands"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
