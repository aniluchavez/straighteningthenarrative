"""
make_podcast_curvature_context_surprisal.py
==============================================
Same event-triggered curvature design as make_podcast_curvature_allwords_vs_midfreq.py,
but the hi/lo SPLIT VARIABLE is a centered rolling-mean ("context") surprisal
instead of single-word surprisal -- testing whether the surprisal-linked
curvature effect (and its neural-vs-semantic peak timing) sharpens when
surprisal is measured with more context, consistent with the timescale
finding in compute_freq_surprisal_full_grid_test.py / make_freq_surp_timescale_summary.py
(surprisal's unique contribution to curvature grows under multi-word
averaging while frequency's shrinks).

Curvature itself stays word-by-word (values array unchanged); only the
variable used to pick "high-surprisal" vs "low-surprisal" events is
context-averaged, over CONTEXT_WINDOW words centered on the event word.

Produces one 2x2 figure per CONTEXT_WINDOW in CONTEXT_WINDOWS (rows =
semantic/neural, cols = all words / mid-freq), all words only used by
default (mid-freq restriction + context-averaging combined would leave too
few eligible windows).

Saves: figures/fig_podcast_curvature_context_surprisal_w{W}.{pdf,png,svg}

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_podcast_curvature_context_surprisal.py
"""
from __future__ import annotations
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
WINDOW = 5
MODEL = "llama-2-7b"
LAYER = 18
REGION = "hippocampus"
HI_PCTILE = 75
LO_PCTILE = 25
CONTEXT_WINDOWS = [1, 5, 10, 15]  # 1 = original word-by-word split (baseline)

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import fit_neural_pca, trajectory_curvature
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

C_HI = "#C0392B"
C_LO = "#2980B9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def centered_rolling_mean(x, w):
    """Centered rolling mean over w words (w=1 -> unchanged). NaN-aware,
    requires at least half the window to be finite."""
    if w <= 1:
        return x.copy()
    n = len(x)
    half = w // 2
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + (w - half))
        seg = x[lo:hi]
        fin = np.isfinite(seg)
        if fin.sum() >= max(2, w // 2):
            out[i] = np.mean(seg[fin])
    return out


def event_windows(values, split_var, n_words):
    finite = np.isfinite(split_var)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_words):
        sj = split_var[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = [values[(j + k) - 2] if 0 <= (j + k) - 2 < m else np.nan for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def _panel(ax, hi, lo, title, event_band=None):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_count = hi.shape[0]
    hi_mu = np.nanmean(hi, axis=0)
    lo_mu = np.nanmean(lo, axis=0)
    if event_band is not None:
        hi_ev, lo_ev = event_band
        hi_se = scipy_sem(hi_ev, axis=0, nan_policy="omit")
        lo_se = scipy_sem(lo_ev, axis=0, nan_policy="omit")
        band_label = f"  ({hi_ev.shape[0]}+{lo_ev.shape[0]} events, SEM)"
    else:
        hi_se = scipy_sem(hi, axis=0, nan_policy="omit")
        lo_se = scipy_sem(lo, axis=0, nan_policy="omit")
        band_label = ""
    ax.plot(lags, hi_mu, color=C_HI, lw=1.8, label="High context-surprisal (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.8, label="Low context-surprisal (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} patients{band_label}", loc="left", fontsize=9.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    # load once per patient, compute per-context-window downstream
    patient_cache = {}
    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data.get("neu_vecs") is None or data.get("emb_path") is None:
            print(f"  {pid}: SKIP (no data)")
            continue
        n = data["n_words"]
        surp = data["surp"][:n]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            print(f"  {pid}: SKIP (no neural PCA)")
            continue
        c_neu = trajectory_curvature(Y_neu)

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        c_sem = trajectory_curvature(emb_L)

        patient_cache[pid] = dict(n=n, surp=surp, c_sem=c_sem, c_neu=c_neu)
        print(f"  {pid}: loaded n_words={n}", flush=True)

    os.makedirs(FIGURES, exist_ok=True)

    for w in CONTEXT_WINDOWS:
        pat_hi_sem, pat_lo_sem = [], []
        pat_hi_neu, pat_lo_neu = [], []
        sem_hi_events, sem_lo_events = None, None

        for pid, d in patient_cache.items():
            surp_ctx = centered_rolling_mean(d["surp"], w)
            hi_s, lo_s = event_windows(d["c_sem"], surp_ctx, d["n"])
            hi_n, lo_n = event_windows(d["c_neu"], surp_ctx, d["n"])
            if hi_s is not None and hi_n is not None:
                pat_hi_sem.append(np.nanmean(hi_s, axis=0)); pat_lo_sem.append(np.nanmean(lo_s, axis=0))
                pat_hi_neu.append(np.nanmean(hi_n, axis=0)); pat_lo_neu.append(np.nanmean(lo_n, axis=0))
                if sem_hi_events is None:
                    sem_hi_events, sem_lo_events = hi_s, lo_s

        pat_hi_sem, pat_lo_sem = np.array(pat_hi_sem), np.array(pat_lo_sem)
        pat_hi_neu, pat_lo_neu = np.array(pat_hi_neu), np.array(pat_lo_neu)
        print(f"\ncontext_window={w}: n_patients={len(pat_hi_sem)}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        _panel(axes[0], pat_hi_sem, pat_lo_sem,
               f"A  Semantic ({MODEL}, layer {LAYER})", event_band=(sem_hi_events, sem_lo_events))
        _panel(axes[1], pat_hi_neu, pat_lo_neu, "B  Neural (spikes)")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06),
                   fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

        ctx_desc = "single-word surprisal (baseline)" if w == 1 else f"{w}-word centered context-mean surprisal"
        fig.suptitle(f"Podcast event-triggered curvature, hippocampus  ·  {MODEL} layer {LAYER}  ·  "
                     f"split on {ctx_desc}", fontsize=9, fontweight="bold", y=1.14)
        fig.tight_layout()

        stem = f"fig_podcast_curvature_context_surprisal_w{w}"
        for ext in ("pdf", "png", "svg"):
            path = os.path.join(FIGURES, f"{stem}.{ext}")
            fig.savefig(path)
            print(f"  -> {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
