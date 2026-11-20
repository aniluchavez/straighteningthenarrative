"""
make_podcast_curvature_chunk15_event_triggered.py
====================================================
Event-triggered curvature, but on the CHUNK-LEVEL (15-word) series used in
compute_freq_surprisal_full_grid_test.py's "chunk15" cells, instead of the
word-by-word series used in make_podcast_curvature_allwords_vs_midfreq.py.

Construction (matches build_chunked_df / _chunk_mean there): word-level
curvature and surprisal are each averaged within non-overlapping 15-word
windows (min 50% finite required) to build one curvature value and one
surprisal value per chunk. Events are then high/low-surprisal CHUNKS (top
vs bottom 25% of chunk-mean surprisal), and lags are in chunk units (each
step = 15 words) instead of word units.

This is the "coarse-grained" analogue of
make_podcast_curvature_context_surprisal_wide.py's wide-lag word-level
figure -- same idea, but curvature itself (not just the surprisal split)
is computed on the chunked series, matching the unique-R2 grid exactly.

Saves: figures/fig_podcast_curvature_chunk15_event_triggered.{pdf,png,svg}

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_podcast_curvature_chunk15_event_triggered.py
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
CHUNK = 15
WINDOW_CHUNKS = 5  # +/- 5 chunks = +/- 75 words
MODEL = "llama-2-7b"
LAYER = 18
REGION = "hippocampus"
HI_PCTILE = 75
LO_PCTILE = 25

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


def chunk_mean(x, cs, min_frac=0.5):
    """Non-overlapping cs-word windows, mean of finite values, dropped if
    < min_frac of the window is finite. Returns chunked array."""
    n = len(x)
    out = []
    for i in range(0, n - cs + 1, cs):
        seg = x[i:i + cs]
        fin = np.isfinite(seg)
        if fin.sum() < cs * min_frac:
            out.append(np.nan)
        else:
            out.append(np.mean(seg[fin]))
    return np.array(out)


def event_windows_chunked(values, split_var, n_chunks):
    finite = np.isfinite(split_var)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)
    lags = list(range(-WINDOW_CHUNKS, WINDOW_CHUNKS + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_chunks):
        sj = split_var[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = [values[j + k] if 0 <= j + k < m else np.nan for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW_CHUNKS:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def _panel(ax, hi, lo, title, event_band=None):
    lags = np.arange(-WINDOW_CHUNKS, WINDOW_CHUNKS + 1)
    n_count = hi.shape[0]
    hi_mu = np.nanmean(hi, axis=0)
    lo_mu = np.nanmean(lo, axis=0)
    if event_band is not None:
        hi_ev, lo_ev = event_band
        hi_se = scipy_sem(hi_ev, axis=0, nan_policy="omit")
        lo_se = scipy_sem(lo_ev, axis=0, nan_policy="omit")
        band_label = f"  ({hi_ev.shape[0]}+{lo_ev.shape[0]} chunk-events, SEM)"
    else:
        hi_se = scipy_sem(hi, axis=0, nan_policy="omit")
        lo_se = scipy_sem(lo, axis=0, nan_policy="omit")
        band_label = ""
    ax.plot(lags, hi_mu, color=C_HI, lw=1.8, marker="o", ms=4, label="High surprisal chunk (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.8, marker="o", ms=4, label="Low surprisal chunk (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel(f"15-word chunks relative to event chunk\n(1 chunk = {CHUNK} words)", fontsize=9)
    ax.set_ylabel("Mean curvature (°), chunk-averaged", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} patients{band_label}", loc="left", fontsize=9.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    pat_hi_sem, pat_lo_sem = [], []
    pat_hi_neu, pat_lo_neu = [], []
    sem_hi_events, sem_lo_events = None, None

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

        # word-level curvature arrays are length n-2 (index i = curvature at word i+1);
        # chunk them directly, and chunk surprisal aligned to the same word range.
        surp_al = surp[2:n]
        c_sem_chunk = chunk_mean(c_sem, CHUNK)
        c_neu_chunk = chunk_mean(c_neu, CHUNK)
        surp_chunk = chunk_mean(surp_al, CHUNK)
        n_chunks = min(len(c_sem_chunk), len(c_neu_chunk), len(surp_chunk))
        c_sem_chunk, c_neu_chunk, surp_chunk = c_sem_chunk[:n_chunks], c_neu_chunk[:n_chunks], surp_chunk[:n_chunks]

        hi_s, lo_s = event_windows_chunked(c_sem_chunk, surp_chunk, n_chunks)
        hi_n, lo_n = event_windows_chunked(c_neu_chunk, surp_chunk, n_chunks)
        if hi_s is not None and hi_n is not None:
            pat_hi_sem.append(np.nanmean(hi_s, axis=0)); pat_lo_sem.append(np.nanmean(lo_s, axis=0))
            pat_hi_neu.append(np.nanmean(hi_n, axis=0)); pat_lo_neu.append(np.nanmean(lo_n, axis=0))
            if sem_hi_events is None:
                sem_hi_events, sem_lo_events = hi_s, lo_s

        print(f"  {pid}: n_words={n}  n_chunks={n_chunks}  done", flush=True)

    pat_hi_sem, pat_lo_sem = np.array(pat_hi_sem), np.array(pat_lo_sem)
    pat_hi_neu, pat_lo_neu = np.array(pat_hi_neu), np.array(pat_lo_neu)
    print(f"\nn_patients: {len(pat_hi_sem)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _panel(axes[0], pat_hi_sem, pat_lo_sem, f"A  Semantic ({MODEL}, layer {LAYER})", event_band=(sem_hi_events, sem_lo_events))
    _panel(axes[1], pat_hi_neu, pat_lo_neu, "B  Neural (spikes)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle(f"Podcast event-triggered curvature, hippocampus  ·  {MODEL} layer {LAYER}  ·  "
                 f"CHUNK-LEVEL (15-word chunks, matching the unique-$R^2$ grid)",
                 fontsize=9, fontweight="bold", y=1.16)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_curvature_chunk15_event_triggered"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
