"""
make_podcast_event_triggered_freq_surp.py
===========================================
Event-triggered curvature, split two ways (word frequency, surprisal) in
two panels: NEURAL space (hippocampus, PCA'd spikes to each patient's
matched n_pcs) and SEMANTIC space (gpt2, fixed K=SEM_K PCA components,
averaged over layers).

Panel A (neural): frequent/infrequent word (orange/green), hi/lo surprisal
(red/blue).
Panel B (semantic, K=20): same 4 lines, computed on the gpt2 embedding.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from podcast_loader import PODCAST_PATIENTS, load_podcast_patient
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import fit_neural_pca, trajectory_curvature

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES  = os.path.join(BASE_DIR, "figures")
WINDOW   = 5
HI_PCTILE = 75
LO_PCTILE = 25
K_MAX = 300
SEM_K = 20
MODEL = "gpt2"

C_FREQ_HI = "#E67E22"   # orange -- most frequent word
C_FREQ_LO = "#27AE60"   # green  -- most infrequent word
C_SURP_HI = "#C0392B"   # red    -- high surprisal
C_SURP_LO = "#2980B9"   # blue   -- low surprisal

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def event_windows(values_by_index, split_var, n_words, index_offset=0):
    """values_by_index: array whose index i corresponds to word (i+index_offset).
    Returns (hi_rows, lo_rows), each [n_events, len(lags)], using split_var at
    the word the value 'arrives at' for the event-defining condition."""
    finite = np.isfinite(split_var)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)

    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values_by_index)
    for j in range(n_words):
        sj = split_var[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = []
        for k in lags:
            idx = (j + k) - index_offset
            row.append(values_by_index[idx] if 0 <= idx < m else np.nan)
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def semantic_curvature_fixedK(data, n, k=SEM_K):
    """Layer-averaged trajectory curvature of the gpt2 embedding, PCA-reduced
    to a fixed K components at each layer (top-K, not matched to neural n_pcs)."""
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]
    c_all = []
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        if emb_L.std() == 0:
            continue
        Xs = StandardScaler().fit_transform(emb_L)
        max_pc = min(Xs.shape[1], Xs.shape[0] - 1, K_MAX)
        if max_pc < k:
            continue
        pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
        Xk = pca.transform(Xs)[:, :k]
        c_all.append(trajectory_curvature(Xk))
    del emb_mm
    if not c_all:
        return None
    return np.mean(c_all, axis=0)


def collect_all():
    pat = {k: [] for k in ("freq_hi", "freq_lo", "surp_hi_neu", "surp_lo_neu",
                            "surp_hi_sem", "surp_lo_sem", "freq_hi_sem", "freq_lo_sem")}
    for short in PODCAST_PATIENTS:
        # gpt2/ctx200 cache (not the default lab-provided raw embeddings) so
        # the 4 extra-source patients are included too.
        data = load_podcast_patient(short, model_tag=MODEL, context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = data["log_freq"][:n]
        Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None or len(Y_neu) < 6:
            continue

        c_neu = trajectory_curvature(Y_neu)   # index i -> word i+2
        offset = 2

        c_sem = semantic_curvature_fixedK(data, n, SEM_K)
        if c_sem is None:
            continue

        fh, fl = event_windows(c_neu, log_freq, n, index_offset=offset)
        sh_n, sl_n = event_windows(c_neu, surp, n, index_offset=offset)
        sh_s, sl_s = event_windows(c_sem, surp, n, index_offset=offset)
        fh_s, fl_s = event_windows(c_sem, log_freq, n, index_offset=offset)
        if any(x is None for x in (fh, fl, sh_n, sl_n, sh_s, sl_s, fh_s, fl_s)):
            continue

        pat["freq_hi"].append(np.nanmean(fh, axis=0))
        pat["freq_lo"].append(np.nanmean(fl, axis=0))
        pat["surp_hi_neu"].append(np.nanmean(sh_n, axis=0))
        pat["surp_lo_neu"].append(np.nanmean(sl_n, axis=0))
        pat["surp_hi_sem"].append(np.nanmean(sh_s, axis=0))
        pat["surp_lo_sem"].append(np.nanmean(sl_s, axis=0))
        pat["freq_hi_sem"].append(np.nanmean(fh_s, axis=0))
        pat["freq_lo_sem"].append(np.nanmean(fl_s, axis=0))
        print(f"  {short}: n_pcs={n_pcs}  done", flush=True)

    return {k: np.array(v) for k, v in pat.items()}


def main():
    pat = collect_all()
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = pat["freq_hi"].shape[0]

    fig, (ax_neu, ax_sem) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.8, wspace=0.28)

    def draw(ax, key, color, ls, label):
        arr = pat[key]
        mu, se = np.nanmean(arr, axis=0), scipy_sem(arr, axis=0, nan_policy="omit")
        ax.plot(lags, mu, color=color, lw=1.6, ls=ls, label=label)
        ax.fill_between(lags, mu - se, mu + se, color=color, alpha=0.15)

    draw(ax_neu, "freq_hi",     C_FREQ_HI, "-", "Most frequent word")
    draw(ax_neu, "freq_lo",     C_FREQ_LO, "-", "Most infrequent word")
    draw(ax_neu, "surp_hi_neu", C_SURP_HI, "-", "High surprisal")
    draw(ax_neu, "surp_lo_neu", C_SURP_LO, "-", "Low surprisal")

    draw(ax_sem, "freq_hi_sem", C_FREQ_HI, "-", "Most frequent word")
    draw(ax_sem, "freq_lo_sem", C_FREQ_LO, "-", "Most infrequent word")
    draw(ax_sem, "surp_hi_sem", C_SURP_HI, "-", "High surprisal")
    draw(ax_sem, "surp_lo_sem", C_SURP_LO, "-", "Low surprisal")

    for ax, title in ((ax_neu, "A  Neural (hippocampus)"), (ax_sem, f"B  Semantic (gpt2, K={SEM_K})")):
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("Words relative to event", fontsize=9.5)
        ax.set_ylabel("Mean curvature (°)", fontsize=9.5)
        ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10, pad=3)
        ax.tick_params(labelsize=8.5)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.8, edgecolor="#DDDDDD")
        ax.set_xticks(lags)

    fig.suptitle("Podcast · event-triggered curvature: word frequency vs. surprisal, neural vs. semantic space",
                 fontsize=9.5, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_event_triggered_freq_surp"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
