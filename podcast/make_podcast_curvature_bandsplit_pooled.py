"""
make_podcast_curvature_bandsplit_pooled.py
=============================================
Fixes the single-patient limitation in
make_convo_podcast_curvature_comparison_surp_midfreq_bandsplit_overlay_pca10.py's
podcast semantic panel (previously YEY only). Pools semantic curvature
across ALL 13 podcast patients, same PCA-10-all-layers-averaged design and
same two narrow SUBTLEX frequency bands (45-50th, 50-55th percentile), to
test whether the semantic/neural "timing decoupling" seen in the
single-patient version survives proper pooling or was a sampling artifact.

Saves: figures/fig_podcast_curvature_bandsplit_pooled.{pdf,png,svg}

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_podcast_curvature_bandsplit_pooled.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from wordfreq import zipf_frequency

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
WINDOW = 5
REGION = "hippocampus"
BANDS = [(45, 50, "45-50%", "-"), (50, 55, "50-55%", "--")]
K = 10
MODEL = "llama-3.1-8b"
HI_PCTILE, LO_PCTILE = 75, 25
C_HI, C_LO = "#C0392B", "#2980B9"

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import trajectory_curvature
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

_zipf_cache = {}
def subtlex_log_freq(words):
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = str(w).lower()
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2:
        return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def curvature_avg_all_layers_pca(emb_mm, n):
    n_layers = emb_mm.shape[0]
    curves = []
    for L in range(n_layers):
        Y = pca_fixed(np.array(emb_mm[L, :n], dtype=np.float64))
        if Y is None:
            continue
        curves.append(trajectory_curvature(Y))
    return np.mean(np.stack(curves, axis=0), axis=0)


def band_masked_surp(surp, log_freq, lo_pctile, hi_pctile):
    finite = np.isfinite(log_freq)
    lo_edge = np.nanpercentile(log_freq[finite], lo_pctile)
    hi_edge = np.nanpercentile(log_freq[finite], hi_pctile)
    in_band = finite & (log_freq >= lo_edge) & (log_freq <= hi_edge)
    out = np.full_like(surp, np.nan, dtype=float)
    out[in_band] = surp[in_band]
    return out, int(in_band.sum())


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


def main():
    results = {b[2]: dict(hi_sem=[], lo_sem=[], hi_neu=[], lo_neu=[]) for b in BANDS}

    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data.get("neu_vecs") is None or data.get("emb_path") is None:
            print(f"  {pid}: SKIP")
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        words = data["words"][:n]
        log_freq = subtlex_log_freq(words)

        Y_neu = pca_fixed(data["neu_vecs"][:n])
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        c_sem_full = curvature_avg_all_layers_pca(emb_mm, n)
        del emb_mm

        for lo, hi, label, _ in BANDS:
            surp_b, n_band = band_masked_surp(surp, log_freq, lo, hi)
            hi_s, lo_s = event_windows(c_sem_full, surp_b, n)
            hi_n, lo_n = event_windows(c_neu_full, surp_b, n)
            if hi_s is None or hi_n is None:
                continue
            results[label]["hi_sem"].append(np.nanmean(hi_s, axis=0)); results[label]["lo_sem"].append(np.nanmean(lo_s, axis=0))
            results[label]["hi_neu"].append(np.nanmean(hi_n, axis=0)); results[label]["lo_neu"].append(np.nanmean(lo_n, axis=0))
        print(f"  {pid}: n_words={n}  done", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lags = np.arange(-WINDOW, WINDOW + 1)

    for ax, key_hi, key_lo, title in [
        (axes[0], "hi_sem", "lo_sem", f"A  Podcast semantic ({MODEL}, PCA-10, all layers avg)\nALL 13 patients pooled"),
        (axes[1], "hi_neu", "lo_neu", "B  Podcast neural (spikes, PCA-10)\nAll 13 patients"),
    ]:
        for lo, hi, label, ls in BANDS:
            hi_arr = np.array(results[label][key_hi])
            lo_arr = np.array(results[label][key_lo])
            n_pat = hi_arr.shape[0]
            hi_mu, hi_se = np.nanmean(hi_arr, axis=0), scipy_sem(hi_arr, axis=0, nan_policy="omit")
            lo_mu, lo_se = np.nanmean(lo_arr, axis=0), scipy_sem(lo_arr, axis=0, nan_policy="omit")
            ax.plot(lags, hi_mu, color=C_HI, lw=1.8, ls=ls, label=f"High surp. ({label})")
            ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.15)
            ax.plot(lags, lo_mu, color=C_LO, lw=1.8, ls=ls, label=f"Low surp. ({label})")
            ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.15)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
        ax.set_ylabel("Mean curvature (°)", fontsize=10)
        ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10, pad=3)
        ax.set_xticks(lags)
        ax.legend(fontsize=7, loc="best", frameon=False)

    fig.suptitle("Podcast event-triggered curvature, hippocampus · MATCHED PCA-10 (both sides) · "
                 f"{MODEL}\nsolid = 45-50th pctile SUBTLEX freq, dashed = 50-55th pctile · semantic now pooled across ALL patients",
                 fontsize=9, fontweight="bold", y=1.05)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_podcast_curvature_bandsplit_pooled"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
