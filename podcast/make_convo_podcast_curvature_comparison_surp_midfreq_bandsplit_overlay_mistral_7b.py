"""
make_convo_podcast_curvature_comparison_surp_midfreq_bandsplit_overlay.py
=============================================================================
Overlay comparison: same event-triggered curvature analysis, computed
separately for the 45-50th and 50-55th percentile SUBTLEX-frequency
sub-bands, plotted together (solid = 45-50%, dashed = 50-55%) to directly
show whether the surprisal effect is consistent across the two halves of
the "middle 10%" frequency band (i.e., not driven by residual frequency
variation within that band).

Saves: figures/fig_convo_podcast_curvature_comparison_surp_midfreq_bandsplit_overlay.{pdf,png,svg}
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
WINDOW = 5
CONVO_MODEL = "mistral-7b"
REGION = "hippocampus"
BANDS = [(45, 50, "45-50%", "-"), (50, 55, "50-55%", "--")]

sys.path.insert(0, PARENT_DIR)
import run_geometry_analysis as _rga
from wordfreq import zipf_frequency
import re as _re

_zipf_cache = {}
def _subtlex_get_log_freq(words):
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = _re.sub(r"[^a-z0-9]", "", str(w).lower())
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out
_rga.get_log_freq = _subtlex_get_log_freq

from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, build_paths,
)
import podcast_loader as _pl
_pl.get_log_freq = _subtlex_get_log_freq
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

PODCAST_SEM_MODEL = "mistral-7b"
PODCAST_SEM_SINGLE_PID = "YEY"
HI_PCTILE = 75
LO_PCTILE = 25
C_HI = "#C0392B"
C_LO = "#2980B9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def curvature_avg_all_layers(emb_mm, n):
    n_layers = emb_mm.shape[0]
    curves = [trajectory_curvature(np.array(emb_mm[L, :n], dtype=np.float64)) for L in range(n_layers)]
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


def load_convo_bandsplit():
    """Compute curvature once per patient, then split into both bands."""
    results = {b[2]: dict(hi_sem=[], lo_sem=[], hi_neu=[], lo_neu=[]) for b in BANDS}
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = data["log_freq"][:n]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)

        paths = build_paths(pid, CONVO_MODEL)
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        c_sem_full = curvature_avg_all_layers(emb_mm, n)
        del emb_mm

        for lo, hi, label, _ in BANDS:
            surp_b, n_band = band_masked_surp(surp, log_freq, lo, hi)
            hi_s, lo_s = event_windows(c_sem_full, surp_b, n)
            hi_n, lo_n = event_windows(c_neu_full, surp_b, n)
            if hi_s is None or hi_n is None:
                continue
            results[label]["hi_sem"].append(np.nanmean(hi_s, axis=0)); results[label]["lo_sem"].append(np.nanmean(lo_s, axis=0))
            results[label]["hi_neu"].append(np.nanmean(hi_n, axis=0)); results[label]["lo_neu"].append(np.nanmean(lo_n, axis=0))
        print(f"  convo {pid}: done", flush=True)
    return results


def load_podcast_bandsplit():
    results = {b[2]: dict(hi_neu=[], lo_neu=[]) for b in BANDS}
    sem_result = {b[2]: dict(hi_sem=[], lo_sem=[]) for b in BANDS}
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, model_tag="gpt2", context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = data["log_freq"][:n]
        X, _ = fit_neural_pca(data["neu_vecs"], n)
        if X is None:
            continue
        c = trajectory_curvature(X)
        for lo, hi, label, _ in BANDS:
            surp_b, n_band = band_masked_surp(surp, log_freq, lo, hi)
            hi_rows, lo_rows = event_windows(c, surp_b, n)
            if hi_rows is None:
                continue
            results[label]["hi_neu"].append(np.nanmean(hi_rows, axis=0))
            results[label]["lo_neu"].append(np.nanmean(lo_rows, axis=0))
        print(f"  podcast neural {short}: done", flush=True)

    # single-patient semantic (YEY)
    data = load_podcast_patient(PODCAST_SEM_SINGLE_PID, model_tag=PODCAST_SEM_MODEL, context_words=200)
    n = data["n_words"]
    surp = data["surp"][:n]
    log_freq = data["log_freq"][:n]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    c_sem_full = curvature_avg_all_layers(emb_mm, n)
    del emb_mm
    for lo, hi, label, _ in BANDS:
        surp_b, n_band = band_masked_surp(surp, log_freq, lo, hi)
        hi_rows, lo_rows = event_windows(c_sem_full, surp_b, n)
        if hi_rows is not None:
            sem_result[label]["hi_sem"] = hi_rows
            sem_result[label]["lo_sem"] = lo_rows
    return results, sem_result


def main():
    convo = load_convo_bandsplit()
    podcast_neu, podcast_sem = load_podcast_bandsplit()

    lags = np.arange(-WINDOW, WINDOW + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_A, ax_B, ax_C, ax_D = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    for lo, hi, label, ls in BANDS:
        r = convo[label]
        if r["hi_sem"]:
            mu_hi = np.nanmean(r["hi_sem"], axis=0); se_hi = scipy_sem(r["hi_sem"], axis=0, nan_policy="omit")
            mu_lo = np.nanmean(r["lo_sem"], axis=0); se_lo = scipy_sem(r["lo_sem"], axis=0, nan_policy="omit")
            ax_A.plot(lags, mu_hi, color=C_HI, ls=ls, lw=1.6, label=f"High surp. ({label})")
            ax_A.fill_between(lags, mu_hi - se_hi, mu_hi + se_hi, color=C_HI, alpha=0.13, lw=0)
            ax_A.plot(lags, mu_lo, color=C_LO, ls=ls, lw=1.6, label=f"Low surp. ({label})")
            ax_A.fill_between(lags, mu_lo - se_lo, mu_lo + se_lo, color=C_LO, alpha=0.13, lw=0)
        if r["hi_neu"]:
            mu_hi = np.nanmean(r["hi_neu"], axis=0); se_hi = scipy_sem(r["hi_neu"], axis=0, nan_policy="omit")
            mu_lo = np.nanmean(r["lo_neu"], axis=0); se_lo = scipy_sem(r["lo_neu"], axis=0, nan_policy="omit")
            ax_B.plot(lags, mu_hi, color=C_HI, ls=ls, lw=1.6, label=f"High surp. ({label})")
            ax_B.fill_between(lags, mu_hi - se_hi, mu_hi + se_hi, color=C_HI, alpha=0.13, lw=0)
            ax_B.plot(lags, mu_lo, color=C_LO, ls=ls, lw=1.6, label=f"Low surp. ({label})")
            ax_B.fill_between(lags, mu_lo - se_lo, mu_lo + se_lo, color=C_LO, alpha=0.13, lw=0)

        rp = podcast_sem[label]
        if len(rp.get("hi_sem", [])):
            # rp["hi_sem"]/["lo_sem"] are per-EVENT rows (single patient), not per-patient means
            mu_hi = np.nanmean(rp["hi_sem"], axis=0); se_hi = scipy_sem(rp["hi_sem"], axis=0, nan_policy="omit")
            mu_lo = np.nanmean(rp["lo_sem"], axis=0); se_lo = scipy_sem(rp["lo_sem"], axis=0, nan_policy="omit")
            ax_C.plot(lags, mu_hi, color=C_HI, ls=ls, lw=1.6, label=f"High surp. ({label})")
            ax_C.fill_between(lags, mu_hi - se_hi, mu_hi + se_hi, color=C_HI, alpha=0.13, lw=0)
            ax_C.plot(lags, mu_lo, color=C_LO, ls=ls, lw=1.6, label=f"Low surp. ({label})")
            ax_C.fill_between(lags, mu_lo - se_lo, mu_lo + se_lo, color=C_LO, alpha=0.13, lw=0)

        rn = podcast_neu[label]
        if rn["hi_neu"]:
            mu_hi = np.nanmean(rn["hi_neu"], axis=0); se_hi = scipy_sem(rn["hi_neu"], axis=0, nan_policy="omit")
            mu_lo = np.nanmean(rn["lo_neu"], axis=0); se_lo = scipy_sem(rn["lo_neu"], axis=0, nan_policy="omit")
            ax_D.plot(lags, mu_hi, color=C_HI, ls=ls, lw=1.6, label=f"High surp. ({label})")
            ax_D.fill_between(lags, mu_hi - se_hi, mu_hi + se_hi, color=C_HI, alpha=0.13, lw=0)
            ax_D.plot(lags, mu_lo, color=C_LO, ls=ls, lw=1.6, label=f"Low surp. ({label})")
            ax_D.fill_between(lags, mu_lo - se_lo, mu_lo + se_lo, color=C_LO, alpha=0.13, lw=0)

    for ax, title in [(ax_A, "A  Convo semantic (mistral-7b, all layers avg)"),
                       (ax_B, "B  Convo neural (spikes)"),
                       (ax_C, "C  Podcast semantic (mistral-7b, all layers avg, YEY only)"),
                       (ax_D, "D  Podcast neural (spikes)")]:
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word")
        ax.set_ylabel("Mean curvature (°)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.legend(fontsize=6.5, framealpha=0.85, ncol=2)

    fig.suptitle("Event-triggered curvature, hippocampus · solid = 45-50th pctile SUBTLEX freq, dashed = 50-55th pctile\n"
                  "(overlay comparison: does the surprisal effect hold consistently across both halves of the mid-frequency band?)",
                  fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_podcast_curvature_comparison_surp_midfreq_bandsplit_overlay_mistral_7b"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
