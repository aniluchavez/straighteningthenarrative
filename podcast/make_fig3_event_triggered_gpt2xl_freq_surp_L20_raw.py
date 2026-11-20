"""
make_fig3_event_triggered_gpt2xl_freq_surp.py
====================================
fig3_event_triggered_pca20_podcast.pdf's design (event-triggered semantic +
neural curvature, hi/lo split, PCA20-reduced trajectory), for gpt2-xl at a
FIXED layer (not the per-patient LOO layer the original pipeline uses),
run twice: once split by SURPRISAL (matching the original fig3) and once
split by WORD FREQUENCY (log_freq, new). 2x2 grid: rows = split variable,
columns = semantic / neural curvature.

Layer = 5 (gpt2-xl's best curvature-alignment layer from fig2's Panel A/D
scan, Delta r = 0.066) -- "that layer" from the surprisal-geometry figure.

Neural side: pipeline-DEFAULT fixed window (not dur80/40 rate-normalized) --
kept consistent with fig2/fig3's existing methodology for direct
comparability, since this is meant as a same-pipeline reproduction.

Saves: figures/fig3_event_triggered_gpt2xl_freq_surp.{pdf,png,svg}
"""
import os, sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis_pca20 import (
    trajectory_curvature, fit_neural_pca, EVENT_WINDOW,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(PARENT_DIR, "figures")

MODEL = "gpt2-xl"
LAYER = 20  # best raw-dim neural-shape match (r=+0.559, peak_lag=+1)
REGION = "hippocampus"
PCTILE_HI, PCTILE_LO = 75, 25

C_HI = "#C0392B"
C_LO = "#2980B9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 10.5,
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


def quantile_event_masks(x, hi_pctile=PCTILE_HI, lo_pctile=PCTILE_LO):
    x = np.asarray(x, dtype=float)
    x_valid = x[np.isfinite(x)]
    hi_thr = float(np.percentile(x_valid, hi_pctile))
    lo_thr = float(np.percentile(x_valid, lo_pctile))
    return x >= hi_thr, x <= lo_thr


def compute_event_triggered_by(data, layer, Y_neu, split_var):
    """Same logic as run_geometry_analysis_pca20.compute_event_triggered, but
    the hi/lo event split uses split_var (surp or log_freq) instead of being
    hardcoded to surprisal."""
    n = data["n_words"]
    W = EVENT_WINDOW

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    if layer >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    Xs = StandardScaler().fit_transform(emb_opt)  # raw dims, NO PCA

    c_sem_full = trajectory_curvature(Xs)
    c_neu_full = trajectory_curvature(Y_neu)[:n - 2] if Y_neu is not None else None

    hi_mask, lo_mask = quantile_event_masks(split_var[:n])

    lags = list(range(-W, W + 1))
    hi_sem = [[] for _ in lags]; lo_sem = [[] for _ in lags]
    hi_neu = [[] for _ in lags]; lo_neu = [[] for _ in lags]

    for j in range(n):
        is_hi, is_lo = hi_mask[j], lo_mask[j]
        if not (is_hi or is_lo):
            continue
        for ki, k in enumerate(lags):
            ci = (j + k) - 2
            if 0 <= ci < len(c_sem_full):
                (hi_sem if is_hi else lo_sem)[ki].append(c_sem_full[ci])
            if c_neu_full is not None and 0 <= ci < len(c_neu_full):
                (hi_neu if is_hi else lo_neu)[ki].append(c_neu_full[ci])

    def _m(lst):
        return float(np.mean(lst)) if lst else float("nan")
    return dict(lags=lags,
                hi_sem=[_m(x) for x in hi_sem], lo_sem=[_m(x) for x in lo_sem],
                hi_neu=[_m(x) for x in hi_neu] if c_neu_full is not None else None,
                lo_neu=[_m(x) for x in lo_neu] if c_neu_full is not None else None)


def collect_all():
    evts_surp, evts_freq = [], []
    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        et_surp = compute_event_triggered_by(data, LAYER, Y_neu, data["surp"])
        et_freq = compute_event_triggered_by(data, LAYER, Y_neu, data["log_freq"])
        if et_surp is not None:
            evts_surp.append(et_surp)
        if et_freq is not None:
            evts_freq.append(et_freq)
        print(f"  {pid}: done", flush=True)
    return evts_surp, evts_freq


def _panel(ax, evts, key_hi, key_lo, title, lbl_hi, lbl_lo):
    evts_valid = [e for e in evts if e[key_hi] is not None]
    if not evts_valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, loc="left")
        return
    lags = np.array(evts_valid[0]["lags"])

    def _stack(key):
        mat = np.array([e[key] for e in evts_valid], dtype=float)
        return np.nanmean(mat, 0), scipy_sem(mat, axis=0, nan_policy="omit")

    hi_mu, hi_se = _stack(key_hi)
    lo_mu, lo_se = _stack(key_lo)
    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label=lbl_hi)
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label=lbl_lo)
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=9.5)
    ax.set_title(f"{title}\n$n$ = {len(evts_valid)} patients", loc="left", fontsize=10, pad=3)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="best", fontsize=7, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)


def main():
    evts_surp, evts_freq = collect_all()

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.9, wspace=0.32, hspace=0.42)

    _panel(axes[0, 0], evts_surp, "hi_sem", "lo_sem", "A  Semantic curvature (surprisal)",
           "High surprisal (top 25%)", "Low surprisal (bottom 25%)")
    _panel(axes[0, 1], evts_surp, "hi_neu", "lo_neu", "B  Neural curvature (surprisal)",
           "High surprisal (top 25%)", "Low surprisal (bottom 25%)")
    _panel(axes[1, 0], evts_freq, "hi_sem", "lo_sem", "C  Semantic curvature (frequency)",
           "Most frequent word (top 25%)", "Most infrequent word (bottom 25%)")
    _panel(axes[1, 1], evts_freq, "hi_neu", "lo_neu", "D  Neural curvature (frequency)",
           "Most frequent word (top 25%)", "Most infrequent word (bottom 25%)")

    fig.suptitle(f"{MODEL}  ·  podcast  ·  event-triggered curvature  ·  raw semantic trajectory (no PCA)  ·  layer {LAYER}  ·  "
                 "lag 0 uses words t-2..t",
                 fontsize=9.5, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig3_event_triggered_gpt2xl_freq_surp_L20_raw"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
