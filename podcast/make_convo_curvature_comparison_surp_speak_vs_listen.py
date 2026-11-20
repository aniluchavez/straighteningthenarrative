"""
make_convo_curvature_comparison_surp_speak_vs_listen.py
=============================================
Convo dataset only (podcast is single-speaker listening-only, no "speaking"
condition to compare against). Splits event words by whether the PATIENT was
speaking (is_speaker==True) or listening (is_speaker==False) at that word,
then within each condition independently, splits into high (top 25%) vs low
(bottom 25%) surprisal -- threshold computed WITHIN that condition, same
convention as the frequency-band overlay scripts.

Hypothesis being tested: surprisal modulation of curvature should be WEAKER
for the patient's own spoken words than for words they're listening to --
the LM's surprisal reflects how unpredictable a word is to an outside
observer, but the patient generated their own speech, so it isn't
"surprising" to them in the same sense it's surprising to a listener.

Same raw/no-PCA embeddings, same loo_layer_neu layer, same event_windows()
lag convention as the rest of this session's event-triggered figures.

Saves: figures/fig_convo_curvature_comparison_surp_speak_vs_listen.{pdf,png,svg}
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem as scipy_sem

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
FIGURES     = os.path.join(BASE_DIR, "figures")
WINDOW      = 5
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"
HI_PCTILE = 75
LO_PCTILE = 25

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, build_paths,
)

C_HI = "#C0392B"   # red  -- high surprisal
C_LO = "#2980B9"   # blue -- low surprisal

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


def speaker_masked_surp(surp, is_speaker, want_speaking):
    """surp, with entries outside the requested speaking/listening condition
    set to NaN."""
    mask = (is_speaker.astype(bool) == want_speaking)
    out = np.full_like(surp, np.nan, dtype=float)
    out[mask] = surp[mask]
    return out, int(mask.sum())


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


def load_convo_condition(want_speaking):
    conv_path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    cached = {r["pid"]: r["loo_layer_neu"] for r in json.load(open(conv_path))}

    pat_hi_sem, pat_lo_sem, pat_hi_neu, pat_lo_neu = [], [], [], []
    n_events_total = 0
    for pid, preg in PATIENTS.items():
        if pid not in cached or REGION not in preg:
            continue
        layer = cached[pid]
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        is_speaker = data["is_speaker"][:n]
        surp_cond, n_cond = speaker_masked_surp(surp, is_speaker, want_speaking)
        if n_cond < 20:
            continue

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)

        paths = build_paths(pid, CONVO_MODEL)
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        c_sem_full = trajectory_curvature(emb_opt)

        hi_s, lo_s = event_windows(c_sem_full, surp_cond, n)
        hi_n, lo_n = event_windows(c_neu_full, surp_cond, n)
        if hi_s is None or hi_n is None:
            continue
        pat_hi_sem.append(np.nanmean(hi_s, axis=0)); pat_lo_sem.append(np.nanmean(lo_s, axis=0))
        pat_hi_neu.append(np.nanmean(hi_n, axis=0)); pat_lo_neu.append(np.nanmean(lo_n, axis=0))
        n_events_total += n_cond
        cond_label = "speaking" if want_speaking else "listening"
        print(f"  convo {pid} [{cond_label}]: layer={layer}  n_words_in_cond={n_cond}  done", flush=True)

    return (np.array(pat_hi_sem), np.array(pat_lo_sem)), (np.array(pat_hi_neu), np.array(pat_lo_neu))


def _panel(ax, hi, lo, title, n_label="patients"):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_count = hi.shape[0]
    hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
    lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")

    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)
    ax.set_ylabel("Mean curvature (°)", fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_count} {n_label}", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.set_xticks(lags)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    hi_swing = float(np.nanmax(hi_mu) - np.nanmin(hi_mu))
    lo_swing = float(np.nanmax(lo_mu) - np.nanmin(lo_mu))
    sep = float(np.nanmax(np.abs(hi_mu - lo_mu)))
    return dict(n=n_count, hi_swing=hi_swing, lo_swing=lo_swing, max_hi_lo_sep=sep)


def main():
    (hi_speak_sem, lo_speak_sem), (hi_speak_neu, lo_speak_neu) = load_convo_condition(want_speaking=True)
    (hi_listen_sem, lo_listen_sem), (hi_listen_neu, lo_listen_neu) = load_convo_condition(want_speaking=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.8, wspace=0.32, hspace=0.6)

    stats = {}
    stats["A_speaking_semantic"] = _panel(axes[0, 0], hi_speak_sem, lo_speak_sem, f"A  Speaking — semantic ({CONVO_MODEL})")
    stats["B_speaking_neural"] = _panel(axes[0, 1], hi_speak_neu, lo_speak_neu, "B  Speaking — neural (spikes)")
    stats["C_listening_semantic"] = _panel(axes[1, 0], hi_listen_sem, lo_listen_sem, f"C  Listening — semantic ({CONVO_MODEL})")
    stats["D_listening_neural"] = _panel(axes[1, 1], hi_listen_neu, lo_listen_neu, "D  Listening — neural (spikes)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.9, edgecolor="#DDDDDD")

    fig.suptitle("Event-triggered curvature, hippocampus  ·  convo only  ·  "
                 "surprisal split within speaking vs listening",
                 fontsize=9, fontweight="bold", y=0.995)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_convo_curvature_comparison_surp_speak_vs_listen"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)

    print("\n=== max |hi-lo| separation (curvature deg), speaking vs listening ===")
    for k, v in stats.items():
        print(f"  {k}: n={v['n']}  hi_swing={v['hi_swing']:.3f}  lo_swing={v['lo_swing']:.3f}  max_sep={v['max_hi_lo_sep']:.3f}")


if __name__ == "__main__":
    main()
