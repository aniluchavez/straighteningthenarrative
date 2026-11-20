"""
make_podcast_curvature_quartiles_gpt2xl_layer29_perpatient.py
================================================================
Per-patient version of make_podcast_curvature_freq_quartiles_gpt2xl_layer29_pseudopop.py.
Full space: semantic = raw gpt2-xl layer 29 embedding; neural = each patient's
own hippocampal units, dur80/40 raw spike counts, z-scored (no PCA). All 13
patients (YFX with the +574.9 s clock fix in podcast_loader).

argv[1] = "freq" (default): columns = SUBTLEX frequency quartiles, split =
          high vs low surprisal (top vs bottom 25% within band)
          "surp": columns = surprisal quartiles, split = high vs low frequency
Bands/splits are computed within each patient's words. Per-patient event
means are averaged across patients; neural band = SEM across patients,
semantic band = SEM across events of the first patient (semantic curvature
is the same transcript for everyone).

Saves: figures/fig_podcast_curvature_{freq|surp}_quartiles_gpt2xl_layer29_perpatient.{pdf,png,svg}
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem as scipy_sem
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

MODE = sys.argv[1] if len(sys.argv) > 1 else "freq"
sys.argv = sys.argv[:1]
import make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 as M
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs

WINDOW = M.WINDOW
if MODE == "freq":
    BANDS = [("Top 25% most frequent", 75, 100), ("50–75th pct frequency", 50, 75),
             ("25–50th pct frequency", 25, 50), ("Bottom 25% (least frequent)", 0, 25)]
    NAMES = ("High surprisal (top 25% within band)", "Low surprisal (bottom 25% within band)")
else:
    BANDS = [("Top 25% most surprising", 75, 100), ("50–75th pct surprisal", 50, 75),
             ("25–50th pct surprisal", 25, 50), ("Bottom 25% (least surprising)", 0, 25)]
    NAMES = ("High frequency (top 25% within band)", "Low frequency (bottom 25% within band)")


def band_mask(var, fin, lo_p, hi_p):
    a, b = np.percentile(var[fin], [lo_p, hi_p])
    return fin & (var >= a) & ((var <= b) if hi_p == 100 else (var < b))


def main():
    # per[(side, band_idx)] = list of (hi_mean, lo_mean) per patient; ev[(band_idx)] = first patient's sem events
    per = {(s, b): [] for s in ("sem", "neu") for b in range(len(BANDS))}
    sem_ev = {}
    pids = []
    for pid in PODCAST_PATIENTS:
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region=M.REGION, model_tag=M.MODEL, context_words=200)
        if data is None or data.get("emb_path") is None:
            continue
        n = min(data["n_words"], len(onset_ms))
        neu = windowed_neu_vecs(pid, M.REGION, _convo_regions(pid), onset_ms[:n], offset_ms[:n], 80, 40, rate=False)
        if neu is None:
            continue
        surp = np.asarray(data["surp"][:n], dtype=float)
        lf = M.subtlex_log_freq(data["words"][:n])
        fin = np.isfinite(lf) & np.isfinite(surp)
        band_var, split_var = (lf, surp) if MODE == "freq" else (surp, lf)
        emb = np.array(np.load(data["emb_path"], mmap_mode="r")[M.LAYER, :n], dtype=np.float64)
        curv = dict(sem=M.trajectory_curvature(emb),
                    neu=M.trajectory_curvature(np.nan_to_num(StandardScaler().fit_transform(neu[:n]))))
        for b, (_, lo_p, hi_p) in enumerate(BANDS):
            m = band_mask(band_var, fin, lo_p, hi_p)
            for side in ("sem", "neu"):
                hi, lo = M.event_windows(curv[side], split_var, m, n)
                if hi is None:
                    continue
                per[(side, b)].append((np.nanmean(hi, axis=0), np.nanmean(lo, axis=0)))
                if side == "sem" and b not in sem_ev:
                    sem_ev[b] = (hi, lo)
        pids.append(pid)
        print(f"  {pid}: n_words={n} units={neu.shape[1]}", flush=True)

    lags = np.arange(-WINDOW, WINDOW + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharey="row")
    for b, (lab, _, _) in enumerate(BANDS):
        for r, side in enumerate(("sem", "neu")):
            ax = axes[r, b]
            H = np.array([x[0] for x in per[(side, b)]])
            L = np.array([x[1] for x in per[(side, b)]])
            for arr, ev_rows, col, name in ((H, sem_ev[b][0], M.C_HI, NAMES[0]), (L, sem_ev[b][1], M.C_LO, NAMES[1])):
                mu = np.nanmean(arr, axis=0)
                se = scipy_sem(ev_rows if side == "sem" else arr, axis=0, nan_policy="omit")
                ax.plot(lags, mu, color=col, lw=1.8, label=name)
                ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.18)
            h, l = np.nanmean(H, axis=0), np.nanmean(L, axis=0)
            print(f"  {side} {lab:30s} n_pat={len(H)}  lag-1 {h[WINDOW - 1]:.1f}/{l[WINDOW - 1]:.1f}  "
                  f"lag0 {h[WINDOW]:.1f}/{l[WINDOW]:.1f}  lag+1 {h[WINDOW + 1]:.1f}/{l[WINDOW + 1]:.1f}  "
                  f"lag+2 {h[WINDOW + 2]:.1f}/{l[WINDOW + 2]:.1f}  base {np.r_[h[:3], l[:3]].mean():.1f}")
            ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_xticks(lags)
            ax.tick_params(labelsize=9)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            space = f"Semantic ({M.MODEL} L{M.LAYER})" if side == "sem" else "Neural (per patient, all units)"
            band_txt = (f"SEM over {sem_ev[b][0].shape[0]}+{sem_ev[b][1].shape[0]} events ({pids[0]})"
                        if side == "sem" else "SEM across patients")
            ax.set_title(f"{'ABCDEFGH'[4 * r + b]}  {space}\n{lab}\nn = {len(H)} patients, {band_txt}",
                         loc="left", fontsize=9.5)
            if b == 0:
                ax.set_ylabel("Mean curvature (°)")
            if r == 1:
                ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word", fontsize=9)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=9.5, frameon=False)
    ttl = ("by surprisal, stratified by SUBTLEX frequency quartile" if MODE == "freq"
           else "by SUBTLEX frequency, stratified by surprisal quartile")
    fig.suptitle(f"Event-triggered curvature {ttl} — full space, {M.MODEL} layer {M.LAYER}, "
                 f"per patient (spike counts, all units)", fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = f"fig_podcast_curvature_{MODE}_quartiles_gpt2xl_layer29_perpatient"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
