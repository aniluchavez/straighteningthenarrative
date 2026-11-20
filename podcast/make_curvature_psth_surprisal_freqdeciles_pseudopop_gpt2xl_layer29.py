"""
make_curvature_psth_surprisal_freqdeciles_pseudopop_gpt2xl_layer29.py
========================================================================
Decile version of make_curvature_psth_surprisal_pseudopop_gpt2xl_layer29.py
(freqbands): event-triggered curvature, high vs low surprisal (top/bottom
25% WITHIN each band), for each SUBTLEX frequency decile of the common words.
Lag convention as before: lag 0 = angle whose second step ends at the event
word; lag +1 = turning angle AT the event word.

Semantic = gpt2-xl layer 29 top K standardized PCs; neural = pseudopop (475
units), fixed [onset+80, onset+330] counts, top K PCs (K = argv[1], default
475). Bands = SEM across events.

argv[2] = "surp": the mirror -- surprisal deciles, split = high vs low SUBTLEX
frequency (top/bottom 25% within decile).

Saves: figures/fig_curvature_psth_{surprisal_freqdeciles|frequency_surpdeciles}_pseudopop_gpt2xl_layer29_pcK.{pdf,png,svg}
       figures/fig_curvature_psth_{...}_summary_pseudopop_gpt2xl_layer29_pcK.{pdf,png,svg}
"""
import os
import sys

import numpy as np
from scipy.stats import sem as scipy_sem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
ARG = sys.argv[1] if len(sys.argv) > 1 else "475"
MODE = sys.argv[2] if len(sys.argv) > 2 else "freq"
TAG = "surprisal_freqdeciles" if MODE == "freq" else "frequency_surpdeciles"
sys.argv = sys.argv[:1]
import make_curvature_psth_surprisal_pseudopop_gpt2xl_layer29 as B
from podcast_loader import load_podcast_patient
from run_geometry_analysis_pca20 import trajectory_curvature

N_DEC = 10
FIGURES = B.FIGURES


def main():
    neu_z, common, _ = B.PP.build_pseudopop()
    n = len(common)
    d = load_podcast_patient(B.PP.REF_PID, region="hippocampus", model_tag=B.MODEL, context_words=200)
    surp = np.asarray(d["surp"], dtype=float)[common]
    lf = B.subtlex_log_freq(np.asarray(d["words"])[common])
    emb = np.array(np.load(d["emb_path"], mmap_mode="r")[B.LAYER, common], dtype=np.float64)
    K = int(ARG)
    Y = dict(sem=B.topk(emb, K), neu=B.topk(neu_z, K))
    at = {s: np.r_[np.nan, trajectory_curvature(y), np.nan][:n] for s, y in Y.items()}
    lags = np.arange(-B.WINDOW, B.WINDOW + 1)
    fin = np.isfinite(surp) & np.isfinite(lf)
    band_var, split_var = (lf, surp) if MODE == "freq" else (surp, lf)
    bname, sname = ("freq", "surprisal") if MODE == "freq" else ("surprisal", "frequency")
    unit = "Zipf" if MODE == "freq" else "surprisal"
    edges = np.percentile(band_var[fin], np.linspace(0, 100, N_DEC + 1))

    res = {}  # (side, dec) -> dict(hi_mu, hi_se, lo_mu, lo_se, n_hi, n_lo)
    for q in range(N_DEC):
        band = fin & (band_var >= edges[q]) & ((band_var <= edges[q + 1]) if q == N_DEC - 1 else (band_var < edges[q + 1]))
        hi_thr, lo_thr = np.percentile(split_var[band], B.HI_P), np.percentile(split_var[band], B.LO_P)
        ev = {}
        for name, mask in (("hi", band & (split_var >= hi_thr)), ("lo", band & (split_var <= lo_thr))):
            js = np.where(mask)[0]
            ev[name] = js[(js - B.WINDOW - 1 >= 1) & (js + B.WINDOW - 1 < n - 1)]
        for side in ("sem", "neu"):
            r = {}
            for name in ("hi", "lo"):
                rows = at[side][ev[name][:, None] + lags[None, :] - 1]
                r[f"{name}_mu"] = np.nanmean(rows, axis=0)
                r[f"{name}_se"] = scipy_sem(rows, axis=0, nan_policy="omit")
                r[f"n_{name}"] = len(ev[name])
            r["zipf"] = (edges[q], edges[q + 1])
            res[(side, q)] = r
            i1 = B.WINDOW + 1
            print(f"  dec{q + 1:2d} ({unit} {edges[q]:.2f}-{edges[q + 1]:.2f}) {side}: "
                  f"lag0 {r['hi_mu'][B.WINDOW]:.1f}/{r['lo_mu'][B.WINDOW]:.1f}  "
                  f"lag+1 {r['hi_mu'][i1]:.1f}/{r['lo_mu'][i1]:.1f}  diff+1 {r['hi_mu'][i1] - r['lo_mu'][i1]:+.1f}  "
                  f"n={r['n_hi']}+{r['n_lo']}", flush=True)

    # grid: rows 0-1 semantic (deciles 1-5, 6-10), rows 2-3 neural
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=True)
    for si, side in enumerate(("sem", "neu")):
        for q in range(N_DEC):
            ax = axes[2 * si + q // 5, q % 5]
            r = res[(side, q)]
            for name, col in (("hi", B.C_HI), ("lo", B.C_LO)):
                ax.plot(lags, r[f"{name}_mu"], color=col, lw=1.5, marker="o", ms=2.5)
                ax.fill_between(lags, r[f"{name}_mu"] - r[f"{name}_se"], r[f"{name}_mu"] + r[f"{name}_se"],
                                color=col, alpha=0.18)
            ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
            ax.axvline(1, color="0.6", lw=0.6, ls=":")
            ax.set_xticks(lags[::2])
            space = "Semantic" if side == "sem" else "Neural"
            ax.set_title(f"{space} · {bname} decile {q + 1} ({unit} {r['zipf'][0]:.2f}–{r['zipf'][1]:.2f})\n"
                         f"{r['n_hi']}+{r['n_lo']} events", loc="left", fontsize=9)
            if q % 5 == 0:
                ax.set_ylabel("Turning angle (°)")
            if 2 * si + q // 5 == 3:
                ax.set_xlabel("Words relative to event\n(+1 = angle at event word)")
    for row in range(4):
        lo = min(a.get_ylim()[0] for a in axes[row]); hi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(lo, hi)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=B.C_HI, lw=2), Line2D([], [], color=B.C_LO, lw=2)],
               [f"High {sname} (top 25% within decile)", f"Low {sname} (bottom 25% within decile)"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    ttl = ("high- vs low-surprisal words, by SUBTLEX frequency decile (1 = rarest)" if MODE == "freq"
           else "high- vs low-frequency words, by surprisal decile (1 = least surprising)")
    fig.suptitle(f"Curvature around {ttl} — "
                 f"{B.MODEL} layer {B.LAYER} and neural pseudopop (fixed 250 ms window), top {K} PCs",
                 fontsize=12, y=1.025)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = f"fig_curvature_psth_{TAG}_pseudopop_gpt2xl_layer29_pc{K}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")

    # summary: high - low at lag 0 and lag +1 by decile
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    x = np.arange(1, N_DEC + 1)
    for ax, side, title in ((axes[0], "sem", f"A  Semantic ({B.MODEL} layer {B.LAYER}, top {K} PCs)"),
                            (axes[1], "neu", f"B  Neural pseudopop (475 units, top {K} PCs)")):
        for lagv, col, mk in ((1, "k", "o"), (0, "0.55", "s")):
            i = B.WINDOW + lagv
            dd = np.array([res[(side, q)]["hi_mu"][i] - res[(side, q)]["lo_mu"][i] for q in range(N_DEC)])
            se = np.array([np.hypot(res[(side, q)]["hi_se"][i], res[(side, q)]["lo_se"][i]) for q in range(N_DEC)])
            lab = "lag +1 (angle at the event word)" if lagv == 1 else "lag 0 (step into the event word)"
            ax.errorbar(x + (0.08 if lagv == 0 else 0), dd, yerr=se, color=col, marker=mk, ms=4.5, lw=1.6, capsize=3,
                        label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xlabel("SUBTLEX frequency decile (1 = rarest, 10 = most frequent)" if MODE == "freq"
                      else "Surprisal decile (1 = least surprising, 10 = most surprising)")
        ax.set_ylabel(f"High − low {sname} turning angle (°)")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{sname.capitalize()} effect on curvature within each {bname} decile (mean ± SE of the difference)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    stem = f"fig_curvature_psth_{TAG}_summary_pseudopop_gpt2xl_layer29_pc{K}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
