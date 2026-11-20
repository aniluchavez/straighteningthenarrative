"""
make_curvature_vs_surprisal_gpt2xl_layer29_pc64_perpatient.py
================================================================
Does curvature go up with surprisal? gpt2-xl layer 29, per patient,
semantic top-64 PCs matched to the neural side (K = min(64, n_units);
neural = duration window [onset+80, offset+40] raw spike counts,
standardized PCA). All 13 patients (YFX clock-fixed).

  A/B  event-triggered curvature, high vs low surprisal (top/bottom 25%),
       lags -5..+5 (lag +1 = turning angle AT the event word)
  C/D  curvature AT each word (lag +1 alignment) by surprisal decile,
       valid words; per-patient decile means, then mean +/- SEM across
       patients (semantic: SEM across words of one patient)

argv[1] = "full": semantic = raw 1600-d embedding, neural = all units z-scored.

Saves: figures/fig_curvature_vs_surprisal_gpt2xl_layer29_{pc64|full}_perpatient.{pdf,png,svg}
"""
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem, spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

FULL = len(sys.argv) > 1 and sys.argv[1] == "full"
sys.argv = sys.argv[:1]
import make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 as M
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs
from compute_curv_disp_pos_gpt2xl_layer29 import pca_fixed

K = 64
N_BINS = 10
WINDOW = M.WINDOW


def main():
    ev = {"sem": ([], []), "neu": ([], [])}
    dec = {"sem": [], "neu": []}
    rho = {"sem": [], "neu": []}
    sem_first = None
    for pid in PODCAST_PATIENTS:
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region=M.REGION, model_tag=M.MODEL, context_words=200)
        if data is None:
            continue
        n = min(data["n_words"], len(on))
        neu = windowed_neu_vecs(pid, M.REGION, _convo_regions(pid), on[:n], off[:n], 80, 40, rate=False)
        if neu is None:
            continue
        surp = np.asarray(data["surp"][:n], dtype=float)
        valid = np.zeros(n, bool)
        v0 = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        valid[:min(n, len(v0))] = v0[:n]
        Yn = np.nan_to_num(StandardScaler().fit_transform(neu[:n])) if FULL else pca_fixed(neu[:n], k=K)
        k = Yn.shape[1]
        emb = np.array(np.load(data["emb_path"], mmap_mode="r")[M.LAYER, :n], dtype=np.float64)
        curv = dict(sem=M.trajectory_curvature(emb if FULL else pca_fixed(emb, k=k)), neu=M.trajectory_curvature(Yn))
        allw = np.isfinite(surp)
        # curvature AT word j = curv[j-1]
        at = {s: np.r_[np.nan, c, np.nan][:n] for s, c in curv.items()}
        m = valid & np.isfinite(surp) & np.isfinite(at["sem"])
        edges = np.percentile(surp[m], np.linspace(0, 100, N_BINS + 1))
        b = np.clip(np.searchsorted(edges, surp[m], side="right") - 1, 0, N_BINS - 1)
        for s in ("sem", "neu"):
            hi, lo = M.event_windows(curv[s], surp, allw, n)
            ev[s][0].append(np.nanmean(hi, axis=0)); ev[s][1].append(np.nanmean(lo, axis=0))
            v = at[s][m]
            dec[s].append([v[b == i].mean() for i in range(N_BINS)])
            rho[s].append(spearmanr(surp[m], v)[0])
        if sem_first is None:
            hi, lo = M.event_windows(curv["sem"], surp, allw, n)
            v = at["sem"][m]
            sem_first = (hi, lo, [scipy_sem(v[b == i]) for i in range(N_BINS)], pid)
        print(f"  {pid}: units={neu.shape[1]} K={k}  spearman(surp, curv at word) sem {rho['sem'][-1]:+.3f} "
              f"neu {rho['neu'][-1]:+.3f}", flush=True)

    lags = np.arange(-WINDOW, WINDOW + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for c, s in enumerate(("sem", "neu")):
        ax = axes[0, c]
        H, L = np.array(ev[s][0]), np.array(ev[s][1])
        for arr, raw, col, lab in ((H, sem_first[0], M.C_HI, "High surprisal (top 25%)"),
                                   (L, sem_first[1], M.C_LO, "Low surprisal (bottom 25%)")):
            mu = arr.mean(axis=0)
            se = scipy_sem(raw, axis=0, nan_policy="omit") if s == "sem" else scipy_sem(arr, axis=0)
            ax.plot(lags, mu, color=col, lw=1.8, label=lab)
            ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.18)
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.axvline(1, color="0.6", lw=0.7, ls=":")
        ax.set_xticks(lags)
        ax.set_xlabel("Words relative to event (lag +1 = angle at the event word)", fontsize=9)
        ax.set_ylabel("Mean curvature (°)")
        name = (f"Semantic ({M.MODEL} L{M.LAYER}, {'full space' if FULL else f'top {K} PCs'})" if s == "sem"
                    else f"Neural (per patient, {'all units' if FULL else 'matched PCs'})")
        ax.set_title(f"{'AB'[c]}  {name}\nn = {len(H)} patients", loc="left", fontsize=10)
        hm, lm = H.mean(axis=0), L.mean(axis=0)
        print(f"  {s} event: lag0 {hm[WINDOW]:.1f}/{lm[WINDOW]:.1f}  lag+1 {hm[WINDOW + 1]:.1f}/{lm[WINDOW + 1]:.1f}  "
              f"lag+2 {hm[WINDOW + 2]:.1f}/{lm[WINDOW + 2]:.1f}")

        ax = axes[1, c]
        D = np.array(dec[s])
        mu = D.mean(axis=0)
        se = np.array(sem_first[2]) if s == "sem" else scipy_sem(D, axis=0)
        x = np.arange(1, N_BINS + 1)
        ax.errorbar(x, mu, yerr=se, color="k", marker="o", ms=4, lw=1.6, capsize=3)
        ax.set_xticks(x)
        ax.set_xlabel("Surprisal decile (1 = least surprising)")
        ax.set_ylabel("Curvature at the word (°)")
        r = np.array(rho[s])
        ax.set_title(f"{'CD'[c]}  Curvature at word vs. surprisal\nSpearman ρ per patient: mean {r.mean():+.3f}, "
                     f"{int((r > 0).sum())}/{len(r)} > 0", loc="left", fontsize=10)
        print(f"  {s} deciles: " + " ".join(f"{v:.1f}" for v in mu) + f"   rho mean {r.mean():+.3f} ({(r > 0).sum()}/{len(r)} > 0)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"Curvature vs. surprisal — {M.MODEL} layer {M.LAYER}, {'full space' if FULL else f'top {K} PCs matched per patient'}, "
                 "duration-window spike counts", fontsize=12, y=1.04)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = f"fig_curvature_vs_surprisal_gpt2xl_layer29_{'full' if FULL else f'pc{K}'}_perpatient"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
