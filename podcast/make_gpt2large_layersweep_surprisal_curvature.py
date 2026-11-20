"""
make_gpt2large_layersweep_surprisal_curvature.py
=============================================
gpt2-large, convo dataset, all 15 patients. Matches compute_scatter_data.py's
r_hi_layers/r_lo_layers convention (the actual logic behind
fig2_surprisal_geometry Panel D), but standalone/fast -- no dependency on the
slow leave-one-out run_geometry_analysis.py pipeline or cached geometry_*.json.

  Panel A: correlation r(semantic curvature, neural curvature) at EVERY
           layer (0..36), computed separately within high- (top 25%) and
           low- (bottom 25%) surprisal words. Best layer = argmax(r_hi - r_lo).
  Panel B: at that best layer, scatter of surprisal (x) vs semantic
           curvature (y), ALL words, pooled across patients, with fitted
           line and Pearson r/p.

Raw (StandardScaler, no PCA) embeddings for semantic; fit_neural_pca
(variance-threshold PCA) for neural -- same feature construction as the rest
of this session. Curvature paired with surprisal at the endpoint word (same
convention as compute_scatter_data.py: trajectory_curvature()[i] pairs with
surp[i+2], so no word after the surprising word enters the curvature
estimate).

Saves: figures/fig_gpt2large_layersweep_surprisal_curvature.{pdf,png,svg}
"""
from __future__ import annotations
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES    = os.path.join(PARENT_DIR, "figures")
CONVO_MODEL = "gpt2-large"
N_LAYERS    = 37
REGION      = "hippocampus"
HI_PCTILE = 75
LO_PCTILE = 25
MIN_N     = 15

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    trajectory_curvature, build_valid_mask, fit_neural_pca, load_patient, PATIENTS, build_paths,
)

C_HI = "#C0392B"
C_LO = "#2980B9"

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


def _safe_r(x, y, mask, min_n=MIN_N):
    m = mask & np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan")
    return float(pearsonr(x[m], y[m])[0])


def main():
    per_patient_r_hi = []  # each: [n_layers]
    per_patient_r_lo = []
    per_patient_r_all = []
    surp_pool_by_layer = {L: [] for L in range(N_LAYERS)}
    curv_pool_by_layer = {L: [] for L in range(N_LAYERS)}
    n_patients_used = 0

    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            print(f"  {pid}: SKIP (no data)")
            continue
        n = data["n_words"]
        surp = data["surp"]
        sp = data["sent_pos"]
        is_turn = data["is_turn"]

        valid = build_valid_mask(n, surp, sp, is_turn)
        if valid.sum() < MIN_N:
            continue

        centre_idx = np.arange(1, n - 1)[valid]
        # trajectory_curvature()[i] pairs with surp[i+2] -- the endpoint of the
        # second displacement, so no word after the surprising word enters
        # the curvature estimate. Same convention as compute_scatter_data.py.
        surp_y = surp[centre_idx + 1]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]

        fin_base = np.isfinite(c_neu) & np.isfinite(surp_y)
        surp_fin = surp_y[fin_base]
        if len(surp_fin) < 20:
            continue
        hi_thr = np.percentile(surp_fin, HI_PCTILE)
        lo_thr = np.percentile(surp_fin, LO_PCTILE)

        paths = build_paths(pid, CONVO_MODEL)
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        if emb_mm.shape[0] < N_LAYERS:
            del emb_mm
            continue

        r_hi_layers, r_lo_layers, r_all_layers = [], [], []
        for L in range(N_LAYERS):
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
            Xs = StandardScaler().fit_transform(emb_L)
            c_semL = trajectory_curvature(Xs)[valid]
            fin_L = np.isfinite(c_semL) & np.isfinite(c_neu) & np.isfinite(surp_y)

            mask_hi = fin_L & (surp_y >= hi_thr)
            mask_lo = fin_L & (surp_y <= lo_thr)
            r_hi_layers.append(_safe_r(c_semL, c_neu, mask_hi))
            r_lo_layers.append(_safe_r(c_semL, c_neu, mask_lo))
            r_all_layers.append(_safe_r(c_semL, c_neu, fin_L))

            surp_pool_by_layer[L].extend(surp_y[fin_L].tolist())
            curv_pool_by_layer[L].extend(c_semL[fin_L].tolist())
        del emb_mm

        per_patient_r_hi.append(r_hi_layers)
        per_patient_r_lo.append(r_lo_layers)
        per_patient_r_all.append(r_all_layers)
        n_patients_used += 1
        print(f"  {pid}: n_valid={valid.sum()}  done", flush=True)

    r_hi_mat = np.array(per_patient_r_hi, dtype=float)  # [n_patients, n_layers]
    r_lo_mat = np.array(per_patient_r_lo, dtype=float)
    r_all_mat = np.array(per_patient_r_all, dtype=float)
    r_hi_by_layer = np.nanmean(r_hi_mat, axis=0)
    r_lo_by_layer = np.nanmean(r_lo_mat, axis=0)
    r_all_by_layer = np.nanmean(r_all_mat, axis=0)
    delta = r_hi_by_layer - r_lo_by_layer
    best_layer = int(np.nanargmax(r_all_by_layer))
    print(f"\nBest layer (max overall r_all, neural-semantic curvature) = {best_layer}  "
          f"(r_all={r_all_by_layer[best_layer]:+.4f}, r_hi={r_hi_by_layer[best_layer]:+.4f}, "
          f"r_lo={r_lo_by_layer[best_layer]:+.4f}, delta={delta[best_layer]:+.4f})")

    surp_best = np.array(surp_pool_by_layer[best_layer])
    curv_best = np.array(curv_pool_by_layer[best_layer])
    fin = np.isfinite(surp_best) & np.isfinite(curv_best)
    surp_best, curv_best = surp_best[fin], curv_best[fin]
    r_val, p_val = pearsonr(surp_best, curv_best)
    print(f"At best layer {best_layer} (ALL words): r(surprisal, curvature) = {r_val:+.4f}  p={p_val:.4g}  n={len(surp_best)}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    layers = np.arange(N_LAYERS)
    axes[0].plot(layers, r_hi_by_layer, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
    axes[0].plot(layers, r_lo_by_layer, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
    axes[0].plot(layers, r_all_by_layer, color="#333333", lw=1.6, ls="-.", label="All words")
    axes[0].axhline(0, color="#BBBBBB", lw=0.7, ls=":")
    axes[0].axvline(best_layer, color="k", lw=0.9, ls="--", alpha=0.6)
    axes[0].text(best_layer, axes[0].get_ylim()[1], f" best layer = {best_layer}",
                 fontsize=8, va="top", ha="left")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("r(semantic curvature, neural curvature)")
    axes[0].set_title(f"A  {CONVO_MODEL}: neural-semantic curvature r by layer, hi vs lo surprisal\n$n$ = {n_patients_used} patients",
                       loc="left", fontsize=10.5)
    axes[0].legend(fontsize=8, framealpha=0.9, edgecolor="#DDDDDD")

    axes[1].scatter(surp_best, curv_best, s=6, alpha=0.12, color="#555555", rasterized=True, linewidths=0)
    m, b = np.polyfit(surp_best, curv_best, 1)
    xr = np.linspace(surp_best.min(), surp_best.max(), 100)
    axes[1].plot(xr, m * xr + b, color="k", lw=1.8, ls="--")
    pstr = f"p = {p_val:.2e}" if p_val >= 0.001 else "p < 0.001"
    axes[1].text(0.03, 0.97, f"r = {r_val:+.3f},  {pstr}\nn = {len(surp_best)}  (all words)",
                 transform=axes[1].transAxes, fontsize=9, va="top")
    axes[1].set_xlabel("Surprisal")
    axes[1].set_ylabel("Semantic curvature (°)")
    axes[1].set_title(f"B  Surprisal vs curvature at layer {best_layer}", loc="left", fontsize=10.5)

    fig.suptitle(f"{CONVO_MODEL}  ·  convo, hippocampus  ·  layer sweep + surprisal-curvature correlation",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_gpt2large_layersweep_surprisal_curvature"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
