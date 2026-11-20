"""
make_multimodel_layersweep_surprisal_curvature_pca20_convo.py
=============================================
Conversation-dataset counterpart to
podcast/make_multimodel_layersweep_surprisal_curvature_pca20_podcast.py:
same PCA-20-matched per-layer r(semantic curvature, neural curvature) sweep
across all 10 contextual models, but on convo data. Uses the same fixed
model->color mapping as make_tentative_paired_pc8_chunk3_convo.py isn't
color-per-model (that one uses metric colors) -- instead matches the
tab10-by-MODELS-list convention from podcast/make_tentative_ABC_pca20_nochunk.py
so convo and podcast layersweep figures stay visually comparable.

Saves: results/multimodel_layersweep_surprisal_curvature_pca20_convo.json
       figures/fig_multimodel_layersweep_surprisal_curvature_pca20_convo.{pdf,png,svg}
"""
from __future__ import annotations
import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION      = "hippocampus"
HI_PCTILE = 75
LO_PCTILE = 25
MIN_N     = 15
PCA_K     = 20

from run_geometry_analysis import (
    trajectory_curvature, build_valid_mask, fit_neural_pca, load_patient,
    PATIENTS, build_paths, MODELS,
)

MODELS_TO_RUN = {k: v for k, v in MODELS.items() if k != "fasttext-wiki"}

MODEL_ORDER = ["bert-base", "roberta-base", "gpt2", "gpt2-medium", "gpt2-xl",
               "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
COLORS = cm.tab10(np.linspace(0, 1, 10))
MODEL_COLOR = {m: COLORS[i] for i, m in enumerate(MODEL_ORDER)}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _safe_r(x, y, mask, min_n=MIN_N):
    m = mask & np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan")
    return float(pearsonr(x[m], y[m])[0])


def _pca_reduce(emb_L, k=PCA_K):
    Xs = StandardScaler().fit_transform(emb_L)
    kk = min(k, Xs.shape[1], Xs.shape[0] - 1)
    if kk < 2:
        return None
    return PCA(n_components=kk, svd_solver="randomized", random_state=0).fit_transform(Xs)


def run_model(model_tag, n_layers):
    per_patient_r_hi, per_patient_r_lo, per_patient_r_all = [], [], []
    n_patients_used = 0

    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, model_tag)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"]
        sp = data["sent_pos"]
        is_turn = data["is_turn"]

        valid = build_valid_mask(n, surp, sp, is_turn)
        if valid.sum() < MIN_N:
            continue

        centre_idx = np.arange(1, n - 1)[valid]
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

        paths = build_paths(pid, model_tag)
        if not os.path.exists(paths["cache"]):
            continue
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        if emb_mm.shape[0] < n_layers:
            del emb_mm
            continue

        r_hi_layers, r_lo_layers, r_all_layers = [], [], []
        for L in range(n_layers):
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
            Xs = _pca_reduce(emb_L)
            if Xs is None:
                r_hi_layers.append(float("nan")); r_lo_layers.append(float("nan")); r_all_layers.append(float("nan"))
                continue
            c_semL = trajectory_curvature(Xs)[valid]
            fin_L = np.isfinite(c_semL) & np.isfinite(c_neu) & np.isfinite(surp_y)

            mask_hi = fin_L & (surp_y >= hi_thr)
            mask_lo = fin_L & (surp_y <= lo_thr)
            r_hi_layers.append(_safe_r(c_semL, c_neu, mask_hi))
            r_lo_layers.append(_safe_r(c_semL, c_neu, mask_lo))
            r_all_layers.append(_safe_r(c_semL, c_neu, fin_L))
        del emb_mm

        per_patient_r_hi.append(r_hi_layers)
        per_patient_r_lo.append(r_lo_layers)
        per_patient_r_all.append(r_all_layers)
        n_patients_used += 1

    if n_patients_used == 0:
        return None

    r_hi_mat = np.array(per_patient_r_hi, dtype=float)
    r_lo_mat = np.array(per_patient_r_lo, dtype=float)
    r_all_mat = np.array(per_patient_r_all, dtype=float)
    r_hi_by_layer = np.nanmean(r_hi_mat, axis=0)
    r_lo_by_layer = np.nanmean(r_lo_mat, axis=0)
    r_all_by_layer = np.nanmean(r_all_mat, axis=0)
    best_layer = int(np.nanargmax(r_all_by_layer))

    return dict(
        model=model_tag, n_layers=n_layers, n_patients=n_patients_used,
        best_layer=best_layer,
        r_all_by_layer=r_all_by_layer.tolist(),
        r_hi_by_layer=r_hi_by_layer.tolist(),
        r_lo_by_layer=r_lo_by_layer.tolist(),
        r_all_best=float(r_all_by_layer[best_layer]),
        r_hi_best=float(r_hi_by_layer[best_layer]),
        r_lo_best=float(r_lo_by_layer[best_layer]),
    )


def main():
    cache_path = os.path.join(RESULTS, "multimodel_layersweep_surprisal_curvature_pca20_convo.json")
    results = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for model_tag, n_layers in MODELS_TO_RUN.items():
        if model_tag in results and results[model_tag] is not None:
            print(f"\n=== {model_tag}: already cached, skipping ===", flush=True)
            continue
        print(f"\n=== {model_tag} ({n_layers} layers, PCA-{PCA_K}, CONVO) ===", flush=True)
        out = run_model(model_tag, n_layers)
        if out is None:
            print("  SKIP (no usable data)")
            results[model_tag] = None
            continue
        print(f"  n_patients={out['n_patients']}  best_layer={out['best_layer']}/{n_layers-1}  "
              f"r_all={out['r_all_best']:+.4f}  r_hi={out['r_hi_best']:+.4f}  r_lo={out['r_lo_best']:+.4f}", flush=True)
        results[model_tag] = out
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(results, open(os.path.join(RESULTS, "multimodel_layersweep_surprisal_curvature_pca20_convo.json"), "w"), indent=2)

    valid_results = {k: v for k, v in results.items() if v is not None}
    print("\n\n=== SUMMARY (sorted by peak r_all, PCA-20, CONVO) ===")
    ranked = sorted(valid_results.values(), key=lambda r: -r["r_all_best"])
    for r in ranked:
        print(f"  {r['model']:15s}  best_layer={r['best_layer']:2d}/{r['n_layers']-1:2d}  "
              f"r_all={r['r_all_best']:+.4f}  r_hi={r['r_hi_best']:+.4f}  r_lo={r['r_lo_best']:+.4f}  "
              f"n_patients={r['n_patients']}")

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for r in ranked:
        n_layers = r["n_layers"]
        x_norm = np.linspace(0, 1, n_layers)
        color = MODEL_COLOR.get(r["model"], "gray")
        ax.plot(x_norm, r["r_all_by_layer"], color=color, lw=1.6,
                label=f"{r['model']} (peak={r['r_all_best']:+.3f} @ L{r['best_layer']}/{n_layers-1})")
    ax.axhline(0, color="#BBBBBB", lw=0.7, ls=":")
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("r(semantic curvature, neural curvature), all words")
    ax.set_title(f"Neural-semantic curvature correlation by layer, across models (PCA-{PCA_K})\nconvo, hippocampus", loc="left", fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#DDDDDD", loc="upper right")

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_multimodel_layersweep_surprisal_curvature_pca20_convo"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
