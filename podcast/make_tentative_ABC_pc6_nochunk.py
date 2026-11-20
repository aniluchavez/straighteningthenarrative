"""
make_tentative_ABC_pc6_nochunk.py
====================================
Panels A-C, fixed top-6 PCA (matched, both neural and semantic sides),
dur80/40 window, spike RATE, WORD-LEVEL (no chunking) correlation, no model
highlighted. Saved as separate figure files in tentative_figures/.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_geometry_analysis_pca20 import (
    trajectory_curvature, build_valid_mask, MIN_PAIRS,
)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(PARENT_DIR, "tentative_figures")
K = 6

MODELS = ["bert-base", "roberta-base", "gpt2", "gpt2-medium", "gpt2-xl",
          "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]

import matplotlib.cm as cm
COLORS = cm.tab10(np.linspace(0, 1, 10))
MODEL_STYLE = {m: dict(color=COLORS[i], label=m, ls="-") for i, m in enumerate(MODELS)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2:
        return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def save_fig(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


def compute_all(models):
    """One pass over all patients/models/layers: mean curvature profile (panel A)
    and word-level r(neural, semantic) profile (panel B), both at fixed top-6 PCA."""
    profile_curv = {m: [] for m in models}     # per model: list of per-patient [n_layers] mean curvature
    profile_r = {m: [] for m in models}        # per model: list of per-patient [n_layers] word-level r

    neu_cache = {}
    for pid in PODCAST_PATIENTS:
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data0 = load_podcast_patient(pid, region="hippocampus", model_tag=models[0], context_words=200)
        if data0 is None:
            continue
        n0 = min(data0["n_words"], len(onset_ms))
        preg = _convo_regions(pid)
        neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n0], offset_ms[:n0], 80, 40, rate=True)
        if neu_vecs is None:
            continue
        neu_cache[pid] = dict(neu_vecs=neu_vecs, n0=n0)

    for model in models:
        print(f"=== {model} ===", flush=True)
        for pid in PODCAST_PATIENTS:
            if pid not in neu_cache:
                continue
            data = load_podcast_patient(pid, region="hippocampus", model_tag=model, context_words=200)
            if data is None:
                continue
            n = min(data["n_words"], neu_cache[pid]["n0"])
            valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
            if valid.sum() < MIN_PAIRS:
                continue

            Y_neu = pca_fixed(neu_cache[pid]["neu_vecs"][:n])
            if Y_neu is None:
                continue
            c_neu = trajectory_curvature(Y_neu)
            m_c = min(len(c_neu), len(valid))
            c_neu = c_neu[:m_c]; v = valid[:m_c]

            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]
            mean_curv_L, r_L = [], []
            for li in range(n_layers):
                emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
                Y_sem = pca_fixed(emb_l)
                if Y_sem is None:
                    mean_curv_L.append(np.nan); r_L.append(np.nan); continue
                c_sem = trajectory_curvature(Y_sem)
                m2 = min(len(c_sem), m_c)
                c_sem_v = c_sem[:m2]; c_neu_v = c_neu[:m2]; vv = v[:m2]
                mean_curv_L.append(float(np.nanmean(c_sem_v[vv])))
                fin = np.isfinite(c_sem_v) & np.isfinite(c_neu_v) & vv
                if fin.sum() < 15:
                    r_L.append(np.nan)
                else:
                    r, _ = pearsonr(c_sem_v[fin], c_neu_v[fin])
                    r_L.append(float(r))
            del emb_mm
            profile_curv[model].append(mean_curv_L)
            profile_r[model].append(r_L)
            print(f"  {pid}: n_layers={n_layers}", flush=True)

    return profile_curv, profile_r


def make_panel_A(profile_curv):
    GRID = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for model, st in MODEL_STYLE.items():
        rows = [r for r in profile_curv[model] if len(r)]
        if not rows:
            continue
        n_l = len(rows[0])
        xs_m = np.linspace(0, 1, n_l)
        mu_c = np.nanmean(np.array(rows, dtype=float), axis=0)
        ax.plot(GRID, np.interp(GRID, xs_m, mu_c),
                color=st["color"], ls=st["ls"], lw=1.6, alpha=0.85, label=st["label"])
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Mean curvature (°)")
    ax.set_title("A  Mean semantic curvature · all models · PC6", loc="left", pad=8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.legend(loc="best", fontsize=9, framealpha=0.85, edgecolor="#DDDDDD", ncol=2)
    fig.tight_layout()
    save_fig(fig, "panelA_mean_curvature_pc6_nochunk")
    plt.close(fig)


def make_panel_B(profile_r):
    GRID = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for model, st in MODEL_STYLE.items():
        rows = [r for r in profile_r[model] if len(r)]
        if not rows:
            continue
        n_l = len(rows[0])
        xs_m = np.linspace(0, 1, n_l)
        mu_r = np.nanmean(np.array(rows, dtype=float), axis=0)
        ax.plot(GRID, np.interp(GRID, xs_m, np.nan_to_num(mu_r)),
                color=st["color"], ls=st["ls"], lw=1.6, alpha=0.85, label=st["label"])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Pearson $r$ (curvature)")
    ax.set_title("B  Neural-LLM alignment · word-level · PC6\n(no chunking)", loc="left", pad=8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.legend(loc="best", fontsize=9, framealpha=0.85, edgecolor="#DDDDDD", ncol=2)
    fig.tight_layout()
    save_fig(fig, "panelB_alignment_pc6_nochunk")
    plt.close(fig)


def _density_hist(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    widths = np.diff(bins)
    total = counts.sum()
    return counts / (total * widths) if total else np.zeros_like(widths, dtype=float)


def _hist_xlim(*arrays):
    joined = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)])
    lo, hi = np.percentile(joined, [0.5, 99.5])
    pad = 0.06 * (hi - lo)
    return max(0.0, lo - pad), min(180.0, hi + pad)


def find_winner(profile_r):
    best = None
    for model, rows in profile_r.items():
        if not rows:
            continue
        mu = np.nanmean(np.array(rows, dtype=float), axis=0)
        if np.all(np.isnan(mu)):
            continue
        layer = int(np.nanargmax(mu[1:])) + 1  # exclude layer 0 (static embedding, not contextual)
        peak = float(mu[layer])
        if best is None or peak > best[2]:
            best = (model, layer, peak)
    return best


def make_panel_C(profile_r):
    winner = find_winner(profile_r)
    if winner is None:
        print("No winner for panel C"); return
    model, layer, peak_r = winner
    print(f"Panel C winner: model={model}  layer={layer}  peak word-level r={peak_r:+.4f}")

    neural_vals, llm_vals = [], []
    for pid in PODCAST_PATIENTS:
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=model, context_words=200)
        if data is None:
            continue
        n = min(data["n_words"], len(onset_ms))
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        if valid.sum() < MIN_PAIRS:
            continue
        preg = _convo_regions(pid)
        neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n], offset_ms[:n], 80, 40, rate=True)
        if neu_vecs is None:
            continue
        Y_neu = pca_fixed(neu_vecs[:n])
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)
        m_c = min(len(c_neu), valid.sum() if False else len(valid))
        c_neu = c_neu[:m_c]; v = valid[:m_c]
        neural_vals.append(c_neu[v & np.isfinite(c_neu)])

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm; continue
        emb_l = np.array(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        Y_sem = pca_fixed(emb_l)
        if Y_sem is None:
            continue
        c_sem = trajectory_curvature(Y_sem)
        m2 = min(len(c_sem), m_c)
        c_sem_v = c_sem[:m2]; vv = v[:m2]
        llm_vals.append(c_sem_v[vv & np.isfinite(c_sem_v)])

    if not neural_vals or not llm_vals:
        print("No histogram data"); return
    neural = np.concatenate(neural_vals)
    llm = np.concatenate(llm_vals)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    bins = np.linspace(0.0, 180.0, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    d_neu = _density_hist(neural, bins)
    d_llm = _density_hist(llm, bins)
    neural_color = "#148F77"; llm_color = "#5DADE2"
    ax.fill_between(centers, d_neu, step="mid", color=neural_color, alpha=0.20)
    ax.plot(centers, d_neu, color=neural_color, lw=1.6, label=f"Hippocampal neural ($N={len(neural):,}$)")
    ax.fill_between(centers, d_llm, step="mid", color=llm_color, alpha=0.18)
    ax.plot(centers, d_llm, color=llm_color, lw=1.6, label=f"{model} layer {layer} ($N={len(llm):,}$)")
    for arr, col in [(neural, neural_color), (llm, llm_color)]:
        mu = float(np.mean(arr)); sd = float(np.std(arr, ddof=1))
        ax.axvspan(mu - sd, mu + sd, color=col, alpha=0.07, lw=0)
        ax.axvline(mu, color=col, lw=1.0)
    xlo, xhi = _hist_xlim(neural, llm)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Curvature $\\kappa$ (°)")
    ax.set_ylabel("Density")
    ax.set_title("C  Curvature distribution\nbest-aligned (model, layer), word-level, PC6", loc="left", pad=8)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.85, edgecolor="#DDDDDD", handlelength=1.6)
    fig.tight_layout()
    save_fig(fig, "panelC_curvature_distribution_pc6_nochunk")
    plt.close(fig)


CACHE_PATH = os.path.join(BASE_DIR, "results", "tentative_pc6_profile_cache.json")

if __name__ == "__main__":
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached profiles from {CACHE_PATH}")
        cached = json.load(open(CACHE_PATH))
        profile_curv, profile_r = cached["profile_curv"], cached["profile_r"]
    else:
        profile_curv, profile_r = compute_all(MODELS)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        json.dump(dict(profile_curv=profile_curv, profile_r=profile_r), open(CACHE_PATH, "w"))
        print(f"Cached profiles to {CACHE_PATH}")
    print("Panel A..."); make_panel_A(profile_curv)
    print("Panel B..."); make_panel_B(profile_r)
    print("Panel C..."); make_panel_C(profile_r)
    print("\nDone.")
