"""
make_tentative_ABC_pca20_nochunk.py
====================================
Panels A-C, PCA20 / dur80-40 window, WORD-LEVEL (no chunking) correlation,
no model highlighted (uniform styling across all 10 models). Saved as
separate figure files in tentative_figures/.

Panel A: mean semantic curvature per normalized layer position, all models.
Panel B: neural-LLM curvature alignment (word-level Pearson r) per layer, all models.
Panel C: curvature distribution at the single best-aligned (model, layer) pair,
         chosen by peak word-level r across all models/layers (not hardcoded).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(PARENT_DIR, "tentative_figures")
CTX_TAG = "_ctx200"

MODELS = ["bert-base", "roberta-base", "gpt2", "gpt2-medium", "gpt2-xl",
          "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]

# uniform styling -- no model singled out
import matplotlib.cm as cm
COLORS = cm.tab10(np.linspace(0, 1, 10))
MODEL_STYLE = {m: dict(color=COLORS[i], label=m, ls="-") for i, m in enumerate(MODELS)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _load_profile(model):
    path = os.path.join(RESULTS, f"geometry_podcast_pca20_{model}{CTX_TAG}_hippocampus.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def _load_cross_space_dur8040(model):
    path = os.path.join(RESULTS, f"cross_space_layers_podcast_pca20_dur8040_{model}_hippocampus.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def save_fig(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


# ---------------- Panel A ----------------
def _load_static_profile(model="fasttext-wiki"):
    """Static (non-contextual) embedder: single-layer profile, no ctx suffix."""
    path = os.path.join(RESULTS, f"geometry_podcast_pca20_{model}_hippocampus.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def make_panel_A():
    GRID = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    temp_null_by_model = []
    for model, st in MODEL_STYLE.items():
        data_m = _load_profile(model)
        if data_m is None:
            continue
        n_l = len(data_m[0]["profile"]["mean_curv"])
        xs_m = np.linspace(0, 1, n_l)
        mu_c = np.array([r["profile"]["mean_curv"] for r in data_m]).mean(0)
        ax.plot(GRID, np.interp(GRID, xs_m, mu_c),
                color=st["color"], ls=st["ls"], lw=1.6, alpha=0.85, label=st["label"])

        # word-order (temporal) shuffle null: same embeddings, permuted word order,
        # already computed per-model/per-layer in compute_profiles()
        tn = [r["profile"]["temp_null_mu"] for r in data_m if "temp_null_mu" in r["profile"]]
        if tn:
            tn_mu = np.array(tn, dtype=float).mean(0)
            temp_null_by_model.append(np.interp(GRID, xs_m, tn_mu))

    # ── Word-order shuffle null, averaged across all contextual models ─────────
    if temp_null_by_model:
        null_mat = np.array(temp_null_by_model)
        null_mu = null_mat.mean(0)
        null_se = scipy_sem(null_mat, axis=0)
        ax.fill_between(GRID, null_mu - null_se, null_mu + null_se,
                         color="#555555", alpha=0.15, lw=0, zorder=0)
        ax.plot(GRID, null_mu, color="#555555", ls=(0, (1.2, 1.2)), lw=1.4,
                zorder=1, label="Word-order shuffle null")

    # ── Static (non-contextual) embedder: fastText, single layer -> flat line ──
    static_data = _load_static_profile("fasttext-wiki")
    if static_data:
        vals = np.array([r["profile"]["mean_curv"][0] for r in static_data], dtype=float)
        mu_s, se_s = vals.mean(), scipy_sem(vals)
        ax.axhline(mu_s, color="#B8860B", ls="--", lw=1.6, zorder=1,
                   label=f"fastText (static, $n$={len(vals)})")
        ax.axhspan(mu_s - se_s, mu_s + se_s, color="#B8860B", alpha=0.12, lw=0, zorder=0)

    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Mean curvature (°)")
    ax.set_title("A  Mean semantic curvature · all models", loc="left", pad=8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.legend(loc="best", fontsize=8.5, framealpha=0.85, edgecolor="#DDDDDD", ncol=2)
    fig.tight_layout()
    save_fig(fig, "panelA_mean_curvature_pca20_nochunk")
    plt.close(fig)


# ---------------- Panel B ----------------
def make_panel_B():
    GRID = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for model, st in MODEL_STYLE.items():
        cs_m = _load_cross_space_dur8040(model)
        if not cs_m:
            continue
        n_l_cs = cs_m[0]["n_layers"]
        xs_cs = np.linspace(0, 1, n_l_cs)
        mat_cs = np.array([r["r_curv_layers"] for r in cs_m if "r_curv_layers" in r], dtype=float)
        if not len(mat_cs):
            continue
        mu_cs = np.nanmean(mat_cs, axis=0)
        ax.plot(GRID, np.interp(GRID, xs_cs, np.nan_to_num(mu_cs)),
                color=st["color"], ls=st["ls"], lw=1.6, alpha=0.85, label=st["label"])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Pearson $r$ (curvature)")
    ax.set_title("B  Neural-LLM alignment · word-level\n(no chunking)", loc="left", pad=8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.legend(loc="best", fontsize=9, framealpha=0.85, edgecolor="#DDDDDD", ncol=2)
    fig.tight_layout()
    save_fig(fig, "panelB_alignment_pca20_nochunk")
    plt.close(fig)

    return {model: cs for model in MODELS if (cs := _load_cross_space_dur8040(model))}


# ---------------- Panel C ----------------
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


def find_winning_model_layer(cross_space_by_model):
    """Pick the (model, layer) with the highest mean word-level r_curv_layers
    across patients -- chosen by the data, not hardcoded."""
    best = None
    for model, cs in cross_space_by_model.items():
        mat = np.array([r["r_curv_layers"] for r in cs if "r_curv_layers" in r], dtype=float)
        if not len(mat):
            continue
        mu = np.nanmean(mat, axis=0)
        layer = int(np.nanargmax(mu[1:])) + 1  # exclude layer 0 (static embedding, not contextual)
        peak = float(mu[layer])
        if best is None or peak > best[2]:
            best = (model, layer, peak)
    return best


def make_panel_C(cross_space_by_model):
    from run_geometry_analysis_pca20 import (
        build_valid_mask, fit_neural_pca, trajectory_curvature, sem_pca_reduce,
    )
    from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs

    winner = find_winning_model_layer(cross_space_by_model)
    if winner is None:
        print("No winner found for panel C"); return
    model, layer, peak_r = winner
    print(f"Panel C winner: model={model}  layer={layer}  peak word-level r={peak_r:+.4f}")

    neural_vals, llm_vals = [], []
    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region="hippocampus", model_tag=model, context_words=200)
        if data is None:
            continue
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        n = min(data["n_words"], len(onset_ms))
        valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
        if valid.sum() == 0:
            continue

        preg = _convo_regions(pid)
        neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n], offset_ms[:n], 80, 40, rate=True)
        if neu_vecs is None:
            continue
        Y_neu, _ = fit_neural_pca(neu_vecs, n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        neural_vals.append(c_neu[np.isfinite(c_neu)])

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm; continue
        emb_layer = np.asarray(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        emb_layer = sem_pca_reduce(emb_layer)
        c_llm = trajectory_curvature(emb_layer)[valid]
        llm_vals.append(c_llm[np.isfinite(c_llm)])

    if not neural_vals or not llm_vals:
        print("No histogram data for panel C"); return
    neural = np.concatenate(neural_vals)
    llm = np.concatenate(llm_vals)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    bins = np.linspace(0.0, 180.0, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    d_neu = _density_hist(neural, bins)
    d_llm = _density_hist(llm, bins)
    neural_color = "#148F77"
    llm_color = "#5DADE2"

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
    ax.set_title(f"C  Curvature distribution\nbest-aligned (model, layer), word-level", loc="left", pad=8)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.85, edgecolor="#DDDDDD", handlelength=1.6)
    fig.tight_layout()
    save_fig(fig, "panelC_curvature_distribution_pca20_nochunk")
    plt.close(fig)


if __name__ == "__main__":
    print("Panel A..."); make_panel_A()
    print("Panel B..."); cs_by_model = make_panel_B()
    print("Panel C..."); make_panel_C(cs_by_model)
    print("\nDone.")
