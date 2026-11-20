"""
make_example_sentence_trajectories.py
====================================
For two example sentences from the podcast stimulus (patient YEY) --
one containing the high-surprisal word "Indianbrooke", one containing the
low-surprisal "one by one" -- plot the neural (hippocampus, dur80/40
rate-normalized) and semantic (llama-3.1-8b, layer 18) trajectories through
that sentence, each PCA-reduced to 2D and Procrustes-aligned (same method
as run_geometry_analysis.find_best_sentence_traj), overlaid with word labels.

Saves: figures/fig_example_sentence_trajectories.{pdf,png,svg}
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import procrustes as scipy_procrustes

from podcast_loader import load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(PARENT_DIR, "figures")

PID = "YEY"
MODEL = "llama-3.1-8b"
LAYER = 18
REGION = "hippocampus"
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40

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

C_SEM = "#D95319"
C_NEU = "#1B7F4A"


def _pca2d(X):
    k = min(2, X.shape[1], X.shape[0] - 1)
    if k < 2:
        return None
    Xc = X - X.mean(0)
    return PCA(n_components=k).fit_transform(Xc)


def sentence_span(sent_id, word_idx):
    sid = sent_id[word_idx]
    wis = np.where(sent_id == sid)[0]
    return wis


def window_span(n, word_idx, pad_before, pad_after):
    lo = max(0, word_idx - pad_before)
    hi = min(n, word_idx + pad_after + 1)
    return np.arange(lo, hi)


def build_trajectory(data, wis, neu_vecs, layer):
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_sent = np.array(emb_mm[layer, wis], dtype=np.float64)
    del emb_mm
    neu_sent = neu_vecs[wis].astype(np.float64)

    llm_2d = _pca2d(emb_sent)
    neu_2d = _pca2d(neu_sent)
    if llm_2d is None or neu_2d is None:
        return None
    try:
        _, neu_aligned, disparity = scipy_procrustes(llm_2d, neu_2d)
        llm_aligned, _, _ = scipy_procrustes(llm_2d, llm_2d)
    except Exception:
        neu_aligned, disparity = neu_2d, float("nan")
        llm_aligned = llm_2d
    return llm_aligned, neu_aligned, disparity


def _draw_panel(ax, words_disp, llm_2d, neu_2d, title):
    ax.plot(llm_2d[:, 0], llm_2d[:, 1], color=C_SEM, lw=1.6, marker="o", ms=4, label="Semantic (llama-3.1-8b, L18)")
    ax.plot(neu_2d[:, 0], neu_2d[:, 1], color=C_NEU, lw=1.6, marker="o", ms=4, label="Neural (hippocampus, dur80/40 rate)")
    for i, w in enumerate(words_disp):
        ax.annotate(w, llm_2d[i], fontsize=7, color=C_SEM, alpha=0.85,
                     xytext=(3, 3), textcoords="offset points")
        ax.annotate(w, neu_2d[i], fontsize=7, color=C_NEU, alpha=0.85,
                     xytext=(3, -9), textcoords="offset points")
    ax.scatter(*llm_2d[0], s=55, facecolor="white", edgecolor=C_SEM, zorder=5, lw=1.3)
    ax.scatter(*neu_2d[0], s=55, facecolor="white", edgecolor=C_NEU, zorder=5, lw=1.3)
    ax.set_title(title, loc="left", fontsize=9.5, pad=4)
    ax.set_xlabel("PC1", fontsize=8.5)
    ax.set_ylabel("PC2", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="best", fontsize=6.5, framealpha=0.8, edgecolor="#DDDDDD")


def main():
    data = load_podcast_patient(PID, region=REGION, model_tag=MODEL, context_words=200)
    n = data["n_words"]
    words = data["words"][:n]
    sent_id = data["sent_id"][:n]

    _, onset_ms, offset_ms = load_words_onset_offset(PID)
    onset_ms, offset_ms = onset_ms[:n], offset_ms[:n]
    preg = _convo_regions(PID)
    neu_vecs = windowed_neu_vecs(PID, REGION, preg, onset_ms, offset_ms, ONSET_PAD_MS, OFFSET_PAD_MS, rate=True)

    hi_idx = int(np.where(words == "indianbrooke")[0][0])
    lo_candidates = [i for i in range(1, n - 1) if words[i] == "one" and words[i - 1] == "went" and words[i + 1] == "by"]
    lo_idx = lo_candidates[0]

    hi_wis = sentence_span(sent_id, hi_idx)
    lo_wis = window_span(n, lo_idx, pad_before=1, pad_after=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.85, wspace=0.32)

    for ax, wis, tag in ((ax1, hi_wis, "High surprisal: “Indianbrooke”"),
                          (ax2, lo_wis, "Low surprisal: “one by one”")):
        res = build_trajectory(data, wis, neu_vecs, LAYER)
        if res is None:
            ax.set_title(f"{tag}\n(insufficient data)", loc="left", fontsize=9.5)
            continue
        llm_2d, neu_2d, disparity = res
        words_disp = words[wis].tolist()
        _draw_panel(ax, words_disp, llm_2d, neu_2d, f"{tag}\nProcrustes disparity={disparity:.3f}")

    fig.suptitle(f"Podcast · {PID} · sentence trajectories, semantic vs. neural (Procrustes-aligned, 2D PCA)",
                 fontsize=9.5, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_example_sentence_trajectories"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
