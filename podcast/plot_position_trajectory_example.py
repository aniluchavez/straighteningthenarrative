"""
plot_position_trajectory_example.py
===================================
What "position" is: for one patient, show the per-word point x_t in semantic
space (GPT-2 XL layer 29, top-8 PCs) and neural space (hippocampus spike
counts, dur80/40 window, top-8 PCs) over a short stretch of the podcast.

Rows: semantic, neural.
Cols: (A) trajectory in the PC1-PC2 plane, colored by time
      (B) PC1-PC3 coordinates of x_t over time  -> x_t is a vector
      (C) the two scalar summaries used as "position" in the analysis:
          PC1 coordinate, and ||x_t - centroid|| over all 8 PCs

Same preprocessing as compute_curv_disp_pos_gpt2xl_layer29.py. No words shown.

    conda run -n gpt2_embed python3 -u plot_position_trajectory_example.py [PID] [START] [N]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

K, MODEL, LAYER = 8, "gpt2-xl", 29
FIG_DIR = os.path.join(BASE_DIR, "figures")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # PC1, PC2, PC3
TIME_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#86b6ef", "#2a78d6", "#0d366b"])


def pca_fixed(X, k=K):
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    return PCA(n_components=min(k, Xs.shape[1], Xs.shape[0] - 1),
               svd_solver="randomized", random_state=0).fit_transform(Xs)


def load(pid):
    _, onset_ms, offset_ms = load_words_onset_offset(pid)
    data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
    if data is None:
        return None
    n0 = min(data["n_words"], len(onset_ms))
    neu = windowed_neu_vecs(pid, "hippocampus", _convo_regions(pid),
                            onset_ms[:n0], offset_ms[:n0], 80, 40, rate=False)
    if neu is None:
        return None
    emb = np.load(data["emb_path"], mmap_mode="r")
    n = min(n0, emb.shape[1], len(neu))
    sem = np.array(emb[LAYER, :n], dtype=np.float64)
    neu = np.asarray(neu[:n], dtype=np.float64)
    ok = np.isfinite(sem).all(1) & np.isfinite(neu).all(1)
    t = np.asarray(onset_ms[:n], dtype=float)[ok] / 1000.0
    # PCA fit on the whole session (as in the analysis), then slice for display
    return t, pca_fixed(sem[ok]), pca_fixed(neu[ok]), neu.shape[1]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def row(axes, t, Y, name, centroid):
    aA, aB, aC = axes
    # A: path in PC1-PC2 plane
    pts = Y[:, :2]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=TIME_CMAP, linewidths=1.2, alpha=0.8)
    lc.set_array(t[:-1])
    aA.add_collection(lc)
    sc = aA.scatter(pts[:, 0], pts[:, 1], c=t, cmap=TIME_CMAP, s=18,
                    edgecolors="white", linewidths=0.6, zorder=3)
    aA.scatter(*pts[0], s=60, facecolors="none", edgecolors=INK, lw=1.2, zorder=4)
    aA.annotate("start", pts[0], xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    aA.scatter(*centroid[:2], marker="+", s=80, color=INK, zorder=4)
    aA.annotate("session centroid", centroid[:2], xytext=(6, -10),
                textcoords="offset points", fontsize=8, color=INK2)
    aA.set_xlabel("PC1", fontsize=9, color=INK2); aA.set_ylabel("PC2", fontsize=9, color=INK2)
    aA.set_title(f"{name}: each dot = one word's x$_t$", fontsize=10, color=INK, loc="left")
    plt.colorbar(sc, ax=aA, fraction=0.046, pad=0.02).set_label("time (s)", fontsize=8, color=INK2)

    # B: coordinates over time
    for j in range(3):
        aB.plot(t, Y[:, j], color=SERIES[j], lw=1.5, label=f"PC{j+1}")
    aB.set_xlabel("time (s)", fontsize=9, color=INK2); aB.set_ylabel("coordinate", fontsize=9, color=INK2)
    aB.set_title(f"x$_t$ is a vector: {K} coordinates per word (3 shown)", fontsize=10, color=INK, loc="left")
    aB.legend(fontsize=8, frameon=False, loc="upper left")

    # C: scalar summaries
    pc1 = Y[:, 0]
    dist = np.linalg.norm(Y - centroid, axis=1)
    aC.plot(t, pc1, color=SERIES[0], lw=1.5, label="PC1 coordinate")
    aC.plot(t, dist, color=INK, lw=1.5, label="‖x$_t$ − centroid‖ (8-D)")
    aC.axhline(0, color=INK2, lw=0.6)
    aC.set_xlabel("time (s)", fontsize=9, color=INK2); aC.set_ylabel("scalar 'position'", fontsize=9, color=INK2)
    aC.set_title("two scalar versions of position", fontsize=10, color=INK, loc="left")
    aC.legend(fontsize=8, frameon=False, loc="upper left")


def main():
    args = sys.argv[1:]
    pids = [args[0]] if args else [p for p in PODCAST_PATIENTS if p not in ("YEX",)]
    start = int(args[1]) if len(args) > 1 else 200
    N = int(args[2]) if len(args) > 2 else 30
    for pid in pids:
        try:
            out = load(pid)
        except (AssertionError, FileNotFoundError, OSError) as e:
            print(f"{pid}: skip ({e.__class__.__name__})"); continue
        if out is None:
            continue
        t_all, S_all, N_all, n_neu = out
        sl = slice(start, start + N)
        t, S, Nn = t_all[sl], S_all[sl], N_all[sl]
        # centroid over the whole session (what position() uses when fed the full trajectory)
        cS, cN = S_all.mean(0), N_all.mean(0)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor="#fcfcfb")
        for ax in axes.ravel():
            ax.set_facecolor("#fcfcfb"); style(ax)
        row(axes[0], t, S, f"Semantic ({MODEL} L{LAYER}, 1600-D → {K} PCs)", cS)
        row(axes[1], t, Nn, f"Neural (hippocampus, {n_neu} neurons → {K} PCs)", cN)
        fig.suptitle(f"{pid}: position x$_t$ in each space, words {start}–{start+len(t)-1}",
                     fontsize=12, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        base = os.path.join(FIG_DIR, f"position_trajectory_example_{pid}_w{start}_{N}")
        fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".pdf")
        print(f"saved {base}.png  (session words={len(t_all)}, neurons={n_neu})")
        break


if __name__ == "__main__":
    main()
