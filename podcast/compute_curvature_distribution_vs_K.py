"""
compute_curvature_distribution_vs_K.py
====================================
Does the declining neural-semantic curvature correlation at high K reflect
high-dimensional angle concentration (random vectors in high-D become
~orthogonal, so turning angles pile up near 90 deg regardless of any real
structure) rather than a genuine loss of shared low-dimensional structure?

For gpt2-xl layer 20 (this session's validated layer), pooled across all 13
podcast patients, computes mean/SD of the semantic curvature distribution
C_LM^K at each K in a wide sweep, plus histograms, to check directly:
  - if the distribution visibly narrows and concentrates near 90 deg as K
    grows, that's the angle-concentration artifact:
  - if the distribution stays broad/well-behaved while correlation with
    HPC still declines, the shared-low-dim-subspace story is more likely.

Saves: results/curvature_distribution_vs_K.json
       figures/fig_curvature_distribution_vs_K.{pdf,png,svg}
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import trajectory_curvature, build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

MODEL = "gpt2-xl"
LAYER = 20
REGION = "hippocampus"
KS = [3, 5, 10, 20, 50, 100, 150, 200, 300]
K_MAX = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(PARENT_DIR, "figures")
OUT_PATH = os.path.join(RESULTS, "curvature_distribution_vs_K.json")

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


def fit_pca_master(X, k_max=K_MAX):
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k_max)
    if max_pc < 2:
        return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def main():
    all_curv_by_k = {k: [] for k in KS}

    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n = data["n_words"]
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        master = fit_pca_master(emb_L)
        if master is None:
            continue

        for k in KS:
            kk = min(k, master.shape[1])
            c = trajectory_curvature(master[:, :kk])[valid]
            all_curv_by_k[k].append(c)
        print(f"  {pid}: done", flush=True)

    stats = {}
    for k in KS:
        arr = np.concatenate(all_curv_by_k[k])
        arr = arr[np.isfinite(arr)]
        stats[k] = dict(mean=float(arr.mean()), sd=float(arr.std()), n=int(len(arr)),
                         p5=float(np.percentile(arr, 5)), p95=float(np.percentile(arr, 95)))
        print(f"K={k:4d}  mean={arr.mean():6.2f} deg  sd={arr.std():5.2f} deg  "
              f"[p5={stats[k]['p5']:.1f}, p95={stats[k]['p95']:.1f}]  n={len(arr)}", flush=True)

    json.dump(stats, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")

    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    cmap = cm.get_cmap("viridis")
    for i, k in enumerate(KS):
        arr = np.concatenate(all_curv_by_k[k])
        arr = arr[np.isfinite(arr)]
        color = cmap(i / (len(KS) - 1))
        axes[0].hist(arr, bins=90, range=(0, 180), histtype="step", density=True,
                     color=color, lw=1.6, label=f"K={k}")
    axes[0].axvline(90, color="k", lw=0.7, ls=":", alpha=0.6)
    axes[0].set_xlabel("Semantic curvature (°)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A  Curvature distribution by K", loc="left", fontsize=11)
    axes[0].legend(fontsize=7, ncol=2, framealpha=0.8, edgecolor="#DDDDDD")

    ks_arr = np.array(KS)
    means = np.array([stats[k]["mean"] for k in KS])
    sds = np.array([stats[k]["sd"] for k in KS])
    axes[1].plot(ks_arr, means, color="#C0392B", marker="o", lw=1.8, label="Mean")
    axes[1].fill_between(ks_arr, means - sds, means + sds, color="#C0392B", alpha=0.15, label="±1 SD")
    axes[1].axhline(90, color="k", lw=0.7, ls=":", alpha=0.6)
    axes[1].set_xlabel("K (PCs)")
    axes[1].set_ylabel("Semantic curvature (°)")
    axes[1].set_title("B  Mean ± SD vs K", loc="left", fontsize=11)
    axes[1].legend(fontsize=8, framealpha=0.8, edgecolor="#DDDDDD")

    fig.suptitle(f"Podcast · {MODEL} layer {LAYER} · semantic curvature distribution vs K (all 13 patients)",
                 fontsize=9.5, fontweight="bold", y=0.98)
    fig.subplots_adjust(wspace=0.3, top=0.85)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_distribution_vs_K"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
