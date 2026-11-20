"""
compute_displacement_distribution_vs_K.py
====================================
Same question as compute_curvature_distribution_vs_K.py, but for
DISPLACEMENT (step magnitude ||X[t+1]-X[t]||) instead of curvature.
Displacement is a norm, not an angle, so the concentration signature looks
different: the RAW magnitude grows with K (summing more squared
components, ~sqrt(K) by the usual norm-concentration result for
high-dimensional isotropic vectors), but its RELATIVE spread (coefficient
of variation, SD/mean) should shrink as K grows if each added dimension
contributes comparably-sized independent variance -- the magnitude
concentrates around its mean in relative terms even as the mean itself
grows.

For gpt2-xl layer 20, pooled across all 13 podcast patients: mean, SD, and
CV of the semantic displacement distribution D_LM^K at each K, plus
histograms of both raw and mean-normalized displacement.

Saves: results/displacement_distribution_vs_K.json
       figures/fig_displacement_distribution_vs_K.{pdf,png,svg}
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
from run_geometry_analysis import trajectory_displacement, build_valid_mask
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
OUT_PATH = os.path.join(RESULTS, "displacement_distribution_vs_K.json")

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
    all_disp_by_k = {k: [] for k in KS}

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
            d = trajectory_displacement(master[:, :kk])[:n - 2][valid]
            all_disp_by_k[k].append(d)
        print(f"  {pid}: done", flush=True)

    stats = {}
    for k in KS:
        arr = np.concatenate(all_disp_by_k[k])
        arr = arr[np.isfinite(arr)]
        mean, sd = arr.mean(), arr.std()
        cv = sd / mean if mean > 1e-12 else float("nan")
        stats[k] = dict(mean=float(mean), sd=float(sd), cv=float(cv), n=int(len(arr)),
                         p5=float(np.percentile(arr, 5)), p95=float(np.percentile(arr, 95)))
        print(f"K={k:4d}  mean={mean:7.3f}  sd={sd:6.3f}  CV={cv:.4f}  "
              f"[p5={stats[k]['p5']:.2f}, p95={stats[k]['p95']:.2f}]  n={len(arr)}", flush=True)

    json.dump(stats, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    cmap = cm.get_cmap("viridis")

    # Panel A: raw displacement histograms (different scales per K)
    for i, k in enumerate(KS):
        arr = np.concatenate(all_disp_by_k[k])
        arr = arr[np.isfinite(arr)]
        color = cmap(i / (len(KS) - 1))
        axes[0].hist(arr, bins=100, histtype="step", density=True, color=color, lw=1.4, label=f"K={k}")
    axes[0].set_xlabel("Semantic displacement (raw)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A  Raw displacement by K", loc="left", fontsize=11)
    axes[0].legend(fontsize=6.5, ncol=2, framealpha=0.8, edgecolor="#DDDDDD")

    # Panel B: mean-normalized displacement (d / mean(d)) -- comparable scale across K
    for i, k in enumerate(KS):
        arr = np.concatenate(all_disp_by_k[k])
        arr = arr[np.isfinite(arr)]
        arr_norm = arr / arr.mean()
        color = cmap(i / (len(KS) - 1))
        axes[1].hist(arr_norm, bins=100, range=(0, 3), histtype="step", density=True, color=color, lw=1.4, label=f"K={k}")
    axes[1].axvline(1, color="k", lw=0.7, ls=":", alpha=0.6)
    axes[1].set_xlabel("Displacement / mean(displacement)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("B  Mean-normalized displacement by K", loc="left", fontsize=11)
    axes[1].legend(fontsize=6.5, ncol=2, framealpha=0.8, edgecolor="#DDDDDD")

    # Panel C: CV vs K
    ks_arr = np.array(KS)
    cvs = np.array([stats[k]["cv"] for k in KS])
    axes[2].plot(ks_arr, cvs, color="#2980B9", marker="o", lw=1.8)
    axes[2].set_xlabel("K (PCs)")
    axes[2].set_ylabel("Coefficient of variation (SD / mean)")
    axes[2].set_title("C  Relative spread vs K", loc="left", fontsize=11)

    fig.suptitle(f"Podcast · {MODEL} layer {LAYER} · semantic displacement distribution vs K (all 13 patients)",
                 fontsize=9.5, fontweight="bold", y=0.98)
    fig.subplots_adjust(wspace=0.32, top=0.85)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_displacement_distribution_vs_K"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
