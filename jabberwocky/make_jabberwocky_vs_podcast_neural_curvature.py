"""
make_jabberwocky_vs_podcast_neural_curvature.py
================================================
Compares overall hippocampal NEURAL trajectory curvature (single PCA
neural space, no layers -- neural has no "layer" axis) between the
Jabberwocky condition (nonce content words, preserved syntax, n=7
patients) and the Podcast condition (natural narrative, passive
listening, n=13 patients).

Same bar + patient-resampling-bootstrap design as the existing
anterior-vs-posterior figure (fig_ap_curvature_podcast.png).

Run:
    cd .../geometry_paper/jabberwocky
    conda run -n gpt2_embed python3 -u make_jabberwocky_vs_podcast_neural_curvature.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
PODCAST_DIR = os.path.join(PARENT_DIR, "podcast")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, PODCAST_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import fit_neural_pca, trajectory_curvature
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

REGION = "hippocampus"
MODEL_TAG = "llama-3.1-8b"  # only used to satisfy loader's emb-cache existence check; not used here


def patient_mean_curvature_jabberwocky(pid):
    data = load_jabberwocky_patient(pid, region=REGION, model_tag=MODEL_TAG, context_words=200)
    if data is None or data.get("neu_vecs") is None:
        return None
    n = data["n_words"]
    Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None
    c = trajectory_curvature(Y_neu)
    return float(np.nanmean(c)), int(len(c))


def patient_mean_curvature_podcast(pid):
    data = load_podcast_patient(pid, region=REGION, model_tag=MODEL_TAG, context_words=200)
    if data is None or data.get("neu_vecs") is None:
        return None
    n = data["n_words"]
    Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None
    c = trajectory_curvature(Y_neu)
    return float(np.nanmean(c)), int(len(c))


def main():
    jab_rows, pod_rows = [], []
    print("Jabberwocky patients:")
    for pid in JABBERWOCKY_PATIENTS:
        res = patient_mean_curvature_jabberwocky(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        mu, n = res
        jab_rows.append((pid, mu, n))
        print(f"  {pid}: mean_curv={mu:.2f}  n={n}")

    print("\nPodcast patients:")
    for pid in PODCAST_PATIENTS:
        res = patient_mean_curvature_podcast(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        mu, n = res
        pod_rows.append((pid, mu, n))
        print(f"  {pid}: mean_curv={mu:.2f}  n={n}")

    jab_vals = np.array([r[1] for r in jab_rows])
    pod_vals_all = np.array([r[1] for r in pod_rows])
    pod_pids_all = [r[0] for r in pod_rows]
    print(f"\nJabberwocky: n_patients={len(jab_vals)}  mean={jab_vals.mean():.3f}  sem={jab_vals.std(ddof=1)/np.sqrt(len(jab_vals)):.3f}")
    print(f"Podcast (all {len(pod_vals_all)}): mean={pod_vals_all.mean():.3f}  sem={pod_vals_all.std(ddof=1)/np.sqrt(len(pod_vals_all)):.3f}")

    # ---- patient-count-matched comparison: subsample podcast to n=len(jab_vals) ----
    # Average over all C(n_pod, n_jab) subsets' bootstrap distributions is expensive;
    # instead resample a matched-size podcast subset *within* each bootstrap draw so
    # the matched comparison has the same n on both sides at every draw.
    n_match = len(jab_vals)
    rng = np.random.default_rng(0)
    n_boot = 5000
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        samp_j = rng.choice(jab_vals, size=n_match, replace=True)
        pod_subset_idx = rng.choice(len(pod_vals_all), size=n_match, replace=False)
        samp_p = rng.choice(pod_vals_all[pod_subset_idx], size=n_match, replace=True)
        boot_diffs[b] = samp_j.mean() - samp_p.mean()

    # Observed point estimate: mean over many random n_match-sized podcast subsets
    # (no resampling-with-replacement, just which patients are in the subset).
    n_subsets = 2000
    subset_means = np.array([
        pod_vals_all[rng.choice(len(pod_vals_all), size=n_match, replace=False)].mean()
        for _ in range(n_subsets)
    ])
    pod_matched_mean = subset_means.mean()
    pod_matched_sem = subset_means.std(ddof=1)
    obs_diff = jab_vals.mean() - pod_matched_mean
    p_boot = 2 * min(np.mean(boot_diffs > 0), np.mean(boot_diffs < 0))
    frac_same_sign = np.mean(np.sign(boot_diffs) == np.sign(obs_diff))
    print(f"\nPodcast (matched, n={n_match}, averaged over {n_subsets} random {n_match}-of-{len(pod_vals_all)} subsets): "
          f"mean={pod_matched_mean:.3f}  sem_across_subsets={pod_matched_sem:.3f}")
    print(f"Observed diff (jabberwocky - podcast[matched n={n_match}]) = {obs_diff:+.3f} deg")
    print(f"Bootstrap p (two-sided) = {p_boot:.4f}   {frac_same_sign*100:.0f}% of draws share the observed sign")

    # keep names used below pointing at the matched comparison
    pod_vals = pod_vals_all

    # ---- figure: bar + bootstrap histogram, same layout as ap_curvature figure ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"Hippocampal neural curvature: Jabberwocky vs. Podcast "
                 f"(patient-count matched, n={n_match} vs n={n_match} of {len(pod_vals_all)})", y=1.03)

    ax = axes[0]
    means = [jab_vals.mean(), pod_matched_mean]
    sems = [jab_vals.std(ddof=1)/np.sqrt(len(jab_vals)), pod_matched_sem]
    ax.bar(["Jabberwocky", "Podcast\n(matched n)"], means, yerr=sems, capsize=6,
           color=["#8E44AD", "#2980B9"])
    ax.set_ylabel("Mean trajectory curvature (deg)")
    ax.set_title(f"Group means (n={n_match} vs n={n_match}, matched)")

    ax = axes[1]
    ax.hist(boot_diffs, bins=40, color="#8E44AD", alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(obs_diff, color="black", linewidth=1.5)
    ax.set_xlabel(f"Delta curvature (jabberwocky - podcast, matched n={n_match})")
    ax.set_ylabel("Count of bootstrap draws")
    ax.set_title(f"Bootstrap stability, matched n (p={p_boot:.4f})")
    ax.text(0.02, 0.95, f"mean = {obs_diff:+.2f}", transform=ax.transAxes, va="top")

    out_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"fig_jabberwocky_vs_podcast_neural_curvature.{ext}")
        fig.savefig(out_path)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
