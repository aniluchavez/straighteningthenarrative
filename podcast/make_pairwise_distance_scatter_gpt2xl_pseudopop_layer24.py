"""
make_pairwise_distance_scatter_gpt2xl_pseudopop.py
======================================================
All-word-pairs neural-vs-semantic distance scatter (RSA-style), matching
the reference paper's Fig. 3 setup as closely as possible:

  - Semantic: gpt2-xl LAYER 36 (the paper's reported layer), cosine
    distance between contextual embeddings.
  - Neural: NOT a single patient's electrodes. Since all 13 podcast
    patients heard the IDENTICAL stimulus, we build a "pseudopopulation"
    by difflib-aligning each patient's word sequence to a reference
    (YEY) and concatenating all patients' hippocampal channels at the
    ~99%-common word backbone -- i.e. one combined neural vector per
    word, pooling across the whole patient cohort, following the same
    construction already used in compute_matched_and_pseudopop_dur8040.py
    for the curvature analyses. This is a much closer match to the
    paper's "population of neurons" than one patient's ~53 channels,
    which is likely why the single-patient version of this figure came
    out null (|r|<0.013 across all 49 layers).

NOTE ON PHI: per established project convention, transcript/word content
is PHI and must never be displayed. No word-pair text labels are added.

Saves: figures/fig_pairwise_distance_scatter_gpt2xl_pseudopop.{pdf,png,svg}

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_pairwise_distance_scatter_gpt2xl_pseudopop.py
"""
from __future__ import annotations
import difflib
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")

REGION = "hippocampus"
MODEL = "gpt2-xl"
LAYER = 24  # exact 50th-percentile depth (24/48 layers)
REF_PID = "YEY"
N_SCATTER = 20000
SEED = 0

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def get_neural(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    preg = _convo_regions(pid)
    n = min(data0["n_words"], len(onset_ms))
    neu_vecs = windowed_neu_vecs(pid, REGION, preg, onset_ms[:n], offset_ms[:n],
                                  ONSET_PAD_MS, OFFSET_PAD_MS, rate=True)
    if neu_vecs is None:
        return None
    return dict(n=n, neu_vecs=neu_vecs, words=data0["words"][:n], surp=data0["surp"][:n],
                sent_pos=data0["sent_pos"][:n], is_turn=data0["is_turn"][:n])


def align_to_ref(ref_words, words):
    sm = difflib.SequenceMatcher(None, ref_words, words, autojunk=False)
    mapping = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def cosine_distance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    Xn = X / norms
    sim = Xn @ Xn.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def main():
    neural = {}
    for pid in PODCAST_PATIENTS:
        rec = get_neural(pid)
        if rec is not None:
            neural[pid] = rec
            print(f"  {pid}: n_words={rec['n']}", flush=True)
    print(f"\n{len(neural)} patients loaded\n", flush=True)

    ref_words = neural[REF_PID]["words"].tolist()
    common = set(range(len(ref_words)))
    mappings = {}
    for pid, rec in neural.items():
        m = align_to_ref(ref_words, rec["words"].tolist())
        mappings[pid] = m
        common &= set(m.keys())
    common_sorted = sorted(common)
    print(f"common words across {len(neural)} patients: {len(common_sorted)}/{len(ref_words)}")

    pids = list(neural.keys())
    combined = []
    for pid in pids:
        m = mappings[pid]
        local_idx = [m[r] for r in common_sorted]
        combined.append(neural[pid]["neu_vecs"][local_idx])
    combined = np.concatenate(combined, axis=1)  # (n_common, sum_channels)
    print(f"combined neural shape: {combined.shape}")

    n_common = len(common_sorted)
    ref_rec = neural[REF_PID]
    valid = build_valid_mask(n_common, ref_rec["surp"][common_sorted], ref_rec["sent_pos"][common_sorted],
                              ref_rec["is_turn"][common_sorted])
    valid_idx = np.where(valid)[0]
    print(f"valid words (after standard exclusions): {len(valid_idx)}/{n_common}")

    neu_z = StandardScaler().fit_transform(combined)[valid_idx]

    RESULTS_EARLY = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_EARLY, exist_ok=True)
    neu_cache_path = os.path.join(RESULTS_EARLY, "pseudopop_neural_cache.npz")
    np.savez_compressed(neu_cache_path, neu_z=neu_z, common_valid=np.array(common_sorted)[valid_idx])
    print(f"  cached neural pseudopop -> {neu_cache_path}")

    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    common_valid = [common_sorted[i] for i in valid_idx]
    emb_L = np.array(emb_mm[LAYER, common_valid], dtype=np.float64)
    del emb_mm

    n = len(valid_idx)
    print(f"\nFinal n_words={n}  neural dim={neu_z.shape[1]}  semantic dim={emb_L.shape[1]}")

    print("Computing neural cosine distance matrix (pseudopopulation)...")
    D_neu = cosine_distance_matrix(neu_z)
    print("Computing semantic cosine distance matrix (gpt2-xl layer 36)...")
    D_sem = cosine_distance_matrix(emb_L)

    iu = np.triu_indices(n, k=1)
    d_neu_all = D_neu[iu]
    d_sem_all = D_sem[iu]
    n_pairs = len(d_neu_all)
    print(f"n_pairs = {n_pairs:,}")

    r, p = pearsonr(d_sem_all, d_neu_all)
    print(f"r = {r:.4f}  p = {p:.4e}")

    RESULTS = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS, exist_ok=True)
    cache_path = os.path.join(RESULTS, "pairwise_distance_scatter_gpt2xl_pseudopop_layer24_cache.npz")
    np.savez_compressed(cache_path, d_sem_all=d_sem_all, d_neu_all=d_neu_all,
                         n_words=n, n_pairs=n_pairs, r=r, p=p,
                         n_channels=neu_z.shape[1], n_patients=len(neural))
    print(f"  cached arrays -> {cache_path}")

    plot_from_arrays(d_sem_all, d_neu_all, r, p, n, n_pairs, neu_z.shape[1], len(neural))


def plot_from_arrays(d_sem_all, d_neu_all, r, p, n, n_pairs, n_channels, n_patients,
                      xlim_pctile=(0.5, 99.5), ylim_pctile=(0.5, 99.5)):
    n_pairs_total = len(d_sem_all)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_pairs_total, size=min(N_SCATTER, n_pairs_total), replace=False)
    x_plot, y_plot = d_sem_all[idx], d_neu_all[idx]

    slope, intercept = np.polyfit(d_sem_all, d_neu_all, 1)
    xlo, xhi = np.percentile(d_sem_all, xlim_pctile)
    ylo, yhi = np.percentile(d_neu_all, ylim_pctile)
    xs = np.linspace(xlo, xhi, 200)
    ys = slope * xs + intercept

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x_plot, y_plot, s=3, color="black", alpha=0.2, linewidths=0, rasterized=True)
    ax.plot(xs, ys, color="#D6006C", lw=2.2, zorder=5)

    p_str = "$p$ < 0.0001" if p < 1e-4 else f"$p$ = {p:.2e}"
    ax.text(0.04, 0.94, f"$r$ = {r:.4f}\n{p_str}", transform=ax.transAxes, fontsize=12, va="top")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"{MODEL} semantic distance (layer {LAYER}, cosine)")
    ax.set_ylabel("Neural cosine distance (hippocampus, pseudopopulation)")
    ax.set_title(f"All word-pair distances, pseudopopulation ({n_patients} patients)\n"
                 f"$n$ = {n:,} words, {n_pairs:,} pairs, {n_channels} combined channels", loc="left", fontsize=10.5)

    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_pairwise_distance_scatter_gpt2xl_pseudopop_layer24"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
