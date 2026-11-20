"""
compute_cross_space_layers.py
==============================
For each model × patient, sweep all layers and compute per-layer Pearson r
between LLM and neural (hippocampal) trajectories for three geometric quantities:

  displacement  — step-to-step Euclidean distance (trajectory_displacement)
  curvature     — turning angle κ in degrees     (trajectory_curvature)
  position      — first-PC coordinate time series (_pc1_series)

Both word-by-word and chunk-averaged (CHUNK_SIZE words) correlations are saved.
Neural metrics are computed once per patient; LLM metrics swept per layer.

Output (one file per model):
  results/cross_space_layers_{model}_hippocampus.json

Each file is a list of per-patient dicts with keys:
  pid, n_layers,
  r_curv_layers / r_disp_layers / r_pos_layers        (word-by-word)
  p_curv_layers / p_disp_layers / p_pos_layers
  r_curv_layers_ch / r_disp_layers_ch / r_pos_layers_ch  (chunk-averaged)
  p_curv_layers_ch / p_disp_layers_ch / p_pos_layers_ch

Run (all models, overwriting existing files):
    cd /scratch/aniluchavez/straighteningthenarrative
    conda run -n gpt2_embed python3 compute_cross_space_layers.py

Run (single model):
    conda run -n gpt2_embed python3 compute_cross_space_layers.py --model llama-3.1-8b
"""

import argparse, json, os, sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement,
    build_valid_mask, fit_neural_pca, load_patient,
    PATIENTS, MIN_PAIRS
)

RESULTS     = os.path.join(BASE_DIR, "results")
REGION      = "hippocampus"
CHUNK_SIZE  = 15
N_PERM      = 200   # word-order shuffles of neural curvature for null distribution

MODELS = [
    "gpt2",
    "gpt2-medium",
    "bert-base",
    "roberta-base",
    "opt-350m",
    "llama-2-7b",
    "llama-3.1-8b",
    "gemma-2-9b",
    "mistral-7b",
]


def _pc1_series(X):
    """Z-scored first-PC time series of matrix X (n_words × d)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return np.full(X.shape[0], np.nan)
    Xs = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=1).fit_transform(Xs)[:, 0]
    s = np.std(pc1)
    if s < 1e-12:
        return np.full(len(pc1), np.nan)
    return (pc1 - pc1.mean()) / s


def _safe_r(x, y, min_n=15):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan"), float("nan")
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p)


def _chunk_r(x, y, chunk_size, min_chunks=5):
    """Chunk-average both arrays then correlate; returns (r, p)."""
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        fx = np.isfinite(x[sl]); fy = np.isfinite(y[sl])
        fin = fx & fy
        if fin.sum() < chunk_size // 2:
            continue
        xa.append(np.mean(x[sl][fin]))
        ya.append(np.mean(y[sl][fin]))
    xa, ya = np.array(xa), np.array(ya)
    if len(xa) < min_chunks:
        return float("nan"), float("nan")
    r, p = pearsonr(xa, ya)
    return float(r), float(p)


def process_model(model):
    out_path = os.path.join(RESULTS, f"cross_space_layers_{model}_{REGION}.json")

    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    out = []
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, model)
        if data is None:
            print(f"  {pid}: load failed")
            continue

        n       = data["n_words"]
        surp    = data["surp"]
        sp      = data["sent_pos"]
        is_turn = data["is_turn"]

        valid = build_valid_mask(n, surp, sp, is_turn)
        if valid.sum() < MIN_PAIRS:
            print(f"  {pid}: too few valid ({valid.sum()})")
            continue

        interior = np.arange(1, n - 1)
        surp_y   = surp[interior + 1][valid]

        # ── Neural metrics (computed once per patient) ─────────────────────────
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            print(f"  {pid}: neural PCA failed")
            continue

        c_neu = trajectory_curvature(Y_neu)[valid]
        d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
        p_neu = _pc1_series(Y_neu)[interior][valid]

        # ── Load LLM embedding cache (mmap) ────────────────────────────────────
        emb_mm   = np.load(data["emb_path"], mmap_mode="r")
        n_layers = emb_mm.shape[0]

        r_curv_L, r_disp_L, r_pos_L = [], [], []
        p_curv_L, p_disp_L, p_pos_L = [], [], []

        r_curv_ch, r_disp_ch, r_pos_ch = [], [], []
        p_curv_ch, p_disp_ch, p_pos_ch = [], [], []

        # store per-layer semantic metrics for null computation
        c_sems, d_sems = [], []

        for L in range(n_layers):
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)

            c_sem = trajectory_curvature(emb_L)[valid]
            d_sem = trajectory_displacement(emb_L)[:n - 2][valid]
            p_sem = _pc1_series(emb_L)[interior][valid]
            c_sems.append(c_sem)
            d_sems.append(d_sem)

            # word-by-word
            rc, pc = _safe_r(c_sem, c_neu)
            rd, pd = _safe_r(d_sem, d_neu)
            rp, pp = _safe_r(p_sem, p_neu)
            r_curv_L.append(rc); p_curv_L.append(pc)
            r_disp_L.append(rd); p_disp_L.append(pd)
            r_pos_L.append(rp);  p_pos_L.append(pp)

            # chunk-averaged
            rcc, pcc = _chunk_r(c_sem, c_neu, CHUNK_SIZE)
            rdc, pdc = _chunk_r(d_sem, d_neu, CHUNK_SIZE)
            rpc, ppc = _chunk_r(p_sem, p_neu, CHUNK_SIZE)
            r_curv_ch.append(rcc); p_curv_ch.append(pcc)
            r_disp_ch.append(rdc); p_disp_ch.append(pdc)
            r_pos_ch.append(rpc);  p_pos_ch.append(ppc)

        del emb_mm

        # ── Shuffle null: permute neural word order N_PERM times ───────────────
        # Each permutation applies the same shuffle to c_neu/d_neu across all
        # layers, giving a (N_PERM × n_layers) null matrix. We store mean ± std.
        null_curv_ch = np.full((N_PERM, n_layers), np.nan)
        null_disp_ch = np.full((N_PERM, n_layers), np.nan)
        for pi in range(N_PERM):
            perm = rng.permutation(len(c_neu))
            c_neu_shuf = c_neu[perm]
            d_neu_shuf = d_neu[perm]
            for L in range(n_layers):
                null_curv_ch[pi, L] = _chunk_r(c_sems[L], c_neu_shuf, CHUNK_SIZE)[0]
                null_disp_ch[pi, L] = _chunk_r(d_sems[L], d_neu_shuf, CHUNK_SIZE)[0]

        null_curv_ch_mean = np.nanmean(null_curv_ch, axis=0).tolist()
        null_curv_ch_std  = np.nanstd(null_curv_ch,  axis=0).tolist()
        null_disp_ch_mean = np.nanmean(null_disp_ch, axis=0).tolist()
        null_disp_ch_std  = np.nanstd(null_disp_ch,  axis=0).tolist()

        def _peak(arr):
            vals = [r for r in arr if not np.isnan(r)]
            return max(vals) if vals else float("nan")

        print(f"  {pid}: {n_layers} layers  "
              f"r_curv={_peak(r_curv_L):+.4f}/{_peak(r_curv_ch):+.4f}  "
              f"r_disp={_peak(r_disp_L):+.4f}/{_peak(r_disp_ch):+.4f}  "
              f"r_pos={_peak(r_pos_L):+.4f}/{_peak(r_pos_ch):+.4f}  "
              f"(word/chunk)")

        out.append(dict(
            pid           = pid,
            n_layers      = n_layers,
            # word-by-word
            r_curv_layers = r_curv_L, p_curv_layers = p_curv_L,
            r_disp_layers = r_disp_L, p_disp_layers = p_disp_L,
            r_pos_layers  = r_pos_L,  p_pos_layers  = p_pos_L,
            # chunk-averaged
            r_curv_layers_ch = r_curv_ch, p_curv_layers_ch = p_curv_ch,
            r_disp_layers_ch = r_disp_ch, p_disp_layers_ch = p_disp_ch,
            r_pos_layers_ch  = r_pos_ch,  p_pos_layers_ch  = p_pos_ch,
            # shuffle null (N_PERM word-order permutations of neural curvature)
            null_curv_layers_ch_mean = null_curv_ch_mean,
            null_curv_layers_ch_std  = null_curv_ch_std,
            null_disp_layers_ch_mean = null_disp_ch_mean,
            null_disp_layers_ch_std  = null_disp_ch_std,
        ))

    if out:
        json.dump(out, open(out_path, "w"))
        print(f"\n  Saved {len(out)} patients → {out_path}")
    else:
        print(f"\n  No data saved for {model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Run a single model (default: all)")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    for model in models:
        process_model(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
