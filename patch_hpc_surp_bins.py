"""Patch geometry JSONs with hippocampal component activation by surprisal.

Adds to each patient record's regression dict:
  hpc_geo_by_surp_bin   = mean hpc kappa_geo in LLM surprisal quintiles
  hpc_nrm_by_surp_bin   = mean hpc kappa_nrm in LLM surprisal quintiles
  hpc_curv_by_surp_bin  = mean total hpc curvature in LLM surprisal quintiles

The surprisal quintiles are constructed the same way as run_geometry_analysis:
using finite LLM geodesic/normal curvature at the selected LOO-neural layer.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

GEOMETRY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GEOMETRY_DIR)

from run_geometry_analysis import (  # noqa: E402
    MODELS,
    PATIENTS,
    REGIONS,
    build_valid_mask,
    fit_neural_pca,
    fit_neural_raw,
    geodesic_curvature,
    load_patient,
    trajectory_curvature,
)

RESULTS = os.path.join(GEOMETRY_DIR, "results")
PATCH_KEYS = (
    "r_neu_geo",
    "p_neu_geo",
    "r_neu_nrm",
    "p_neu_nrm",
    "hpc_geo_by_surp_bin",
    "hpc_nrm_by_surp_bin",
    "hpc_curv_by_surp_bin",
)
N_BINS = 5


def _nan_bins():
    return [float("nan")] * N_BINS


def _q5_q1(vals):
    vals = np.asarray(vals, dtype=float)
    if not np.isfinite(vals[0]) or abs(vals[0]) <= 1e-12:
        return float("nan")
    return float(vals[-1] / vals[0])


def _nan_result():
    return {
        "r_neu_geo": float("nan"),
        "p_neu_geo": float("nan"),
        "r_neu_nrm": float("nan"),
        "p_neu_nrm": float("nan"),
        "hpc_geo_by_surp_bin": _nan_bins(),
        "hpc_nrm_by_surp_bin": _nan_bins(),
        "hpc_curv_by_surp_bin": _nan_bins(),
    }


def compute_hpc_bins(data, layer):
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
    if valid.sum() < N_BINS * 5:
        return _nan_result()

    interior = np.arange(1, n - 1)
    surp_y = data["surp"][interior + 1][valid]

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm

    geo_arr, nrm_arr = geodesic_curvature(emb_opt, k_neighbors=20, tangent_dim=10)
    c_geo = geo_arr[valid]
    c_nrm = nrm_arr[valid]

    Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return _nan_result()

    Y_neu_raw = fit_neural_raw(data["neu_vecs"], n)
    neu_for_geo = Y_neu_raw if Y_neu_raw is not None else Y_neu
    geo_neu_arr, nrm_neu_arr = geodesic_curvature(
        neu_for_geo,
        k_neighbors=min(15, len(neu_for_geo) - 1),
        tangent_dim=min(5, neu_for_geo.shape[1] - 1),
    )
    c_geo_neu = geo_neu_arr[valid]
    c_nrm_neu = nrm_neu_arr[valid]
    c_neu = trajectory_curvature(Y_neu)[valid]

    fin_llm = np.isfinite(c_geo) & np.isfinite(c_nrm) & np.isfinite(surp_y)
    fin_hpc = (np.isfinite(c_geo_neu) & np.isfinite(c_nrm_neu) &
               np.isfinite(c_neu) & np.isfinite(surp_y))
    if fin_llm.sum() < N_BINS * 5 or fin_hpc.sum() < N_BINS * 5:
        return _nan_result()

    q_edges = np.percentile(surp_y[fin_llm], np.linspace(0, 100, N_BINS + 1))
    q_edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(surp_y, q_edges) - 1, 0, N_BINS - 1)

    def by_bin(x):
        return [
            float(np.mean(x[fin_hpc & (bin_idx == b)]))
            if (fin_hpc & (bin_idx == b)).sum() > 0 else float("nan")
            for b in range(N_BINS)
        ]

    fin_geo = np.isfinite(c_geo_neu) & np.isfinite(surp_y)
    fin_nrm = np.isfinite(c_nrm_neu) & np.isfinite(surp_y)
    r_geo, p_geo = (pearsonr(c_geo_neu[fin_geo], surp_y[fin_geo])
                    if fin_geo.sum() > 10 else (float("nan"), float("nan")))
    r_nrm, p_nrm = (pearsonr(c_nrm_neu[fin_nrm], surp_y[fin_nrm])
                    if fin_nrm.sum() > 10 else (float("nan"), float("nan")))

    return {
        "r_neu_geo": float(r_geo),
        "p_neu_geo": float(p_geo),
        "r_neu_nrm": float(r_nrm),
        "p_neu_nrm": float(p_nrm),
        "hpc_geo_by_surp_bin": by_bin(c_geo_neu),
        "hpc_nrm_by_surp_bin": by_bin(c_nrm_neu),
        "hpc_curv_by_surp_bin": by_bin(c_neu),
    }


def patch_json(json_path, model_tag, region, force=False):
    with open(json_path) as f:
        records = json.load(f)

    changed = False
    for rec in records:
        reg = rec.get("regression")
        if reg is None:
            continue
        if not force and all(k in reg for k in PATCH_KEYS):
            print(f"    {rec['pid']} -- already patched, skipping")
            continue

        pid = rec["pid"]
        preg = PATIENTS.get(pid)
        if preg is None:
            print(f"    {pid} -- not in PATIENTS dict, skipping")
            continue

        data = load_patient(pid, preg, region, model_tag)
        if data is None:
            print(f"    {pid} -- load_patient returned None, skipping")
            continue

        result = compute_hpc_bins(data, reg["layer"])
        reg.update(result)
        changed = True
        print(f"    {pid} layer={reg['layer']}  "
              f"hpc_geo Q5/Q1={_q5_q1(result['hpc_geo_by_surp_bin']):.3f}  "
              f"hpc_nrm Q5/Q1={_q5_q1(result['hpc_nrm_by_surp_bin']):.3f}")

    if changed:
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {json_path}")
    else:
        print(f"  No changes needed for {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch geometry JSONs with hippocampal surprisal-bin component curves."
    )
    parser.add_argument("--model", default="llama-3.1-8b",
                        help="Model tag to patch; ignored with --all.")
    parser.add_argument("--region", default="hippocampus",
                        help="Region to patch; ignored with --all.")
    parser.add_argument("--all", action="store_true",
                        help="Patch all model/region JSONs.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if hpc bin keys already exist.")
    args = parser.parse_args()

    targets = ((m, r) for m in MODELS for r in REGIONS) if args.all else [(args.model, args.region)]
    for model_tag, region in targets:
        json_path = os.path.join(RESULTS, f"geometry_{model_tag}_{region}.json")
        if not os.path.exists(json_path):
            print(f"Missing: {json_path}, skipping")
            continue
        print(f"\n{model_tag} / {region}")
        patch_json(json_path, model_tag, region, force=args.force)


if __name__ == "__main__":
    main()
