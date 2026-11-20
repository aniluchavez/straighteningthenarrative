"""Patch existing geometry JSONs with the 2x2 component coupling correlations.

Adds to each patient record's regression dict:
  r_geo_sem_neu      = r(LLM kappa_geo, hpc kappa_geo)
  r_nrm_sem_neu      = r(LLM kappa_nrm, hpc kappa_nrm)
  r_geo_sem_nrm_neu  = r(LLM kappa_geo, hpc kappa_nrm)
  r_nrm_sem_geo_neu  = r(LLM kappa_nrm, hpc kappa_geo)

Skips records that already have these fields.
Runs in minutes — no shuffle permutations, no per-layer sweep.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

GEOMETRY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GEOMETRY_DIR)

from run_geometry_analysis import (
    PATIENTS,
    REGIONS,
    MODELS,
    build_valid_mask,
    fit_neural_pca,
    fit_neural_raw,
    geodesic_curvature,
    load_patient,
)

RESULTS = os.path.join(GEOMETRY_DIR, "results")
NEW_KEYS = (
    "r_geo_sem_neu",
    "r_nrm_sem_neu",
    "r_geo_sem_nrm_neu",
    "r_nrm_sem_geo_neu",
)


def compute_coupling(data, layer):
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
    if valid.sum() < 10:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm

    geo_arr, nrm_arr = geodesic_curvature(emb_opt, k_neighbors=20, tangent_dim=10)
    c_geo = geo_arr[valid]
    c_nrm = nrm_arr[valid]

    Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    Y_neu_raw = fit_neural_raw(data["neu_vecs"], n)
    neu_for_geo = Y_neu_raw if Y_neu_raw is not None else Y_neu
    geo_neu_arr, nrm_neu_arr = geodesic_curvature(
        neu_for_geo,
        k_neighbors=min(15, len(neu_for_geo) - 1),
        tangent_dim=min(5, neu_for_geo.shape[1] - 1),
    )
    c_geo_neu = geo_neu_arr[valid]
    c_nrm_neu = nrm_neu_arr[valid]

    def _r(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return float("nan"), float("nan")
        return pearsonr(x[mask], y[mask])

    r_geo_sn,      p_geo_sn      = _r(c_geo,     c_geo_neu)
    r_nrm_sn,      p_nrm_sn      = _r(c_nrm,     c_nrm_neu)
    r_geo_nrm_sn,  p_geo_nrm_sn  = _r(c_geo,     c_nrm_neu)
    r_nrm_geo_sn,  p_nrm_geo_sn  = _r(c_nrm,     c_geo_neu)

    return dict(
        r_geo_sem_neu=float(r_geo_sn),       p_geo_sem_neu=float(p_geo_sn),
        r_nrm_sem_neu=float(r_nrm_sn),       p_nrm_sem_neu=float(p_nrm_sn),
        r_geo_sem_nrm_neu=float(r_geo_nrm_sn), p_geo_sem_nrm_neu=float(p_geo_nrm_sn),
        r_nrm_sem_geo_neu=float(r_nrm_geo_sn), p_nrm_sem_geo_neu=float(p_nrm_geo_sn),
    )


def patch_json(json_path, model_tag, region, force=False):
    with open(json_path) as f:
        records = json.load(f)

    changed = False
    for rec in records:
        reg = rec.get("regression")
        if reg is None:
            continue
        if not force and all(k in reg for k in NEW_KEYS):
            print(f"    {rec['pid']} — already patched, skipping")
            continue

        pid = rec["pid"]
        layer = reg["layer"]
        preg = PATIENTS.get(pid)
        if preg is None:
            print(f"    {pid} — not in PATIENTS dict, skipping")
            continue

        data = load_patient(pid, preg, region, model_tag)
        if data is None:
            print(f"    {pid} — load_patient returned None, skipping")
            continue

        result = compute_coupling(data, layer)
        if result is None:
            print(f"    {pid} — no neural data or too few valid points")
            for k in NEW_KEYS:
                reg[k] = float("nan")
                reg[f"p{k[1:]}"] = float("nan")
            changed = True
            continue

        reg.update(result)
        changed = True
        print(f"    {pid} layer={layer}  "
              f"r_geo×geo={result['r_geo_sem_neu']:.3f}  "
              f"r_nrm×nrm={result['r_nrm_sem_neu']:.3f}  "
              f"r_geo×nrm={result['r_geo_sem_nrm_neu']:.3f}  "
              f"r_nrm×geo={result['r_nrm_sem_geo_neu']:.3f}")

    if changed:
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {json_path}")
    else:
        print(f"  No changes needed for {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch geometry JSONs with LLM x hpc component coupling metrics."
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Model tags to patch.")
    parser.add_argument("--regions", nargs="+", default=list(REGIONS),
                        help="Regions to patch.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if coupling keys already exist.")
    args = parser.parse_args()

    for model_tag in args.models:
        if model_tag not in MODELS:
            print(f"Unknown model: {model_tag}, skipping")
            continue
        for region in args.regions:
            if region not in REGIONS:
                print(f"Unknown region: {region}, skipping")
                continue
            json_path = os.path.join(RESULTS, f"geometry_{model_tag}_{region}.json")
            if not os.path.exists(json_path):
                print(f"Missing: {json_path}, skipping")
                continue
            print(f"\n{model_tag} / {region}")
            patch_json(json_path, model_tag, region, force=args.force)


if __name__ == "__main__":
    main()
