"""Recompute hippocampal geodesic decomposition using raw-standardized neural data.

Replaces the PCA-based geodesic quantities in every JSON with versions computed
on the full-dimensional standardized spike vectors. This gives the geo/nrm split
a larger ambient space so the perpendicular component is geometrically meaningful.

Overwrites these regression fields for every patient record:
  r_neu_geo, p_neu_geo          — hpc kappa_geo vs surprisal
  r_neu_nrm, p_neu_nrm          — hpc kappa_nrm vs surprisal
  r_geo_sem_neu, p_geo_sem_neu  — LLM kappa_geo vs hpc kappa_geo
  r_nrm_sem_neu, p_nrm_sem_neu  — LLM kappa_nrm vs hpc kappa_nrm
  r_geo_sem_nrm_neu, ...        — cross terms
  r_nrm_sem_geo_neu, ...
"""

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
    fit_neural_raw,
    geodesic_curvature,
    load_patient,
)

RESULTS = os.path.join(GEOMETRY_DIR, "results")

GEODESIC_KEYS = (
    "r_neu_geo", "p_neu_geo",
    "r_neu_nrm", "p_neu_nrm",
    "r_geo_sem_neu", "p_geo_sem_neu",
    "r_nrm_sem_neu", "p_nrm_sem_neu",
    "r_geo_sem_nrm_neu", "p_geo_sem_nrm_neu",
    "r_nrm_sem_geo_neu", "p_nrm_sem_geo_neu",
)


def _r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    return pearsonr(x[mask], y[mask])


def compute_raw_geodesic(data, layer, reg):
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
    if valid.sum() < 10:
        return None

    surp_y = data["surp"][np.arange(1, n - 1) + 1][valid]

    # LLM geodesic decomposition at the LOO layer (recompute from embedding cache)
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    geo_llm, nrm_llm = geodesic_curvature(emb_opt, k_neighbors=20, tangent_dim=10)
    c_geo = geo_llm[valid]
    c_nrm = nrm_llm[valid]

    # Neural geodesic decomposition using raw-standardized data
    Y_neu_raw = fit_neural_raw(data["neu_vecs"], n)
    if Y_neu_raw is None:
        return None
    geo_neu, nrm_neu = geodesic_curvature(
        Y_neu_raw,
        k_neighbors=min(15, len(Y_neu_raw) - 1),
        tangent_dim=min(10, Y_neu_raw.shape[1] - 1),
    )
    c_geo_neu = geo_neu[valid]
    c_nrm_neu = nrm_neu[valid]

    r_neu_geo,      p_neu_geo      = _r(c_geo_neu, surp_y)
    r_neu_nrm,      p_neu_nrm      = _r(c_nrm_neu, surp_y)
    r_geo_sem_neu,  p_geo_sem_neu  = _r(c_geo,     c_geo_neu)
    r_nrm_sem_neu,  p_nrm_sem_neu  = _r(c_nrm,     c_nrm_neu)
    r_geo_nrm_sn,   p_geo_nrm_sn   = _r(c_geo,     c_nrm_neu)
    r_nrm_geo_sn,   p_nrm_geo_sn   = _r(c_nrm,     c_geo_neu)

    return dict(
        r_neu_geo=float(r_neu_geo),             p_neu_geo=float(p_neu_geo),
        r_neu_nrm=float(r_neu_nrm),             p_neu_nrm=float(p_neu_nrm),
        r_geo_sem_neu=float(r_geo_sem_neu),     p_geo_sem_neu=float(p_geo_sem_neu),
        r_nrm_sem_neu=float(r_nrm_sem_neu),     p_nrm_sem_neu=float(p_nrm_sem_neu),
        r_geo_sem_nrm_neu=float(r_geo_nrm_sn),  p_geo_sem_nrm_neu=float(p_geo_nrm_sn),
        r_nrm_sem_geo_neu=float(r_nrm_geo_sn),  p_nrm_sem_geo_neu=float(p_nrm_geo_sn),
    )


def patch_json(json_path, model_tag, region):
    with open(json_path) as f:
        records = json.load(f)

    changed = False
    for rec in records:
        reg = rec.get("regression")
        if reg is None:
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

        result = compute_raw_geodesic(data, layer, reg)
        if result is None:
            print(f"    {pid} — no neural data or too few valid points")
            for k in GEODESIC_KEYS:
                reg[k] = float("nan")
            changed = True
            continue

        reg.update(result)
        changed = True
        print(f"    {pid} layer={layer}  "
              f"r_neu_geo={result['r_neu_geo']:+.3f}  "
              f"r_neu_nrm={result['r_neu_nrm']:+.3f}  "
              f"r_nrm×nrm={result['r_nrm_sem_neu']:+.3f}  "
              f"r_geo×nrm={result['r_geo_sem_nrm_neu']:+.3f}")

    if changed:
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {json_path}")
    else:
        print(f"  No changes for {json_path}")


def main():
    for model_tag in MODELS:
        for region in REGIONS:
            json_path = os.path.join(RESULTS, f"geometry_{model_tag}_{region}.json")
            if not os.path.exists(json_path):
                print(f"Missing: {json_path}, skipping")
                continue
            print(f"\n{model_tag} / {region}")
            patch_json(json_path, model_tag, region)


if __name__ == "__main__":
    main()
