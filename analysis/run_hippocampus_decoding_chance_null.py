"""Label-permutation chance null for hippocampus cross-space decoding.

Tests whether the cross-validated global_r2 reported in
run_geometry_paper_curvature_alignment.py is above chance, using a full
random-permutation null (not circular-shift): the target sequence's row
correspondence to the source sequence is fully shuffled, the same
cross-validated ridge decoder is refit on the shuffled correspondence, and
the resulting null R2 distribution is compared to the real R2.

This is intentionally a *different* question from the word-order /
flanking-word shuffle nulls already in run_geometry_paper_curvature_alignment.py,
which hold decoding accuracy fixed and only test whether curvature predicts it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(THIS_DIR)
GEOMETRY_PAPER_DIR = os.path.join(
    WORKSPACE, "neural_network_similarity", "Experiments", "p6", "geometry_paper"
)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if GEOMETRY_PAPER_DIR not in sys.path:
    sys.path.insert(0, GEOMETRY_PAPER_DIR)

import run_geometry_analysis as geom  # noqa: E402
from curvature_alignment import align_by_target_lag, fit_linear_cross_space_decoder  # noqa: E402
from run_geometry_paper_curvature_alignment import (  # noqa: E402
    load_patient_arrays,
    load_loo_neural_layers,
)

REGION = "hippocampus"
MODEL = "llama-3.1-8b"
DIRECTIONS = ("LLM_to_neural", "neural_to_LLM")


def run_patient_direction(
    pid: str,
    layer: int,
    direction: str,
    target_lag: int,
    n_splits: int,
    shuffle_cv: bool,
    decoder: str,
    random_state: int,
    n_shuffles: int,
) -> Dict[str, object]:
    Z_llm, Z_neural, meta = load_patient_arrays(
        pid=pid,
        region=REGION,
        model=MODEL,
        layer=layer,
        neural_space="pca",
        llm_pcs=64,
        random_state=random_state,
        match_pcs=True,
    )
    X_source, Y_target = (Z_llm, Z_neural) if direction == "LLM_to_neural" else (Z_neural, Z_llm)
    X_aligned, Y_aligned, _, _ = align_by_target_lag(X_source, Y_target, target_lag=target_lag)

    t0 = time.time()
    Y_pred = fit_linear_cross_space_decoder(
        X_aligned, Y_aligned, n_splits=n_splits, shuffle_cv=shuffle_cv,
        random_state=random_state, decoder=decoder,
    )
    real_r2 = float(r2_score(Y_aligned, Y_pred, multioutput="variance_weighted"))
    fit_seconds = time.time() - t0

    rng = np.random.default_rng(abs(hash((pid, direction))) % (2**32))
    null_r2 = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        perm = rng.permutation(len(Y_aligned))
        Y_null = Y_aligned[perm]
        Y_pred_null = fit_linear_cross_space_decoder(
            X_aligned, Y_null, n_splits=n_splits, shuffle_cv=shuffle_cv,
            random_state=random_state, decoder=decoder,
        )
        null_r2[i] = r2_score(Y_null, Y_pred_null, multioutput="variance_weighted")

    p_value = float((1 + np.sum(null_r2 >= real_r2)) / (1 + n_shuffles))
    return dict(
        pid=pid,
        region=REGION,
        direction=direction,
        n_timepoints=int(len(Y_aligned)),
        real_r2=real_r2,
        null_mean=float(null_r2.mean()),
        null_std=float(null_r2.std(ddof=1)),
        null_p95=float(np.percentile(null_r2, 95)),
        p_value=p_value,
        n_shuffles=n_shuffles,
        fit_seconds=fit_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--target-lag", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--decoder", default="ridge")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument(
        "--output",
        default=os.path.join(
            THIS_DIR, "..", "figures", "results", "geometry_paper_curvature_alignment",
            "hippocampus_decoding_chance_null.csv",
        ),
    )
    args = parser.parse_args()

    loo_layers = load_loo_neural_layers(MODEL, REGION)
    patients = [
        pid for pid in geom.PATIENTS
        if REGION in geom.PATIENTS[pid] and pid in loo_layers
        and (args.patients is None or pid in args.patients)
    ]

    rows: List[Dict[str, object]] = []
    for pid in patients:
        layer = loo_layers[pid]
        for direction in DIRECTIONS:
            print(f"{pid} {direction} layer={layer} ...", flush=True)
            try:
                row = run_patient_direction(
                    pid=pid,
                    layer=layer,
                    direction=direction,
                    target_lag=args.target_lag,
                    n_splits=args.n_splits,
                    shuffle_cv=True,
                    decoder=args.decoder,
                    random_state=args.random_state,
                    n_shuffles=args.n_shuffles,
                )
            except Exception as exc:
                print(f"  skipped: {exc}", flush=True)
                continue
            print(
                f"  real_r2={row['real_r2']:+.4f}  null_mean={row['null_mean']:+.4f} "
                f"null_std={row['null_std']:.4f}  p={row['p_value']:.4f}  "
                f"({row['fit_seconds']:.2f}s/fit)",
                flush=True,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
