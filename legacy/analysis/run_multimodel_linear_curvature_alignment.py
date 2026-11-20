"""Run linear cross-space curvature alignment across multiple LLMs.

This script asks whether brain--model future-state readout is explained better
by absolute model flatness or by brain-model geometry matching.

Default analysis:
    LLM_t    -> neural_{t+1}
    neural_t -> LLM_{t+1}

using cross-validated Ridge readout for every requested model/patient/region.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from curvature_alignment import _safe_partial_pearson, _safe_pearson, run_bidirectional_curvature_alignment  # noqa: E402
from run_geometry_paper_curvature_alignment import (  # noqa: E402
    add_metadata,
    geom,
    load_loo_neural_layers,
    load_patient_arrays,
    parse_float_list,
)


CAUSAL_MODELS = [
    "gpt2",
    "gpt2-medium",
    "opt-350m",
    "llama-2-7b",
    "llama-3.1-8b",
    "gemma-2-9b",
    "mistral-7b",
]


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / (np.nanstd(values, ddof=1) + 1e-12)


def standardized_regression(
    y: np.ndarray,
    predictors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Return standardized OLS betas and R2 for finite rows."""
    names = list(predictors)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.asarray(predictors[name], dtype=float) for name in names])
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() <= len(names) + 1:
        return {
            "n_regression_models": int(mask.sum()),
            "regression_r2": float("nan"),
            **{f"beta_{name}": float("nan") for name in names},
        }

    y_z = zscore(y[mask])
    X_z = np.column_stack([zscore(X[mask, idx]) for idx in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(X_z, y_z, rcond=None)
    y_hat = X_z @ beta
    ss_res = float(np.sum((y_z - y_hat) ** 2))
    ss_tot = float(np.sum((y_z - np.mean(y_z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n_regression_models": int(mask.sum()),
        "regression_r2": float(r2),
        **{f"beta_{name}": float(value) for name, value in zip(names, beta)},
    }


def geometry_metrics(pointwise: pd.DataFrame, direction: str) -> Dict[str, float]:
    valid = pointwise.loc[pointwise["valid"]].copy()
    if valid.empty:
        return {
            "mean_decoding_corr": float("nan"),
            "median_decoding_corr": float("nan"),
            "mean_target_mse": float("nan"),
            "mean_llm_curvature": float("nan"),
            "median_llm_curvature": float("nan"),
            "mean_neural_curvature": float("nan"),
            "mean_curvature_mismatch": float("nan"),
            "median_curvature_mismatch": float("nan"),
            "curvature_match_r": float("nan"),
            "curvature_match_p": float("nan"),
            "mean_llm_displacement": float("nan"),
            "mean_neural_displacement": float("nan"),
            "mean_displacement_mismatch": float("nan"),
        }

    if direction == "LLM_to_neural":
        llm_curv = valid["curvature_source"].to_numpy(dtype=float)
        neural_curv = valid["curvature_target"].to_numpy(dtype=float)
        llm_disp = valid["displacement_source"].to_numpy(dtype=float)
        neural_disp = valid["displacement_target"].to_numpy(dtype=float)
    elif direction == "neural_to_LLM":
        llm_curv = valid["curvature_target"].to_numpy(dtype=float)
        neural_curv = valid["curvature_source"].to_numpy(dtype=float)
        llm_disp = valid["displacement_target"].to_numpy(dtype=float)
        neural_disp = valid["displacement_source"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Expected cross-space direction, got {direction!r}")

    match_r, match_p = _safe_pearson(llm_curv, neural_curv)
    return {
        "mean_decoding_corr": float(valid["target_corr"].mean()),
        "median_decoding_corr": float(valid["target_corr"].median()),
        "mean_target_mse": float(valid["target_mse"].mean()),
        "mean_llm_curvature": float(np.nanmean(llm_curv)),
        "median_llm_curvature": float(np.nanmedian(llm_curv)),
        "mean_neural_curvature": float(np.nanmean(neural_curv)),
        "mean_curvature_mismatch": float(valid["curvature_mismatch"].mean()),
        "median_curvature_mismatch": float(valid["curvature_mismatch"].median()),
        "curvature_match_r": float(match_r),
        "curvature_match_p": float(match_p),
        "mean_llm_displacement": float(np.nanmean(llm_disp)),
        "mean_neural_displacement": float(np.nanmean(neural_disp)),
        "mean_displacement_mismatch": float(valid["displacement_mismatch"].mean()),
    }


def summarize_model_level(patient_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metrics = [
        "global_r2",
        "mean_decoding_corr",
        "mean_target_mse",
        "mean_llm_curvature",
        "median_llm_curvature",
        "mean_neural_curvature",
        "mean_curvature_mismatch",
        "median_curvature_mismatch",
        "curvature_match_r",
        "mean_llm_displacement",
        "mean_neural_displacement",
        "mean_displacement_mismatch",
        "pearson_joint_curvature_vs_mse_r",
        "pearson_joint_curvature_vs_decoding_corr_r",
        "partial_joint_curvature_vs_mse_control_displacement_r",
        "partial_joint_curvature_vs_decoding_corr_control_displacement_r",
    ]
    for (model, region, direction), group in patient_summary.groupby(
        ["model", "region", "direction"]
    ):
        row: Dict[str, object] = {
            "model": model,
            "region": region,
            "direction": direction,
            "n_patients": int(group["pid"].nunique()),
            "mean_loo_layer_neu": float(group["loo_layer_neu"].mean()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sem"] = sem(values)
            row[f"{metric}_median"] = float(np.nanmedian(values))
        rows.append(row)
    return pd.DataFrame(rows)


def predictor_tests(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    outcomes = ["global_r2_mean", "mean_decoding_corr_mean"]
    predictors = {
        "llm_curvature": "mean_llm_curvature_mean",
        "curvature_mismatch": "mean_curvature_mismatch_mean",
        "curvature_match": "curvature_match_r_mean",
        "llm_displacement": "mean_llm_displacement_mean",
    }

    for (region, direction), group in model_summary.groupby(["region", "direction"]):
        for outcome in outcomes:
            y = group[outcome].to_numpy(dtype=float)
            for label, column in predictors.items():
                x = group[column].to_numpy(dtype=float)
                r, p = _safe_pearson(x, y)
                rows.append(
                    {
                        "region": region,
                        "direction": direction,
                        "outcome": outcome,
                        "test": f"{label}_vs_{outcome}",
                        "n_models": int(np.isfinite(x).sum()),
                        "r": float(r),
                        "p": float(p),
                    }
                )

            flatness_control = group[["mean_curvature_mismatch_mean"]].to_numpy(dtype=float)
            r_flat_partial, p_flat_partial = _safe_partial_pearson(
                group["mean_llm_curvature_mean"].to_numpy(dtype=float),
                y,
                flatness_control,
            )
            mismatch_control = group[["mean_llm_curvature_mean"]].to_numpy(dtype=float)
            r_mismatch_partial, p_mismatch_partial = _safe_partial_pearson(
                group["mean_curvature_mismatch_mean"].to_numpy(dtype=float),
                y,
                mismatch_control,
            )
            rows.extend(
                [
                    {
                        "region": region,
                        "direction": direction,
                        "outcome": outcome,
                        "test": f"llm_curvature_vs_{outcome}_control_mismatch",
                        "n_models": int(len(group)),
                        "r": float(r_flat_partial),
                        "p": float(p_flat_partial),
                    },
                    {
                        "region": region,
                        "direction": direction,
                        "outcome": outcome,
                        "test": f"curvature_mismatch_vs_{outcome}_control_llm_curvature",
                        "n_models": int(len(group)),
                        "r": float(r_mismatch_partial),
                        "p": float(p_mismatch_partial),
                    },
                ]
            )

            regression = standardized_regression(
                y,
                {
                    "llm_curvature": group["mean_llm_curvature_mean"].to_numpy(dtype=float),
                    "curvature_mismatch": group[
                        "mean_curvature_mismatch_mean"
                    ].to_numpy(dtype=float),
                },
            )
            rows.append(
                {
                    "region": region,
                    "direction": direction,
                    "outcome": outcome,
                    "test": f"standardized_regression_{outcome}",
                    "n_models": regression.pop("n_regression_models"),
                    "r": float("nan"),
                    "p": float("nan"),
                    **regression,
                }
            )
    return pd.DataFrame(rows)


def direction_tag(directions: Tuple[str, ...]) -> str:
    return (
        "both"
        if set(directions) == {"LLM_to_neural", "neural_to_LLM"}
        else "-".join(directions)
    )


def model_tag(models: Iterable[str]) -> str:
    models = list(models)
    if set(models) == set(geom.MODELS):
        return "all-models"
    if set(models) == set(CAUSAL_MODELS):
        return "causal-models"
    if len(models) <= 3:
        return "-".join(models)
    return f"{len(models)}models"


def run(
    models: Iterable[str],
    regions: Iterable[str],
    patients: Optional[Iterable[str]],
    neural_space: str,
    output_dir: str,
    n_bins: int,
    n_splits: int,
    alphas: Optional[np.ndarray],
    llm_pcs: Optional[int],
    shuffle_cv: bool,
    directions: Tuple[str, ...],
    target_lag: int,
    save_pointwise: bool,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    requested_patients = set(patients) if patients else None
    directions = tuple(directions)

    all_summary: List[pd.DataFrame] = []
    all_bins: List[pd.DataFrame] = []
    all_pointwise: List[pd.DataFrame] = []

    for model in models:
        for region in regions:
            try:
                layers = load_loo_neural_layers(model, region)
            except FileNotFoundError as exc:
                print(f"Skipping {model} {region}: {exc}", flush=True)
                continue

            for pid in geom.PATIENTS:
                if requested_patients is not None and pid not in requested_patients:
                    continue
                if region not in geom.PATIENTS[pid] or pid not in layers:
                    continue

                layer = layers[pid]
                print(
                    f"{model} {region} {pid} layer={layer} neural_space={neural_space}",
                    flush=True,
                )
                try:
                    Z_llm, Z_neural, meta = load_patient_arrays(
                        pid=pid,
                        region=region,
                        model=model,
                        layer=layer,
                        neural_space=neural_space,
                        llm_pcs=llm_pcs,
                        random_state=random_state,
                    )
                    print(f"  Z_llm={Z_llm.shape}  Z_neural={Z_neural.shape}", flush=True)
                    out = run_bidirectional_curvature_alignment(
                        Z_llm=Z_llm,
                        Z_neural=Z_neural,
                        n_bins=n_bins,
                        alphas=alphas,
                        n_splits=n_splits,
                        shuffle_cv=shuffle_cv,
                        decoder="ridge",
                        directions=directions,
                        target_lag=target_lag,
                        random_state=random_state,
                    )
                except Exception as exc:
                    print(f"  skipped: {exc}", flush=True)
                    continue

                summary = out.summary.copy()
                for direction in summary["direction"].unique():
                    pointwise = out.pointwise[out.pointwise["direction"] == direction]
                    metrics = geometry_metrics(pointwise, direction)
                    for key, value in metrics.items():
                        summary.loc[summary["direction"] == direction, key] = value

                all_summary.append(add_metadata(summary, meta))
                all_bins.append(add_metadata(out.bins, meta))
                if save_pointwise:
                    all_pointwise.append(add_metadata(out.pointwise, meta))

    patient_summary = (
        pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    )
    bins = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    pointwise = (
        pd.concat(all_pointwise, ignore_index=True)
        if save_pointwise and all_pointwise
        else pd.DataFrame()
    )
    model_summary = (
        summarize_model_level(patient_summary)
        if not patient_summary.empty
        else pd.DataFrame()
    )
    tests = predictor_tests(model_summary) if not model_summary.empty else pd.DataFrame()

    region_name = "-".join(regions)
    patient_name = "selected" if requested_patients is not None else "all-patients"
    llm_tag = f"llmpc{llm_pcs}" if llm_pcs is not None else "llmraw"
    cv_tag = "shufflecv" if shuffle_cv else "temporalfold"
    lag_tag = f"lag{target_lag:+d}".replace("+", "plus").replace("-", "minus")
    stem = (
        f"multimodel_linear_curvature_alignment_{model_tag(models)}_{region_name}_"
        f"{patient_name}_{neural_space}_{llm_tag}_{cv_tag}_ridge_"
        f"{direction_tag(directions)}_{lag_tag}"
    )

    patient_summary_path = os.path.join(output_dir, f"{stem}_patient_summary.csv")
    bins_path = os.path.join(output_dir, f"{stem}_bins.csv")
    pointwise_path = os.path.join(output_dir, f"{stem}_pointwise.csv")
    model_summary_path = os.path.join(output_dir, f"{stem}_model_summary.csv")
    tests_path = os.path.join(output_dir, f"{stem}_model_predictor_tests.csv")

    patient_summary.to_csv(patient_summary_path, index=False)
    bins.to_csv(bins_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    if save_pointwise:
        pointwise.to_csv(pointwise_path, index=False)

    print(f"\nSaved patient summary: {patient_summary_path}")
    print(f"Saved bins:            {bins_path}")
    print(f"Saved model summary:   {model_summary_path}")
    print(f"Saved predictor tests: {tests_path}")
    if save_pointwise:
        print(f"Saved pointwise:       {pointwise_path}")

    return patient_summary, model_summary, tests, bins


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run linear cross-space curvature alignment across multiple models "
            "and test whether flatness or brain-model geometry matching predicts readout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=CAUSAL_MODELS,
        choices=list(geom.MODELS),
        help="Models to include. Defaults to causal/decoder-style models.",
    )
    parser.add_argument("--regions", nargs="+", default=geom.REGIONS, choices=geom.REGIONS)
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--neural-space", choices=["pca", "raw"], default="pca")
    parser.add_argument("--llm-pcs", type=int, default=64)
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["LLM_to_neural", "neural_to_LLM"],
        default=["LLM_to_neural", "neural_to_LLM"],
    )
    parser.add_argument("--target-lag", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--shuffle-cv", action="store_true")
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--save-pointwise", action="store_true")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(THIS_DIR, "results", "geometry_paper_curvature_alignment"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patient_summary, model_summary, tests, _ = run(
        models=tuple(args.models),
        regions=tuple(args.regions),
        patients=args.patients,
        neural_space=args.neural_space,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        n_splits=args.n_splits,
        alphas=parse_float_list(args.alphas),
        llm_pcs=args.llm_pcs,
        shuffle_cv=args.shuffle_cv,
        directions=tuple(args.directions),
        target_lag=args.target_lag,
        save_pointwise=args.save_pointwise,
        random_state=args.random_state,
    )

    if not model_summary.empty:
        preview_cols = [
            "model",
            "region",
            "direction",
            "n_patients",
            "global_r2_mean",
            "mean_decoding_corr_mean",
            "mean_llm_curvature_mean",
            "mean_curvature_mismatch_mean",
            "curvature_match_r_mean",
        ]
        print("\nModel-level summary:")
        print(model_summary[preview_cols].to_string(index=False))
    if not tests.empty:
        print("\nAcross-model predictor tests:")
        print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
