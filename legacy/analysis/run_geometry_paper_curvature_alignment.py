"""Run curvature alignment on the geometry-paper patient data.

This bridges the geometry-paper loaders to curvature_alignment.py:

    Z_llm    = selected word-aligned LLM layer embeddings
    Z_neural = word-aligned neural representation for one patient/region

Each patient is run separately because raw neural feature dimensions differ
across patients. The outputs are combined into CSV tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(THIS_DIR)
GEOMETRY_PAPER_DIR = os.path.join(
    WORKSPACE,
    "neural_network_similarity",
    "Experiments",
    "p6",
    "geometry_paper",
)

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if GEOMETRY_PAPER_DIR not in sys.path:
    sys.path.insert(0, GEOMETRY_PAPER_DIR)

import run_geometry_analysis as geom  # noqa: E402
from curvature_alignment import (  # noqa: E402
    align_by_target_lag,
    local_curvature_shuffled_flanks,  # noqa: F401 (used in flanking_word_shuffle_null)
    make_curvature_bins,
    make_curvature_matched_displacement_bins,
    make_displacement_matched_curvature_bins,
    make_flanks_shuffle_bins,
    make_pointwise_readout_table,
    pointwise_decoding_error,
    run_bidirectional_curvature_alignment,
    summarize_pointwise_readout_metrics,
)


NULL_METRICS = {
    "pearson_joint_curvature_vs_mse_r": "greater",
    "spearman_joint_curvature_vs_mse_r": "greater",
    "pearson_joint_curvature_vs_decoding_corr_r": "less",
    "pearson_curvature_mismatch_vs_mse_r": "greater",
    "partial_joint_curvature_vs_mse_control_displacement_r": "greater",
    "partial_joint_curvature_vs_decoding_corr_control_displacement_r": "less",
    "partial_joint_curvature_vs_mse_control_source_target_displacement_r": "greater",
    "partial_joint_curvature_vs_decoding_corr_control_source_target_displacement_r": "less",
}


def is_scalar_metadata_value(value: object) -> bool:
    return value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            np.integer,
            np.floating,
            np.bool_,
        ),
    )


def metadata_items(
    meta: Dict[str, object],
    include_emb_path: bool = False,
):
    for key, value in meta.items():
        if key.startswith("_"):
            continue
        if key == "emb_path" and not include_emb_path:
            continue
        if not is_scalar_metadata_value(value):
            continue
        yield key, value


def load_loo_neural_layers(model: str, region: str) -> Dict[str, int]:
    path = os.path.join(geom.RESULTS, f"geometry_{model}_{region}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run geometry_paper/run_geometry_analysis.py first."
        )
    with open(path, "r") as f:
        records = json.load(f)
    return {rec["pid"]: int(rec["loo_layer_neu"]) for rec in records}


def finite_row_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr).all(axis=1)
    return mask


def parse_float_list(text: Optional[str]) -> Optional[np.ndarray]:
    if text is None or str(text).strip() == "":
        return None
    return np.asarray(
        [float(part.strip()) for part in str(text).split(",") if part.strip()],
        dtype=float,
    )


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError(f"Expected comma-separated positive integers, got {text!r}.")
    return values


def load_patient_arrays(
    pid: str,
    region: str,
    model: str,
    layer: int,
    neural_space: str,
    llm_pcs: Optional[int],
    random_state: int,
    match_pcs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Load and preprocess LLM and neural arrays for one patient.

    When match_pcs=True, both spaces are reduced to the same dimensionality:
    min(max_llm_pcs, max_neural_pcs), where max_neural_pcs uses ALL available
    components (no variance threshold) so geometry measures are computed in
    matched spaces.
    """
    data = geom.load_patient(pid, geom.PATIENTS[pid], region, model)
    if data is None or data.get("neu_vecs") is None:
        raise ValueError("No geometry-paper data/neural vectors found.")

    n_words = int(data["n_words"])
    row_indices = np.arange(n_words, dtype=int)
    is_speaker = np.asarray(data.get("is_speaker", np.zeros(n_words, dtype=bool)), dtype=bool)[:n_words]
    if len(is_speaker) != n_words:
        raise ValueError(f"is_speaker length {len(is_speaker)} does not match n_words={n_words}.")
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    Z_llm_raw = np.asarray(emb_mm[layer, :n_words], dtype=np.float32)
    del emb_mm

    # ── Neural PCA ──────────────────────────────────────────────────────────
    if neural_space == "pca":
        neu_raw = np.asarray(data["neu_vecs"][:n_words], dtype=np.float32)
        max_neural_pcs = min(neu_raw.shape[1], n_words - 1)
        if max_neural_pcs < 2:
            raise ValueError("Too few neurons/words for neural PCA.")
        neu_scaled = StandardScaler().fit_transform(neu_raw)

        if match_pcs:
            # Use all available neural PCs — threshold applied after matching
            max_llm_pcs = min(int(llm_pcs) if llm_pcs else Z_llm_raw.shape[1],
                              n_words - 1, Z_llm_raw.shape[1])
            n_neural_pcs = min(max_llm_pcs, max_neural_pcs)
        else:
            # Original behaviour: keep PCs explaining 90% variance
            pca_full = PCA(n_components=max_neural_pcs).fit(neu_scaled)
            cum = np.cumsum(pca_full.explained_variance_ratio_)
            n_neural_pcs = min(int(np.searchsorted(cum, 0.90) + 1), max_neural_pcs)

        pca_neu = PCA(n_components=n_neural_pcs, random_state=random_state)
        Z_neural = pca_neu.fit_transform(neu_scaled).astype(np.float32)
        n_neural_features = n_neural_pcs
        neural_feature_label = "n_neural_pcs"
    elif neural_space == "raw":
        Z_neural = np.asarray(data["neu_vecs"][:n_words], dtype=np.float32)
        n_neural_features = int(Z_neural.shape[1])
        neural_feature_label = "n_neurons"
    else:
        raise ValueError(f"Unknown neural_space: {neural_space}")

    # ── LLM PCA ─────────────────────────────────────────────────────────────
    llm_space = "raw"
    llm_pca_explained_variance = np.nan
    Z_llm = Z_llm_raw
    if llm_pcs is not None or match_pcs:
        if match_pcs:
            n_llm_pcs = Z_neural.shape[1]  # matched to neural
        else:
            n_llm_pcs = min(int(llm_pcs), Z_llm_raw.shape[0] - 1, Z_llm_raw.shape[1])
        if n_llm_pcs < 1:
            raise ValueError(f"PC count {n_llm_pcs} invalid for LLM shape {Z_llm_raw.shape}.")
        Z_llm_scaled = StandardScaler().fit_transform(Z_llm_raw)
        pca_llm = PCA(n_components=n_llm_pcs, random_state=random_state)
        Z_llm = pca_llm.fit_transform(Z_llm_scaled).astype(np.float32)
        llm_space = f"pca{n_llm_pcs}"
        llm_pca_explained_variance = float(pca_llm.explained_variance_ratio_.sum())

    mask = finite_row_mask(Z_llm, Z_neural)
    if mask.sum() < 3:
        raise ValueError("Fewer than 3 finite aligned rows.")
    if not mask.all():
        Z_llm = Z_llm[mask]
        Z_neural = Z_neural[mask]
        row_indices = row_indices[mask]
        is_speaker = is_speaker[mask]

    meta = {
        "pid": pid,
        "region": region,
        "model": model,
        "loo_layer_neu": int(layer),
        "llm_space": llm_space,
        "n_llm_features": int(Z_llm.shape[1]),
        "llm_pca_explained_variance": llm_pca_explained_variance,
        "neural_space": neural_space,
        "match_pcs": match_pcs,
        neural_feature_label: int(n_neural_features),
        "n_words_original": n_words,
        "n_words_used": int(mask.sum()),
        "n_speaker1_words_used": int(is_speaker.sum()),
        "n_other_words_used": int((~is_speaker).sum()),
        "emb_path": data["emb_path"],
        "_row_index_used": row_indices,
        "_is_speaker_used": is_speaker,
    }
    return Z_llm, Z_neural, meta


def add_metadata(df: pd.DataFrame, meta: Dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for key, value in reversed(list(metadata_items(meta, include_emb_path=False))):
        df.insert(0, key, value)
    return df


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return float("nan"), float("nan")
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


def curvature_decoding_metrics_from_pointwise(
    Z_source: np.ndarray,
    Z_target: np.ndarray,
    target_mse_t: np.ndarray,
    target_corr_t: np.ndarray,
    direction_name: str,
    target_lag: int,
) -> Dict[str, float]:
    pointwise = make_pointwise_readout_table(
        X_source=Z_source,
        Y_target=Z_target,
        target_mse_t=target_mse_t,
        target_corr_t=target_corr_t,
        direction_name=direction_name,
        target_lag=target_lag,
    )
    return summarize_pointwise_readout_metrics(pointwise)


def word_label_shuffle_null(
    out,
    Z_llm: np.ndarray,
    Z_neural: np.ndarray,
    n_shuffles: int,
    random_state: int,
    directions: Tuple[str, ...],
    target_lag: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    mse_neural = corr_neural = mse_llm = corr_llm = None
    if "LLM_to_neural" in directions:
        _, Z_neural_target, _, _ = align_by_target_lag(Z_llm, Z_neural, target_lag)
        mse_neural, corr_neural = pointwise_decoding_error(Z_neural_target, out.pred_neural)
    if "neural_to_LLM" in directions:
        _, Z_llm_target, _, _ = align_by_target_lag(Z_neural, Z_llm, target_lag)
        mse_llm, corr_llm = pointwise_decoding_error(Z_llm_target, out.pred_llm)

    rows: List[Dict[str, float]] = []
    for shuffle_idx in range(n_shuffles):
        if "LLM_to_neural" in directions:
            perm = rng.permutation(len(mse_neural))
            metrics_l2n = curvature_decoding_metrics_from_pointwise(
                Z_source=Z_llm,
                Z_target=Z_neural,
                target_mse_t=mse_neural[perm],
                target_corr_t=corr_neural[perm],
                direction_name="LLM_to_neural",
                target_lag=target_lag,
            )
            rows.append({"shuffle_index": shuffle_idx, "direction": "LLM_to_neural", **metrics_l2n})

        if "neural_to_LLM" in directions:
            perm = rng.permutation(len(mse_llm))
            metrics_n2l = curvature_decoding_metrics_from_pointwise(
                Z_source=Z_neural,
                Z_target=Z_llm,
                target_mse_t=mse_llm[perm],
                target_corr_t=corr_llm[perm],
                direction_name="neural_to_LLM",
                target_lag=target_lag,
            )
            rows.append({"shuffle_index": shuffle_idx, "direction": "neural_to_LLM", **metrics_n2l})

    return pd.DataFrame(rows)


def trajectory_word_order_shuffle_null(
    out,
    Z_llm: np.ndarray,
    Z_neural: np.ndarray,
    n_shuffles: int,
    random_state: int,
    directions: Tuple[str, ...],
    target_lag: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    mse_neural = corr_neural = mse_llm = corr_llm = None
    if "LLM_to_neural" in directions:
        _, Z_neural_target, _, _ = align_by_target_lag(Z_llm, Z_neural, target_lag)
        mse_neural, corr_neural = pointwise_decoding_error(Z_neural_target, out.pred_neural)
    if "neural_to_LLM" in directions:
        _, Z_llm_target, _, _ = align_by_target_lag(Z_neural, Z_llm, target_lag)
        mse_llm, corr_llm = pointwise_decoding_error(Z_llm_target, out.pred_llm)

    rows: List[Dict[str, float]] = []
    for shuffle_idx in range(n_shuffles):
        trajectory_perm = rng.permutation(Z_llm.shape[0])

        if "LLM_to_neural" in directions:
            error_perm = rng.permutation(len(mse_neural))
            metrics_l2n = curvature_decoding_metrics_from_pointwise(
                Z_source=Z_llm[trajectory_perm],
                Z_target=Z_neural[trajectory_perm],
                target_mse_t=mse_neural[error_perm],
                target_corr_t=corr_neural[error_perm],
                direction_name="LLM_to_neural",
                target_lag=target_lag,
            )
            rows.append({"shuffle_index": shuffle_idx, "direction": "LLM_to_neural", **metrics_l2n})

        if "neural_to_LLM" in directions:
            error_perm = rng.permutation(len(mse_llm))
            metrics_n2l = curvature_decoding_metrics_from_pointwise(
                Z_source=Z_neural[trajectory_perm],
                Z_target=Z_llm[trajectory_perm],
                target_mse_t=mse_llm[error_perm],
                target_corr_t=corr_llm[error_perm],
                direction_name="neural_to_LLM",
                target_lag=target_lag,
            )
            rows.append({"shuffle_index": shuffle_idx, "direction": "neural_to_LLM", **metrics_n2l})

    return pd.DataFrame(rows)


def summarize_word_shuffle_null(
    real_summary: pd.DataFrame,
    null_values: pd.DataFrame,
    meta: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, real_row in real_summary.iterrows():
        direction = real_row["direction"]
        null_dir = null_values[null_values["direction"] == direction]
        for metric, alternative in NULL_METRICS.items():
            real_value = float(real_row[metric])
            null = pd.to_numeric(null_dir[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(null) == 0 or not np.isfinite(real_value):
                null_mean = null_std = z_vs_null = p_directional = p_two_sided = float("nan")
            else:
                null_mean = float(np.mean(null))
                null_std = float(np.std(null, ddof=1)) if len(null) > 1 else float("nan")
                z_vs_null = (
                    float((real_value - null_mean) / null_std)
                    if np.isfinite(null_std) and null_std > 0
                    else float("nan")
                )
                p_ge = float((1 + np.sum(null >= real_value)) / (1 + len(null)))
                p_le = float((1 + np.sum(null <= real_value)) / (1 + len(null)))
                p_directional = p_ge if alternative == "greater" else p_le
                p_two_sided = min(1.0, 2.0 * min(p_ge, p_le))

            row = {
                "direction": direction,
                "metric": metric,
                "alternative": alternative,
                "real_value": real_value,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_vs_null": z_vs_null,
                "empirical_p_directional": p_directional,
                "empirical_p_two_sided": p_two_sided,
                "n_word_shuffles": int(len(null)),
            }
            for key, value in reversed(list(metadata_items(meta, include_emb_path=False))):
                row = {key: value, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def sem(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def pointwise_metric_columns(pointwise: pd.DataFrame) -> List[str]:
    preferred = [
        "target_mse",
        "target_corr",
        "ridge_target_mse",
        "nonlinear_target_mse",
        "mlp_target_mse",
        "ridge_target_corr",
        "nonlinear_target_corr",
        "mlp_target_corr",
        "mse_gain_nonlinear_over_ridge",
        "corr_gain_nonlinear_over_ridge",
        "mse_gain_mlp_over_ridge",
        "corr_gain_mlp_over_ridge",
        "joint_curvature",
        "curvature_mismatch",
        "joint_displacement",
        "displacement_mismatch",
    ]
    return [
        column
        for column in preferred
        if column in pointwise.columns and pd.api.types.is_numeric_dtype(pointwise[column])
    ]


def annotate_pointwise_with_speaker(pointwise: pd.DataFrame, meta: Dict[str, object]) -> pd.DataFrame:
    """Attach source/target Speaker1 flags to a lag-aligned pointwise table."""
    df = pointwise.copy()
    is_speaker = meta.get("_is_speaker_used")
    if is_speaker is None or df.empty:
        return df

    is_speaker = np.asarray(is_speaker, dtype=bool)
    row_index = np.asarray(meta.get("_row_index_used", np.arange(len(is_speaker))), dtype=int)
    for prefix in ("source", "target"):
        index_col = f"{prefix}_time_index"
        if index_col not in df.columns:
            continue
        idx_float = pd.to_numeric(df[index_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(idx_float)
        idx = np.zeros(len(df), dtype=int)
        idx[finite] = idx_float[finite].astype(int)
        valid_idx = finite & (idx >= 0) & (idx < len(is_speaker))

        speaker = pd.Series(pd.NA, index=df.index, dtype="boolean")
        original_word_index = pd.Series(pd.NA, index=df.index, dtype="Int64")
        if valid_idx.any():
            valid_positions = df.index[valid_idx]
            speaker.loc[valid_positions] = is_speaker[idx[valid_idx]]
            original_word_index.loc[valid_positions] = row_index[idx[valid_idx]]
        df[f"{prefix}_is_speaker1"] = speaker
        df[f"{prefix}_original_word_index"] = original_word_index
    return df


def summarize_speaker_split(pointwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize pointwise decoding separately for Speaker1 and all other words."""
    if pointwise.empty or "valid" not in pointwise.columns:
        return pd.DataFrame()

    metrics = pointwise_metric_columns(pointwise)
    if not metrics:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for reference in ("target", "source"):
        speaker_col = f"{reference}_is_speaker1"
        if speaker_col not in pointwise.columns:
            continue

        valid_df = pointwise.loc[
            pointwise["valid"].astype(bool) & pointwise[speaker_col].notna()
        ].copy()
        if valid_df.empty:
            continue

        group_cols = ["direction", "target_lag", speaker_col]
        for group_values, group in valid_df.groupby(group_cols, dropna=True):
            direction, target_lag, is_speaker = group_values
            row: Dict[str, object] = {
                "direction": direction,
                "target_lag": int(target_lag),
                "speaker_reference": reference,
                "speaker_group": "Speaker1" if bool(is_speaker) else "other",
                "is_speaker1": bool(is_speaker),
                "n_points": int(len(group)),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(values.mean())
                row[f"median_{metric}"] = float(values.median())
                row[f"sem_{metric}"] = sem(values)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_speaker_contrasts(split_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute patient-level Speaker1 minus other contrasts from split summaries."""
    if split_summary.empty:
        return pd.DataFrame()

    id_cols = ["direction", "target_lag", "speaker_reference"]
    mean_cols = [column for column in split_summary.columns if column.startswith("mean_")]
    rows: List[Dict[str, object]] = []
    for group_values, group in split_summary.groupby(id_cols, dropna=False):
        speaker = group[group["speaker_group"] == "Speaker1"]
        other = group[group["speaker_group"] == "other"]
        if speaker.empty or other.empty:
            continue
        row = dict(zip(id_cols, group_values))
        row["n_speaker1_points"] = int(speaker["n_points"].iloc[0])
        row["n_other_points"] = int(other["n_points"].iloc[0])
        for mean_col in mean_cols:
            metric = mean_col[len("mean_") :]
            speaker_value = float(speaker[mean_col].iloc[0])
            other_value = float(other[mean_col].iloc[0])
            row[f"speaker1_{metric}"] = speaker_value
            row[f"other_{metric}"] = other_value
            row[f"delta_{metric}_speaker1_minus_other"] = speaker_value - other_value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_speaker_group_stats(speaker_contrasts: pd.DataFrame) -> pd.DataFrame:
    """One-sample group tests on patient-level Speaker1-minus-other contrasts."""
    if speaker_contrasts.empty:
        return pd.DataFrame()

    group_cols = [
        column
        for column in ["model", "region", "direction", "speaker_reference"]
        if column in speaker_contrasts.columns
    ]
    delta_cols = [
        column
        for column in speaker_contrasts.columns
        if column.startswith("delta_") and column.endswith("_speaker1_minus_other")
    ]
    if not group_cols or not delta_cols:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for group_values, group in speaker_contrasts.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            group_values = (group_values,)
        base = dict(zip(group_cols, group_values))
        for delta_col in delta_cols:
            values = pd.to_numeric(group[delta_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) >= 2:
                test = ttest_1samp(values, popmean=0.0, nan_policy="omit")
                t_stat = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                t_stat = p_value = float("nan")
            rows.append(
                {
                    **base,
                    "metric": delta_col[len("delta_") : -len("_speaker1_minus_other")],
                    "n_patients": int(len(values)),
                    "mean_delta_speaker1_minus_other": float(np.mean(values)) if len(values) else float("nan"),
                    "sem_delta_speaker1_minus_other": sem(values),
                    "t_vs_zero": t_stat,
                    "p_vs_zero": p_value,
                }
            )
    return pd.DataFrame(rows)


def flanking_word_shuffle_null(
    out,
    Z_llm: np.ndarray,
    Z_neural: np.ndarray,
    n_shuffles: int,
    random_state: int,
    directions: Tuple[str, ...],
    target_lag: int,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Per-patient flanking-word shuffle null for curvature-binned readout.

    For each direction, lag-aligns the trajectories (matching what was done
    in the real analysis), then calls make_flanks_shuffle_bins to produce a
    null distribution of binned decoding quality under shuffled curvature.
    """
    rng = np.random.default_rng(random_state)
    rows: List[pd.DataFrame] = []

    for direction in directions:
        if direction == "LLM_to_neural":
            Z_src, Z_tgt = Z_llm, Z_neural
        elif direction == "neural_to_LLM":
            Z_src, Z_tgt = Z_neural, Z_llm
        else:
            continue

        X_al, Y_al, _, _ = align_by_target_lag(Z_src, Z_tgt, target_lag=target_lag)
        pw = out.pointwise[out.pointwise["direction"] == direction]
        if pw.empty:
            continue

        null_bins = make_flanks_shuffle_bins(
            pointwise=pw,
            X_aligned=X_al,
            Y_aligned=Y_al,
            n_shuffles=n_shuffles,
            rng=rng,
            n_bins=n_bins,
        )
        rows.append(null_bins)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(
    model: str,
    regions: Iterable[str],
    patients: Optional[Iterable[str]],
    neural_space: str,
    output_dir: str,
    n_bins: int,
    n_splits: int,
    alphas: Optional[np.ndarray],
    llm_pcs: Optional[int],
    shuffle_cv: bool,
    decoder: str,
    directions: Tuple[str, ...],
    elasticnet_l1_ratios: Tuple[float, ...],
    poisson_alpha: float,
    rbf_gamma: Optional[float],
    rbf_n_components: int,
    mlp_hidden_layer_sizes: Tuple[int, ...],
    mlp_alpha: float,
    mlp_max_iter: int,
    mlp_standardize_target: bool,
    target_lag: int,
    n_word_shuffles: int,
    word_shuffle_mode: str,
    n_flanks_shuffles: int,
    match_pcs: bool,
    random_state: int,
    force_layer: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    requested_patients = set(patients) if patients else None

    all_summary: List[pd.DataFrame] = []
    all_bins: List[pd.DataFrame] = []
    all_disp_matched_curv_bins: List[pd.DataFrame] = []
    all_curv_matched_disp_bins: List[pd.DataFrame] = []
    all_pointwise: List[pd.DataFrame] = []
    all_speaker_summary: List[pd.DataFrame] = []
    all_speaker_contrasts: List[pd.DataFrame] = []
    all_null_values: List[pd.DataFrame] = []
    all_null_summary: List[pd.DataFrame] = []
    all_flanks_null_bins: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, object]] = []

    for region in regions:
        loo_layers = load_loo_neural_layers(model, region)
        for pid in geom.PATIENTS:
            if requested_patients is not None and pid not in requested_patients:
                continue
            if region not in geom.PATIENTS[pid] or pid not in loo_layers:
                continue

            layer = force_layer if force_layer is not None else loo_layers[pid]
            print(f"{model} {region} {pid} layer={layer} neural_space={neural_space}", flush=True)
            try:
                Z_llm, Z_neural, meta = load_patient_arrays(
                    pid=pid,
                    region=region,
                    model=model,
                    layer=layer,
                    neural_space=neural_space,
                    llm_pcs=llm_pcs,
                    random_state=random_state,
                    match_pcs=match_pcs,
                )
                print(f"  Z_llm={Z_llm.shape}  Z_neural={Z_neural.shape}", flush=True)
                out = run_bidirectional_curvature_alignment(
                    Z_llm=Z_llm,
                    Z_neural=Z_neural,
                    n_bins=n_bins,
                    alphas=alphas,
                    n_splits=n_splits,
                    shuffle_cv=shuffle_cv,
                    decoder=decoder,
                    directions=directions,
                    elasticnet_l1_ratios=elasticnet_l1_ratios,
                    poisson_alpha=poisson_alpha,
                    rbf_gamma=rbf_gamma,
                    rbf_n_components=rbf_n_components,
                    mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
                    mlp_alpha=mlp_alpha,
                    mlp_max_iter=mlp_max_iter,
                    mlp_standardize_target=mlp_standardize_target,
                    target_lag=target_lag,
                    random_state=random_state,
                )
            except Exception as exc:
                print(f"  skipped: {exc}", flush=True)
                continue

            pointwise = annotate_pointwise_with_speaker(out.pointwise, meta)
            speaker_summary = summarize_speaker_split(pointwise)
            speaker_contrasts = summarize_speaker_contrasts(speaker_summary)

            all_summary.append(add_metadata(out.summary, meta))
            all_bins.append(add_metadata(out.bins, meta))
            all_pointwise.append(add_metadata(pointwise, meta))

            # Displacement-matched curvature bins and curvature-matched displacement bins
            for direction in out.pointwise["direction"].unique():
                pt_dir = out.pointwise[out.pointwise["direction"] == direction]
                dm_curv = make_displacement_matched_curvature_bins(pt_dir, n_bins=n_bins)
                cm_disp = make_curvature_matched_displacement_bins(pt_dir, n_bins=n_bins)
                if not dm_curv.empty:
                    all_disp_matched_curv_bins.append(add_metadata(dm_curv, meta))
                if not cm_disp.empty:
                    all_curv_matched_disp_bins.append(add_metadata(cm_disp, meta))
            if not speaker_summary.empty:
                all_speaker_summary.append(add_metadata(speaker_summary, meta))
            if not speaker_contrasts.empty:
                all_speaker_contrasts.append(add_metadata(speaker_contrasts, meta))
            metadata_rows.append(dict(metadata_items(meta, include_emb_path=True)))

            if n_word_shuffles > 0:
                null_fn = (
                    word_label_shuffle_null
                    if word_shuffle_mode == "label"
                    else trajectory_word_order_shuffle_null
                )
                null_values = null_fn(
                    out=out,
                    Z_llm=Z_llm,
                    Z_neural=Z_neural,
                    n_shuffles=n_word_shuffles,
                    random_state=random_state + len(metadata_rows),
                    directions=tuple(out.summary["direction"].unique()),
                    target_lag=target_lag,
                )
                all_null_values.append(add_metadata(null_values, meta))
                all_null_summary.append(summarize_word_shuffle_null(out.summary, null_values, meta))

            if n_flanks_shuffles > 0:
                flanks_null = flanking_word_shuffle_null(
                    out=out,
                    Z_llm=Z_llm,
                    Z_neural=Z_neural,
                    n_shuffles=n_flanks_shuffles,
                    random_state=random_state + len(metadata_rows) + 9999,
                    directions=tuple(out.summary["direction"].unique()),
                    target_lag=target_lag,
                    n_bins=n_bins,
                )
                if not flanks_null.empty:
                    all_flanks_null_bins.append(add_metadata(flanks_null, meta))

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    bins = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    pointwise = pd.concat(all_pointwise, ignore_index=True) if all_pointwise else pd.DataFrame()
    speaker_summary = (
        pd.concat(all_speaker_summary, ignore_index=True)
        if all_speaker_summary
        else pd.DataFrame()
    )
    speaker_contrasts = (
        pd.concat(all_speaker_contrasts, ignore_index=True)
        if all_speaker_contrasts
        else pd.DataFrame()
    )
    speaker_group_stats = summarize_speaker_group_stats(speaker_contrasts)
    null_values = pd.concat(all_null_values, ignore_index=True) if all_null_values else pd.DataFrame()
    null_summary = pd.concat(all_null_summary, ignore_index=True) if all_null_summary else pd.DataFrame()
    flanks_null_bins = pd.concat(all_flanks_null_bins, ignore_index=True) if all_flanks_null_bins else pd.DataFrame()
    disp_matched_curv_bins = pd.concat(all_disp_matched_curv_bins, ignore_index=True) if all_disp_matched_curv_bins else pd.DataFrame()
    curv_matched_disp_bins = pd.concat(all_curv_matched_disp_bins, ignore_index=True) if all_curv_matched_disp_bins else pd.DataFrame()
    metadata = pd.DataFrame(metadata_rows)

    region_tag = "-".join(regions)
    patient_tag = "selected" if requested_patients is not None else "all-patients"
    llm_tag = f"llmpc{llm_pcs}" if llm_pcs is not None else "llmraw"
    cv_tag = "shufflecv" if shuffle_cv else "temporalfold"
    lag_tag = f"lag{target_lag:+d}".replace("+", "plus").replace("-", "minus")
    direction_tag = (
        "both"
        if set(directions) == {"LLM_to_neural", "neural_to_LLM"}
        else "-".join(directions)
    )
    pc_tag = "matchedpcs" if match_pcs else ""
    layer_tag = f"layer{force_layer}" if force_layer is not None else ""
    stem = f"curvature_alignment_{model}_{region_tag}_{patient_tag}_{neural_space}_{llm_tag}_{cv_tag}_{decoder}_{direction_tag}_{lag_tag}"
    if pc_tag:
        stem = f"{stem}_{pc_tag}"
    if layer_tag:
        stem = f"{stem}_{layer_tag}"
    if n_word_shuffles > 0:
        stem = f"{stem}_{word_shuffle_mode}wordshuffle{n_word_shuffles}"

    flanks_null_path = os.path.join(output_dir, f"{stem}_flanksshuffle{n_flanks_shuffles}_bins.csv") if n_flanks_shuffles > 0 else None
    disp_matched_curv_bins_path = os.path.join(output_dir, f"{stem}_disp_matched_curv_bins.csv")
    curv_matched_disp_bins_path = os.path.join(output_dir, f"{stem}_curv_matched_disp_bins.csv")
    summary_path = os.path.join(output_dir, f"{stem}_summary.csv")
    bins_path = os.path.join(output_dir, f"{stem}_bins.csv")
    pointwise_path = os.path.join(output_dir, f"{stem}_pointwise.csv")
    speaker_summary_path = os.path.join(output_dir, f"{stem}_speaker_split_summary.csv")
    speaker_contrast_path = os.path.join(output_dir, f"{stem}_speaker_split_contrasts.csv")
    speaker_group_stats_path = os.path.join(output_dir, f"{stem}_speaker_split_group_stats.csv")
    metadata_path = os.path.join(output_dir, f"{stem}_metadata.csv")
    null_values_path = os.path.join(output_dir, f"{stem}_word_shuffle_null_values.csv")
    null_summary_path = os.path.join(output_dir, f"{stem}_word_shuffle_null_summary.csv")

    summary.to_csv(summary_path, index=False)
    bins.to_csv(bins_path, index=False)
    pointwise.to_csv(pointwise_path, index=False)
    if not speaker_summary.empty:
        speaker_summary.to_csv(speaker_summary_path, index=False)
    if not speaker_contrasts.empty:
        speaker_contrasts.to_csv(speaker_contrast_path, index=False)
    if not speaker_group_stats.empty:
        speaker_group_stats.to_csv(speaker_group_stats_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    if not null_values.empty:
        null_values.to_csv(null_values_path, index=False)
        null_summary.to_csv(null_summary_path, index=False)
    if not flanks_null_bins.empty and flanks_null_path is not None:
        flanks_null_bins.to_csv(flanks_null_path, index=False)
    if not disp_matched_curv_bins.empty:
        disp_matched_curv_bins.to_csv(disp_matched_curv_bins_path, index=False)
    if not curv_matched_disp_bins.empty:
        curv_matched_disp_bins.to_csv(curv_matched_disp_bins_path, index=False)

    print(f"\nSaved summary:  {summary_path}")
    print(f"Saved bins:     {bins_path}")
    print(f"Saved pointwise:{pointwise_path}")
    if not speaker_summary.empty:
        print(f"Saved speaker split summary:   {speaker_summary_path}")
    if not speaker_contrasts.empty:
        print(f"Saved speaker split contrasts: {speaker_contrast_path}")
    if not speaker_group_stats.empty:
        print(f"Saved speaker split group stats: {speaker_group_stats_path}")
    print(f"Saved metadata: {metadata_path}")
    if not null_values.empty:
        print(f"Saved null values:  {null_values_path}")
        print(f"Saved null summary: {null_summary_path}")
    if not flanks_null_bins.empty and flanks_null_path is not None:
        print(f"Saved flanks-shuffle null bins: {flanks_null_path}")
    return summary, bins, pointwise, null_summary, speaker_summary, speaker_contrasts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run curvature_alignment.py across geometry-paper patients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama-3.1-8b", choices=list(geom.MODELS))
    parser.add_argument("--regions", nargs="+", default=geom.REGIONS, choices=geom.REGIONS)
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--neural-space", choices=["pca", "raw"], default="pca")
    parser.add_argument(
        "--llm-pcs",
        type=int,
        default=None,
        help="Optionally PCA-reduce LLM embeddings before alignment/decoding.",
    )
    parser.add_argument(
        "--decoder",
        choices=["ridge", "poly2_ridge", "rbf_ridge", "elasticnet", "poisson", "mlp"],
        default="ridge",
        help="Decoder family. Ridge is L2; poly2_ridge is quadratic ridge; rbf_ridge is RBF random-features Ridge; Poisson requires nonnegative targets.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["LLM_to_neural", "neural_to_LLM"],
        default=["LLM_to_neural", "neural_to_LLM"],
    )
    parser.add_argument(
        "--alphas",
        default=None,
        help="Optional comma-separated regularization grid for ridge/elasticnet.",
    )
    parser.add_argument(
        "--elasticnet-l1-ratios",
        default="0.05,0.1,0.5",
        help="Comma-separated L1 ratio grid for Elastic Net.",
    )
    parser.add_argument("--poisson-alpha", type=float, default=1.0)
    parser.add_argument(
        "--rbf-gamma",
        type=float,
        default=None,
        help="RBF gamma for --decoder rbf_ridge. Defaults to 1 / source feature count.",
    )
    parser.add_argument("--rbf-components", type=int, default=256)
    parser.add_argument(
        "--mlp-hidden",
        default="128",
        help="Comma-separated hidden-layer widths for --decoder mlp.",
    )
    parser.add_argument("--mlp-alpha", type=float, default=1e-3)
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument(
        "--mlp-standardize-target",
        action="store_true",
        help="Z-score targets within each training fold for MLPRegressor.",
    )
    parser.add_argument(
        "--target-lag",
        type=int,
        default=0,
        help="Target word offset. Use 1 for source word t predicting target word t+1.",
    )
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--shuffle-cv",
        action="store_true",
        help="Shuffle cross-validation folds. Useful when temporal block drift makes R2 overly pessimistic.",
    )
    parser.add_argument(
        "--n-word-shuffles",
        type=int,
        default=0,
        help="Run a within-patient word-order shuffle null with this many permutations.",
    )
    parser.add_argument(
        "--word-shuffle-mode",
        choices=["label", "trajectory"],
        default="label",
        help=(
            "'label' keeps the real trajectory curvature fixed and shuffles word-level decoding errors; "
            "'trajectory' shuffles the full word trajectory before recomputing curvature."
        ),
    )
    parser.add_argument(
        "--match-pcs",
        action="store_true",
        help=(
            "Match LLM and neural PCA dimensionality to min(max_llm_pcs, max_neural_pcs) "
            "per patient. Neural PCA uses ALL available components (no variance threshold) "
            "so both spaces are compared in the same number of dimensions."
        ),
    )
    parser.add_argument(
        "--n-flanks-shuffles",
        type=int,
        default=0,
        help=(
            "Run a flanking-word shuffle null with this many permutations. "
            "For each shuffle, the center word is kept fixed but its flanking embeddings "
            "are replaced with randomly drawn embeddings from elsewhere in the trajectory, "
            "breaking sequential structure while preserving marginal embedding distributions."
        ),
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--force-layer",
        type=int,
        default=None,
        help=(
            "Override the LOO-optimal layer for all patients with this fixed layer index. "
            "Useful for layer-sweep analyses."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(THIS_DIR, "results", "geometry_paper_curvature_alignment"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, bins, _, null_summary, speaker_summary, speaker_contrasts = run(
        model=args.model,
        regions=args.regions,
        patients=args.patients,
        neural_space=args.neural_space,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        n_splits=args.n_splits,
        alphas=parse_float_list(args.alphas),
        llm_pcs=args.llm_pcs,
        shuffle_cv=args.shuffle_cv,
        decoder=args.decoder,
        directions=tuple(args.directions),
        elasticnet_l1_ratios=tuple(parse_float_list(args.elasticnet_l1_ratios).tolist()),
        poisson_alpha=args.poisson_alpha,
        rbf_gamma=args.rbf_gamma,
        rbf_n_components=args.rbf_components,
        mlp_hidden_layer_sizes=parse_int_tuple(args.mlp_hidden),
        mlp_alpha=args.mlp_alpha,
        mlp_max_iter=args.mlp_max_iter,
        mlp_standardize_target=args.mlp_standardize_target,
        target_lag=args.target_lag,
        n_word_shuffles=args.n_word_shuffles,
        word_shuffle_mode=args.word_shuffle_mode,
        n_flanks_shuffles=args.n_flanks_shuffles,
        match_pcs=args.match_pcs,
        random_state=args.random_state,
        force_layer=args.force_layer,
    )
    if not summary.empty:
        print("\nMain stats:")
        print(summary.to_string(index=False))
    if not bins.empty:
        print("\nCurvature bins:")
        print(bins.head(20).to_string(index=False))
    if not null_summary.empty:
        print("\nWord-order shuffle null:")
        print(null_summary.to_string(index=False))
    if not speaker_summary.empty:
        print("\nSpeaker split summary:")
        print(speaker_summary.head(20).to_string(index=False))
    if not speaker_contrasts.empty:
        print("\nSpeaker1-minus-other contrasts:")
        print(speaker_contrasts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
