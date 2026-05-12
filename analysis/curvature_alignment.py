"""Curvature-conditioned brain/LLM alignment analyses.

This module tests the idea that local trajectory curvature predicts when a
linear map between representational spaces works well.

Expected inputs
---------------
Z_llm : array, shape [T, D_model]
    Word-aligned LLM hidden states.
Z_neural : array, shape [T, N_neurons]
    Word-aligned neural population activity.

Core hypothesis
---------------
Higher local curvature, or greater LLM-neural curvature mismatch, should be
associated with worse linear cross-space decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as student_t
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import MultiTaskElasticNetCV, PoissonRegressor, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class CurvatureAlignmentOutput:
    """Container for bidirectional curvature-alignment results."""

    summary: pd.DataFrame
    bins: pd.DataFrame
    pointwise: pd.DataFrame
    pred_neural: np.ndarray
    pred_llm: np.ndarray
    curvature_llm: np.ndarray
    curvature_neural: np.ndarray


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan
    return spearmanr(x[mask], y[mask])


def local_curvature(Z: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local discrete curvature from a trajectory.

    Curvature is the turning angle (degrees) between successive displacement
    vectors — identical to the definition in run_geometry_analysis.py:

        kappa_t = arccos(cos(z_{t+1} - z_t, z_{t+2} - z_{t+1}))   [degrees]

    Previously used 1 - cos(...), which is monotonically related but on a
    different scale (0–2 vs 0–180°). Unified to degrees so all analyses in
    the paper use the same metric.

    Parameters
    ----------
    Z
        Array of shape [T, D].
    eps
        Small value to avoid division by zero.

    Returns
    -------
    curvature
        Array of shape [T - 2]. Turning angle in degrees; higher = sharper bending.
    straightness
        Array of shape [T - 2]. Cosine similarity between successive steps.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D [T, D], got shape {Z.shape}")
    if Z.shape[0] < 3:
        raise ValueError("Need at least 3 timepoints to compute curvature.")

    v1 = Z[1:-1] - Z[:-2]
    v2 = Z[2:] - Z[1:-1]

    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + eps
    straightness = np.sum(v1 * v2, axis=1) / denom
    straightness = np.clip(straightness, -1.0, 1.0)
    curvature = np.degrees(np.arccos(straightness))
    return curvature, straightness


def local_curvature_shuffled_flanks(
    Z: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Flanking-word shuffle null for local curvature.

    For each center position t, keeps Z[t] fixed but replaces Z[t-1] and
    Z[t+1] with two randomly drawn embeddings from positions outside
    {t-1, t, t+1}.  This breaks temporal adjacency while preserving the
    marginal distribution of individual word embeddings — the appropriate
    null for the claim that flat moments reflect sequential trajectory
    structure rather than word-type properties.

    Returns curvature in degrees, shape [T - 2].
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = np.asarray(Z, dtype=float)
    T = Z.shape[0]
    if T < 3:
        raise ValueError("Need at least 3 timepoints.")

    n_centers = T - 2
    centers = np.arange(1, T - 1)   # center positions in Z

    left_idx = np.empty(n_centers, dtype=int)
    right_idx = np.empty(n_centers, dtype=int)
    remaining = np.ones(n_centers, dtype=bool)

    while remaining.any():
        ri = remaining.nonzero()[0]
        t_ri = centers[ri]
        cand_l = rng.integers(0, T, size=len(ri))
        cand_r = rng.integers(0, T, size=len(ri))
        bad = (
            (cand_l == t_ri - 1) | (cand_l == t_ri) | (cand_l == t_ri + 1) |
            (cand_r == t_ri - 1) | (cand_r == t_ri) | (cand_r == t_ri + 1) |
            (cand_l == cand_r)
        )
        good = ri[~bad]
        left_idx[good] = cand_l[~bad]
        right_idx[good] = cand_r[~bad]
        remaining[good] = False

    v1 = Z[centers] - Z[left_idx]
    v2 = Z[right_idx] - Z[centers]
    eps = 1e-8
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + eps
    cos_a = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def local_displacement(Z: np.ndarray) -> np.ndarray:
    """Compute centered local step size from a trajectory.

    For the same center points used by local_curvature, this returns the mean
    norm of the incoming and outgoing displacement vectors.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D [T, D], got shape {Z.shape}")
    if Z.shape[0] < 3:
        raise ValueError("Need at least 3 timepoints to compute local displacement.")

    step_norms = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    return 0.5 * (step_norms[:-1] + step_norms[1:])


def align_by_target_lag(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    target_lag: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align source and target arrays so source[t] predicts target[t + lag]."""
    X_source = np.asarray(X_source, dtype=float)
    Y_target = np.asarray(Y_target, dtype=float)
    if X_source.ndim != 2 or Y_target.ndim != 2:
        raise ValueError("X_source and Y_target must both be 2D arrays.")
    if X_source.shape[0] != Y_target.shape[0]:
        raise ValueError(
            f"X_source and Y_target must share time dimension. "
            f"Got {X_source.shape[0]} and {Y_target.shape[0]}"
        )

    target_lag = int(target_lag)
    n_time = X_source.shape[0]
    if abs(target_lag) >= n_time:
        raise ValueError(f"target_lag={target_lag} leaves no aligned samples.")

    if target_lag > 0:
        source_idx = np.arange(0, n_time - target_lag)
        target_idx = np.arange(target_lag, n_time)
    elif target_lag < 0:
        source_idx = np.arange(-target_lag, n_time)
        target_idx = np.arange(0, n_time + target_lag)
    else:
        source_idx = np.arange(n_time)
        target_idx = np.arange(n_time)

    if len(source_idx) < 3:
        raise ValueError(
            f"target_lag={target_lag} leaves fewer than 3 aligned samples."
        )
    return X_source[source_idx], Y_target[target_idx], source_idx, target_idx


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _safe_partial_pearson(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    n_control = controls.shape[1]
    if mask.sum() <= n_control + 2:
        return np.nan, np.nan

    x_resid = _residualize(x[mask], controls[mask])
    y_resid = _residualize(y[mask], controls[mask])
    if np.nanstd(x_resid) == 0 or np.nanstd(y_resid) == 0:
        return np.nan, np.nan

    r, _ = pearsonr(x_resid, y_resid)
    dof = mask.sum() - n_control - 2
    if dof <= 0 or not np.isfinite(r):
        return float(r), np.nan
    if abs(r) >= 1:
        return float(r), 0.0
    t_stat = r * np.sqrt(dof / max(1e-12, 1.0 - r**2))
    p = 2.0 * student_t.sf(abs(t_stat), dof)
    return float(r), float(p)


def summarize_pointwise_readout_metrics(pointwise: pd.DataFrame) -> Dict[str, float]:
    """Summarize curvature/displacement relationships to decoding quality."""
    joint_curv = pointwise["joint_curvature"].to_numpy(dtype=float)
    curv_mismatch = pointwise["curvature_mismatch"].to_numpy(dtype=float)
    source_disp = pointwise["displacement_source"].to_numpy(dtype=float)
    target_disp = pointwise["displacement_target"].to_numpy(dtype=float)
    joint_disp = pointwise["joint_displacement"].to_numpy(dtype=float)
    disp_mismatch = pointwise["displacement_mismatch"].to_numpy(dtype=float)
    mse_curv = pointwise["target_mse"].to_numpy(dtype=float)
    corr_curv = pointwise["target_corr"].to_numpy(dtype=float)

    valid = (
        np.isfinite(joint_curv)
        & np.isfinite(curv_mismatch)
        & np.isfinite(joint_disp)
        & np.isfinite(disp_mismatch)
        & np.isfinite(mse_curv)
        & np.isfinite(corr_curv)
    )

    r_curv_mse, p_curv_mse = _safe_pearson(joint_curv, mse_curv)
    sr_curv_mse, sp_curv_mse = _safe_spearman(joint_curv, mse_curv)
    r_curv_corr, p_curv_corr = _safe_pearson(joint_curv, corr_curv)
    r_mismatch_mse, p_mismatch_mse = _safe_pearson(curv_mismatch, mse_curv)

    r_disp_mse, p_disp_mse = _safe_pearson(joint_disp, mse_curv)
    sr_disp_mse, sp_disp_mse = _safe_spearman(joint_disp, mse_curv)
    r_disp_corr, p_disp_corr = _safe_pearson(joint_disp, corr_curv)
    r_disp_mismatch_mse, p_disp_mismatch_mse = _safe_pearson(disp_mismatch, mse_curv)

    pr_curv_mse_disp, pp_curv_mse_disp = _safe_partial_pearson(
        joint_curv, mse_curv, joint_disp
    )
    pr_curv_corr_disp, pp_curv_corr_disp = _safe_partial_pearson(
        joint_curv, corr_curv, joint_disp
    )
    separate_displacements = np.column_stack([source_disp, target_disp])
    pr_curv_mse_separate_disp, pp_curv_mse_separate_disp = _safe_partial_pearson(
        joint_curv, mse_curv, separate_displacements
    )
    pr_curv_corr_separate_disp, pp_curv_corr_separate_disp = _safe_partial_pearson(
        joint_curv, corr_curv, separate_displacements
    )
    pr_disp_mse_curv, pp_disp_mse_curv = _safe_partial_pearson(
        joint_disp, mse_curv, joint_curv
    )

    return {
        "n_valid_curvature_points": int(valid.sum()),
        "pearson_joint_curvature_vs_mse_r": r_curv_mse,
        "pearson_joint_curvature_vs_mse_p": p_curv_mse,
        "spearman_joint_curvature_vs_mse_r": sr_curv_mse,
        "spearman_joint_curvature_vs_mse_p": sp_curv_mse,
        "pearson_joint_curvature_vs_decoding_corr_r": r_curv_corr,
        "pearson_joint_curvature_vs_decoding_corr_p": p_curv_corr,
        "pearson_curvature_mismatch_vs_mse_r": r_mismatch_mse,
        "pearson_curvature_mismatch_vs_mse_p": p_mismatch_mse,
        "pearson_joint_displacement_vs_mse_r": r_disp_mse,
        "pearson_joint_displacement_vs_mse_p": p_disp_mse,
        "spearman_joint_displacement_vs_mse_r": sr_disp_mse,
        "spearman_joint_displacement_vs_mse_p": sp_disp_mse,
        "pearson_joint_displacement_vs_decoding_corr_r": r_disp_corr,
        "pearson_joint_displacement_vs_decoding_corr_p": p_disp_corr,
        "pearson_displacement_mismatch_vs_mse_r": r_disp_mismatch_mse,
        "pearson_displacement_mismatch_vs_mse_p": p_disp_mismatch_mse,
        "partial_joint_curvature_vs_mse_control_displacement_r": pr_curv_mse_disp,
        "partial_joint_curvature_vs_mse_control_displacement_p": pp_curv_mse_disp,
        "partial_joint_curvature_vs_decoding_corr_control_displacement_r": pr_curv_corr_disp,
        "partial_joint_curvature_vs_decoding_corr_control_displacement_p": pp_curv_corr_disp,
        "partial_joint_curvature_vs_mse_control_source_target_displacement_r": pr_curv_mse_separate_disp,
        "partial_joint_curvature_vs_mse_control_source_target_displacement_p": pp_curv_mse_separate_disp,
        "partial_joint_curvature_vs_decoding_corr_control_source_target_displacement_r": pr_curv_corr_separate_disp,
        "partial_joint_curvature_vs_decoding_corr_control_source_target_displacement_p": pp_curv_corr_separate_disp,
        "partial_joint_displacement_vs_mse_control_curvature_r": pr_disp_mse_curv,
        "partial_joint_displacement_vs_mse_control_curvature_p": pp_disp_mse_curv,
    }


def make_pointwise_readout_table(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    target_mse_t: np.ndarray,
    target_corr_t: np.ndarray,
    direction_name: str,
    target_lag: int = 0,
) -> pd.DataFrame:
    """Build the per-word table used for curvature/displacement readout tests."""
    X_aligned, Y_aligned, source_idx, target_idx = align_by_target_lag(
        X_source,
        Y_target,
        target_lag=target_lag,
    )
    target_mse_t = np.asarray(target_mse_t, dtype=float)
    target_corr_t = np.asarray(target_corr_t, dtype=float)
    if len(target_mse_t) != X_aligned.shape[0] or len(target_corr_t) != X_aligned.shape[0]:
        raise ValueError(
            "target_mse_t and target_corr_t must match the lag-aligned sample count. "
            f"Got {len(target_mse_t)}, {len(target_corr_t)}, and {X_aligned.shape[0]}."
        )

    curvature_source, _ = local_curvature(X_aligned)
    curvature_target, _ = local_curvature(Y_aligned)
    displacement_source = local_displacement(X_aligned)
    displacement_target = local_displacement(Y_aligned)

    joint_curv = 0.5 * (curvature_source + curvature_target)
    curv_mismatch = np.abs(curvature_source - curvature_target)
    joint_displacement = 0.5 * (displacement_source + displacement_target)
    displacement_mismatch = np.abs(displacement_source - displacement_target)

    center_idx = np.arange(1, X_aligned.shape[0] - 1)
    mse_curv = target_mse_t[1:-1]
    corr_curv = target_corr_t[1:-1]

    valid = (
        np.isfinite(joint_curv)
        & np.isfinite(curv_mismatch)
        & np.isfinite(joint_displacement)
        & np.isfinite(displacement_mismatch)
        & np.isfinite(mse_curv)
        & np.isfinite(corr_curv)
    )

    return pd.DataFrame(
        {
            "direction": direction_name,
            "target_lag": int(target_lag),
            "time_index": center_idx,
            "source_time_index": source_idx[1:-1],
            "target_time_index": target_idx[1:-1],
            "curvature_source": curvature_source,
            "curvature_target": curvature_target,
            "joint_curvature": joint_curv,
            "curvature_mismatch": curv_mismatch,
            "displacement_source": displacement_source,
            "displacement_target": displacement_target,
            "joint_displacement": joint_displacement,
            "displacement_mismatch": displacement_mismatch,
            "target_mse": mse_curv,
            "target_corr": corr_curv,
            "valid": valid,
        }
    )


def fit_linear_cross_space_decoder(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    n_splits: int = 5,
    standardize: bool = True,
    shuffle_cv: bool = False,
    random_state: int = 0,
    decoder: str = "ridge",
    elasticnet_l1_ratios: Tuple[float, ...] = (0.05, 0.1, 0.5),
    poisson_alpha: float = 1.0,
    rbf_gamma: Optional[float] = None,
    rbf_n_components: int = 256,
    mlp_hidden_layer_sizes: Tuple[int, ...] = (128,),
    mlp_alpha: float = 1e-3,
    mlp_max_iter: int = 500,
    mlp_standardize_target: bool = False,
) -> np.ndarray:
    """Cross-validated linear decoder from source space X to target space Y.

    Parameters
    ----------
    X
        Source representations, shape [T, D_source].
    Y
        Target representations, shape [T, D_target].
    alphas
        Ridge regularization values for inner RidgeCV.
    n_splits
        Number of folds. Uses non-shuffled folds by default to respect temporal
        structure.
    standardize
        Whether to z-score source features within each training fold.
    shuffle_cv
        Whether to shuffle folds. Default False.
    random_state
        Used only when shuffle_cv=True.
    decoder
        Decoder family: "ridge" (L2), "poly2_ridge", "rbf_ridge",
        "elasticnet", "poisson", or "mlp". Poisson requires nonnegative
        target values.
    elasticnet_l1_ratios
        L1/L2 mixing values for MultiTaskElasticNetCV.
    poisson_alpha
        L2 penalty strength for PoissonRegressor.
    rbf_gamma
        RBF kernel width for rbf_ridge. If None, uses 1 / n_source_features.
    rbf_n_components
        Number of random Fourier features for rbf_ridge.
    mlp_hidden_layer_sizes
        Hidden-layer widths for MLPRegressor.
    mlp_alpha
        L2 penalty strength for MLPRegressor.
    mlp_max_iter
        Maximum MLP training iterations within each outer fold.
    mlp_standardize_target
        Whether to z-score targets inside each training fold for MLPRegressor.

    Returns
    -------
    Y_pred
        Cross-validated predictions, shape [T, D_target].
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must share time dimension. Got {X.shape[0]} and {Y.shape[0]}")

    decoder = decoder.lower()
    if decoder not in {"ridge", "poly2_ridge", "rbf_ridge", "elasticnet", "poisson", "mlp"}:
        raise ValueError("decoder must be one of: ridge, poly2_ridge, rbf_ridge, elasticnet, poisson, mlp")
    if decoder == "poisson" and np.nanmin(Y) < 0:
        raise ValueError("Poisson decoder requires nonnegative target values.")

    if alphas is None and decoder in {"ridge", "poly2_ridge", "rbf_ridge"}:
        alphas = np.logspace(-3, 6, 20)
    elif alphas is None and decoder == "elasticnet":
        alphas = np.logspace(-4, 1, 8)

    n_splits = min(n_splits, X.shape[0])
    if n_splits < 2:
        raise ValueError("Need at least 2 samples/folds for cross-validated decoding.")

    kf = KFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state if shuffle_cv else None)
    Y_pred = np.zeros_like(Y, dtype=float)

    for train_idx, test_idx in kf.split(X):
        if decoder == "ridge":
            estimator = RidgeCV(alphas=alphas)
        elif decoder == "poly2_ridge":
            steps = []
            if standardize:
                steps.append(StandardScaler())
            steps.extend(
                [
                    PolynomialFeatures(degree=2, include_bias=False),
                    StandardScaler(),
                    RidgeCV(alphas=alphas),
                ]
            )
            model = make_pipeline(*steps)
            model.fit(X[train_idx], Y[train_idx])
            Y_pred[test_idx] = model.predict(X[test_idx])
            continue
        elif decoder == "rbf_ridge":
            gamma = (1.0 / X.shape[1]) if rbf_gamma is None else float(rbf_gamma)
            steps = []
            if standardize:
                steps.append(StandardScaler())
            steps.extend(
                [
                    RBFSampler(
                        gamma=gamma,
                        n_components=int(rbf_n_components),
                        random_state=random_state,
                    ),
                    StandardScaler(),
                    RidgeCV(alphas=alphas),
                ]
            )
            model = make_pipeline(*steps)
            model.fit(X[train_idx], Y[train_idx])
            Y_pred[test_idx] = model.predict(X[test_idx])
            continue
        elif decoder == "elasticnet":
            estimator = MultiTaskElasticNetCV(
                alphas=alphas,
                l1_ratio=list(elasticnet_l1_ratios),
                cv=min(3, max(2, len(train_idx))),
                max_iter=5000,
                random_state=random_state,
            )
        elif decoder == "poisson":
            estimator = MultiOutputRegressor(
                PoissonRegressor(alpha=poisson_alpha, max_iter=1000)
            )
        else:
            estimator = MLPRegressor(
                hidden_layer_sizes=tuple(mlp_hidden_layer_sizes),
                activation="relu",
                solver="adam",
                alpha=mlp_alpha,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=random_state,
            )

        model = make_pipeline(StandardScaler(), estimator) if standardize else estimator
        if decoder == "mlp" and mlp_standardize_target:
            model = TransformedTargetRegressor(
                regressor=model,
                transformer=StandardScaler(),
            )
        model.fit(X[train_idx], Y[train_idx])
        Y_pred[test_idx] = model.predict(X[test_idx])

    return Y_pred


def pointwise_decoding_error(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pointwise MSE and target-dimension correlation per timepoint."""
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Y_true and Y_pred must match. Got {Y_true.shape} and {Y_pred.shape}")

    mse_t = np.mean((Y_true - Y_pred) ** 2, axis=1)
    corr_t = np.full(Y_true.shape[0], np.nan, dtype=float)
    for t in range(Y_true.shape[0]):
        corr_t[t], _ = _safe_pearson(Y_true[t], Y_pred[t])
    return mse_t, corr_t


def curvature_conditioned_decoding(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    n_bins: int = 5,
    direction_name: str = "source_to_target",
    alphas: Optional[np.ndarray] = None,
    n_splits: int = 5,
    standardize: bool = True,
    shuffle_cv: bool = False,
    random_state: int = 0,
    decoder: str = "ridge",
    elasticnet_l1_ratios: Tuple[float, ...] = (0.05, 0.1, 0.5),
    poisson_alpha: float = 1.0,
    rbf_gamma: Optional[float] = None,
    rbf_n_components: int = 256,
    mlp_hidden_layer_sizes: Tuple[int, ...] = (128,),
    mlp_alpha: float = 1e-3,
    mlp_max_iter: int = 500,
    mlp_standardize_target: bool = False,
    target_lag: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Test whether linear decoding performance depends on local curvature."""
    X_source = np.asarray(X_source, dtype=float)
    Y_target = np.asarray(Y_target, dtype=float)
    X_aligned, Y_aligned, _, _ = align_by_target_lag(
        X_source,
        Y_target,
        target_lag=target_lag,
    )

    Y_pred = fit_linear_cross_space_decoder(
        X_aligned,
        Y_aligned,
        alphas=alphas,
        n_splits=n_splits,
        standardize=standardize,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
        decoder=decoder,
        elasticnet_l1_ratios=elasticnet_l1_ratios,
        poisson_alpha=poisson_alpha,
        rbf_gamma=rbf_gamma,
        rbf_n_components=rbf_n_components,
        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        mlp_alpha=mlp_alpha,
        mlp_max_iter=mlp_max_iter,
        mlp_standardize_target=mlp_standardize_target,
    )

    mse_t, corr_t = pointwise_decoding_error(Y_aligned, Y_pred)
    pointwise = make_pointwise_readout_table(
        X_source=X_source,
        Y_target=Y_target,
        target_mse_t=mse_t,
        target_corr_t=corr_t,
        direction_name=direction_name,
        target_lag=target_lag,
    )
    metrics = summarize_pointwise_readout_metrics(pointwise)

    summary = pd.DataFrame([
        {
            "direction": direction_name,
            "decoder": decoder,
            "target_lag": int(target_lag),
            "n_timepoints": int(X_aligned.shape[0]),
            "global_r2": float(r2_score(Y_aligned, Y_pred, multioutput="variance_weighted")),
            **metrics,
        }
    ])

    bins = make_curvature_bins(pointwise, n_bins=n_bins)
    return summary, bins, pointwise, Y_pred


def make_curvature_bins(pointwise: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Summarize decoding performance across quantile bins of joint curvature."""
    rows: List[Dict[str, float]] = []
    direction = pointwise["direction"].iloc[0]
    valid_df = pointwise.loc[pointwise["valid"]].copy()
    if valid_df.empty:
        return pd.DataFrame()

    # qcut can fail with duplicate edges; duplicates='drop' handles flat curvature.
    valid_df["curvature_bin"] = pd.qcut(
        valid_df["joint_curvature"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    for b, df_b in valid_df.groupby("curvature_bin", dropna=True):
        rows.append(
            {
                "direction": direction,
                "bin": int(b),
                "n": int(len(df_b)),
                "mean_joint_curvature": float(df_b["joint_curvature"].mean()),
                "mean_curvature_mismatch": float(df_b["curvature_mismatch"].mean()),
                "mean_joint_displacement": float(df_b["joint_displacement"].mean()),
                "mean_displacement_mismatch": float(df_b["displacement_mismatch"].mean()),
                "mean_mse": float(df_b["target_mse"].mean()),
                "mean_decoding_corr": float(df_b["target_corr"].mean()),
                "sem_mse": float(df_b["target_mse"].sem()),
                "sem_decoding_corr": float(df_b["target_corr"].sem()),
            }
        )
    return pd.DataFrame(rows)


def make_displacement_matched_curvature_bins(
    pointwise: pd.DataFrame,
    n_bins: int = 5,
    n_disp_strata: int = 5,
) -> pd.DataFrame:
    """Curvature-binned readout with displacement matched across curvature bins.

    Within each displacement stratum, words are re-binned by curvature quintile
    and mean readout is computed.  Results are then averaged across strata, so
    each curvature bin contains words drawn from the same displacement
    distribution.  This tests whether curvature predicts readout independently
    of step size.
    """
    valid_df = pointwise[pointwise["valid"]].copy()
    if valid_df.empty:
        return pd.DataFrame()
    direction = valid_df["direction"].iloc[0]

    try:
        valid_df["disp_stratum"] = pd.qcut(
            valid_df["joint_displacement"], q=n_disp_strata,
            labels=False, duplicates="drop"
        )
    except Exception:
        return pd.DataFrame()

    stratum_rows: List[Dict] = []
    for stratum, sdf in valid_df.groupby("disp_stratum", dropna=True):
        if len(sdf) < n_bins * 2:
            continue
        try:
            sdf = sdf.copy()
            sdf["curvature_bin"] = pd.qcut(
                sdf["joint_curvature"], q=n_bins,
                labels=False, duplicates="drop"
            )
        except Exception:
            continue
        for b, bdf in sdf.groupby("curvature_bin", dropna=True):
            stratum_rows.append({
                "disp_stratum": int(stratum),
                "bin": int(b),
                "mean_decoding_corr": float(bdf["target_corr"].mean()),
                "n": int(len(bdf)),
            })

    if not stratum_rows:
        return pd.DataFrame()

    sdf_all = pd.DataFrame(stratum_rows)
    rows: List[Dict] = []
    for b, g in sdf_all.groupby("bin"):
        rows.append({
            "direction": direction,
            "bin": int(b),
            "n_strata": int(len(g)),
            "n": int(g["n"].sum()),
            "mean_decoding_corr": float(g["mean_decoding_corr"].mean()),
            "sem_decoding_corr": float(g["mean_decoding_corr"].sem()),
        })
    return pd.DataFrame(rows)


def make_curvature_matched_displacement_bins(
    pointwise: pd.DataFrame,
    n_bins: int = 5,
    n_curv_strata: int = 5,
) -> pd.DataFrame:
    """Displacement-binned readout with curvature matched across displacement bins.

    Mirror of make_displacement_matched_curvature_bins — tests whether
    displacement predicts readout independently of curvature.
    """
    valid_df = pointwise[pointwise["valid"]].copy()
    if valid_df.empty:
        return pd.DataFrame()
    direction = valid_df["direction"].iloc[0]

    try:
        valid_df["curv_stratum"] = pd.qcut(
            valid_df["joint_curvature"], q=n_curv_strata,
            labels=False, duplicates="drop"
        )
    except Exception:
        return pd.DataFrame()

    stratum_rows: List[Dict] = []
    for stratum, sdf in valid_df.groupby("curv_stratum", dropna=True):
        if len(sdf) < n_bins * 2:
            continue
        try:
            sdf = sdf.copy()
            sdf["displacement_bin"] = pd.qcut(
                sdf["joint_displacement"], q=n_bins,
                labels=False, duplicates="drop"
            )
        except Exception:
            continue
        for b, bdf in sdf.groupby("displacement_bin", dropna=True):
            stratum_rows.append({
                "curv_stratum": int(stratum),
                "bin": int(b),
                "mean_decoding_corr": float(bdf["target_corr"].mean()),
                "n": int(len(bdf)),
            })

    if not stratum_rows:
        return pd.DataFrame()

    sdf_all = pd.DataFrame(stratum_rows)
    rows: List[Dict] = []
    for b, g in sdf_all.groupby("bin"):
        rows.append({
            "direction": direction,
            "bin": int(b),
            "n_strata": int(len(g)),
            "n": int(g["n"].sum()),
            "mean_decoding_corr": float(g["mean_decoding_corr"].mean()),
            "sem_decoding_corr": float(g["mean_decoding_corr"].sem()),
        })
    return pd.DataFrame(rows)


def make_flanks_shuffle_bins(
    pointwise: pd.DataFrame,
    X_aligned: np.ndarray,
    Y_aligned: np.ndarray,
    n_shuffles: int,
    rng: Optional[np.random.Generator] = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Flanking-word shuffle null for the curvature-binned readout.

    Keeps per-word decoding quality (target_corr / target_mse) fixed.  For
    each shuffle, replaces source and target curvature with
    local_curvature_shuffled_flanks values, recomputes joint_curvature, then
    re-bins the unchanged readout.  Returns one row per (shuffle_index, bin).

    Parameters
    ----------
    pointwise
        Pointwise table from make_pointwise_readout_table for one direction.
        Must contain columns: time_index, target_corr, target_mse, valid.
    X_aligned, Y_aligned
        The lag-aligned source/target trajectory arrays used to produce
        `pointwise`.  Shape [T_aligned, D].
    n_shuffles
        Number of independent flank-shuffle permutations.
    rng
        Random generator.  Created fresh if None.
    n_bins
        Number of curvature quantile bins (passed to make_curvature_bins).
    """
    if rng is None:
        rng = np.random.default_rng()

    direction = pointwise["direction"].iloc[0] if "direction" in pointwise.columns else "unknown"
    t_min = int(pointwise["time_index"].min())  # should be 1

    all_bins: List[pd.DataFrame] = []
    for shuffle_idx in range(n_shuffles):
        c_src = local_curvature_shuffled_flanks(X_aligned, rng)
        c_tgt = local_curvature_shuffled_flanks(Y_aligned, rng)
        joint_c = 0.5 * (c_src + c_tgt)

        pw = pointwise.copy()
        pw["joint_curvature"] = [
            joint_c[int(t) - t_min] if 0 <= int(t) - t_min < len(joint_c) else np.nan
            for t in pw["time_index"]
        ]
        pw["valid"] = pw["valid"] & pw["joint_curvature"].notna()

        b = make_curvature_bins(pw, n_bins=n_bins)
        if not b.empty:
            b.insert(0, "shuffle_index", shuffle_idx)
            all_bins.append(b)

    return pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()


def run_bidirectional_curvature_alignment(
    Z_llm: np.ndarray,
    Z_neural: np.ndarray,
    n_bins: int = 5,
    alphas: Optional[np.ndarray] = None,
    n_splits: int = 5,
    standardize: bool = True,
    shuffle_cv: bool = False,
    random_state: int = 0,
    decoder: str = "ridge",
    elasticnet_l1_ratios: Tuple[float, ...] = (0.05, 0.1, 0.5),
    poisson_alpha: float = 1.0,
    rbf_gamma: Optional[float] = None,
    rbf_n_components: int = 256,
    mlp_hidden_layer_sizes: Tuple[int, ...] = (128,),
    mlp_alpha: float = 1e-3,
    mlp_max_iter: int = 500,
    mlp_standardize_target: bool = False,
    directions: Tuple[str, ...] = ("LLM_to_neural", "neural_to_LLM"),
    target_lag: int = 0,
) -> CurvatureAlignmentOutput:
    """Run cross-space or within-space curvature-conditioned decoding.

    target_lag=1 means source word t predicts target word t+1.
    """
    Z_llm = np.asarray(Z_llm, dtype=float)
    Z_neural = np.asarray(Z_neural, dtype=float)
    if Z_llm.shape[0] != Z_neural.shape[0]:
        raise ValueError(
            f"Z_llm and Z_neural must have same number of rows/timepoints. "
            f"Got {Z_llm.shape[0]} and {Z_neural.shape[0]}."
        )

    k_llm, _ = local_curvature(Z_llm)
    k_neural, _ = local_curvature(Z_neural)

    directions = tuple(directions)
    if not directions:
        raise ValueError("At least one direction must be requested.")
    valid_directions = {
        "LLM_to_neural",
        "neural_to_LLM",
        "LLM_to_LLM",
        "neural_to_neural",
    }
    invalid = set(directions) - valid_directions
    if invalid:
        raise ValueError(f"Unknown directions: {sorted(invalid)}")

    summaries = []
    bins = []
    pointwise = []
    pred_neural = np.full_like(Z_neural, np.nan, dtype=float)
    pred_llm = np.full_like(Z_llm, np.nan, dtype=float)

    if "LLM_to_neural" in directions:
        summary_l2n, bins_l2n, point_l2n, pred_neural = curvature_conditioned_decoding(
            Z_llm,
            Z_neural,
            n_bins=n_bins,
            direction_name="LLM_to_neural",
            alphas=alphas,
            n_splits=n_splits,
            standardize=standardize,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
            decoder=decoder,
            elasticnet_l1_ratios=elasticnet_l1_ratios,
            poisson_alpha=poisson_alpha,
            rbf_gamma=rbf_gamma,
            rbf_n_components=rbf_n_components,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_alpha=mlp_alpha,
            mlp_max_iter=mlp_max_iter,
            mlp_standardize_target=mlp_standardize_target,
            target_lag=target_lag,
        )
        summaries.append(summary_l2n)
        bins.append(bins_l2n)
        pointwise.append(point_l2n)

    if "neural_to_LLM" in directions:
        summary_n2l, bins_n2l, point_n2l, pred_llm = curvature_conditioned_decoding(
            Z_neural,
            Z_llm,
            n_bins=n_bins,
            direction_name="neural_to_LLM",
            alphas=alphas,
            n_splits=n_splits,
            standardize=standardize,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
            decoder=decoder,
            elasticnet_l1_ratios=elasticnet_l1_ratios,
            poisson_alpha=poisson_alpha,
            rbf_gamma=rbf_gamma,
            rbf_n_components=rbf_n_components,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_alpha=mlp_alpha,
            mlp_max_iter=mlp_max_iter,
            mlp_standardize_target=mlp_standardize_target,
            target_lag=target_lag,
        )
        summaries.append(summary_n2l)
        bins.append(bins_n2l)
        pointwise.append(point_n2l)

    if "LLM_to_LLM" in directions:
        summary_l2l, bins_l2l, point_l2l, pred_llm = curvature_conditioned_decoding(
            Z_llm,
            Z_llm,
            n_bins=n_bins,
            direction_name="LLM_to_LLM",
            alphas=alphas,
            n_splits=n_splits,
            standardize=standardize,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
            decoder=decoder,
            elasticnet_l1_ratios=elasticnet_l1_ratios,
            poisson_alpha=poisson_alpha,
            rbf_gamma=rbf_gamma,
            rbf_n_components=rbf_n_components,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_alpha=mlp_alpha,
            mlp_max_iter=mlp_max_iter,
            mlp_standardize_target=mlp_standardize_target,
            target_lag=target_lag,
        )
        summaries.append(summary_l2l)
        bins.append(bins_l2l)
        pointwise.append(point_l2l)

    if "neural_to_neural" in directions:
        summary_n2n, bins_n2n, point_n2n, pred_neural = curvature_conditioned_decoding(
            Z_neural,
            Z_neural,
            n_bins=n_bins,
            direction_name="neural_to_neural",
            alphas=alphas,
            n_splits=n_splits,
            standardize=standardize,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
            decoder=decoder,
            elasticnet_l1_ratios=elasticnet_l1_ratios,
            poisson_alpha=poisson_alpha,
            rbf_gamma=rbf_gamma,
            rbf_n_components=rbf_n_components,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_alpha=mlp_alpha,
            mlp_max_iter=mlp_max_iter,
            mlp_standardize_target=mlp_standardize_target,
            target_lag=target_lag,
        )
        summaries.append(summary_n2n)
        bins.append(bins_n2n)
        pointwise.append(point_n2n)

    return CurvatureAlignmentOutput(
        summary=pd.concat(summaries, ignore_index=True),
        bins=pd.concat(bins, ignore_index=True),
        pointwise=pd.concat(pointwise, ignore_index=True),
        pred_neural=pred_neural,
        pred_llm=pred_llm,
        curvature_llm=k_llm,
        curvature_neural=k_neural,
    )


def save_curvature_alignment_outputs(
    out: CurvatureAlignmentOutput,
    out_prefix: str,
) -> None:
    """Save tabular outputs to CSV and arrays to NPZ."""
    out.summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    out.bins.to_csv(f"{out_prefix}_bins.csv", index=False)
    out.pointwise.to_csv(f"{out_prefix}_pointwise.csv", index=False)
    np.savez_compressed(
        f"{out_prefix}_arrays.npz",
        pred_neural=out.pred_neural,
        pred_llm=out.pred_llm,
        curvature_llm=out.curvature_llm,
        curvature_neural=out.curvature_neural,
    )


if __name__ == "__main__":
    # Minimal smoke test with fake aligned trajectories.
    rng = np.random.default_rng(0)
    Z_llm = rng.normal(size=(100, 64))
    W = rng.normal(size=(64, 20)) * 0.1
    Z_neural = Z_llm @ W + 0.1 * rng.normal(size=(100, 20))

    out = run_bidirectional_curvature_alignment(Z_llm, Z_neural, n_bins=5)
    print(out.summary)
    print(out.bins)
