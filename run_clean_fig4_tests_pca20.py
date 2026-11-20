"""
Clean Figure 4 tests.

This script separates the regression figure into the questions we actually
want to ask:

1. Is curvature or displacement more aligned between semantic and neural space?
2. Around surprising words, is shared trajectory change better attributed to
   curvature or displacement?
3. Does the curvature result survive controls for surprisal, sentence position,
   sentence start, turn changes, speaker role, word length, and frequency?
4. Optional neural-surprisal version: is surprisal predicted by neural curvature
   or neural displacement?

It reuses the same cached embeddings, patients, neural PCA, and LOO-neural
layers as the current geometry-paper Figure 4.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as t_dist

import run_geometry_analysis_pca20 as geom


def safe_pearson(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def centered_displacement(X: np.ndarray) -> np.ndarray:
    """Local displacement aligned to curvature centers; returns len(X)-2."""
    steps = geom.trajectory_displacement(X)
    if len(steps) < 2:
        return np.zeros(0, dtype=float)
    return 0.5 * (steps[:-1] + steps[1:])


def causal_displacement(X: np.ndarray) -> np.ndarray:
    """Step arriving at each word (right-aligned, backward-only); returns len(X)-2.

    value[j] = ||X[j+2] - X[j+1]||  (step arriving at word j+2).
    Matches causal_curvature indexing: both use only words up to j+2,
    so predicting surprisal at j+3 (lag=1) has zero window overlap.
    """
    steps = geom.trajectory_displacement(X)  # n-1 values
    if len(steps) < 2:
        return np.zeros(0, dtype=float)
    return steps[1:]  # n-2 values, arriving at words 2..n-1


def distance_to_turn(is_turn: np.ndarray) -> np.ndarray:
    n = len(is_turn)
    turn_locs = np.where(is_turn)[0]
    if len(turn_locs) == 0:
        return np.full(n, float(n))
    idx = np.searchsorted(turn_locs, np.arange(n))
    before = np.abs(np.arange(n) - turn_locs[np.clip(idx - 1, 0, len(turn_locs) - 1)])
    after = np.abs(np.arange(n) - turn_locs[np.clip(idx, 0, len(turn_locs) - 1)])
    return np.minimum(before, after).astype(float)


def ols_clean(y, predictors: Dict[str, np.ndarray], *, standardize=True) -> Dict:
    names = list(predictors)
    y = np.asarray(y, dtype=float).ravel()
    X_cols = [np.asarray(predictors[n], dtype=float).ravel() for n in names]
    mask = np.isfinite(y)
    for col in X_cols:
        mask &= np.isfinite(col)
    y = y[mask]
    X_cols = [col[mask] for col in X_cols]
    n = len(y)
    if n < max(5, len(names) + 2):
        return dict(n=int(n), r2=float("nan"), r2_adj=float("nan"),
                    betas={k: float("nan") for k in names},
                    pvals={k: float("nan") for k in names})

    if standardize:
        y = (y - y.mean()) / (y.std() + 1e-12)
        X_cols = [(x - x.mean()) / (x.std() + 1e-12) for x in X_cols]

    Xm = np.column_stack([np.ones(n)] + X_cols)
    k = Xm.shape[1]
    b, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    yhat = Xm @ b
    resid = y - yhat
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = float(np.dot(resid, resid))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1) if ss_tot > 0 else float("nan")

    s2 = ss_res / max(n - k, 1)
    try:
        cov_b = s2 * np.linalg.inv(Xm.T @ Xm)
        se_b = np.sqrt(np.diag(cov_b))
        tvals = b / np.where(se_b > 0, se_b, np.nan)
        pvals = 2 * t_dist.sf(np.abs(tvals), df=max(n - k, 1))
    except np.linalg.LinAlgError:
        pvals = np.full(k, np.nan)

    return dict(
        n=int(n),
        r2=float(r2),
        r2_adj=float(r2_adj),
        betas={name: float(beta) for name, beta in zip(names, b[1:])},
        pvals={name: float(p) for name, p in zip(names, pvals[1:])},
    )


def load_existing_layers(model: str, region: str) -> Dict[str, int]:
    path = os.path.join(geom.RESULTS, f"geometry_pca20_{model}_{region}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing existing geometry result {path}. Run run_geometry_analysis.py first."
        )
    with open(path) as f:
        records = json.load(f)
    return {rec["pid"]: int(rec["loo_layer_neu"]) for rec in records}


def build_patient_table(pid: str, region: str, model: str, layer: int, surprisal_lag: int,
                        causal: bool = False) -> Optional[pd.DataFrame]:
    data = geom.load_patient(pid, geom.PATIENTS[pid], region, model)
    if data is None or data.get("neu_vecs") is None:
        return None

    n = data["n_words"]
    Y_neu, n_pcs = geom.fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    E = np.asarray(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    E = geom.sem_pca_reduce(E)

    c_sem = geom.trajectory_curvature(E)
    c_neu = geom.trajectory_curvature(Y_neu)

    if causal:
        # Right-aligned: value[j] is at word j+2, uses only words j..j+2.
        # With lag=1, target = j+3 → zero window overlap with target.
        d_sem = causal_displacement(E)
        d_neu = causal_displacement(Y_neu)
        m = min(len(c_sem), len(d_sem), len(c_neu), len(d_neu))
        center_idx = np.arange(2, 2 + m)   # "current" word = right edge of window
    else:
        d_sem = centered_displacement(E)
        d_neu = centered_displacement(Y_neu)
        m = min(len(c_sem), len(d_sem), len(c_neu), len(d_neu))
        center_idx = np.arange(1, 1 + m)

    target_idx = center_idx + int(surprisal_lag)
    valid_target = (target_idx >= 0) & (target_idx < n)

    sent_pos = np.asarray(data["sent_pos"])
    is_turn = np.asarray(data["is_turn"], dtype=bool)
    is_speaker = np.asarray(data["is_speaker"], dtype=bool)
    word_len = np.asarray(data["word_len"], dtype=float)
    log_freq = np.asarray(data["log_freq"], dtype=float)
    dist_turn = distance_to_turn(is_turn)
    surp = np.asarray(data["surp"], dtype=float)

    center_idx = center_idx[valid_target]
    target_idx = target_idx[valid_target]

    table = pd.DataFrame({
        "pid": pid,
        "region": region,
        "model": model,
        "loo_layer_neu": layer,
        "n_neural_pcs": n_pcs,
        "center_word_index": center_idx,
        "target_word_index": target_idx,
        "surprisal_lag": int(surprisal_lag),
        "surprisal": surp[target_idx],
        "c_sem": c_sem[:m][valid_target],
        "d_sem": d_sem[:m][valid_target],
        "c_neu": c_neu[:m][valid_target],
        "d_neu": d_neu[:m][valid_target],
        "sent_pos": sent_pos[target_idx].astype(float),
        "sentence_start": (sent_pos[target_idx] == 0).astype(float),
        "near_sentence_start": (sent_pos[target_idx] <= 1).astype(float),
        "turn_boundary": is_turn[target_idx].astype(float),
        "turn_boundary_center": is_turn[center_idx].astype(float),
        "is_speaker": is_speaker[target_idx].astype(float),
        "word_len": word_len[target_idx],
        "log_freq": log_freq[target_idx],
        "dist_turn": dist_turn[target_idx],
    })
    return table


BASE_CONTROLS = [
    "surprisal",
    "sent_pos",
    "sentence_start",
    "turn_boundary",
    "is_speaker",
    "word_len",
    "log_freq",
    "dist_turn",
]

NO_SURP_CONTROLS = [c for c in BASE_CONTROLS if c != "surprisal"]


def summarize_patient(table: pd.DataFrame) -> Dict:
    pid = str(table["pid"].iloc[0])
    region = str(table["region"].iloc[0])
    model = str(table["model"].iloc[0])
    layer = int(table["loo_layer_neu"].iloc[0])

    r_dd, p_dd = safe_pearson(table["d_sem"], table["d_neu"])
    r_cc, p_cc = safe_pearson(table["c_sem"], table["c_neu"])

    # Main attribution: cross-space curvature/displacement with controls.
    m_cneu = ols_clean(
        table["c_neu"],
        {"c_sem": table["c_sem"], "d_sem": table["d_sem"],
         **{c: table[c] for c in BASE_CONTROLS}},
    )
    m_dneu = ols_clean(
        table["d_neu"],
        {"d_sem": table["d_sem"], "c_sem": table["c_sem"],
         **{c: table[c] for c in BASE_CONTROLS}},
    )

    # Surprisal interpretation, semantic and optional neural versions.
    m_surp_sem = ols_clean(
        table["surprisal"],
        {"c_sem": table["c_sem"], "d_sem": table["d_sem"],
         **{c: table[c] for c in NO_SURP_CONTROLS}},
    )
    m_surp_neu = ols_clean(
        table["surprisal"],
        {"c_neu": table["c_neu"], "d_neu": table["d_neu"],
         **{c: table[c] for c in NO_SURP_CONTROLS}},
    )
    m_surp_all = ols_clean(
        table["surprisal"],
        {"c_sem": table["c_sem"], "d_sem": table["d_sem"],
         "c_neu": table["c_neu"], "d_neu": table["d_neu"],
         **{c: table[c] for c in NO_SURP_CONTROLS}},
    )

    def b(model_dict, name):
        return model_dict["betas"].get(name, float("nan"))

    def p(model_dict, name):
        return model_dict["pvals"].get(name, float("nan"))

    return dict(
        pid=pid,
        region=region,
        model=model,
        loo_layer_neu=layer,
        n=int(len(table)),
        r_dsem_dneu=r_dd,
        p_dsem_dneu=p_dd,
        r_csem_cneu=r_cc,
        p_csem_cneu=p_cc,
        # Main cross-space models.
        r2_cneu_model=m_cneu["r2"],
        beta_csem_to_cneu=b(m_cneu, "c_sem"),
        p_csem_to_cneu=p(m_cneu, "c_sem"),
        beta_dsem_to_cneu=b(m_cneu, "d_sem"),
        p_dsem_to_cneu=p(m_cneu, "d_sem"),
        beta_surprisal_to_cneu=b(m_cneu, "surprisal"),
        p_surprisal_to_cneu=p(m_cneu, "surprisal"),
        r2_dneu_model=m_dneu["r2"],
        beta_dsem_to_dneu=b(m_dneu, "d_sem"),
        p_dsem_to_dneu=p(m_dneu, "d_sem"),
        beta_csem_to_dneu=b(m_dneu, "c_sem"),
        p_csem_to_dneu=p(m_dneu, "c_sem"),
        beta_surprisal_to_dneu=b(m_dneu, "surprisal"),
        p_surprisal_to_dneu=p(m_dneu, "surprisal"),
        # Surprisal interpretation models.
        r2_surp_sem_model=m_surp_sem["r2"],
        beta_csem_to_surp=b(m_surp_sem, "c_sem"),
        p_csem_to_surp=p(m_surp_sem, "c_sem"),
        beta_dsem_to_surp=b(m_surp_sem, "d_sem"),
        p_dsem_to_surp=p(m_surp_sem, "d_sem"),
        r2_surp_neu_model=m_surp_neu["r2"],
        beta_cneu_to_surp=b(m_surp_neu, "c_neu"),
        p_cneu_to_surp=p(m_surp_neu, "c_neu"),
        beta_dneu_to_surp=b(m_surp_neu, "d_neu"),
        p_dneu_to_surp=p(m_surp_neu, "d_neu"),
        r2_surp_all_model=m_surp_all["r2"],
    )


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "r_dsem_dneu",
        "r_csem_cneu",
        "beta_csem_to_cneu",
        "beta_dsem_to_cneu",
        "beta_dsem_to_dneu",
        "beta_csem_to_dneu",
        "beta_csem_to_surp",
        "beta_dsem_to_surp",
        "beta_cneu_to_surp",
        "beta_dneu_to_surp",
    ]
    rows = []
    for region, sub in df.groupby("region"):
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            rows.append({
                "region": region,
                "metric": metric,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                "median": float(np.median(vals)),
            })
    return pd.DataFrame(rows)


def run(model: str, regions: Iterable[str], output_dir: str, surprisal_lag: int,
        causal: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    all_patient_rows = []
    all_word_tables = []

    for region in regions:
        layers = load_existing_layers(model, region)
        for pid in geom.PATIENTS:
            if region not in geom.PATIENTS[pid] or pid not in layers:
                continue
            print(f"  {model} {region} {pid} layer={layers[pid]}")
            table = build_patient_table(pid, region, model, layers[pid], surprisal_lag, causal=causal)
            if table is None or table.empty:
                print("    skipped")
                continue
            all_word_tables.append(table)
            all_patient_rows.append(summarize_patient(table))

    patient_df = pd.DataFrame(all_patient_rows)
    word_df = pd.concat(all_word_tables, ignore_index=True) if all_word_tables else pd.DataFrame()
    summary_df = group_summary(patient_df) if not patient_df.empty else pd.DataFrame()

    stem = f"clean_fig4_{model}_lag{surprisal_lag}{'_causal' if causal else ''}"
    patient_path = os.path.join(output_dir, f"{stem}_patient_summary.csv")
    word_path = os.path.join(output_dir, f"{stem}_word_table.csv")
    group_path = os.path.join(output_dir, f"{stem}_group_summary.csv")
    patient_df.to_csv(patient_path, index=False)
    word_df.to_csv(word_path, index=False)
    summary_df.to_csv(group_path, index=False)

    print(f"\nSaved patient summary: {patient_path}")
    print(f"Saved word table:      {word_path}")
    print(f"Saved group summary:   {group_path}")
    return patient_df, summary_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run clean Figure 4 curvature-vs-displacement tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="llama-3.1-8b", choices=list(geom.MODELS))
    p.add_argument("--regions", nargs="+", default=geom.REGIONS)
    p.add_argument("--output_dir", default=os.path.join(geom.RESULTS, "clean_fig4_pca20"))
    p.add_argument("--surprisal_lag", type=int, default=1,
                   help="Use surprisal at center_word + lag. lag=1 matches current fig4_regression.")
    p.add_argument("--causal", action="store_true",
                   help="Use backward-only (causal) curvature and displacement: geometry at t "
                        "uses only words t-2..t, so with lag=1 the target (t+1) is never in "
                        "the predictor window.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    patient_df, summary_df = run(args.model, args.regions, args.output_dir, args.surprisal_lag,
                                 causal=args.causal)

    if not patient_df.empty:
        cols = [
            "pid", "region", "n",
            "r_dsem_dneu", "r_csem_cneu",
            "beta_csem_to_cneu", "p_csem_to_cneu",
            "beta_dsem_to_cneu", "p_dsem_to_cneu",
            "beta_csem_to_surp", "p_csem_to_surp",
            "beta_dsem_to_surp", "p_dsem_to_surp",
            "beta_cneu_to_surp", "p_cneu_to_surp",
            "beta_dneu_to_surp", "p_dneu_to_surp",
        ]
        print("\nPatient-level key columns:")
        print(patient_df[cols].to_string(index=False))

    if not summary_df.empty:
        print("\nGroup summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

